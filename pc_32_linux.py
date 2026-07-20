#!/usr/bin/env python3
"""Linux entry point for the PC <-> STM32 USB CDC bridge.

The frame protocol and bridge implementation are shared with ``pc_32.py``.
This entry point adds Linux-oriented serial-port defaults, device discovery,
and useful error messages for missing devices and serial permissions.
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import sys
import time
from collections import deque
from datetime import datetime
from collections.abc import Callable
from pathlib import Path
from typing import TextIO

from pc_32 import (
	PolicyOutput,
	RobotBridge,
	RobotCommand,
	RobotState,
	SerialBridge,
	clamp,
)


DEFAULT_PORT = "/dev/ttyACM0"
DEFAULT_POLICY_ROOT = Path(
	"/home/cx/mjlab-recovered/logs/rsl_rl/custom_biped_velocity_nolinvel"
)
LINUX_PORT_PATTERNS = (
	"/dev/ttyACM*",       # USB CDC ACM devices (typical for STM32)
	"/dev/ttyUSB*",       # USB-to-serial adapters
	"/dev/serial/by-id/*",  # Stable names managed by udev
)

JOINT_LIMITS_DEG = (
	(-60.0, 60.0),    # FDCAN1 ID 1
	(-120.0, 0.0),    # FDCAN1 ID 2
	(-45.0, 45.0),    # FDCAN1 ID 3
	(-60.0, 60.0),    # FDCAN2 ID 1
	(-120.0, 0.0),    # FDCAN2 ID 2
	(-45.0, 45.0),    # FDCAN2 ID 3
)
MOTION_TEST_AMPLITUDE_DEG = 20.0
MOTION_TEST_INITIAL_HOLD_S = 2.0
MOTION_TEST_PHASE_S = 1.5

POLICY_OBSERVATION_SIZE = 65
POLICY_ACTION_SIZE = 6
POLICY_HISTORY_LENGTH = 4
POLICY_CONTROL_PERIOD_S = 0.02
POLICY_GAIT_CYCLE_S = 1.0
IMU_REQUIRED_STATUS = 0x0003

# Joint order is identical to the STM32 state/command frame:
# left hip, left knee, left ankle, right hip, right knee, right ankle.
DEFAULT_JOINT_POS_RAD = tuple(
	math.radians(value) for value in (10.0, -20.0, 10.0, 10.0, -20.0, 10.0)
)
ACTION_SCALE_RAD = (
	0.2473661710,
	0.3957858736,
	0.2671554647,
	0.2473661710,
	0.3957858736,
	0.2671554647,
)


class TeeOutput:
	"""Write output to both the terminal and a log file."""

	def __init__(self, terminal: TextIO, log_file: TextIO) -> None:
		self.terminal = terminal
		self.log_file = log_file

	def write(self, text: str) -> int:
		self.terminal.write(text)
		self.log_file.write(text)
		return len(text)

	def flush(self) -> None:
		self.terminal.flush()
		self.log_file.flush()

	def isatty(self) -> bool:
		return self.terminal.isatty()


def run_with_log() -> int:
	"""Run the bridge while saving stdout and stderr to a timestamped file."""
	log_dir = Path(__file__).resolve().parent / "log"
	log_dir.mkdir(parents=True, exist_ok=True)
	start_time = datetime.now()
	log_path = log_dir / f"{start_time:%Y-%m-%d_%H-%M-%S}.txt"

	original_stdout = sys.stdout
	original_stderr = sys.stderr
	with log_path.open("w", encoding="utf-8", buffering=1) as log_file:
		sys.stdout = TeeOutput(original_stdout, log_file)
		sys.stderr = TeeOutput(original_stderr, log_file)
		try:
			print(f"Log file: {log_path}")
			return main()
		finally:
			sys.stdout = original_stdout
			sys.stderr = original_stderr


def find_linux_serial_ports() -> list[str]:
	"""Return likely Linux serial devices, with STM32-style ports first."""
	ports: list[str] = []
	for pattern in LINUX_PORT_PATTERNS:
		ports.extend(sorted(glob.glob(pattern)))
	return list(dict.fromkeys(ports))


def resolve_port(port: str) -> str:
	"""Resolve ``auto`` to the first likely USB serial device."""
	if port != "auto":
		return port

	ports = find_linux_serial_ports()
	if not ports:
		raise RuntimeError(
			"No Linux USB serial port was found. Connect the STM32 and check "
			"`ls /dev/ttyACM* /dev/ttyUSB*`."
		)
	return ports[0]


def parse_target_pos(value: str) -> list[float]:
	values = [part.strip() for part in value.split(",")]
	if len(values) != 6:
		raise argparse.ArgumentTypeError(
			"target position must contain exactly 6 comma-separated values"
		)
	try:
		return [float(part) for part in values]
	except ValueError as exc:
		raise argparse.ArgumentTypeError(
			"target position values must be numbers"
		) from exc


def parse_velocity_command(value: str) -> list[float]:
	values = [part.strip() for part in value.split(",")]
	if len(values) != 3:
		raise argparse.ArgumentTypeError(
			"velocity command must contain VX,VY,WZ"
		)
	try:
		command = [float(part) for part in values]
	except ValueError as exc:
		raise argparse.ArgumentTypeError(
			"velocity command values must be numbers"
		) from exc
	if not all(math.isfinite(item) for item in command):
		raise argparse.ArgumentTypeError(
			"velocity command values must be finite"
		)
	return command


def find_latest_onnx(policy_root: Path = DEFAULT_POLICY_ROOT) -> Path:
	"""Return the most recently modified exported ONNX policy."""
	models = list(policy_root.glob("*/*.onnx"))
	if not models:
		raise RuntimeError(f"No ONNX policy found below {policy_root}")
	return max(models, key=lambda path: path.stat().st_mtime)


class OnnxPolicy:
	"""Build the 65-D mjlab actor observation and run ONNX inference."""

	def __init__(self, model_path: Path, velocity_command: list[float]) -> None:
		try:
			import numpy as np
			import onnxruntime as ort
		except ImportError as exc:
			raise RuntimeError(
				"ONNX deployment requires numpy and onnxruntime. Install them with "
				"`python -m pip install numpy onnxruntime`."
			) from exc

		self._np = np
		self.model_path = model_path.resolve()
		if not self.model_path.is_file():
			raise RuntimeError(f"ONNX model does not exist: {self.model_path}")

		self._session = ort.InferenceSession(
			str(self.model_path), providers=["CPUExecutionProvider"]
		)
		inputs = self._session.get_inputs()
		outputs = self._session.get_outputs()
		if len(inputs) != 1 or inputs[0].shape != [1, POLICY_OBSERVATION_SIZE]:
			raise RuntimeError(
				f"Expected one ONNX input with shape [1, {POLICY_OBSERVATION_SIZE}], "
				f"got {[(item.name, item.shape) for item in inputs]}"
			)
		if len(outputs) != 1 or outputs[0].shape != [1, POLICY_ACTION_SIZE]:
			raise RuntimeError(
				f"Expected one ONNX output with shape [1, {POLICY_ACTION_SIZE}], "
				f"got {[(item.name, item.shape) for item in outputs]}"
			)

		self._input_name = inputs[0].name
		self._output_name = outputs[0].name
		self._velocity_command = np.asarray(velocity_command, dtype=np.float32)
		self._default_joint_pos = np.asarray(
			DEFAULT_JOINT_POS_RAD, dtype=np.float32
		)
		self._action_scale = np.asarray(ACTION_SCALE_RAD, dtype=np.float32)
		self._joint_pos_history: deque[object] = deque(
			maxlen=POLICY_HISTORY_LENGTH
		)
		self._joint_vel_history: deque[object] = deque(
			maxlen=POLICY_HISTORY_LENGTH
		)
		self._step_count = 0
		self._waiting_for_imu = False

		metadata = self._session.get_modelmeta().custom_metadata_map
		observation_names = metadata.get("observation_names", "")
		expected_names = (
			"base_ang_vel,projected_gravity,joint_pos,joint_vel,"
			"actions,command,gait_phase"
		)
		if observation_names and observation_names != expected_names:
			raise RuntimeError(
				"ONNX observation order does not match this deployment code: "
				f"{observation_names}"
			)
		metadata_scale = metadata.get("action_scale", "")
		if metadata_scale:
			exported_scale = np.fromstring(metadata_scale, sep=",", dtype=np.float32)
			if exported_scale.shape != (POLICY_ACTION_SIZE,) or not np.allclose(
				exported_scale, self._action_scale, atol=5.0e-4, rtol=0.0
			):
				raise RuntimeError(
					"ONNX action scale does not match this deployment code: "
					f"{metadata_scale}"
				)

		print(
			f"Loaded ONNX policy: {self.model_path}\n"
			f"  input={self._input_name}[1,{POLICY_OBSERVATION_SIZE}] "
			f"output={self._output_name}[1,{POLICY_ACTION_SIZE}]\n"
			f"  velocity_command={velocity_command}",
			flush=True,
		)

	def _append_history(self, history: deque[object], value: object) -> None:
		if not history:
			for _ in range(POLICY_HISTORY_LENGTH):
				history.append(value.copy())
		else:
			history.append(value.copy())

	def _hold_current_position(
		self, state: RobotState, last_action: list[float]
	) -> PolicyOutput:
		return PolicyOutput(
			command=RobotCommand(target_joint_pos=list(state.joint_pos)),
			raw_action=list(last_action),
		)

	def __call__(
		self, state: RobotState, last_action: list[float]
	) -> PolicyOutput:
		np = self._np
		if (state.status & IMU_REQUIRED_STATUS) != IMU_REQUIRED_STATUS:
			if not self._waiting_for_imu:
				print(
					"ONNX policy waiting for valid gyro and quaternion; holding "
					"the measured joint positions.",
					flush=True,
				)
				self._waiting_for_imu = True
			return self._hold_current_position(state, last_action)
		if self._waiting_for_imu:
			print("ONNX policy: IMU valid, inference started.", flush=True)
			self._waiting_for_imu = False

		joint_pos_deg = np.asarray(state.joint_pos, dtype=np.float32)
		joint_vel_deg_s = np.asarray(state.joint_vel, dtype=np.float32)
		base_ang_vel = np.asarray(state.base_ang_vel, dtype=np.float32)
		projected_gravity = np.asarray(
			state.gravity_orientation, dtype=np.float32
		)
		previous_action = np.asarray(last_action, dtype=np.float32)
		state_values = (
			joint_pos_deg,
			joint_vel_deg_s,
			base_ang_vel,
			projected_gravity,
			previous_action,
		)
		if any(array.shape != (size,) for array, size in zip(
			state_values, (6, 6, 3, 3, 6), strict=True
		)):
			raise RuntimeError("Robot state dimensions do not match the policy")
		if not all(np.isfinite(array).all() for array in state_values):
			raise RuntimeError("Robot state contains NaN or infinity")

		joint_pos_rel = np.deg2rad(joint_pos_deg) - self._default_joint_pos
		joint_vel = np.deg2rad(joint_vel_deg_s)
		self._append_history(self._joint_pos_history, joint_pos_rel)
		self._append_history(self._joint_vel_history, joint_vel)

		phase = (
			2.0
			* math.pi
			* self._step_count
			* POLICY_CONTROL_PERIOD_S
			/ POLICY_GAIT_CYCLE_S
		)
		gait_phase = np.asarray(
			(math.sin(phase), math.cos(phase)), dtype=np.float32
		)
		observation = np.concatenate(
			(
				base_ang_vel,
				projected_gravity,
				np.concatenate(tuple(self._joint_pos_history)),
				np.concatenate(tuple(self._joint_vel_history)),
				previous_action,
				self._velocity_command,
				gait_phase,
			)
		).astype(np.float32, copy=False)
		if observation.shape != (POLICY_OBSERVATION_SIZE,):
			raise RuntimeError(
				f"Built observation has shape {observation.shape}, expected "
				f"({POLICY_OBSERVATION_SIZE},)"
			)

		raw_action = self._session.run(
			[self._output_name],
			{self._input_name: observation.reshape(1, -1)},
		)[0][0]
		if raw_action.shape != (POLICY_ACTION_SIZE,) or not np.isfinite(raw_action).all():
			raise RuntimeError("ONNX policy returned an invalid action")

		target_rad = self._default_joint_pos + raw_action * self._action_scale
		target_deg = np.rad2deg(target_rad)
		clamped_target_deg = [
			clamp(float(value), lower, upper)
			for value, (lower, upper) in zip(
				target_deg, JOINT_LIMITS_DEG, strict=True
			)
		]
		self._step_count += 1
		return PolicyOutput(
			command=RobotCommand(target_joint_pos=clamped_target_deg),
			raw_action=[float(value) for value in raw_action],
		)


def make_passthrough_policy(
	target_pos: list[float] | None,
) -> Callable[[RobotState, list[float]], RobotCommand]:
	def policy(state: RobotState, last_action: list[float]) -> RobotCommand:
		del last_action
		target = target_pos if target_pos is not None else state.joint_pos
		return RobotCommand(target_joint_pos=list(target), seq=0, flags=0)

	return policy


def make_motion_test_policy() -> Callable[[RobotState, list[float]], RobotCommand]:
	"""Move each joint a small amount in turn, then return to its start."""
	initial_pos: list[float] | None = None
	test_offsets: list[float] = []
	start_time = 0.0
	last_phase = ""

	def announce(phase: str) -> None:
		nonlocal last_phase
		if phase != last_phase:
			print(f"motion_test: {phase}", flush=True)
			last_phase = phase

	def policy(state: RobotState, last_action: list[float]) -> RobotCommand:
		nonlocal initial_pos, test_offsets, start_time
		del last_action

		if initial_pos is None:
			if len(state.joint_pos) != len(JOINT_LIMITS_DEG):
				raise RuntimeError("motion test requires exactly six joint positions")

			initial_pos = list(state.joint_pos)
			for index, (position, limits) in enumerate(
				zip(initial_pos, JOINT_LIMITS_DEG), start=1
			):
				lower, upper = limits
				if not math.isfinite(position):
					raise RuntimeError(
						f"joint {index} position is not finite: {position}"
					)
				if not lower <= position <= upper:
					raise RuntimeError(
						f"joint {index} position {position:.3f} deg is outside "
						f"[{lower:.1f}, {upper:.1f}] deg"
					)

				if position + MOTION_TEST_AMPLITUDE_DEG <= upper:
					test_offsets.append(MOTION_TEST_AMPLITUDE_DEG)
				elif position - MOTION_TEST_AMPLITUDE_DEG >= lower:
					test_offsets.append(-MOTION_TEST_AMPLITUDE_DEG)
				else:
					raise RuntimeError(
						f"joint {index} has insufficient room for the motion test"
					)

			start_time = time.monotonic()
			print(
				"motion_test: initial joint positions="
				f"{[round(value, 3) for value in initial_pos]}",
				flush=True,
			)

		assert initial_pos is not None
		target = list(initial_pos)
		elapsed = time.monotonic() - start_time

		if elapsed < MOTION_TEST_INITIAL_HOLD_S:
			announce("holding initial positions")
			return RobotCommand(target_joint_pos=target)

		test_elapsed = elapsed - MOTION_TEST_INITIAL_HOLD_S
		joint_period = 2.0 * MOTION_TEST_PHASE_S
		joint_index = int(test_elapsed // joint_period)

		if joint_index >= len(initial_pos):
			announce("complete; holding initial positions (press Ctrl+C to stop)")
			return RobotCommand(target_joint_pos=target)

		phase_elapsed = test_elapsed - joint_index * joint_period
		if phase_elapsed < MOTION_TEST_PHASE_S:
			lower, upper = JOINT_LIMITS_DEG[joint_index]
			target[joint_index] = max(
				lower,
				min(upper, initial_pos[joint_index] + test_offsets[joint_index]),
			)
			announce(
				f"joint {joint_index + 1} -> {target[joint_index]:.3f} deg"
			)
		else:
			announce(f"joint {joint_index + 1} returning to start")

		return RobotCommand(target_joint_pos=target)

	return policy


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		description="Linux PC <-> STM32 USB CDC bridge"
	)
	parser.add_argument(
		"--port",
		default=DEFAULT_PORT,
		help=(
			"Linux serial device (default: /dev/ttyACM0); use 'auto' to "
			"select the first /dev/ttyACM*, /dev/ttyUSB*, or by-id device"
		),
	)
	parser.add_argument("--baudrate", type=int, default=921600)
	parser.add_argument(
		"--rate", type=float, default=200.0, help="Bridge loop rate in Hz"
	)
	parser.add_argument(
		"--policy",
		choices=("passthrough", "hold", "motion-test", "onnx"),
		default="hold",
		help=(
			"Local policy: hold, passthrough, sequential six-joint motion "
			"test, or ONNX actor inference"
		),
	)
	parser.add_argument(
		"--target-pos",
		"--target_pos",
		dest="target_pos",
		type=parse_target_pos,
		default=None,
		metavar="P1,P2,P3,P4,P5,P6",
		help="Six joint target positions; used by the passthrough policy",
	)
	parser.add_argument(
		"--model",
		type=Path,
		default=None,
		help=(
			"ONNX model path. With --policy onnx, the newest model below "
			f"{DEFAULT_POLICY_ROOT} is used when omitted."
		),
	)
	parser.add_argument(
		"--command",
		type=parse_velocity_command,
		default=[0.15, 0.0, 0.0],
		metavar="VX,VY,WZ",
		help=(
			"Velocity command supplied to the ONNX actor in m/s,m/s,rad/s "
			"(default: 0.15,0,0)"
		),
	)
	parser.add_argument(
		"--list-ports",
		action="store_true",
		help="List likely Linux USB serial devices and exit",
	)
	return parser


def main() -> int:
	parser = build_parser()
	args = parser.parse_args()

	if not sys.platform.startswith("linux"):
		print(
			f"warning: this entry point is configured for Linux (current: {sys.platform})",
			file=sys.stderr,
		)

	if args.list_ports:
		ports = find_linux_serial_ports()
		if ports:
			print("\n".join(ports))
			return 0
		print("No likely Linux USB serial devices found.", file=sys.stderr)
		return 1

	if args.rate <= 0:
		parser.error("--rate must be greater than zero")
	if args.target_pos is not None and args.policy != "passthrough":
		parser.error("--target-pos requires --policy passthrough")
	if args.model is not None and args.policy != "onnx":
		parser.error("--model requires --policy onnx")

	try:
		if args.policy == "passthrough":
			policy = make_passthrough_policy(args.target_pos)
		elif args.policy == "motion-test":
			policy = make_motion_test_policy()
		elif args.policy == "onnx":
			model_path = args.model if args.model is not None else find_latest_onnx()
			policy = OnnxPolicy(model_path, args.command)
		else:
			policy = None

		port = resolve_port(args.port)
		if not os.path.exists(port):
			raise RuntimeError(
				f"Serial device {port!r} does not exist. Connect the STM32, "
				"use `--list-ports`, or pass `--port auto`."
			)

		serial_bridge = SerialBridge(port=port, baudrate=args.baudrate)
		print(
			f"Connected to {port} at {args.baudrate} baud; "
			f"loop rate {args.rate:g} Hz."
		)
		RobotBridge(serial_bridge=serial_bridge, policy_fn=policy).run_forever(
			rate_hz=args.rate
		)
		return 0
	except PermissionError as exc:
		print(
			f"Permission denied opening the serial port: {exc}\n"
			"Add the current user to the dialout group with:\n"
			"  sudo usermod -aG dialout $USER\n"
			"Then log out and log back in.",
			file=sys.stderr,
		)
		return 2
	except OSError as exc:
		if getattr(exc, "errno", None) == 13 or "Permission denied" in str(exc):
			print(
				f"Permission denied opening the serial port: {exc}\n"
				"Add the current user to the dialout group with:\n"
				"  sudo usermod -aG dialout $USER\n"
				"Then log out and log back in.",
				file=sys.stderr,
			)
			return 2
		print(f"error: {exc}", file=sys.stderr)
		return 1
	except RuntimeError as exc:
		print(f"error: {exc}", file=sys.stderr)
		return 1


if __name__ == "__main__":
	raise SystemExit(run_with_log())
