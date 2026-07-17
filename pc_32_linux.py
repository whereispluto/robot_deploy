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
from datetime import datetime
from collections.abc import Callable
from pathlib import Path
from typing import TextIO

from pc_32 import RobotBridge, RobotCommand, RobotState, SerialBridge


DEFAULT_PORT = "/dev/ttyACM0"
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
		choices=("passthrough", "hold", "motion-test"),
		default="hold",
		help=(
			"Local policy: hold, passthrough, or a sequential six-joint "
			"2-degree motion test"
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

	try:
		port = resolve_port(args.port)
		if not os.path.exists(port):
			raise RuntimeError(
				f"Serial device {port!r} does not exist. Connect the STM32, "
				"use `--list-ports`, or pass `--port auto`."
			)

		serial_bridge = SerialBridge(port=port, baudrate=args.baudrate)
		if args.policy == "passthrough":
			policy = make_passthrough_policy(args.target_pos)
		elif args.policy == "motion-test":
			policy = make_motion_test_policy()
		else:
			policy = None
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
