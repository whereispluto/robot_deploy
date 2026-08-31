#!/usr/bin/env python3
"""Run the six-joint ONNX policy in a standalone MuJoCo simulation.

The policy observation, action mapping, startup sequence, and command-line
arguments are shared with ``pc_32_linux.py``.  ``--port`` and ``--baudrate``
are accepted for command compatibility but no serial device is opened.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from pc_32 import RobotState
from pc_32_linux import (
	DEFAULT_JOINT_POS_RAD,
	IMU_REQUIRED_STATUS,
	MOTOR_POSITION_DAMPING,
	MOTOR_POSITION_STIFFNESS,
	MOTOR_RATED_TORQUE_NM,
	OnnxPolicy,
	POLICY_ACTION_SIZE,
	POLICY_CONTROL_PERIOD_S,
	POLICY_JOINT_NAMES,
	TeeOutput,
	build_parser as build_linux_parser,
	find_latest_onnx,
)


DEFAULT_MJLAB_ROOT = Path("/home/cx/mjlab-recovered")
DEFAULT_MUJOCO_PYTHON = DEFAULT_MJLAB_ROOT / ".venv" / "bin" / "python"
DEFAULT_ROBOT_XML = (
	DEFAULT_MJLAB_ROOT
	/ "src/mjlab/asset_zoo/robots/custom_biped/mjcf/biped.xml"
)

PHYSICS_TIMESTEP_S = 0.005
PHYSICS_STEPS_PER_CONTROL = 4
JOINT_NAMES = POLICY_JOINT_NAMES
INITIAL_BASE_HEIGHT_M = 0.522

# Radian-based gains equivalent to the M4438_30 int16 motor codes (19, 19)
# produced when the firmware calls the FDCAN API with Kp=Kd=1.0.
MOTOR_KP = MOTOR_POSITION_STIFFNESS
MOTOR_KD = MOTOR_POSITION_DAMPING
MOTOR_EFFORT_LIMIT_NM = MOTOR_RATED_TORQUE_NM
MOTOR_STALL_TORQUE_NM = 10.0
MOTOR_NO_LOAD_SPEED_RAD_S = 160.0 * 2.0 * math.pi / 60.0


def _load_runtime() -> tuple[Any, Any]:
	"""Import simulation dependencies, using the mjlab venv when necessary."""
	try:
		import mujoco
		import mujoco.viewer
		import numpy as np
		import onnxruntime as ort
	except ImportError as exc:
		candidate = DEFAULT_MUJOCO_PYTHON
		try:
			current_python = Path(sys.executable).resolve()
			candidate_python = candidate.resolve()
		except OSError:
			current_python = Path(sys.executable)
			candidate_python = candidate
		if candidate.is_file() and current_python != candidate_python:
			print(
				f"Simulation packages are unavailable in {sys.executable}; "
				f"restarting with {candidate}.",
				flush=True,
			)
			os.execv(
				str(candidate),
				[str(candidate), str(Path(__file__).resolve()), *sys.argv[1:]],
			)
		raise RuntimeError(
			"MuJoCo simulation requires numpy, mujoco, and onnxruntime. "
			"Install them with `python -m pip install numpy mujoco onnxruntime`."
		) from exc
	del ort
	return np, mujoco


def build_parser() -> argparse.ArgumentParser:
	parser = build_linux_parser()
	# The simulated robot is reset directly into the training home pose, so it can
	# start inference immediately just like MjLab play.  The real-robot entry point
	# retains its safe move-and-hold startup defaults.
	parser.set_defaults(startup_move_time=0.0, startup_hold_time=0.0)
	parser.description = (
		"Standalone MuJoCo sim2sim runner for the six-joint ONNX policy"
	)
	parser.add_argument(
		"--robot-xml",
		type=Path,
		default=DEFAULT_ROBOT_XML,
		help=f"Robot MJCF path (default: {DEFAULT_ROBOT_XML})",
	)
	parser.add_argument(
		"--headless",
		action="store_true",
		help="Run without opening the MuJoCo viewer",
	)
	parser.add_argument(
		"--duration",
		type=float,
		default=None,
		metavar="SECONDS",
		help=(
			"Stop after this many seconds of simulation time "
			"(default: run until closed)"
		),
	)
	parser.add_argument(
		"--no-realtime",
		action="store_true",
		help="Do not pace simulation time to wall-clock time",
	)
	parser.add_argument(
		"--reset-on-fall",
		action="store_true",
		help="Reset when the base is too low or its pitch exceeds 50 degrees",
	)
	parser.add_argument(
		"--log-interval",
		type=float,
		default=1.0,
		metavar="SECONDS",
		help="Simulation-time interval between state summaries; 0 disables them",
	)
	return parser


def _add_torque_motor(spec: Any, mujoco: Any, joint_name: str) -> None:
	spec.add_actuator(
		name=joint_name,
		target=joint_name,
		trntype=mujoco.mjtTrn.mjTRN_JOINT,
		dyntype=mujoco.mjtDyn.mjDYN_NONE,
		gaintype=mujoco.mjtGain.mjGAIN_FIXED,
		biastype=mujoco.mjtBias.mjBIAS_NONE,
		gear=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
		ctrllimited=True,
		ctrlrange=[-MOTOR_EFFORT_LIMIT_NM, MOTOR_EFFORT_LIMIT_NM],
		forcelimited=True,
		forcerange=[-MOTOR_EFFORT_LIMIT_NM, MOTOR_EFFORT_LIMIT_NM],
	)


def build_model(robot_xml: Path, mujoco: Any) -> Any:
	if not robot_xml.is_file():
		raise RuntimeError(f"Robot MJCF does not exist: {robot_xml}")

	spec = mujoco.MjSpec.from_file(str(robot_xml.resolve()))
	spec.option.timestep = PHYSICS_TIMESTEP_S
	spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
	spec.option.solver = mujoco.mjtSolver.mjSOL_NEWTON
	spec.option.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
	spec.option.jacobian = mujoco.mjtJacobian.mjJAC_AUTO
	spec.option.iterations = 10
	spec.option.tolerance = 1.0e-8
	spec.option.ls_iterations = 20
	spec.option.ls_tolerance = 0.01
	spec.option.ccd_iterations = 50
	spec.worldbody.add_geom(
		name="terrain",
		type=mujoco.mjtGeom.mjGEOM_PLANE,
		pos=[0.0, 0.0, 0.0],
		size=[0.0, 0.0, 0.01],
		condim=3,
		friction=[1.0, 0.005, 0.0001],
		solref=[0.02, 1.0],
		solimp=[0.9, 0.95, 0.001, 0.5, 2.0],
		rgba=[0.2, 0.3, 0.4, 1.0],
		group=0,
	)
	for joint_name in JOINT_NAMES:
		_add_torque_motor(spec, mujoco, joint_name)

	model = spec.compile()
	if not math.isclose(
		model.opt.timestep, PHYSICS_TIMESTEP_S, rel_tol=0.0, abs_tol=1.0e-12
	):
		raise RuntimeError(f"Unexpected MuJoCo timestep: {model.opt.timestep}")
	if PHYSICS_STEPS_PER_CONTROL * model.opt.timestep != POLICY_CONTROL_PERIOD_S:
		raise RuntimeError("Physics and ONNX policy control periods do not match")
	expected_options = (
		("integrator", model.opt.integrator, mujoco.mjtIntegrator.mjINT_IMPLICITFAST),
		("solver", model.opt.solver, mujoco.mjtSolver.mjSOL_NEWTON),
		("cone", model.opt.cone, mujoco.mjtCone.mjCONE_PYRAMIDAL),
		("jacobian", model.opt.jacobian, mujoco.mjtJacobian.mjJAC_AUTO),
	)
	for option_name, actual, expected in expected_options:
		if actual != expected:
			raise RuntimeError(
				f"Unexpected MuJoCo {option_name}: {actual} (expected {expected})"
			)
	return model


def _joint_addresses(model: Any) -> tuple[list[int], list[int]]:
	qpos_addresses: list[int] = []
	qvel_addresses: list[int] = []
	for name in JOINT_NAMES:
		joint_id = model.joint(name).id
		qpos_addresses.append(int(model.jnt_qposadr[joint_id]))
		qvel_addresses.append(int(model.jnt_dofadr[joint_id]))
	return qpos_addresses, qvel_addresses


def reset_data(model: Any, data: Any, mujoco: Any) -> None:
	mujoco.mj_resetData(model, data)
	initial_positions = {
		"base_x": 0.0,
		"base_z": INITIAL_BASE_HEIGHT_M,
		"base_pitch": 0.0,
		**dict(zip(JOINT_NAMES, DEFAULT_JOINT_POS_RAD, strict=True)),
	}
	for joint_name, position in initial_positions.items():
		joint_id = model.joint(joint_name).id
		data.qpos[int(model.jnt_qposadr[joint_id])] = position
	data.ctrl[:] = 0.0
	mujoco.mj_forward(model, data)


def dc_motor_torque(np: Any, target: Any, position: Any, velocity: Any) -> Any:
	"""Apply the same PD law and torque-speed clipping as MjLab DcMotorActuator."""
	desired = MOTOR_KP * (target - position) - MOTOR_KD * velocity
	corner_speed = MOTOR_NO_LOAD_SPEED_RAD_S * (
		1.0 + MOTOR_EFFORT_LIMIT_NM / MOTOR_STALL_TORQUE_NM
	)
	clipped_velocity = np.clip(velocity, -corner_speed, corner_speed)
	torque_speed_upper = MOTOR_STALL_TORQUE_NM * (
		1.0 - clipped_velocity / MOTOR_NO_LOAD_SPEED_RAD_S
	)
	torque_speed_lower = MOTOR_STALL_TORQUE_NM * (
		-1.0 - clipped_velocity / MOTOR_NO_LOAD_SPEED_RAD_S
	)
	upper = np.minimum(torque_speed_upper, MOTOR_EFFORT_LIMIT_NM)
	lower = np.maximum(torque_speed_lower, -MOTOR_EFFORT_LIMIT_NM)
	return np.clip(desired, lower, upper)


def make_robot_state(
	data: Any,
	np: Any,
	qpos_addresses: list[int],
	qvel_addresses: list[int],
	command: list[float],
) -> RobotState:
	joint_pos_deg = np.rad2deg(data.qpos[qpos_addresses])
	joint_vel_deg_s = np.rad2deg(data.qvel[qvel_addresses])
	base_quat = np.asarray(data.sensor("orientation").data, dtype=np.float64)
	base_ang_vel = np.asarray(
		data.sensor("angular-velocity").data, dtype=np.float64
	)
	base_lin_vel = np.asarray(data.sensor("linear-velocity").data, dtype=np.float64)
	values = (joint_pos_deg, joint_vel_deg_s, base_quat, base_ang_vel, base_lin_vel)
	if not all(np.isfinite(value).all() for value in values):
		raise RuntimeError("MuJoCo produced NaN or infinity in the policy state")
	return RobotState(
		joint_pos=joint_pos_deg.tolist(),
		joint_vel=joint_vel_deg_s.tolist(),
		base_lin_vel=base_lin_vel.tolist(),
		base_ang_vel=base_ang_vel.tolist(),
		base_quat=base_quat.tolist(),
		cmd=list(command),
		timestamp_ms=int(round(data.time * 1000.0)),
		status=IMU_REQUIRED_STATUS,
	)


def log_robot_state(state: RobotState) -> None:
	"""Log the policy input state using the same format as sim2real."""
	print(
		f"joint_pos={state.joint_pos} "
		f"joint_vel={state.joint_vel} "
		f"imu_ang_vel={state.base_ang_vel} "
		f"imu_quat_wxyz={state.base_quat} "
		f"status=0x{state.status:04x}",
		flush=True,
	)


def _has_fallen(model: Any, data: Any) -> bool:
	base_height = float(data.body("base_link").xpos[2])
	pitch_id = model.joint("base_pitch").id
	pitch = float(data.qpos[int(model.jnt_qposadr[pitch_id])])
	return base_height < 0.01 or abs(pitch) > math.radians(50.0)


def run_simulation(args: argparse.Namespace, np: Any, mujoco: Any) -> int:
	model_path = args.model if args.model is not None else find_latest_onnx()
	model = build_model(args.robot_xml, mujoco)
	data = mujoco.MjData(model)
	qpos_addresses, qvel_addresses = _joint_addresses(model)
	reset_data(model, data, mujoco)

	def new_policy() -> OnnxPolicy:
		return OnnxPolicy(
			model_path,
			args.command,
			startup_move_duration=args.startup_move_time,
			startup_hold_duration=args.startup_hold_time,
			clock=lambda: float(data.time),
		)

	policy = new_policy()
	last_action = [0.0] * POLICY_ACTION_SIZE
	target = np.asarray(DEFAULT_JOINT_POS_RAD, dtype=np.float64)
	reset_requested = threading.Event()

	def key_callback(keycode: int) -> None:
		if keycode in (ord("r"), ord("R")):
			reset_requested.set()

	viewer = None
	if not args.headless:
		viewer = mujoco.viewer.launch_passive(
			model, data, key_callback=key_callback, show_left_ui=False
		)
		viewer.cam.trackbodyid = model.body("base_link").id
		viewer.cam.distance = 1.5
		viewer.cam.azimuth = 90.0
		viewer.cam.elevation = -10.0

	print(
		"MuJoCo sim2sim started\n"
		f"  robot={args.robot_xml.resolve()}\n"
		f"  physics={1.0 / model.opt.timestep:g} Hz, policy="
		f"{1.0 / POLICY_CONTROL_PERIOD_S:g} Hz\n"
		f"  command={args.command}\n"
		"  press R in the viewer to reset; close the viewer or press Ctrl+C to stop",
		flush=True,
	)

	next_wall_tick = time.perf_counter()
	next_log_time = 0.0
	try:
		while viewer is None or viewer.is_running():
			if args.duration is not None and data.time >= args.duration:
				break

			if reset_requested.is_set() or (
				args.reset_on_fall and _has_fallen(model, data)
			):
				reset_requested.clear()
				reset_data(model, data, mujoco)
				policy = new_policy()
				last_action = [0.0] * POLICY_ACTION_SIZE
				target = np.asarray(DEFAULT_JOINT_POS_RAD, dtype=np.float64)
				next_wall_tick = time.perf_counter()
				next_log_time = 0.0
				print("Simulation reset.", flush=True)

			state = make_robot_state(
				data, np, qpos_addresses, qvel_addresses, args.command
			)
			log_robot_state(state)
			policy_output = policy(state, last_action)
			last_action = policy_output.raw_action
			target = np.deg2rad(
				np.asarray(policy_output.command.target_joint_pos, dtype=np.float64)
			)

			for _ in range(PHYSICS_STEPS_PER_CONTROL):
				position = data.qpos[qpos_addresses]
				velocity = data.qvel[qvel_addresses]
				data.ctrl[:] = dc_motor_torque(np, target, position, velocity)
				mujoco.mj_step(model, data)

			if viewer is not None:
				viewer.sync()

			if args.log_interval > 0.0 and data.time + 1.0e-12 >= next_log_time:
				base = data.body("base_link").xpos
				pitch_id = model.joint("base_pitch").id
				pitch = data.qpos[int(model.jnt_qposadr[pitch_id])]
				print(
					f"sim t={data.time:8.2f}s  x={base[0]: .3f}m "
					f"z={base[2]: .3f}m pitch={math.degrees(pitch): .2f}deg "
					f"action={[round(value, 3) for value in last_action]}",
					flush=True,
				)
				next_log_time = data.time + args.log_interval

			if not args.no_realtime:
				next_wall_tick += POLICY_CONTROL_PERIOD_S
				sleep_time = next_wall_tick - time.perf_counter()
				if sleep_time > 0.0:
					time.sleep(sleep_time)
				elif sleep_time < -POLICY_CONTROL_PERIOD_S:
					next_wall_tick = time.perf_counter()
	except KeyboardInterrupt:
		pass
	finally:
		if viewer is not None:
			viewer.close()
	return 0


def run_with_log() -> int:
	"""Run sim2sim while saving stdout and stderr to a timestamped file."""
	log_dir = Path(__file__).resolve().parent / "log_sim2sim"
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


def main() -> int:
	parser = build_parser()
	args = parser.parse_args()
	if args.policy != "onnx":
		parser.error("MuJoCo sim2sim currently requires --policy onnx")
	if args.target_pos is not None:
		parser.error("--target-pos is not used by --policy onnx")
	if args.rate <= 0.0:
		parser.error("--rate must be greater than zero")
	if args.startup_move_time < 0.0 or args.startup_hold_time < 0.0:
		parser.error("ONNX startup times must not be negative")
	if args.duration is not None and args.duration <= 0.0:
		parser.error("--duration must be greater than zero")
	if args.log_interval < 0.0:
		parser.error("--log-interval must not be negative")
	if args.list_ports:
		print("MuJoCo sim2sim does not use a serial port.")
		return 0
	if args.port != "auto":
		print(f"note: --port {args.port!r} is accepted for compatibility and ignored.")
	if args.rate != 200.0:
		print(
			"note: --rate controls serial polling in pc_32_linux.py and is ignored; "
			"the ONNX policy remains at 50 Hz."
		)

	try:
		np, mujoco = _load_runtime()
		return run_simulation(args, np, mujoco)
	except (OSError, RuntimeError, ValueError) as exc:
		print(f"error: {exc}", file=sys.stderr)
		return 1


if __name__ == "__main__":
	raise SystemExit(run_with_log())
