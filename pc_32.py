"""PC <-> STM32 USB CDC communication bridge.

Protocol summary
----------------
Transport: USB CDC virtual serial port (pyserial on Windows/Linux).

Frame layout:
	+---------+---------+---------+---------+---------+---------+---------+
	| magic   | version | msg_id  | seq     | length  | payload | crc16   |
	+---------+---------+---------+---------+---------+---------+---------+
	| 4 bytes | 1 byte  | 1 byte  | 2 bytes | 2 bytes | N bytes | 2 bytes |

Endian: little-endian for integers/floats.

Message IDs:
	0x01  State frame    STM32 -> PC
	0x02  Command frame  PC -> STM32
	0x03  Heartbeat      both directions

State payload (STM32 -> PC):
	joint_pos[6]      float32
	joint_vel[6]      float32
	base_lin_vel[3]   float32
	base_ang_vel[3]   float32
	base_quat[4]      float32   (w, x, y, z)
	cmd[3]            float32
	timestamp_ms      uint32
	status            uint16

Command payload (PC -> STM32):
	target_joint_pos[6] float32
	seq                 uint16
	flags               uint16

Notes:
	- last_action is kept on the PC side and inserted into the policy input.
	- If you later want to add IMU calibration or a state estimator, do it here.
	- Install dependency: pip install pyserial
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import struct
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

try:
	import serial
except ImportError as exc:  # pragma: no cover - runtime dependency
	serial = None
	_serial_import_error = exc
else:
	_serial_import_error = None


MAGIC = b"RB32"
VERSION = 1
MSG_STATE = 0x01
MSG_COMMAND = 0x02
MSG_HEARTBEAT = 0x03

STATE_FLOAT_COUNT = 6 + 6 + 3 + 3 + 4 + 3
STATE_PAYLOAD_FORMAT = "<" + ("f" * STATE_FLOAT_COUNT) + "IH"
STATE_PAYLOAD_SIZE = struct.calcsize(STATE_PAYLOAD_FORMAT)
COMMAND_PAYLOAD_FORMAT = "<6fHH"
COMMAND_PAYLOAD_SIZE = struct.calcsize(COMMAND_PAYLOAD_FORMAT)
FRAME_HEADER_FORMAT = "<4sBBHH"
FRAME_HEADER_SIZE = struct.calcsize(FRAME_HEADER_FORMAT)
FRAME_CRC_SIZE = 2


def crc16_ccitt(data: bytes, initial: int = 0xFFFF) -> int:
	crc = initial
	for byte in data:
		crc ^= byte << 8
		for _ in range(8):
			if crc & 0x8000:
				crc = ((crc << 1) ^ 0x1021) & 0xFFFF
			else:
				crc = (crc << 1) & 0xFFFF
	return crc


def clamp(value: float, lower: float, upper: float) -> float:
	return max(lower, min(upper, value))


def quat_to_gravity(quat_wxyz: tuple[float, float, float, float]) -> tuple[float, float, float]:
	qw, qx, qy, qz = quat_wxyz
	return (
		2.0 * (-qz * qx + qw * qy),
		-2.0 * (qz * qy + qw * qx),
		1.0 - 2.0 * (qw * qw + qz * qz),
	)


@dataclass
class RobotState:
	joint_pos: list[float]
	joint_vel: list[float]
	base_lin_vel: list[float]
	base_ang_vel: list[float]
	base_quat: list[float]
	cmd: list[float]
	timestamp_ms: int
	status: int

	@property
	def gravity_orientation(self) -> tuple[float, float, float]:
		return quat_to_gravity(tuple(self.base_quat))


@dataclass
class RobotCommand:
	target_joint_pos: list[float]
	seq: int = 0
	flags: int = 0


class FrameError(RuntimeError):
	pass


def pack_frame(msg_id: int, payload: bytes, seq: int = 0) -> bytes:
	header = struct.pack(FRAME_HEADER_FORMAT, MAGIC, VERSION, msg_id, seq & 0xFFFF, len(payload))
	crc = crc16_ccitt(header + payload)
	return header + payload + struct.pack("<H", crc)


def unpack_frame(frame: bytes) -> tuple[int, int, bytes]:
	if len(frame) < FRAME_HEADER_SIZE + FRAME_CRC_SIZE:
		raise FrameError("frame too short")

	magic, version, msg_id, seq, length = struct.unpack_from(FRAME_HEADER_FORMAT, frame, 0)
	if magic != MAGIC:
		raise FrameError("bad magic")
	if version != VERSION:
		raise FrameError(f"unsupported version: {version}")

	expected_size = FRAME_HEADER_SIZE + length + FRAME_CRC_SIZE
	if len(frame) != expected_size:
		raise FrameError("frame length mismatch")

	payload = frame[FRAME_HEADER_SIZE:FRAME_HEADER_SIZE + length]
	recv_crc, = struct.unpack_from("<H", frame, FRAME_HEADER_SIZE + length)
	calc_crc = crc16_ccitt(frame[:FRAME_HEADER_SIZE] + payload)
	if recv_crc != calc_crc:
		raise FrameError("crc mismatch")

	return msg_id, seq, payload


def pack_state(state: RobotState, seq: int = 0) -> bytes:
	payload = struct.pack(
		STATE_PAYLOAD_FORMAT,
		*state.joint_pos,
		*state.joint_vel,
		*state.base_lin_vel,
		*state.base_ang_vel,
		*state.base_quat,
		*state.cmd,
		int(state.timestamp_ms) & 0xFFFFFFFF,
		int(state.status) & 0xFFFF,
	)
	return pack_frame(MSG_STATE, payload, seq)


def unpack_state(payload: bytes) -> RobotState:
	if len(payload) != STATE_PAYLOAD_SIZE:
		raise FrameError("state payload size mismatch")

	values = struct.unpack(STATE_PAYLOAD_FORMAT, payload)
	joint_pos = list(values[0:6])
	joint_vel = list(values[6:12])
	base_lin_vel = list(values[12:15])
	base_ang_vel = list(values[15:18])
	base_quat = list(values[18:22])
	cmd = list(values[22:25])
	timestamp_ms = int(values[25])
	status = int(values[26])

	return RobotState(
		joint_pos=joint_pos,
		joint_vel=joint_vel,
		base_lin_vel=base_lin_vel,
		base_ang_vel=base_ang_vel,
		base_quat=base_quat,
		cmd=cmd,
		timestamp_ms=timestamp_ms,
		status=status,
	)


def pack_command(command: RobotCommand) -> bytes:
	payload = struct.pack(
		COMMAND_PAYLOAD_FORMAT,
		*command.target_joint_pos,
		int(command.seq) & 0xFFFF,
		int(command.flags) & 0xFFFF,
	)
	return pack_frame(MSG_COMMAND, payload, command.seq)


def unpack_command(payload: bytes) -> RobotCommand:
	if len(payload) != COMMAND_PAYLOAD_SIZE:
		raise FrameError("command payload size mismatch")

	values = struct.unpack(COMMAND_PAYLOAD_FORMAT, payload)
	target_joint_pos = list(values[0:6])
	seq = int(values[6])
	flags = int(values[7])
	return RobotCommand(target_joint_pos=target_joint_pos, seq=seq, flags=flags)


class SerialBridge:
	def __init__(
		self,
		port: str,
		baudrate: int = 921600,
		timeout: float = 0.01,
	) -> None:
		if serial is None:
			raise RuntimeError(
				"pyserial is not installed. Run `pip install pyserial` first."
			) from _serial_import_error

		self._serial = serial.Serial(port=port, baudrate=baudrate, timeout=timeout)
		self._rx_buffer = bytearray()
		self._lock = threading.Lock()

	@property
	def is_open(self) -> bool:
		return self._serial.is_open

	def close(self) -> None:
		if self._serial.is_open:
			self._serial.close()

	def send(self, frame: bytes) -> None:
		self._serial.write(frame)

	def send_command(self, command: RobotCommand) -> None:
		self.send(pack_command(command))

	def read_frames(self) -> list[bytes]:
		frames: list[bytes] = []
		chunk = self._serial.read(4096)
		if chunk:
			self._rx_buffer.extend(chunk)

		while True:
			start = self._rx_buffer.find(MAGIC)
			if start < 0:
				self._rx_buffer.clear()
				break
			if start > 0:
				del self._rx_buffer[:start]
			if len(self._rx_buffer) < FRAME_HEADER_SIZE:
				break

			_, _, _, _, length = struct.unpack_from(FRAME_HEADER_FORMAT, self._rx_buffer, 0)
			frame_size = FRAME_HEADER_SIZE + length + FRAME_CRC_SIZE
			if len(self._rx_buffer) < frame_size:
				break

			frame = bytes(self._rx_buffer[:frame_size])
			del self._rx_buffer[:frame_size]
			frames.append(frame)

		return frames


class RobotBridge:
	def __init__(
		self,
		serial_bridge: SerialBridge,
		policy_fn: Optional[Callable[[RobotState, list[float]], RobotCommand]] = None,
		command_limit: float = 100.0,
	) -> None:
		self.serial_bridge = serial_bridge
		self.policy_fn = policy_fn
		self.command_limit = command_limit
		self.last_action = [0.0] * 6
		self.last_state: Optional[RobotState] = None
		self._seq = 0

	def _default_policy(self, state: RobotState) -> RobotCommand:
		target = [clamp(value, -self.command_limit, self.command_limit) for value in state.joint_pos]
		return RobotCommand(target_joint_pos=target, seq=self._seq, flags=0)

	def handle_state(self, state: RobotState) -> RobotCommand:
		self.last_state = state
		if self.policy_fn is None:
			command = self._default_policy(state)
		else:
			command = self.policy_fn(state, self.last_action)

		self.last_action = list(command.target_joint_pos)
		self._seq = (self._seq + 1) & 0xFFFF
		command.seq = self._seq
		return command

	def step(self) -> None:
		for frame in self.serial_bridge.read_frames():
			try:
				msg_id, _, payload = unpack_frame(frame)
			except FrameError:
				continue

			if msg_id == MSG_STATE:
				state = unpack_state(payload)
				print(
					f"joint_pos={state.joint_pos} "
					f"joint_vel={state.joint_vel} "
					f"imu_ang_vel={state.base_ang_vel} "
					f"imu_quat_wxyz={state.base_quat}",
					flush=True,
				)
				command = self.handle_state(state)
				self.serial_bridge.send_command(command)

	def run_forever(self, rate_hz: float = 200.0) -> None:
		period = 1.0 / rate_hz
		next_tick = time.perf_counter()
		try:
			while True:
				self.step()
				next_tick += period
				sleep_time = next_tick - time.perf_counter()
				if sleep_time > 0:
					time.sleep(sleep_time)
				else:
					next_tick = time.perf_counter()
		except KeyboardInterrupt:
			pass
		finally:
			self.serial_bridge.close()


def passthrough_policy(state: RobotState, last_action: list[float]) -> RobotCommand:

	# 新增：支持外部target_pos参数
	import sys
	target_pos_arg = None
	for i, arg in enumerate(sys.argv):
		if arg == '--target_pos' and i + 1 < len(sys.argv):
			try:
				target_pos_arg = [float(x) for x in sys.argv[i+1].split(',')]
			except Exception:
				target_pos_arg = None
			break

	def passthrough_policy(state: RobotState, last_action: list[float]) -> RobotCommand:
		del last_action
		if target_pos_arg is not None and len(target_pos_arg) == 6:
			return RobotCommand(target_joint_pos=list(target_pos_arg), seq=0, flags=0)
		else:
			return RobotCommand(target_joint_pos=list(state.joint_pos), seq=0, flags=0)


def main() -> None:

	parser = argparse.ArgumentParser(description="PC <-> STM32 USB CDC bridge")
	parser.add_argument("--port", default="COM10", help="Serial port, default COM10 (e.g. COM5 or /dev/ttyACM0)")
	parser.add_argument("--baudrate", type=int, default=921600)
	parser.add_argument("--rate", type=float, default=200.0, help="Bridge loop rate in Hz")
	parser.add_argument(
		"--policy",
		choices=("passthrough", "hold"),
		default="hold",
		help="Simple local policy used before model inference is wired in",
	)
	parser.add_argument(
		"--target_pos",
		type=str,
		default=None,
		help="Comma separated 6 joint target positions, e.g. 30,0,0,0,0,0"
	)
	args = parser.parse_args()

	bridge = SerialBridge(port=args.port, baudrate=args.baudrate)
	policy = passthrough_policy if args.policy == "passthrough" else None
	robot_bridge = RobotBridge(serial_bridge=bridge, policy_fn=policy)
	robot_bridge.run_forever(rate_hz=args.rate)


if __name__ == "__main__":
	main()
