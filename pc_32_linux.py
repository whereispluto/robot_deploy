#!/usr/bin/env python3
"""Linux entry point for the PC <-> STM32 USB CDC bridge.

The frame protocol and bridge implementation are shared with ``pc_32.py``.
This entry point adds Linux-oriented serial-port defaults, device discovery,
and useful error messages for missing devices and serial permissions.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from collections.abc import Callable

from pc_32 import RobotBridge, RobotCommand, RobotState, SerialBridge


DEFAULT_PORT = "/dev/ttyACM0"
LINUX_PORT_PATTERNS = (
	"/dev/ttyACM*",       # USB CDC ACM devices (typical for STM32)
	"/dev/ttyUSB*",       # USB-to-serial adapters
	"/dev/serial/by-id/*",  # Stable names managed by udev
)


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
		choices=("passthrough", "hold"),
		default="hold",
		help="Local policy used before model inference is connected",
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
		policy = (
			make_passthrough_policy(args.target_pos)
			if args.policy == "passthrough"
			else None
		)
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
	raise SystemExit(main())
