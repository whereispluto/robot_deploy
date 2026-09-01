#!/usr/bin/env python3
"""Safely collect a single-joint internal-PD step response.

This program requires the matching STM32 firmware gain-test message (0x04).
All positions use the logical joint direction exported by the firmware.
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from pc_32 import (
    FrameError,
    GainTestCommand,
    MSG_STATE,
    RobotState,
    SerialBridge,
    unpack_frame,
    unpack_state,
)


JOINT_NAMES = (
    "left_hip",
    "left_knee",
    "left_ankle",
    "right_hip",
    "right_knee",
    "right_ankle",
)
JOINT_LIMITS_DEG = (
    (-60.0, 60.0),
    (-120.0, 0.0),
    (-45.0, 45.0),
    (-60.0, 60.0),
    (-120.0, 0.0),
    (-45.0, 45.0),
)
PORT_PATTERNS = (
    "/dev/ttyACM*",
    "/dev/ttyUSB*",
    "/dev/serial/by-id/*",
)
STATUS_FEEDBACK_FRESH = 0x0100
STATUS_FATAL_MOTOR = 0x0E00
STATUS_GAIN_TEST_ACTIVE = 0x1000
PID_MODEL_SCALE = 0.5256  # M4438_30, from src/convert/convert.c


@dataclass(frozen=True)
class Phase:
    name: str
    target_deg: float
    duration_s: float
    is_step: bool


@dataclass(frozen=True)
class Sample:
    host_time_s: float
    mcu_time_ms: int
    phase: str
    target_deg: float
    position_deg: float
    velocity_deg_s: float
    error_deg: float
    feedback_torque_nm: float
    status: int


def find_ports() -> list[str]:
    ports: list[str] = []
    for pattern in PORT_PATTERNS:
        ports.extend(sorted(glob.glob(pattern)))
    return list(dict.fromkeys(ports))


def resolve_port(port: str) -> str:
    if port != "auto":
        return port
    ports = find_ports()
    if not ports:
        raise RuntimeError("没有找到 /dev/ttyACM* 或 /dev/ttyUSB* 串口")
    return ports[0]


def effective_gain(requested_si: float) -> tuple[int, float]:
    """Predict the M4438_30 int16 code and quantized SI gain."""

    protocol_gain = requested_si * (2.0 * math.pi)
    raw_code = math.trunc(protocol_gain / PID_MODEL_SCALE * 10.0)
    effective_si = raw_code / 10.0 * PID_MODEL_SCALE / (2.0 * math.pi)
    return raw_code, effective_si


def build_phases(
    baseline_deg: float,
    step_deg: float,
    direction: str,
    cycles: int,
    hold_s: float,
    step_s: float,
    return_s: float,
    limits: tuple[float, float],
) -> list[Phase]:
    offsets = {
        "positive": (step_deg,),
        "negative": (-step_deg,),
        "both": (step_deg, -step_deg),
    }[direction]
    for offset in offsets:
        target = baseline_deg + offset
        if not limits[0] <= target <= limits[1]:
            raise ValueError(
                f"目标 {target:.3f}° 超出关节限位 [{limits[0]:.1f}, {limits[1]:.1f}]°"
            )

    phases = [Phase("initial_hold", baseline_deg, hold_s, False)]
    for cycle in range(1, cycles + 1):
        for offset in offsets:
            direction_name = "positive" if offset > 0.0 else "negative"
            phases.append(
                Phase(
                    f"cycle_{cycle}_{direction_name}_step",
                    baseline_deg + offset,
                    step_s,
                    True,
                )
            )
            phases.append(
                Phase(
                    f"cycle_{cycle}_{direction_name}_return",
                    baseline_deg,
                    return_s,
                    False,
                )
            )
    return phases


def latest_states(bridge: SerialBridge) -> list[RobotState]:
    states: list[RobotState] = []
    for frame in bridge.read_frames():
        try:
            msg_id, _, payload = unpack_frame(frame)
            if msg_id == MSG_STATE:
                states.append(unpack_state(payload))
        except FrameError:
            continue
    return states


def check_state(state: RobotState, joint_index: int, max_torque_nm: float) -> None:
    if (state.status & STATUS_FEEDBACK_FRESH) == 0:
        raise RuntimeError(f"电机反馈不新鲜，status=0x{state.status:04x}")
    if (state.status & STATUS_FATAL_MOTOR) != 0:
        raise RuntimeError(
            f"检测到电机通信故障标志，status=0x{state.status:04x}；"
            "排除故障并重启 STM32 后再测试"
        )

    lower, upper = JOINT_LIMITS_DEG[joint_index]
    position = state.joint_pos[joint_index]
    if not lower - 1.0 <= position <= upper + 1.0:
        raise RuntimeError(f"Joint {joint_index + 1} 反馈 {position:.3f}° 超出限位")

    # This catches a lost/ineffective maximum-torque register when the response
    # actually reaches the configured limit. Keep a margin for feedback noise.
    if abs(state.cmd[0]) > max_torque_nm * 1.5 + 0.10:
        raise RuntimeError(
            f"反馈力矩 {state.cmd[0]:.3f} N·m 明显超过设置值 "
            f"{max_torque_nm:.3f} N·m，已停止测试"
        )


def write_csv(
    path: Path,
    samples: list[Sample],
    kp: float,
    kd: float,
    max_torque: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            (
                "host_time_s",
                "mcu_time_ms",
                "phase",
                "target_deg",
                "position_deg",
                "velocity_deg_s",
                "error_deg",
                "kp_nm_per_rad",
                "kd_nms_per_rad",
                "max_torque_nm",
                "feedback_torque_nm",
                "status_hex",
            )
        )
        for sample in samples:
            writer.writerow(
                (
                    f"{sample.host_time_s:.6f}",
                    sample.mcu_time_ms,
                    sample.phase,
                    f"{sample.target_deg:.6f}",
                    f"{sample.position_deg:.6f}",
                    f"{sample.velocity_deg_s:.6f}",
                    f"{sample.error_deg:.6f}",
                    f"{kp:.9f}",
                    f"{kd:.9f}",
                    f"{max_torque:.6f}",
                    f"{sample.feedback_torque_nm:.6f}",
                    f"0x{sample.status:04x}",
                )
            )


def print_metrics(samples: list[Sample], phases: list[Phase], baseline: float) -> None:
    print("\n阶跃结果（50 Hz 状态反馈下的近似指标）：")
    for phase in phases:
        if not phase.is_step:
            continue
        phase_samples = [sample for sample in samples if sample.phase == phase.name]
        if len(phase_samples) < 2:
            print(f"  {phase.name}: 样本不足")
            continue

        amplitude = phase.target_deg - baseline
        direction = 1.0 if amplitude > 0.0 else -1.0
        progress = [
            direction * (sample.position_deg - baseline) / abs(amplitude)
            for sample in phase_samples
        ]
        start_time = phase_samples[0].host_time_s
        t10 = next(
            (
                sample.host_time_s - start_time
                for sample, value in zip(phase_samples, progress, strict=True)
                if value >= 0.1
            ),
            None,
        )
        t90 = next(
            (
                sample.host_time_s - start_time
                for sample, value in zip(phase_samples, progress, strict=True)
                if value >= 0.9
            ),
            None,
        )
        rise_time = t90 - t10 if t10 is not None and t90 is not None else None
        overshoot = max(0.0, (max(progress) - 1.0) * 100.0)
        tail_count = max(1, len(phase_samples) // 5)
        final_error = (
            sum(abs(sample.error_deg) for sample in phase_samples[-tail_count:])
            / tail_count
        )
        max_speed = max(abs(sample.velocity_deg_s) for sample in phase_samples)
        max_torque = max(abs(sample.feedback_torque_nm) for sample in phase_samples)
        rise_text = f"{rise_time:.3f} s" if rise_time is not None else "未到 90%"
        print(
            f"  {phase.name}: rise(10-90%)={rise_text}, "
            f"overshoot={overshoot:.1f}%, final_error={final_error:.3f}°, "
            f"max_speed={max_speed:.1f}°/s, max_torque={max_torque:.3f} N·m"
        )


def make_command(
    joint_index: int,
    target_deg: float,
    kp: float,
    kd: float,
    max_torque: float,
    enabled: bool,
    seq: int,
) -> GainTestCommand:
    return GainTestCommand(
        joint_index=joint_index,
        target_position_deg=target_deg,
        kp_nm_per_rad=kp,
        kd_nms_per_rad=kd,
        max_torque_nm=max_torque,
        enabled=enabled,
        seq=seq,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="单关节真运控（电机内部 PD）阶跃响应测试"
    )
    parser.add_argument("--port", default="auto")
    parser.add_argument("--baudrate", type=int, default=921600)
    parser.add_argument("--joint", type=int, choices=range(1, 7))
    parser.add_argument("--kp", type=float, help="N·m/rad")
    parser.add_argument("--kd", type=float, help="N·m·s/rad")
    parser.add_argument("--max-torque", type=float, default=0.2, help="N·m")
    parser.add_argument("--step-deg", type=float, default=3.0)
    parser.add_argument(
        "--direction",
        choices=("positive", "negative", "both"),
        default="both",
    )
    parser.add_argument("--cycles", type=int, default=2)
    parser.add_argument("--initial-hold", type=float, default=2.0)
    parser.add_argument("--step-duration", type=float, default=2.0)
    parser.add_argument("--return-duration", type=float, default=1.5)
    parser.add_argument("--command-rate", type=float, default=50.0)
    parser.add_argument("--state-timeout", type=float, default=0.25)
    parser.add_argument("--abort-velocity", type=float, default=300.0)
    parser.add_argument(
        "--confirm-suspended",
        action="store_true",
        help="确认机器人已悬空且被测关节不会碰撞后才允许运行",
    )
    parser.add_argument("--list-ports", action="store_true")
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if not args.confirm_suspended:
        parser.error("必须先悬空/固定机器人，并传入 --confirm-suspended")
    if args.joint is None or args.kp is None or args.kd is None:
        parser.error("运行测试必须同时指定 --joint、--kp 和 --kd")
    if not math.isfinite(args.kp) or not 0.0 <= args.kp <= 100.0:
        parser.error("--kp 必须在 [0, 100] N·m/rad 内")
    if not math.isfinite(args.kd) or not 0.0 <= args.kd <= 10.0:
        parser.error("--kd 必须在 [0, 10] N·m·s/rad 内")
    if not math.isfinite(args.max_torque) or not 0.0 < args.max_torque <= 2.0:
        parser.error("--max-torque 必须在 (0, 2] N·m 内")
    if not math.isfinite(args.step_deg) or not 0.0 < args.step_deg <= 10.0:
        parser.error("--step-deg 必须在 (0, 10]° 内")
    if args.cycles < 1 or args.cycles > 20:
        parser.error("--cycles 必须在 [1, 20] 内")
    for name in (
        "initial_hold",
        "step_duration",
        "return_duration",
        "command_rate",
        "state_timeout",
        "abort_velocity",
    ):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0.0:
            parser.error(f"--{name.replace('_', '-')} 必须大于 0")
    if args.command_rate < 20.0 or args.command_rate > 100.0:
        parser.error("--command-rate 必须在 [20, 100] Hz 内")


def run_test(args: argparse.Namespace) -> int:
    joint_index = args.joint - 1
    port = resolve_port(args.port)
    if not os.path.exists(port):
        raise RuntimeError(f"串口不存在：{port}")

    kp_code, kp_effective = effective_gain(args.kp)
    kd_code, kd_effective = effective_gain(args.kd)
    print(f"串口: {port}; Joint {args.joint} ({JOINT_NAMES[joint_index]})")
    print(
        f"请求 KP={args.kp:.6f}, KD={args.kd:.6f}; "
        f"预计协议代码=({kp_code}, {kd_code}); "
        f"量化后 SI 增益≈({kp_effective:.6f}, {kd_effective:.6f})"
    )
    print(
        f"步长={args.step_deg:.3f}°, 最大力矩={args.max_torque:.3f} N·m；"
        "按 Ctrl+C 可随时发送零 PD 并退出"
    )

    bridge = SerialBridge(port=port, baudrate=args.baudrate, timeout=0.005)
    seq = 0
    baseline = 0.0
    samples: list[Sample] = []
    phases: list[Phase] = []
    log_path = (
        Path(__file__).resolve().parent
        / "log_gain_test"
        / (
            f"{datetime.now():%Y-%m-%d_%H-%M-%S}_joint{args.joint}_"
            f"kp{args.kp:g}_kd{args.kd:g}.csv"
        )
    )

    try:
        print("等待新鲜电机反馈，并验证 STM32 已支持 gain-test 协议……")
        deadline = time.monotonic() + 5.0
        next_send = 0.0
        latest_state: RobotState | None = None
        while time.monotonic() < deadline:
            now = time.monotonic()
            if now >= next_send:
                seq = (seq + 1) & 0xFFFF
                bridge.send_gain_test_command(
                    make_command(
                        joint_index,
                        0.0,
                        args.kp,
                        args.kd,
                        args.max_torque,
                        False,
                        seq,
                    )
                )
                next_send = now + 1.0 / args.command_rate
            for state in latest_states(bridge):
                latest_state = state
                if (state.status & STATUS_FEEDBACK_FRESH) != 0 and (
                    state.status & STATUS_GAIN_TEST_ACTIVE
                ) != 0:
                    check_state(state, joint_index, args.max_torque)
                    baseline = state.joint_pos[joint_index]
                    break
            if latest_state is not None and (
                (latest_state.status & STATUS_FEEDBACK_FRESH) != 0
                and (latest_state.status & STATUS_GAIN_TEST_ACTIVE) != 0
            ):
                break
            time.sleep(0.001)
        else:
            raise RuntimeError(
                "5 秒内未收到 gain-test 激活状态；请先烧录本次修改后的 STM32 固件"
            )

        phases = build_phases(
            baseline,
            args.step_deg,
            args.direction,
            args.cycles,
            args.initial_hold,
            args.step_duration,
            args.return_duration,
            JOINT_LIMITS_DEG[joint_index],
        )
        print(
            f"基准位置={baseline:.3f}°，逻辑限位="
            f"{JOINT_LIMITS_DEG[joint_index]}°，开始测试"
        )

        start = time.monotonic()
        phase_start = start
        phase_index = 0
        phase = phases[phase_index]
        last_state_time = start
        next_send = start
        print(f"phase: {phase.name}, target={phase.target_deg:.3f}°")

        while phase_index < len(phases):
            now = time.monotonic()
            if now - phase_start >= phase.duration_s:
                phase_start += phase.duration_s
                phase_index += 1
                if phase_index >= len(phases):
                    break
                phase = phases[phase_index]
                print(f"phase: {phase.name}, target={phase.target_deg:.3f}°")

            if now >= next_send:
                seq = (seq + 1) & 0xFFFF
                bridge.send_gain_test_command(
                    make_command(
                        joint_index,
                        phase.target_deg,
                        args.kp,
                        args.kd,
                        args.max_torque,
                        True,
                        seq,
                    )
                )
                next_send += 1.0 / args.command_rate
                if next_send < now:
                    next_send = now + 1.0 / args.command_rate

            for state in latest_states(bridge):
                last_state_time = now
                check_state(state, joint_index, args.max_torque)
                velocity = state.joint_vel[joint_index]
                if abs(velocity) > args.abort_velocity:
                    raise RuntimeError(
                        f"关节速度 {velocity:.1f}°/s 超过终止阈值 "
                        f"{args.abort_velocity:.1f}°/s"
                    )
                position = state.joint_pos[joint_index]
                samples.append(
                    Sample(
                        host_time_s=now - start,
                        mcu_time_ms=state.timestamp_ms,
                        phase=phase.name,
                        target_deg=phase.target_deg,
                        position_deg=position,
                        velocity_deg_s=velocity,
                        error_deg=phase.target_deg - position,
                        feedback_torque_nm=state.cmd[0],
                        status=state.status,
                    )
                )

            if now - last_state_time > args.state_timeout:
                raise RuntimeError(f"超过 {args.state_timeout:.3f} s 未收到状态反馈")
            time.sleep(0.001)

        print_metrics(samples, phases, baseline)
        return 0
    finally:
        # A disabled service command makes the firmware send kp=kd=tqe=0 to all
        # six motors. Repetition covers a transient USB packet loss.
        for _ in range(6):
            try:
                seq = (seq + 1) & 0xFFFF
                bridge.send_gain_test_command(
                    make_command(
                        joint_index,
                        baseline,
                        args.kp,
                        args.kd,
                        args.max_torque,
                        False,
                        seq,
                    )
                )
            except OSError:
                break
            time.sleep(0.02)
        bridge.close()
        if samples:
            write_csv(log_path, samples, args.kp, args.kd, args.max_torque)
            print(f"CSV 日志: {log_path}")
        print("已发送零 PD 停止命令。")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.list_ports:
        ports = find_ports()
        print("\n".join(ports) if ports else "未找到串口")
        return 0 if ports else 1
    validate_args(parser, args)
    try:
        return run_test(args)
    except KeyboardInterrupt:
        print("\n用户中止测试。", file=sys.stderr)
        return 130
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
