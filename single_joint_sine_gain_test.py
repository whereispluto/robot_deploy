#!/usr/bin/env python3
"""Safely collect a single-joint internal-PD sinusoidal response.

This program uses the STM32 gain-test message (0x04). Positions are logical
joint angles: motor installation direction is already handled by the firmware.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from pc_32 import RobotState, SerialBridge
from single_joint_gain_test import (
    JOINT_LIMITS_DEG,
    JOINT_NAMES,
    STATUS_FEEDBACK_FRESH,
    STATUS_GAIN_TEST_ACTIVE,
    check_state,
    effective_gain,
    find_ports,
    latest_states,
    make_command,
    resolve_port,
)


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


@dataclass(frozen=True)
class SineProfile:
    baseline_deg: float
    amplitude_deg: float
    frequency_hz: float
    initial_hold_s: float
    ramp_cycles: int
    measure_cycles: int
    final_hold_s: float

    @property
    def ramp_duration_s(self) -> float:
        return self.ramp_cycles / self.frequency_hz

    @property
    def measure_duration_s(self) -> float:
        return self.measure_cycles / self.frequency_hz

    @property
    def sine_duration_s(self) -> float:
        return 2.0 * self.ramp_duration_s + self.measure_duration_s

    @property
    def total_duration_s(self) -> float:
        return self.initial_hold_s + self.sine_duration_s + self.final_hold_s

    @property
    def measure_start_s(self) -> float:
        return self.initial_hold_s + self.ramp_duration_s

    @property
    def measure_end_s(self) -> float:
        return self.measure_start_s + self.measure_duration_s

    def target_at(self, elapsed_s: float) -> tuple[str, float]:
        """Return phase name and target at a host elapsed time."""

        if elapsed_s < self.initial_hold_s:
            return "initial_hold", self.baseline_deg

        sine_time = elapsed_s - self.initial_hold_s
        if sine_time < self.ramp_duration_s:
            progress = sine_time / self.ramp_duration_s
            envelope = smoothstep(progress)
            phase = "ramp_up"
        elif sine_time < self.ramp_duration_s + self.measure_duration_s:
            envelope = 1.0
            phase = "measure"
        elif sine_time < self.sine_duration_s:
            cooldown_time = sine_time - self.ramp_duration_s - self.measure_duration_s
            progress = cooldown_time / self.ramp_duration_s
            envelope = 1.0 - smoothstep(progress)
            phase = "ramp_down"
        else:
            return "final_hold", self.baseline_deg

        angle = 2.0 * math.pi * self.frequency_hz * sine_time
        target = self.baseline_deg + envelope * self.amplitude_deg * math.sin(angle)
        return phase, target


def smoothstep(value: float) -> float:
    value = min(1.0, max(0.0, value))
    return value * value * (3.0 - 2.0 * value)


def write_csv(
    path: Path,
    samples: list[Sample],
    kp: float,
    kd: float,
    max_torque: float,
    frequency_hz: float,
    amplitude_deg: float,
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
                "frequency_hz",
                "amplitude_deg",
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
                    f"{frequency_hz:.6f}",
                    f"{amplitude_deg:.6f}",
                    f"{sample.feedback_torque_nm:.6f}",
                    f"0x{sample.status:04x}",
                )
            )


def sine_fit(
    times_s: list[float], values: list[float], frequency_hz: float
) -> tuple[float, float, float]:
    """Least-squares fit of offset + a*sin(wt) + b*cos(wt).

    Returns offset, amplitude and phase in ``amplitude*sin(wt + phase)`` form.
    The measurement contains an integer number of cycles, so the three basis
    functions are approximately orthogonal at the fixed-rate sample times.
    """

    count = len(values)
    offset = sum(values) / count
    omega = 2.0 * math.pi * frequency_hz
    centered = [value - offset for value in values]
    sin_values = [math.sin(omega * value) for value in times_s]
    cos_values = [math.cos(omega * value) for value in times_s]
    sin_power = sum(value * value for value in sin_values)
    cos_power = sum(value * value for value in cos_values)
    a = sum(y * basis for y, basis in zip(centered, sin_values, strict=True))
    b = sum(y * basis for y, basis in zip(centered, cos_values, strict=True))
    a /= sin_power
    b /= cos_power
    return offset, math.hypot(a, b), math.atan2(b, a)


def print_metrics(
    samples: list[Sample], profile: SineProfile, start_time_s: float
) -> None:
    measured = [sample for sample in samples if sample.phase == "measure"]
    if len(measured) < 10:
        print("\n稳定正弦段样本不足，无法计算指标。")
        return

    errors = [sample.error_deg for sample in measured]
    rmse = math.sqrt(sum(value * value for value in errors) / len(errors))
    max_error = max(abs(value) for value in errors)
    max_speed = max(abs(sample.velocity_deg_s) for sample in measured)
    max_torque = max(abs(sample.feedback_torque_nm) for sample in measured)
    # Use sine time, whose zero is the start of ramp-up, for target and response.
    times = [
        sample.host_time_s - start_time_s - profile.initial_hold_s
        for sample in measured
    ]
    positions = [sample.position_deg for sample in measured]
    offset, response_amplitude, phase_rad = sine_fit(
        times, positions, profile.frequency_hz
    )
    phase_deg = math.degrees(phase_rad)
    # Wrap to the most useful lag representation. A positive value means the
    # measured joint response occurs after the target.
    lag_deg = (-phase_deg + 180.0) % 360.0 - 180.0
    lag_ms = lag_deg / 360.0 / profile.frequency_hz * 1000.0
    amplitude_ratio = response_amplitude / profile.amplitude_deg

    print("\n稳定正弦段结果（基于状态反馈的近似指标）：")
    print(
        f"  tracking_rmse={rmse:.3f}°, max_error={max_error:.3f}°, "
        f"amplitude_ratio={amplitude_ratio:.3f}"
    )
    print(
        f"  phase_lag={lag_deg:.1f}° ({lag_ms:.1f} ms), response_center={offset:.3f}°"
    )
    print(
        f"  max_speed={max_speed:.1f}°/s, "
        f"max_torque={max_torque:.3f} N·m, samples={len(measured)}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="单关节真运控（电机内部 PD）正弦跟踪测试"
    )
    parser.add_argument("--port", default="auto")
    parser.add_argument("--baudrate", type=int, default=921600)
    parser.add_argument("--joint", type=int, choices=range(1, 7))
    parser.add_argument("--kp", type=float, help="N·m/rad")
    parser.add_argument("--kd", type=float, help="N·m·s/rad")
    parser.add_argument("--max-torque", type=float, default=0.2, help="N·m")
    parser.add_argument("--amplitude-deg", type=float, default=3.0)
    parser.add_argument("--frequency-hz", type=float, default=0.5)
    parser.add_argument(
        "--cycles",
        type=int,
        default=6,
        help="满幅正弦的测量周期数（不含增幅和减幅周期）",
    )
    parser.add_argument(
        "--ramp-cycles",
        type=int,
        default=1,
        help="正弦开始和结束时各自用于平滑增减幅的周期数",
    )
    parser.add_argument("--initial-hold", type=float, default=2.0)
    parser.add_argument("--final-hold", type=float, default=1.0)
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
    if not math.isfinite(args.amplitude_deg) or not 0.0 < args.amplitude_deg <= 10.0:
        parser.error("--amplitude-deg 必须在 (0, 10]° 内")
    if not math.isfinite(args.frequency_hz) or not 0.05 <= args.frequency_hz <= 5.0:
        parser.error("--frequency-hz 必须在 [0.05, 5] Hz 内")
    if args.cycles < 2 or args.cycles > 100:
        parser.error("--cycles 必须在 [2, 100] 内")
    if args.ramp_cycles < 1 or args.ramp_cycles > 10:
        parser.error("--ramp-cycles 必须在 [1, 10] 内")
    for name in (
        "initial_hold",
        "final_hold",
        "command_rate",
        "state_timeout",
        "abort_velocity",
    ):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0.0:
            parser.error(f"--{name.replace('_', '-')} 必须大于 0")
    if not 20.0 <= args.command_rate <= 100.0:
        parser.error("--command-rate 必须在 [20, 100] Hz 内")
    if args.command_rate / args.frequency_hz < 20.0:
        parser.error("每周期至少需要 20 个命令点；请降低频率或提高命令频率")


def wait_for_service(
    bridge: SerialBridge,
    joint_index: int,
    args: argparse.Namespace,
    seq: int,
) -> tuple[int, RobotState]:
    deadline = time.monotonic() + 5.0
    next_send = 0.0
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
            if (state.status & STATUS_FEEDBACK_FRESH) != 0 and (
                state.status & STATUS_GAIN_TEST_ACTIVE
            ) != 0:
                check_state(state, joint_index, args.max_torque)
                return seq, state
        time.sleep(0.001)
    raise RuntimeError(
        "5 秒内未收到 gain-test 激活状态；请先烧录支持 0x04 协议的 STM32 固件"
    )


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
        f"正弦幅值={args.amplitude_deg:.3f}°, 频率={args.frequency_hz:.3f} Hz, "
        f"稳定段={args.cycles} 周期, 最大力矩={args.max_torque:.3f} N·m"
    )
    print("按 Ctrl+C 可随时发送零 PD 并退出。")

    bridge = SerialBridge(port=port, baudrate=args.baudrate, timeout=0.005)
    seq = 0
    baseline = 0.0
    samples: list[Sample] = []
    profile: SineProfile | None = None
    start = 0.0
    log_path = (
        Path(__file__).resolve().parent
        / "log_sine_gain_test"
        / (
            f"{datetime.now():%Y-%m-%d_%H-%M-%S}_joint{args.joint}_sine_"
            f"{args.frequency_hz:g}hz_kp{args.kp:g}_kd{args.kd:g}.csv"
        )
    )

    try:
        print("等待新鲜电机反馈，并验证 STM32 gain-test 服务……")
        seq, initial_state = wait_for_service(bridge, joint_index, args, seq)
        baseline = initial_state.joint_pos[joint_index]
        lower, upper = JOINT_LIMITS_DEG[joint_index]
        if (
            baseline - args.amplitude_deg < lower
            or baseline + args.amplitude_deg > upper
        ):
            raise RuntimeError(
                f"基准位置 {baseline:.3f}° 加减幅值后超出逻辑限位 "
                f"[{lower:.1f}, {upper:.1f}]°；请减小幅值或调整初始姿态"
            )

        profile = SineProfile(
            baseline_deg=baseline,
            amplitude_deg=args.amplitude_deg,
            frequency_hz=args.frequency_hz,
            initial_hold_s=args.initial_hold,
            ramp_cycles=args.ramp_cycles,
            measure_cycles=args.cycles,
            final_hold_s=args.final_hold,
        )
        commanded_peak_speed = 2.0 * math.pi * args.frequency_hz * args.amplitude_deg
        if commanded_peak_speed >= args.abort_velocity:
            raise RuntimeError(
                f"目标正弦峰值速度 {commanded_peak_speed:.1f}°/s 不低于终止阈值 "
                f"{args.abort_velocity:.1f}°/s"
            )
        print(
            f"基准位置={baseline:.3f}°，逻辑限位=({lower:.1f}, {upper:.1f})°，"
            f"预计总时长={profile.total_duration_s:.1f} s"
        )

        start = time.monotonic()
        next_send = start
        last_state_time = start
        last_phase = ""
        active_phase = "initial_hold"
        active_target = baseline

        while True:
            now = time.monotonic()
            elapsed = now - start
            if elapsed >= profile.total_duration_s:
                break
            active_phase, active_target = profile.target_at(elapsed)
            if active_phase != last_phase:
                print(f"phase: {active_phase}")
                last_phase = active_phase

            if now >= next_send:
                seq = (seq + 1) & 0xFFFF
                bridge.send_gain_test_command(
                    make_command(
                        joint_index,
                        active_target,
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
                if (state.status & STATUS_GAIN_TEST_ACTIVE) == 0:
                    raise RuntimeError(
                        f"gain-test 服务意外退出，status=0x{state.status:04x}"
                    )
                velocity = state.joint_vel[joint_index]
                if abs(velocity) > args.abort_velocity:
                    raise RuntimeError(
                        f"关节速度 {velocity:.1f}°/s 超过终止阈值 "
                        f"{args.abort_velocity:.1f}°/s"
                    )
                position = state.joint_pos[joint_index]
                samples.append(
                    Sample(
                        host_time_s=elapsed,
                        mcu_time_ms=state.timestamp_ms,
                        phase=active_phase,
                        target_deg=active_target,
                        position_deg=position,
                        velocity_deg_s=velocity,
                        error_deg=active_target - position,
                        feedback_torque_nm=state.cmd[0],
                        status=state.status,
                    )
                )

            if now - last_state_time > args.state_timeout:
                raise RuntimeError(f"超过 {args.state_timeout:.3f} s 未收到状态反馈")
            time.sleep(0.001)

        print_metrics(samples, profile, 0.0)
        return 0
    finally:
        # A disabled command makes the firmware send kp=kd=tqe=0 to all motors.
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
            write_csv(
                log_path,
                samples,
                args.kp,
                args.kd,
                args.max_torque,
                args.frequency_hz,
                args.amplitude_deg,
            )
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
