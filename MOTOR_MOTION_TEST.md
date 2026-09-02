# 六关节限位与微动测试

## 关节方向约定

- 正向安装的电机保持原始正方向。
- 反向安装的电机以其物理反方向作为逻辑正方向。
- STM32 固件负责反向映射，PC 端使用统一的逻辑关节方向，不再重复取反。

软件中反向的电机：

| FDCAN | 电机 ID | PC 关节编号 |
|---|---:|---:|
| FDCAN1 / PORT1 | 1 | Joint 1（左髋） |
| FDCAN2 / PORT2 | 2 | Joint 5（右膝） |
| FDCAN2 / PORT2 | 3 | Joint 6（右踝） |

> 该表已与当前 `motor_many.c` 和 `motor.c` 的方向映射核对。PC 测试程序
> 始终发送逻辑角度，不应再次手动取反。

## 六关节限位

以下限位均使用逻辑角度，单位为度（°）：

| PC 关节 | FDCAN | 电机 ID | 最小角度 | 最大角度 |
|---|---|---:|---:|---:|
| Joint 1 | FDCAN1 / PORT1 | 1 | -60° | +60° |
| Joint 2 | FDCAN1 / PORT1 | 2 | -120° | 0° |
| Joint 3 | FDCAN1 / PORT1 | 3 | -45° | +45° |
| Joint 4 | FDCAN2 / PORT2 | 1 | -60° | +60° |
| Joint 5 | FDCAN2 / PORT2 | 2 | -120° | 0° |
| Joint 6 | FDCAN2 / PORT2 | 3 | -45° | +45° |

## 微动测试命令

```bash
/home/cx/anaconda3/envs/robot_deploy/bin/python \
  /home/cx/robot_deploy/pc_32_linux.py \
  --policy motion-test
```


## 传入策略测试

```bash
python pc_32_linux.py \
  --port auto \
  --policy onnx \
  --command 0.15,0,0
```

```bash
python pc_32_linux.py \
  --port auto \
  --policy onnx \
  --model /home/cx/mjlab-recovered/logs/rsl_rl/custom_biped_velocity_nolinvel/2026-07-20_17-02-11/2026-07-20_17-02-11.onnx \
  --command 0.1,0,0
```

```bash
cd /home/cx/robot_deploy

python pc_32_mujoco.py \
  --port auto \
  --policy onnx \
  --model /home/cx/mjlab-recovered/logs/rsl_rl/custom_biped_velocity_nolinvel/2026-07-20_17-02-11/2026-07-20_17-02-11.onnx \
  --command 0.1,0,0
```


测试默认行为：

1. 保持启动时的位置 2 秒。
2. Joint 1 在当前位置附近移动 20°，然后返回起点。
3. 按相同方式依次测试 Joint 2～Joint 6。
4. 如果当前位置距离上限不足 20°，程序自动向负方向测试。
5. 全部完成后保持初始位置，按 `Ctrl+C` 停止程序。

完整测试约需 20 秒。终端会显示当前测试阶段，例如：

```text
motion_test: joint 1 -> 20.000 deg
motion_test: joint 1 returning to start
```

每次运行的关节位置和 IMU 返回数据会自动保存在：

```text
/home/cx/robot_deploy/log/YYYY-MM-DD_HH-MM-SS.txt
```

反转电机：左腿髋、右腿膝、右腿踝

## 单关节 KP/KD 阶跃测试

`single_joint_gain_test.py` 使用单独的 USB 消息 `0x04`。因此运行前必须先编译并
烧录同时修改过的 STM32 固件 `/home/cx/robot`；旧固件不会响应此测试协议。

安全措施：

1. 只给指定关节发送非零 KP/KD；其他五个关节发送 `KP=KD=前馈力矩=0`。
2. KP/KD 使用 SI 单位输入，固件乘 `2π` 转为协议的按圈增益，再经过原厂
   `pid_adjust()` 编码。
3. 目标位置在 PC 和 STM32 两侧均按上表限位。
4. PC 命令持续以 50 Hz 发送；超过 100 ms 未收到新命令，STM32 自动向六个电机
   发送零 PD。
5. 电机反馈丢失、通信故障、速度过高或反馈力矩明显超过上限时立即停止。
6. `Ctrl+C`、正常结束和异常退出都会重复发送六次零 PD 停止命令。

第一次测试前，机器人必须悬空并可靠固定躯干；被测关节运动范围内不能有人或
障碍物。未固定的另外五个关节没有保持力矩，必要时应做机械支撑。

建议从很小的阶跃和力矩开始，例如 Joint 3：

```bash
cd /home/cx/robot_deploy
/home/cx/anaconda3/envs/robot_deploy/bin/python \
  single_joint_gain_test.py \
  --port auto \
  --joint 1 \
  --kp 0.5 \
  --kd 0.0 \
  --max-torque 0.2 \
  --step-deg 3 \
  --cycles 2 \
  --confirm-suspended
```

调参顺序：

1. 固定 `KD=0`，从低 KP 开始。每次只增加一个档位并保存 CSV；只有在响应太慢、
   最终误差较大且力矩没有长期饱和时才增加 KP。
2. 出现持续振荡、明显撞击、过冲继续增大或力矩长期顶到上限时，不再增加 KP，
   回到前一个安全值。
3. 固定选定的 KP，从小 KD 开始增加。KD 的目标是减少振荡和过冲；如果噪声、
   高频抖动或峰值力矩反而增加，则 KD 已过大。
4. 低力矩下克服不了重力/静摩擦时，先改变关节姿态或增加机械支撑；确有需要再把
   `--max-torque` 小步增加，而不是同时增大 KP、KD 和力矩上限。
5. 六个关节分别重复。最终把日志中“量化后 SI 增益”填入 MuJoCo 执行器配置，
   用相同阶跃比较仿真和实机，再开始重新训练。

CSV 保存在 `/home/cx/robot_deploy/log_gain_test/`，包含目标角、位置、速度、误差、
反馈力矩、状态字和运行时 KP/KD。终端同时给出上升时间、过冲、末端误差、最大
速度和最大反馈力矩的近似值。

注意：测试会监测反馈力矩是否明显超过 `--max-torque`，但如果阶跃从未触及限幅，
只能说明“本次未观察到超限”，不能单独证明 0x90 设置的最大力矩寄存器在切换到
0xB0 后一定保留。若要专门验证寄存器保留，需要受控的测功装置或可靠外部力矩计。

## 单关节 KP/KD 正弦测试

`single_joint_sine_gain_test.py` 复用上述 `0x04` 服务协议、关节逻辑方向、限位和
故障保护。程序先保持启动位置，然后平滑增加正弦幅值，采集若干个满幅周期，最后
平滑减幅并回到启动位置。每次运行只测试一组 KP/KD，避免无人确认时自动切换到
更激进的增益。

从低频、小幅值和小力矩开始，例如 Joint 1：

```bash
cd /home/cx/robot_deploy
uv run --python /home/cx/anaconda3/envs/robot_deploy/bin/python \
  single_joint_sine_gain_test.py \
  --port auto \
  --joint 1 \
  --kp 0.5 \
  --kd 0.0 \
  --max-torque 0.2 \
  --amplitude-deg 3 \
  --frequency-hz 0.5 \
  --cycles 6 \
  --confirm-suspended
```

默认在正弦前后各使用一个周期平滑增幅/减幅。`--cycles` 只计算中间满幅段的周期
数；可用 `--ramp-cycles` 修改过渡周期数。目标角在运行前按对应关节的上下限检查，
并检查正弦目标的理论峰值速度低于 `--abort-velocity`。

CSV 保存在 `log_sine_gain_test/`，文件名含关节、频率、KP 和 KD。终端会给出
稳定段的跟踪 RMSE、最大误差、幅值比、相位滞后、最大速度和最大反馈力矩。比较
增益时应对
同一关节保持姿态、幅值、频率、力矩上限和周期数一致；先固定 KD 逐步选择 KP，
再固定 KP 逐步增加 KD。每次修改增益前都应检查上一组波形和力矩，并重新人工确认
测试环境安全。
