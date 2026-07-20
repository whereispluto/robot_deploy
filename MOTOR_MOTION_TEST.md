# 六关节限位与微动测试

## 关节方向约定

- 正向安装的电机保持原始正方向。
- 反向安装的电机以其物理反方向作为逻辑正方向。
- STM32 固件负责反向映射，PC 端使用统一的逻辑关节方向，不再重复取反。

软件中反向的电机：

| FDCAN | 电机 ID | PC 关节编号 |
|---|---:|---:|
| FDCAN1 / PORT1 | 2 | Joint 2 |
| FDCAN1 / PORT1 | 3 | Joint 3 |
| FDCAN2 / PORT2 | 1 | Joint 4 |

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


## 传入策略测试
python pc_32_linux.py \
  --port auto \
  --policy onnx \
  --command 0.15,0,0
```
python pc_32_linux.py \
  --port auto \
  --policy onnx \
  --model /home/cx/mjlab-recovered/logs/rsl_rl/custom_biped_velocity_nolinvel/2026-07-20_17-02-11/2026-07-20_17-02-11.onnx \
  --command 0.1,0,0


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
