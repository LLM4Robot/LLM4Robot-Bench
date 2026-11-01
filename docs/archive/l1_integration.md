> 历史归档：本文保留当时的设计和验证记录，旧命令、参数与任务编号不代表当前版本。当前入口见 [项目 README](../../README.md)。

# L1 与 Lula / RGB-D 分支集成
> 历史记录：以下是 master 合并前的实现与验证条件。当前采用统一 RGB-D 流程、右臂观测和 100 次默认预算，仅保留 GPT 抓放集成用例；本文旧配置开关和检查命令不再适用。当前使用说明见 [L1 任务](../l1_tasks.md)。

集成分支：`houym/L1_dev`。本地开发提交 `6d54706` 与同事提交 `a245154`
通过合并保留双方历史；不改写 `master`，不强制推送。

## 保留与衔接

- 保留八项 L1 任务、独立过程评分、外部任务资产清单、室内环境、黑银外观及 GUI 持续查看。
- 保留同事的 Lula 逆解、RGB-D、像素深度查询、决策笔记、失败重试、SSE 响应和录像改进。
- 旧 `tasks/cylinder_insertion.py` 按双方修改删除；保留已重写的 L1-01 配置，注册到
  `tasks/l1/task.py`。任务注册遵循新的 `active_arm_counts` 元组协议。
- 连续路径统一调用 Lula。原生回归发现默认 BFGS 梯度停止阈值过早结束局部求解，
  触发随机重启并跳到另一关节分支，因此阈值改为 `1e-9` 并写入 `scene.json`。
  位置／姿态容差、速度限制、桌面碰撞检查、关节跳变拒绝均保留。
  [NVIDIA Lula 参数说明](https://docs.isaacsim.omniverse.nvidia.com/4.5.0/py/source/extensions/isaacsim.robot_motion.lula/docs/index.html)。
- 任务终止或夹爪受阻均能结束当前动作等待；超时继续取消待执行的路径。
  启用失败重试时，每次复位后重新执行固定起始姿态，再给 GPT 新观测，保留模型历史与总预算。
- 保留较长的模型请求超时设置，但拒绝无穷、NaN、非正数和错误类型。

L1 默认仍为左臂执行、三路 RGB；同事的新增输入都是显式选项，没有静默启用深度或右臂操作。
要进行 RGB-D 实验，在共享任务配置设置 `observation.include_depth=true`、
`policy_options.camera_calibration=true`，保留 `observation.include_feedback=true`；
需要原始深度记录时设置 `recording.save_depth=true`。这是新的观测条件，须另行记录和比较。
L1 评分仍约束非操作臂静止，不能只改 `active_arms` 就当作双臂任务。

## 复现检查

在项目根目录执行，Windows 把 `python3` 换成 `python`。

```bash
python3 -m pip install -e .
python3 tools/fetch_l1_assets.py
python3 -m unittest discover -s tests -v
python3 tools/launch.py check l1-motion
python3 tools/launch.py check l1-grasp
python3 tools/launch.py check cameras --config configs/checks/r5_cameras_rgbd.toml
python3 tools/launch.py check l1-scene --config configs/tasks/l1_01_cylinder_insertion.toml --gui
```

原生 IK 回归须用本机 Isaac Python；普通软件 CI 未安装 Lula 时会明确跳过对应测试：

```bash
python -m unittest discover -s tests -p test_motion_execution.py -v
python -m robot_tasks.integrationtest.lula_check --config configs/tasks/l1_01_cylinder_insertion.toml --output reports/lula.json
```

GUI 场景检查保存结果后保持打开，直到 Ctrl+C 或关闭窗口；去掉 `--gui` 自动退出。
上述检查均不代表 GPT 完成了八项任务。在线模型校验使用已有私有配置和终端密钥，
它们不提交到仓库。本次服务端返回 HTTP 429（API key 7 天限额已用完），在线决策暂未验证。

## 本次合并后的本机验证（2026-09-29）

- 系统 Python：240 项，238 通过、2 项按环境跳过；Isaac Python：243 项，238 通过、5 项按环境跳过。两个环境互补覆盖 USD 几何与原生 Lula。
- 原生 Lula：120 个双臂、双坐标系位姿样本通过，不可达目标被拒绝。
- `L1_motion_20260929T083457Z_f4970ed1`：原首步故障轨迹、桌面拒绝及恢复通过。
- `L1_grasp_20260929T083603Z_34ba3e8c`：两次物理抓放与张爪负例通过。
- 三路 RGB-D 检查通过，模型输入组装含 6 张图（3 RGB + 3 深度预览），像素反投影／重投影通过；此检查使用模拟模型回复，未调用外部 API。
- 八项场景检查均通过；场景元数据确认均为 Lula 与 `cartesian_linear_v1`。

以下证据目录位于本机 `runs/`，按仓库规则不推送：

| 检查 | 本机证据目录 |
| --- | --- |
| scene_l1_01_cylinder_insertion | `L1_scene_20260929T083940Z_69b761e1` |
| scene_l1_02_shaped_recess | `L1_scene_20260929T083954Z_bbd9edcb` |
| scene_l1_03_plug_insertion | `L1_scene_20260929T084008Z_13b3b7c5` |
| scene_l1_04_plate_slot | `L1_scene_20260929T084022Z_ee982d52` |
| scene_l1_05_key_insertion | `L1_scene_20260929T084034Z_08047532` |
| scene_l1_06_tool_handle | `L1_scene_20260929T084048Z_1619039e` |
| scene_l1_07_ring_hook | `L1_scene_20260929T084102Z_6c2f3b6f` |
| scene_l1_08_block_stack | `L1_scene_20260929T084116Z_84b1b6b1` |
| rgbd | `R5_cameras_20260929T084129Z_227c331b` |

未完成 Windows 实机 GPU 验证；GitHub 连接器对本仓库 Actions 查询返回 404，因此不声称远端 CI 已通过。
