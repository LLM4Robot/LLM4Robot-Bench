# 独立工程诊断

本目录不接入正式 GPT 决策循环，不计入模型成功率。所有命令均在仓库根目录运行。

拓境实验室公共场景验收（需先安装锁定的 L20 私有交付包）：

```bash
python -m robot_tasks.integrationtest.diagnostics.tuojing_lab --gui
# 验收后保留窗口供鼠标查看：
python -m robot_tasks.integrationtest.diagnostics.tuojing_lab --gui --hold
```

复用生产 `IsaacSimBackend`、`R5Adapter` 与规划器，检查左右臂关节运动、开爪、
末端上移、URDF/物理位姿一致性、相机随机械臂跟随以及三路同步 RGB-D。
使用 `configs/checks/tuojing_lab.toml`；不调用 GPT、不加载任务道具或评分器。
独立子进程运行，父进程检查 `result.json`，避免将 Kit 的零退出码误读为通过。
证据在 `build/tuojing_lab_check/<时间>/`，包括图像、深度、标定、检查结果和日志。
这项验收不代表完整 L1–L3 任务或真实接触已验证。

批量场景诊断 `scene_smoke.py` 不注册为新的 GPT 用例，不调用模型：

```bash
python -m robot_tasks.integrationtest.diagnostics.scene_smoke --all
python -m robot_tasks.integrationtest.diagnostics.scene_smoke --all --no-preparation
```


前者验证共同准备动作、场景、RGB-D 和初始评分；后者跳过规划器，只验证物理场景。
加 `--prop-probe` 可对有检查定义的空腔承重、滑轨和螺纹做物理夹具诊断。
L1-02 会先调用同目录 `plug_holder_probe.py` 验证自由底座对充电头的承托与拔出，再检查插线板接触几何；这些检查不调用 GPT，也不代表机器人自主完成任务。
`prop_physics.py` 会明确设置诊断初态或施加测试驱动，不能作为机器人任务成功视频或 GPT 成功率。
结果位于 `runs/`，与正式测评分别保存。

L2 原生资产诊断：`python -m robot_tasks.integrationtest.diagnostics.scene_smoke L2-02 --no-preparation --prop-probe`。ID 可换为 L2-01、L2-03、L2-04、L2-05。`native_l2_probe.py` 检查拔帽、抽书、纸杯珍珠初态和原生滑轨；`orange_probe.py` 检查软体接触、压缩及释放。施加的是诊断夹具驱动，不是 GPT 或机器人任务成功。

笔帽／绿盒接触回归：`python -m robot_tasks.integrationtest.diagnostics.marker_contact --baseline`。覆盖原释放姿态、已穿透状态压力测试以及四种落点／朝向，检查后段接触力、速度和相对原外观表面的穿透；默认输出 `build/marker_contact/result.json`。不调用 GPT，不改变任务成功条件。

`python -m robot_tasks.integrationtest.diagnostics.l2_clearance` 在规划器环境检查五个任务的采样抓取／操作姿态，默认输出 `reports/l2_asset_validation/clearance.json`。范围包括关节限位、机械臂／相机／支架对桌面和固定道具的包围盒间隙；不证明连续路径、全身自碰撞或真实抓持稳定性。

局部 R5 接触与执行检查（Isaac Python，不调用模型）：

```bash
python -m robot_tasks.integrationtest.diagnostics.l2_execution_probe L2-05 --poses reports/l2_asset_validation/clearance.json --output reports/l2_protocol_validation/L2-05-motion
```

它使用上一项的显式采样位姿布置张开的夹指，再通过生产控制器闭合、沿目标方向移动 20 mm，持续检查物理状态并保存三视角图片。ID 可换成五个 L2 任务。`--width` 检查不同夹爪开口，`--travel` 调整局部位移（通常不超过 50 mm，橘子提起检查允许到 80 mm），`--velocity-iterations` 隔离求解器参数实验。这些参数只影响工程检查。它不经过视觉选抓点，不是脚本任务参照，不构成完整机器人任务或 GPT 成功证据。正式任务的初态、指令和评分不被该检查修改。

`L3-01` 的原生东方树叶瓶由左臂扶持、右臂取盖；空间采样诊断检查一组扶瓶与抓盖姿态的可达性及双臂、相机支架间距。在包含 NumPy、SciPy、trimesh 的规划器环境运行：

```bash
python -m robot_tasks.integrationtest.diagnostics.bottle_clearance
# 在 Isaac Python 中渲染上一步报告保存的检查姿态：
```

使用 URDF 网格包围盒和相机／支架 CAD 尺寸检查双臂互相及相机与道具的间距，只排除预期的扶瓶夹指—瓶身接触。数值 IK 只用于此诊断，不接入生产控制；结果保存在 `reports/native_bottle_clearance.json`。这不是完整轨迹、自碰撞认证、实际夹持或 GPT 成功证明。旧方瓶的 1,420 姿态报告不适用于现行东方树叶瓶。

笔、纸杯和空顶帽的夹持回归可在 Isaac Python 中串行执行：

```bash
python -m robot_tasks.integrationtest.diagnostics.native_grasp L1-01 --output build/native_grasp/L1-01
python -m robot_tasks.integrationtest.diagnostics.native_grasp L1-04 --output build/native_grasp/L1-04
python -m robot_tasks.integrationtest.diagnostics.native_grasp L1-03 --output build/native_grasp/L1-03
```

检查各目录 `grasp.json` 的 `passed`。该诊断仅在开始时设置机械臂检查姿态，
随后靠真实关节驱动闭合、抬升和释放，检查双侧接触、初始无穿透、物体随动，
以及接触点到原始可见夹指 STL 的距离、生产 RGB-D 抓持验证是否通过；
不会调用 GPT，也不是自主任务成功结果。加 `--check-planner` 可检查从初始姿态
到抓持姿态的规划，诊断的实际夹持仍从重置姿态开始，不代表执行过整条接近路径。
需要 Isaac Python 环境中的 SciPy、trimesh 和 rtree。
L1 使用独立机器人配置和贴合表面的凸分解，名义开口为原始 URDF 允许的 0–70 mm；
帽圈与杯壁可由竖直夹指从上方夹持，物体原生尺寸与空腔保留。

当前六个 L1 已使用拓境场景。逐项验收可用：

```bash
python -m robot_tasks.integrationtest.diagnostics.scene_smoke L1-02 --prop-probe --output build/tuojing_l1_migration/L1-02
```

将 ID 替换为 L1-01、L1-02、L1-03、L1-04、L1-05。覆盖双臂准备、三路 RGB-D、
初态评分、插座/杯腔/挂杆与刚体玫瑰承托；无需模型令牌。检查 `result.json` 的
`passed`，不能只看 Isaac 进程退出码。无任务执行轨迹或模型成功率含义。

L2-04 物性对照沿用 `l2_execution_probe`：`--soft-physics <JSON>` 只覆盖本次诊断的 `[orange_physics]` 参数，`result.json` 保存原生值和实际生效值。固定抓取姿态和评分条件，对比闭合、抬升；当前 `--travel 0.075 --finish` 检查承台上夹起 5 cm，不再搬运释放；源资产和正式任务配置不被诊断改写。工程夹具检查不调用 GPT，也不证明自主选抓点成功。

## L3 新道具接触检查

所有诊断都不调用模型，结果不要并入 GPT 成功率。`initial_gallery` 检查场景、RGB-D 和未完成初态；运行它可查看新摆放。

```bash
python -m robot_tasks.integrationtest.diagnostics.initial_gallery L2-07 --output reports/asset_checks/L2-07
python -m robot_tasks.integrationtest.diagnostics.l3_asset_probe L3-02 --output reports/l3_assets_20261008/L3-02-contact
python -m robot_tasks.integrationtest.diagnostics.l3_asset_probe L2-07 --output reports/asset_checks/L2-07-contact
python -m robot_tasks.integrationtest.diagnostics.l3_asset_probe L3-04 --output reports/l3_assets_20261008/L3-05-contact
```

`l3_asset_probe` 只做隔离部件诊断：螺丝用额外同轴导向和外力矩验证真实旋转推进；板擦通过测试速度和单独的抓持标志检查接触擦除，生产评分的未抓持状态仍不得擦除；簸箕预置方块验证承托/篮内判定，不代替扫入过程。正式任务中没有这些诊断导向、速度或标志。

规划器 Python 可运行 `nut_clearance` 和 `l3_tool_clearance`，检查若干夹爪/相机包围盒候选姿态与固定物的间隙；它们不生成供 GPT 使用的目标，不证明连续无碰撞轨迹或稳定抓持。

当前 L3-04 接触检查使用固定簸箕，只注入用于检验篮内范围与承托的方块初态；不代表扫入成功。L3-01 的 `l3_asset_probe` 使用临时同轴夹具和有限扭矩/拉力校验盖的物理退出，不会进入生产场景。

L3-01 瓶盖被动接触回归（当前原版和扰动版）：

```bash
python -m robot_tasks.integrationtest.diagnostics.bottle_contact L3-01 --output build/bottle_contact/original
python -m robot_tasks.integrationtest.diagnostics.bottle_contact L3-01_variant01 --output build/bottle_contact/variant01
```

使用 Isaac Python 串行运行，检查 `bottle_contact.json` 和 `result.json` 均为通过。
前者包含静置、0.05 N 侧推、1/5 N 下压、0.15 N 直接上拉、旋出及瓶口上方 5 cm
下落的状态；后者检查生产复位、三路图像与初始评分。测试的外力和力矩只存在于
这个独立诊断中；旋出用的强力矩脉冲不代表真实夹爪动作或实测开盖扭矩。
诊断没有模型调用，也不证明机器人抓持稳定或任务成功。接触修订说明见
[瓶盖物理设置](../../../../docs/l3_01_bottle_unscrew.md)。

### L3-02 原生松螺母（2026-10-09）

当前任务是双臂旋下预松螺母，旧的粗螺丝旋入诊断不再作为任务入口。

```bash
python -m robot_tasks.integrationtest.diagnostics.nut_clearance --output reports/nut_clearance.json
python -m robot_tasks.integrationtest.diagnostics.nut_probe --output reports/nut_thread
python -m robot_tasks.integrationtest.diagnostics.nut_grasp_probe --poses reports/nut_clearance.json --output reports/nut_grasp
```

依次验证采样净空、受限夹具中的真实螺纹旋出、预置机器人姿态下双指夹持与 10 mm 搬动；均无 GPT，不代表完整任务成功。只在独立诊断中设置夹具和机器人初态，正式任务不调用它们。

L1-04 稳定夹持与脱手观测回归（不调用 GPT）：

```bash
PYTHONPATH=src python -m robot_tasks.integrationtest.diagnostics.native_grasp L1-04 --hold-s 30 --grasp-style wall --output build/grip_control/wall
PYTHONPATH=src python -m robot_tasks.integrationtest.diagnostics.native_grasp L1-04 --hold-s 30 --grasp-style body --output build/grip_control/body
```

使用 Isaac Python 串行运行。该检查复用生产夹爪控制，诊断复位后抬起并保持 30 秒，
张爪后做 5 mm 竖直诊断试退。它不执行完整套杯任务，也不向 GPT 提供诊断姿态。
检查 `grasp.json` 的 `control_passed`、`stability`、`actual_finger_limits`、
`release_physically_complete` 和 `release_after_probe`。杯身抓法在最大开口下可能仍卡住，
此时整体 `passed=false` 是需要保留的物理释放失败证据，不应通过增大杯口或关闭碰撞掩盖。
`release_after_probe` 只用 RGB-D 与机器人状态作提示；没有接触力、物体真值或评分输入。
遮挡、纹理重复和位移过小可返回未知，`stationary_after_probe` 也不认证承托或任务完成。

L1-04 和 L1-05 的 `stable_gripper_drive=true` 启用软件对称驱动、物理关节限速与闭合误差限制。
`finger_max_acceleration` 单位 m/s²，`finger_closing_lead` 为每指最大闭合领先量，
`finger_effort_limit` 为每指驱动力上限 N。软件同步不代表已确认真实夹爪机械联动。
L2 与常规 L3 保留 12–70 mm，L3-02 保留 10–70 mm；L2／L3 使用各自的夹指凸分解碰撞覆盖，不叠加 L1 参数。其它任务默认保持其原有控制方式。新结果记录 `release_check_protocol`，避免与旧测评混淆。

加 `--place-nested` 会在保持检查之后调用生产规划器执行升高、搬运与浅入杯诊断。
规划失败写入 `placement_failure.json` 并使诊断失败；不会关闭碰撞检查或改用直接传送。
该选项使用明确的诊断几何目标，不属于 GPT 自主任务或成功率统计。

L1-05 原生花茎规划与稳定夹持回归（不调用 GPT）：

```bash
PYTHONPATH=src python -m robot_tasks.integrationtest.diagnostics.native_grasp L1-05 --plan-lift --hold-s 30 --output build/rose_grasp/full_lift
```

在 Isaac Python 中执行。沿花头长轴的诊断抓取姿态由复位设定，随后闭爪、规划抬升、保持和释放
使用物理驱动。`--plan-lift` 经过生产规划器的携带物碰撞检查；失败写入 `lift_failure.json`。
`--lift-m` 只改变诊断的抬升距离，不会写入任务提示词或向 GPT 提供抓取坐标。
检查 `grasp.json` 中的 `lift_plan`、`rise_m`、`control_passed` 和 `passed`，不能仅看
`result.json` 的场景检查或进程退出码。长花茎可能在短距离提花头后仍触桌，不能将花头
移动或双指有接触等同于整朵花离桌。

L1-05 静置启动回归（不调用 GPT）：

```bash
PYTHONPATH=src python -m robot_tasks.integrationtest.diagnostics.rose_rest --output build/rose_rest
```

复用生产初始化后，再自由推进 30 秒，检查 `rest_result.json` 与 `result.json` 均通过。
复位姿态使用锁定原生碰撞形状与重心的低势能落座方向：长轴在世界 XZ 平面，
在俯仰和绕长轴转角中最小化承托后的重心高度，最后给最低表面 0.01 mm 初始净空。
它只是场景的初始布置，不是抓取位姿或动作脚本。原资产质量、惯量、摩擦、阻尼未改。
生产入口最多等待 120 秒仿真时间，要求连续 2 秒线速度低于 1 mm/s、角速度低于
0.015 rad/s、位置漂移不超过 0.5 mm、完整姿态角变化不超过 0.01 rad，且由桌面承托、
没有夹指接触；不满足时在调用模型前报错，不锁定物体或强行清零速度。
这项启动检查使用仿真测量，只决定场景何时就绪，不将物体真值交给 GPT。

场景检查 `scene_smoke` 和初态图库默认跟随任务的 RGB 设置；`scene_smoke --depth` 同时验证深度。专门验证深度抓持的 `native_grasp`、`tuojing_lab` 诊断显式开启深度，不改变正式任务默认值。

`wrist_collision <run目录> --step 37 --output <证据目录>` 回放已记录的关节路径，独立验证腕相机/支架的物理碰撞与执行中止；不调用 GPT，故意绕过规划器，不能算任务成功。`--isolate-box` 仅在诊断中关闭刷子接触以单独检查绿盒；`--baseline` 仅在诊断中关闭腕部附件碰撞以复现旧问题。两者均不修改任务配置或源资产。

`--check-retreat` 在接触停止后另行下发返回原清空姿态的诊断动作，检查接触反馈不会将后续退出动作锁死；生产流程仍由模型选择恢复动作。
