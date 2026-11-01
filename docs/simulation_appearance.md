# 室内环境与 R5 黑银外观

L1 八项任务默认使用 `lab_room_v1`：5.2 × 5.4 × 3 米房间、
墙体、门窗、顶灯、储物柜，以及带木纹台面和金属桌腿的工作台。
R5 使用 `r5_black_silver_v1` 黑色机身／银色金属分区材质。

## 启动

在仓库根目录执行；Windows 将 `python3` 换成 `python`：

```bash
# 可选：校验或恢复贴图和机器人视觉资产（正常克隆已包含这些文件）
python3 tools/fetch_appearance_assets.py

# 正式任务：使用已有模型配置与密钥环境变量
python3 tools/launch.py run L1-01 --gui
```

`run` 会调用真实模型；三路录像和场景信息保存到该次 `runs/` 目录。可在 Isaac Sim 的 Perspective 视口中缩放／旋转，视口操作不修改模型接收的三路相机。当前抓放任务保留原场景，L1 使用本节的室内外观。

## 配置与实验可比性

共享任务配置的 `[simulation]` 中分别选择：

```toml
environment = "lab_room_v1"          # 或 plain
robot_appearance = "r5_black_silver_v1" # 或 vendor_white
```

未指定时保留 `plain`／`vendor_white` 默认行为。Windows 与 Linux 使用相同
资产和配置；本机 Python 路径仍只放在 `configs/runtime/local.toml`。

机器人基座、台面顶面 Z=0、工件坐标、相机参数、关节、质量、摩擦、原有
碰撞模型、运动规划、动作协议和评分逻辑不因外观配置而改变。
原台面碰撞体保留，仅隐藏其渲染并覆盖等尺寸带 UV 的台面。
房间地板与墙体在工作区外具有碰撞；桌腿、储物柜、门窗是静态视觉装饰，
不作为可操作物体，不能据此评估它们的接触行为。

**背景、反光和照明改变了 RGB 观测条件。** 固定本版本配置后再开展比较，
不要把旧白底与新室内环境的成功率混在一起。
`scene.json` 记录环境、机器人外观、资产清单哈希及 link5 视觉补偿；
`config.toml` 保存任务配置。三路传感器保持原参数，没有新增感知模型。

## 公开资料与实现选择

- [ARX 官方 R5a 模型](https://github.com/ARXroboticsX/ARX_Model/tree/74386441845f12af4da203843b6400951d6c5530/R5/R5a)
  提供现用几何与 URDF，检查到的该版本未包含厂家原始 UV 图片。
- [公开黑银 R5 GLB](https://github.com/fiveages-sim/robot-descriptions-arx/tree/71b069961eafe3d71524d199d99ac24c752b348a/arx5_description/meshes/r5)
  提供黑银 PBR 分区。本实现导入其材质和视觉网格，并修正与厂家 R5a
  不同的 link5 原点；不会替换物理或运动学模型。原始许可随资产保留。
- [NVIDIA 室内场景目录](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/assets/usd_assets_environments.html)
  提供 Simple Room 等场景。本项目采用本地程序化房间以保持现有桌面坐标，
  避免启动依赖远程 USD 资源；没有引入该目录里的资产。
- [Poly Haven 木纹](https://polyhaven.com/a/kitchen_wood)、
  [地板](https://polyhaven.com/a/concrete_floor_02)使用
  [CC0](https://polyhaven.com/license)贴图，台面与地面有实际 UV 坐标及
  漫反射、粗糙度、法线贴图。

当前实现对应已确认的标准黑银外观，**不是厂家原始 UV 贴图，也不是对真机
颜色、反射率和贴纸的精确标定复刻**。若以后获得官方材质资产，可只替换视觉
配置，并更新版本与资产哈希。

## 合并前的历史验证（2026-09-29，Ubuntu / Isaac Sim 5.1）

以下运行使用 master 当时的执行器与检查入口，不代表当前公共流程的复测结果。

- 软件回归：190 项，189 通过、1 项按环境条件跳过。新增测试核对资产哈希、
  视觉网格与厂家 STL 对齐、物理对象不被视觉覆盖修改，以及台面尺寸与 UV。
- `runs/L1_scene_20260929T075759Z_18a851ff/`：最终照明下三路 RGB、
  初始非成功、试管架支撑、夹具落座和同种子复位均通过。
- `runs/L1_grasp_20260929T075448Z_4432e7bd/`：室内环境与黑银视觉网格下，
  两次抓放正例及一次张爪负例均通过。此抓放录像使用调亮前的灯光；随后只调整
  灯强并增加窗边光源，以前述场景检查确认最终渲染，未再改动物理模型。

这些是工程回归，未调用 GPT，不表示八项 L1 任务已经由 GPT 完成。
