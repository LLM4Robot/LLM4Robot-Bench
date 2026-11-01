# L1-01～L1-08 任务资产调研

调研日期：2026-09-28。任务范围：飞书文档第 111 版前八项刚性任务，排除 L1-09。

本文保留资产候选调研时的结论。后续实际选用的资产、参数化替代和本地验证见 [L1 实现说明](l1_tasks.md) 与 [验证结果](archive/l1_validation.md)；运行不依赖本文提到的临时下载目录。

结论：有可复用资产，但没有已核实可直接覆盖八项任务、适配本项目 R5、符合全部评分条件的一套现成场景。优先复用 RoboDojo 的 USDZ 和 ManiSkill 的嵌槽网格；薄片、工具柄、挂环需要补齐或调整几何与评分。所有选择均为候选，尚未完成 R5 抓取、插入或物理稳定性验收。

## 来源与版本

| 来源 | 本次读取版本 | 用途 |
| --- | --- | --- |
| [RoboDojo GitHub](https://github.com/RoboDojo-Benchmark/RoboDojo) | `726e9aabfaa642203722eb126f5eaf0f37f3e1ad` | 任务定义、配置、加载器 |
| [RoboDojo 资产库](https://huggingface.co/datasets/RoboDojo-Benchmark/RoboDojo/tree/43dacb13d3051a9ccd421f4e7e08e4e3eb29dfab/Assets/Object/RoboDojo) | `43dacb13d3051a9ccd421f4e7e08e4e3eb29dfab` | USDZ、metadata.json |
| [ManiSkill](https://github.com/mani-skill/ManiSkill) | `62ff3a5896b4d5b4cf0ac4c8d79afe600c9404a3` | AssemblingKits 的 OBJ 网格和目标描述 |
| [Isaac Lab](https://github.com/isaac-sim/IsaacLab) | `b0542fe2d45bf91c4e1d9ef6952b9c709c80b4e8` | Factory PegInsert 对照候选 |

使用了用户指定的 GitHub 插件读取代码；Browser 插件完成连接，但两次页面导航均超时，没有取得可视页面或截图。网页目录通过官方公开页面及 Hugging Face API 补充核实。不能将本次结果描述为已在 Browser 中完成视觉验证。

RoboProbe 是控制接口与评测结果仓库，其 [setup.md](https://github.com/RoboProbe/RoboProbe/blob/main/docs/setup.md) 明确说明仿真环境另行提供。实际任务资产来源是上表 RoboDojo，不能只克隆 RoboProbe 就认为资产齐全。

## 八项对应关系

下表路径均相对于 RoboDojo 资产库的 `Assets/Object/RoboDojo/`。路径存在与物理任务通过是两个不同的验证阶段。

| 任务 | 匹配来源与资产 | 匹配程度及必须调整的内容 |
| --- | --- | --- |
| L1-01 圆柱插孔 | [insert_tubes](https://github.com/RoboDojo-Benchmark/RoboDojo/blob/726e9aabfaa642203722eb126f5eaf0f37f3e1ad/task/RoboDojo/config/insert_tubes.yml)：`Rigid/test_tube/00002`、`Geometry/test_tube_slot/00000` | 有配套资产，约 1.50 MB。原任务三根试管、多孔架；本任务需明确数量、目标孔、间隙、释放与稳定时间。试管不等于理想实心圆柱，若使用应声明任务版本。 |
| L1-02 异形块嵌槽 | [ManiSkill AssemblingKits](https://github.com/mani-skill/ManiSkill/blob/62ff3a5896b4d5b4cf0ac4c8d79afe600c9404a3/mani_skill/envs/tasks/tabletop/assembling_kits.py) | 形状与槽位高度匹配。OBJ 分离视觉、碰撞模型；需转换或导入 USD，替换 Panda/SAPIEN 接口，保留几何对称性定义，增加释放后稳定条件。 |
| L1-03 插头插接 | [plug_in_charger](https://github.com/RoboDojo-Benchmark/RoboDojo/blob/726e9aabfaa642203722eb126f5eaf0f37f3e1ad/task/RoboDojo/config/plug_in_charger.yml)：`Rigid/charger/00000`、`Rigid/socket/00000` | 有配套资产，约 1.20 MB。需把插座固定，核实插针/孔的低阻配合，加入释放保持；不是通电测试。 |
| L1-04 薄片入槽 | [deposit_coin](https://github.com/RoboDojo-Benchmark/RoboDojo/blob/726e9aabfaa642203722eb126f5eaf0f37f3e1ad/task/RoboDojo/config/deposit_coin.yml)：`Rigid/coin/00000`、`Geometry/piggy_bank/00000` | 部分匹配。原任务从硬币架取币，不能原样替代桌面薄片。需明确薄片形状，调整初始摆放，判定整个薄片经过狭缝并进入接收区域，测试高速穿透。 |
| L1-05 钥匙插锁 | [insert_key](https://github.com/RoboDojo-Benchmark/RoboDojo/blob/726e9aabfaa642203722eb126f5eaf0f37f3e1ad/task/RoboDojo/config/insert_key.yml)：`Rigid/key/00000`、`Geometry/key_slot/00000` | 有配套资产，约 15.28 MB。需只保留插入目标，使用固定锁孔、方向和深度条件，不额外要求旋转开锁。 |
| L1-06 工具柄定点抓取 | [store_tools_in_toolbox](https://github.com/RoboDojo-Benchmark/RoboDojo/blob/726e9aabfaa642203722eb126f5eaf0f37f3e1ad/task/RoboDojo/config/store_tools_in_toolbox.yml) 提供 hammer、pliers、wrench 等 | 仅相关候选。本次未核实符合刷子/刮刀要求且带柄部标注的成套资产。应补刷子或刮刀刚性模型、柄部区域、工作端轴线与目标朝向评分；不能把收纳工具当成本任务。 |
| L1-07 挂环上钩 | [hang_mugs](https://github.com/RoboDojo-Benchmark/RoboDojo/blob/726e9aabfaa642203722eb126f5eaf0f37f3e1ad/task/RoboDojo/config/hang_mugs.yml)：`Rigid/mug/*`、`Geometry/cup_holder/00000` | 部分匹配。杯柄穿挂可参考，但杯子不等于独立刚性环。优先复用或简化挂架，单独建立环，验证环孔未被碰撞凸包封闭及释放后承重。 |
| L1-08 刚性积木定姿堆叠 | [stack_blocks](https://github.com/RoboDojo-Benchmark/RoboDojo/blob/726e9aabfaa642203722eb126f5eaf0f37f3e1ad/task/RoboDojo/config/stack_blocks.yml)：`Rigid/block/*` | 有积木资产。原任务三块不同纹理积木；需锁定少量积木、指定结构与朝向。原评分不能直接替代本任务的姿态和持续稳定判定。也可用轻量参数化积木。 |

## 已做的文件检查

从固定资产版本下载了七个 USDZ 及 metadata.json，共约 19.1 MB，存于 `/tmp/l1-asset-survey/`。用本机 Isaac Sim 所带 USD 库离线打开，不需要模型 API。下面是资产中声明的碰撞类型，不代表已经完成 PhysX 烘焙或插入成功验证。

| 样本 | USD 声明 | 导入注意事项 |
| --- | --- | --- |
| test_tube/00002 | 1 个 convexDecomposition 网格 | 长轴与放置姿态必须从实际变换核实 |
| test_tube_slot/00000 | 1 个 SDF 网格 | 原始包围盒约 20500×9600×6000，元数据目标约 0.205×0.096×0.060 m，相差约 10 万倍；不能直接使用原始变换 |
| charger/00000 | 5 个 SDF 网格 | 保留插针结构，核实 R5 可夹持宽度 |
| socket/00000 | 3 个 SDF 网格，2 处刚体 API | 固定插座时需检查整个层级，不能仅处理根节点 |
| key/00000 | 2 个凸包网格、2 个 Cube、1 个 SDF 网格 | 混合碰撞，需要核实功能端的真实接触 |
| key_slot/00000 | 1 个 SDF 网格 | 锁孔静态固定与尺度核实 |
| cup_holder/00000 | 1 个 convexDecomposition 网格 | 挂架高约 0.349 m；需验证 R5 可达性及挂钩碰撞 |

上游 [GeometryObject](https://github.com/RoboDojo-Benchmark/RoboDojo/blob/726e9aabfaa642203722eb126f5eaf0f37f3e1ad/env/scene_manager/objects/geometry.py) 和 [RigidObject](https://github.com/RoboDojo-Benchmark/RoboDojo/blob/726e9aabfaa642203722eb126f5eaf0f37f3e1ad/env/scene_manager/objects/rigid.py) 会另行设置缩放、姿态与物理参数。因此原始 USDZ 不等于上游运行时场景。

样本摘要与 SHA-256 保存于 `docs/l1_asset_samples.json`，未将第三方二进制模型加入 Git。临时目录不是运行时依赖。

## ManiSkill 与 Factory 的取舍

[ManiSkill 资产清单](https://github.com/mani-skill/ManiSkill/blob/62ff3a5896b4d5b4cf0ac4c8d79afe600c9404a3/mani_skill/utils/assets/data.py) 提供：

- 下载地址：<https://storage1.ucsd.edu/datasets/ManiSkill2022-assets/assembling_kits_v1.zip>
- SHA-256：`e3371f17a07a012edaa3a0b3604fb1577f3fb921876c3d5ed59733dd75a6b4a0`
- 文件组织：`kits/*.obj`、`kits/*.json`、`models/collision/*.obj`、`models/visual/*.obj`、`episodes.json`。
- 原代码对固定槽使用非凸碰撞，对动态块使用多凸体碰撞。转入 Isaac 时不能统一改成单凸包。本次核实了源码和下载清单，未下载并验收该 ZIP。

[Factory PegInsert 配置](https://github.com/isaac-sim/IsaacLab/blob/b0542fe2d45bf91c4e1d9ef6952b9c709c80b4e8/source/isaaclab_tasks/isaaclab_tasks/direct/factory/factory_tasks_cfg.py) 引用 `factory_peg_8mm.usd` 与 `factory_hole_8mm.usd`。杆直径 7.986 mm、孔直径 8.1 mm，直径差 0.114 mm。默认工件配置关闭重力，参数围绕手中工件插装设置。因此它适合参考插装物理配置，不能原样作为宽松配合、从桌面自主抓取的 L1 基线。

## 许可记录

- RoboDojo GitHub 根 LICENSE 为 MIT；Hugging Face 资产库页面标记 Apache-2.0。代码和数据许可分别记录，不能混用。
- ManiSkill 代码根 LICENSE 为 Apache-2.0；下载的 AssemblingKits 数据还需检查包内来源和许可文件，不能仅凭代码许可推断所有网格许可。
- Isaac Lab Factory 源文件声明 BSD-3-Clause；其外部 NVIDIA USD 资产应另行记录对应资产条款。本次未下载这些 USD。
- 后续引入资产时保留许可、来源、固定 revision、校验和及本地修改说明。

## 后续接入方案

1. 共用资产清单与版本，按文件下载并校验 SHA-256；本机只配置缓存路径。Windows/Linux 使用同一模型、尺度、碰撞与任务参数，不按操作系统暗中换资产。
2. 在独立导入层处理尺寸、坐标轴、刚体层级、固定夹具、碰撞和材质，避免把 RoboDojo 或 ManiSkill 全套环境搬入项目。
3. 先对 L1-01/03/05 做资产加载和物理配合验证，再迁移 L1-02；L1-04/06/07 补齐文档要求的几何；L1-08 锁定数量和朝向标记。
4. 资产元数据中的真实目标、功能点、柄部区域只用于构建与独立评分。GPT 仍从 RGB、自身状态与任务文字选择目标、抓取姿态和动作；不增加 YOLO 或其他感知模型。
5. 先验证空孔通行、错误姿态拒绝、释放承重、初始状态不误判成功和种子复现，再做 GPT 闭环。读取 USD 成功不能当作实验成功。
