# 脚本目录索引

发布整理新增 `reproduce.py` 与 `release/`：复现计划、行为基线检查、初态渲染、资产恢复和干净打包。使用说明见 `docs/release/`。

以下是既有工具的用途索引。原项目、历史实验和任务资产保持原样。

本机日常运行主要使用 `launch.py`、`reproduce.py` 和 `batch_run.py`。远端部署按需使用 [remote-deploy skill](../skills/r5-remote-deploy/SKILL.md)。资产已随项目提供，启动任务不需要先执行资产生成、修复或安装脚本。

```text
tools/
├── launch.py、batch_run.py       日常运行入口
├── remote/                      旧命令兼容入口；实现位于可选 skill
├── diagnostics/                 视角检查、L2 初态对照图
├── recover_l3.py                基于兼容记录恢复 L3
├── generate_l1_position_variants.py
├── prepare_*、build_*、fetch_*…   资产安装、重建、导出
├── l1_*、l2_*、video_*、mug_*…    资产构建依赖模块
└── *.ps1、setup_curobo.sh        本机环境与检查工具
```

## 本机运行与可选远端部署

| 入口 | 用途 |
| --- | --- |
| [launch.py](launch.py) | 本机任务和工程检查启动器。 |
| [batch_run.py](batch_run.py) | 本机批量运行，创建 `runs/` 记录。 |
| [reproduce.py](reproduce.py) | 按已有任务配置生成计划或显式执行。 |
| [setup_runtime.sh](setup_runtime.sh) | 为本机 Linux 的既有 Isaac/cuRobo 安装创建包装器。 |
| [recover_l3.py](recover_l3.py) | 兼容特定测量状态记录包的历史恢复入口。 |

SSH 配置、同步、远端状态、桥接和记录收集集中在
[skills/r5-remote-deploy](../skills/r5-remote-deploy/SKILL.md)。正常本机实验不需要它。
`remote/fleet.py`、`remote/remote_entry.sh` 和 `remote/recovery_planner.sh` 只转发旧命令，保留已有部署的兼容性。

## 检查与预览

以下工具不调用 GPT。它们需要已有渲染／真机证据，或在 Isaac 中生成初态。

| 脚本 | 用途与输入 |
| --- | --- |
| [diagnostics/audit_reference_image_camera.py](diagnostics/audit_reference_image_camera.py) | 对锁定的 L3-01 用户参考图检查原生渲染误差；默认读取旧对齐目录下的 `overview.png`，需要 OpenCV。 |
| [diagnostics/audit_shared_camera_framing.py](diagnostics/audit_shared_camera_framing.py) | 检查公共机位、桌面／背景边缘和逐任务构图；需要已有初态图片、USD 和 OpenCV。 |
| [diagnostics/audit_view_pose_alignment.py](diagnostics/audit_view_pose_alignment.py) | 检查真机／仿真视角与机械臂物理复位；可指定 `--initial-root`、`--output`，仍需对应真机证据。 |
| [diagnostics/preview_l2_initial_layouts.py](diagnostics/preview_l2_initial_layouts.py) | 渲染 L2 原版／P01 初态并合成对照图；需 Isaac 与中文字体。`--compose-only` 仅合成已有图片。 |

上述四个预览脚本从 `tools/` 根目录移动到 `tools/diagnostics/`，文件名保持不变；引用和测试已随路径调整。它们是可复用工具，但默认的历史输入目录不会随干净的 Git 检出提供。

例如，在配置好 Isaac 的 Python 环境中，从项目根目录预览：

```bash
PYTHONPATH=src python tools/diagnostics/preview_l2_initial_layouts.py \
  --tasks L2-02 L2-03 --output build/l2_initial_preview
```

## 资产安装、构建与校验入口

这些是手动维护工具。部分仍面向旧版本的资产包，**不构成 Combined 当前全部资产的一键重建流程**。重建后需要核对输出、任务清单和运行资产锁，不能把重建等同于验证通过。

| 脚本 | 用途与执行影响 |
| --- | --- |
| [diagnostics/export_planner_geometry.py](diagnostics/export_planner_geometry.py) | 从已有运行的缓存导出规划几何；会更新规划参考资产。 |
| [verify_runtime_assets.py](verify_runtime_assets.py) | 默认校验当前运行输入；`--usd` 检查 USD 引用，`--write-lock` 才改写资产锁。 |
| [prepare_video_scenes.py](prepare_video_scenes.py) | 从原始 USDZ 生成运行层，串联外观、碰撞体和摆放修订；会改资产和元数据。 |
| [prepare_combined_original_scenes.py](prepare_combined_original_scenes.py) | 从记录的可逆变换重建 L3-01／05／11 原版覆盖层；会更新场景及元数据。 |
| [build_video_scoring_geometry.py](build_video_scoring_geometry.py) | 从完整场景提取评分几何，写入 `video_scoring_geometry.json`。 |
| [build_l1_scoring_geometry.py](build_l1_scoring_geometry.py) | 从旧锁定 L1 道具提取评分几何，写入 `l1_scoring_geometry.json`；保留旧代理用途。 |
| [install_video_scenes.py](install_video_scenes.py) | 安装 10 月 15 日完整场景包；会替换场景和清单，不是日常任务前置步骤。 |
| [tuojing_assets.py](tuojing_assets.py) | 安装／验证锁定的实验室资产包，同时被下载工具依赖。 |
| [fetch_tuojing_release.py](fetch_tuojing_release.py) | 通过 GitHub CLI 下载指定资产发布包并安装；需相应仓库访问权限。 |
| [fetch_l1_assets.py](fetch_l1_assets.py) | 下载锁定的旧 L1 道具资产。 |
| [fetch_r5_assets.py](fetch_r5_assets.py) | 下载锁定的 R5 机器人描述。 |
| [fetch_appearance_assets.py](fetch_appearance_assets.py) | 恢复锁定的房间贴图与机器人外观。 |
| [convert_camera_cad.py](convert_camera_cad.py) | STEP 相机外壳转可视 USD；需独立 CadQuery 环境，配置见 [camera_assets_environment.yml](camera_assets_environment.yml)。 |
| [export_yagu_marker.py](export_yagu_marker.py) | 从 L1-01 当前运行层导出马克笔 USDZ；会写入 `appearance/`。 |
| [export_l103_measured_holder.py](export_l103_measured_holder.py) | 导出 L1-02 实测托盒及尺寸来源；会写入 `appearance/`。 |

## 资产构建依赖模块：19 个

这些不是可以直接清除的临时脚本。`prepare_video_scenes.py` 等构建工具及几何测试仍在导入它们，本轮保留模块名和位置。

| 模块 | 负责内容 |
| --- | --- |
| [video_scene_metadata.py](video_scene_metadata.py) | 从 USD 提取刚体坐标和场景记录。 |
| [video_scene_repairs.py](video_scene_repairs.py) | 场景物理绑定和修订入口。 |
| [video_scene_contents.py](video_scene_contents.py) | 玻璃球等任务内容数量及初态。 |
| [video_visual_r3.py](video_visual_r3.py) | 房间、桌面、背景外观修订。 |
| [l1_02_measured_base.py](l1_02_measured_base.py) | 实测插头托盒。 |
| [l1_02_height_spacer.py](l1_02_height_spacer.py) | 按真机画面估计的场景垫高件。 |
| [l1_03_contact_geometry.py](l1_03_contact_geometry.py) | 帽圈／支撑盒碰撞几何。 |
| [l1_03_initial_layout.py](l1_03_initial_layout.py) | 帽子任务初始摆放。 |
| [l1_05_flower_collision.py](l1_05_flower_collision.py) | 花头接触外壳。 |
| [l1_05_grip_dynamics.py](l1_05_grip_dynamics.py) | 花茎抓取摩擦和质量参数。 |
| [l1_05_passive_leaves.py](l1_05_passive_leaves.py) | 被动叶片和花茎接触近似。 |
| [l1_05_supported_layout.py](l1_05_supported_layout.py) | 花／充电器在杯内的支撑初态。 |
| [l2_measured_support.py](l2_measured_support.py) | 实测公共方盒尺寸。 |
| [l2_book_rim_fit.py](l2_book_rim_fit.py) | 书本与盒沿贴合、弯曲外观和碰撞近似。 |
| [l2_marker_asset.py](l2_marker_asset.py) | L2-01 可分离笔杆／笔帽外观。 |
| [l2_video_placement.py](l2_video_placement.py) | L2 书本、杯子、橘瓣、抽屉初态。 |
| [mug17_profile.py](mug17_profile.py) | L1-01 杯体轮廓。 |
| [mug_rim_profile.py](mug_rim_profile.py) | 杯口和边缘外观。 |
| [yagu_marker_appearance.py](yagu_marker_appearance.py) | 马克笔标签和材质。 |

## 初态配置生成：1 个

[generate_l1_position_variants.py](generate_l1_position_variants.py) 生成固定种子的 L1 P03 候选摆放和拒绝记录，进行几何／可达性筛选，不调用 GPT。它会写入 `configs/task_references/`，不是每次运行前自动采样；当前原版／`_v3` 直接读取已确定的配置。

## 环境与 Windows 入口：4 个

| 脚本 | 用途 |
| --- | --- |
| [setup_curobo.sh](setup_curobo.sh) | 安装独立 cuRobo 运行环境；会修改指定环境目录。 |
| [check_framework.ps1](check_framework.ps1) | Windows 软件测试、机器人资产和配置检查。 |
| [run_r5_check.ps1](run_r5_check.ps1) | Windows 工程检查／调试入口；模型检查可能调用 GPT。 |
| [research_storage.ps1](research_storage.ps1) | Windows 缓存路径设置，依赖机器本地 `configs/runtime/windows.json`。 |

## 工具目录外的脚本和临时文件

- `assets/tuojing/source/common/lab/` 中的 **9 个 Python 文件**是供应商实验室资产来源的一部分：`build_and_validate.py`、`launch_isaac.py`、`validate_delivery.py`、`mounts/convert_step.py`，以及 `reconstruction/` 下的 `constrained_ba_v2.py`、`fit_assets.py`、`metric_fit.py`、`prepare_scene.py`、`reconstruct.py`。它们属于锁定资产包，保留原位；当前公共启动器不会执行这些供应商脚本。
- `src/` 是公共运行实现，`tests/` 是回归测试，不归为临时脚本。
- 本次合并使用的一次性传输、比对和验证脚本在本机 `/tmp/gpt_merge_audit_20261020/`、`/tmp/gpt_hym_merge_20261020/` 中，未放入发布项目。它们是本机审计文件，不是运行依赖。
- 新的一次性试验脚本和输出统一放 `build/<事项>_<日期>/`；临时下载放资产缓存目录；录像、结果按 `runs/`、`reports/`、`exports/` 分类。这些目录已被 Git 忽略，远端同步也排除它们。
- 验证通过、需要长期复用的工具再迁入 `tools/` 对应分类，并在本索引登记。不要将正式运行必需的输入放到上述临时目录中。

本次清理移除了已停用的旧专题报告脚本，以及 `build/initial_gallery_20261021/` 中已完成的截图采集、分发、旧任务预览、收集和排版脚本与传输包；已生成的照片、检查结果和历史来源记录保留。
