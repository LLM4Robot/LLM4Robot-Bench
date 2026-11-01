# GPT 批量运行与录像

从仓库根目录使用 `python3 tools/batch_run.py`。入口由普通 Python 3.11+
启动，自动使用 `configs/runtime/local.toml` 配置的 Isaac Python 执行任务，
无需先激活 Isaac 环境。默认串行运行全部 16 个已实现仿真任务，包含 L2-07 接触擦除代理；L3-16 已移除。不改变任务、评分或 GPT 控制流程。

## 先验证入口

```bash
cd /home/hym/Documents/ChatGPT/Tuojing/GPT_Benchmark
python3 tools/batch_run.py --all --dry-run --output runs/batch_flow_check
```

`--dry-run` 使用现有占位流程，不读取 `.env`、不调用 API、不启动 Isaac，
不生成任务录像，也不能作为任务成功证据。该批次的模式会锁定，不能续跑成正式实验。
场景、RGB-D、准备动作的物理预检仍使用：

```bash
PYTHONPATH=src /home/hym/anaconda3/envs/env_isaaclab/bin/python \
  -m robot_tasks.integrationtest.diagnostics.scene_smoke --all
```

## 两个任务试跑

```bash
python3 tools/batch_run.py --tasks L1-01 L2-01 \
  --max-calls 10 --wall-time-s 1200 --output runs/gpt_pilot
```

自动读取仓库 `.env` 中的赋值，将凭据仅传给子进程。已有终端环境变量优先；
支持 `export KEY=value`、单/双引号和行尾注释，不执行 shell 代码，也不展开变量。
其他凭据文件用 `--env-file /path/to/.env`；其他模型配置用
`--model-config /path/to/model.toml`。不保存或显示 API Key。

## 全量正式测评

```bash
python3 tools/batch_run.py --all --output runs/gpt_master_baseline
```

默认读取 `configs/models/gpt.toml`，思考强度沿用该文件（当前为 `medium`）；
`--reasoning-effort` 可显式覆盖。每任务默认最多 100 次模型调用、3600 秒回合墙钟时间、seed=0。
外部进程 watchdog 从启动开始计时，预算为墙钟上限 + 300 秒启动/清理余量；
必要时用 `--startup-grace-s` 调整。超时先请求正常退出，等待最多 30 秒保存结果
及关闭视频，再强制结束本批次拥有的进程。Linux 下也跟踪独立会话的 cuRobo
子进程。无法确认清理完成时停止队列，避免继续占用 GPU。

正式运行强制保存 RGB、深度和相机视频；默认后台运行，`--gui` 可显示窗口。
不自动运行或重试场景预检。批量层不因操作失败重跑任务；任务内部原有的
重置重试机制保留，仍共享单回合总调用预算。报告区分首次成功与最终成功。

新批次目录必须不存在，以免覆盖证据。启动时冻结任务、公共参数和模型配置，
保存代码及输入配置哈希；不修改原始 TOML。续跑时条件变化会报错，应创建新批次。

## 中断与续跑

按 Ctrl+C 会清理当前任务并更新汇总，然后停止队列。恢复时使用：

```bash
python3 tools/batch_run.py --resume runs/gpt_master_baseline
```

续跑跳过所有已结束任务，包括操作失败、框架错误和超时；继续尚未执行的任务。
主动中断的任务会从初始场景重新开始，新回合独立保存，旧日志和录像保留。
不恢复中途的物理状态或 GPT 对话。若旧 worker/规划器仍存活，拒绝重复启动。
续跑不能覆盖预算、模型、种子、任务列表或模式。

认证、权限或明确额度耗尽会暂停整个队列；修复凭据/额度后续跑。
触发错误的回合仍作为已结束记录保留。没有“重跑直到成功”的隐式重试。
同一批次由 OS 文件锁保护，禁止两个入口同时运行或修改其汇总。

## 输出与结果

```text
runs/gpt_master_baseline/
├── batch.json               模型、实验条件、任务清单、输入哈希
├── configs/                 本批次冻结配置，无 API Key
├── state.json               队列进度与 worker 身份
├── summary.json             原始汇总、评分指标、Token 和录像检查
├── summary.csv              可在表格中比较的结果与消耗
├── summary.md               逐任务报告及三视角录像链接
└── tasks/<任务ID>/run_<ID>/
    ├── process.log / process.json
    └── episodes/<原有回合目录>/
        ├── result.json / report.md / episode.jsonl
        └── video/<相机名>/motion.mp4
```

主表展示当前选定回合，中断后重开的旧回合仍在各 `run_<ID>` 目录。
`result.json` 是评分依据；进程退出码 0 不能代表任务成功。
`succeeded`/`failed` 是已完成评估的任务结果；`infrastructure_error`、`timeout`、
`interrupted` 分开统计。首次成功、最终成功、尝试/重置次数都保留。
Token 是返回了 `usage` 的请求之和，结合 `usage_samples` 与 `model_calls` 判断
完整程度；缺失值留空，不当成零，不估算费用。

使用 `ffprobe` 检查每个预期相机 MP4 的非空、视频流和时长；
`video_complete` 只有全部通过时才为 true。这是容器/元数据检查，未逐帧解码。
缺少 `ffprobe` 时录像标为未验证，不会虚报完整。录像按仿真时间播放，
API 等待时物理暂停；墙钟时间另存。

已完成的批次可以离线重新检查视频并汇总：

```bash
python3 tools/batch_run.py --summarize runs/gpt_master_baseline
```

批量入口退出码：0 表示队列完成且所需录像检查通过（任务本身可能失败）；
2 表示配置/环境错误、超时、缺失录像或未完成队列；130 表示主动中断。
dry-run 的 0 只表示批量流程验证完成。
