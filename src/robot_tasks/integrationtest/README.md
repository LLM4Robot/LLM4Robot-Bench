# GPT 抓放集成检查

这里只保留 `gpt-pick-place` 一个用例，直接复用 `GPT-PICK-PLACE` 的场景、GPT 策略、执行循环和评分。

```bash
python -m robot_tasks check gpt-pick-place --gui
```

默认配置为 `configs/checks/gpt_single_arm_pick_place.toml`，它引用公共 workflow。
支持 `--config`、`--model-config` 和 `--output`；输出仍统一放在本轮 `runs/` 目录。
该检查会启动 Isaac Sim 并真实调用模型；只有实测任务成功才返回通过。

| 文件 | 职责 |
|---|---|
| `gpt_pick_place_check.py` | 唯一用例，将任务实测结果转为集成检查结果 |
| `registry.py` | 检查名称、配置和入口解析 |
| `simulation_process.py` | 独立进程、日志、超时监控与退出处理 |
| `simulation_worker.py` | 子进程入口 |
| `debug_check.py` | WSL 编辑器连接 Windows Isaac Python 的调试入口 |

后四项是启动设施，不是额外测试场景。新任务通过主任务注册表和 `run <任务ID>` 运行，
不需要复制这里的抓放用例。离线软件回归保留在根目录 `tests/`，不启动真实仿真或模型请求。

独立物理诊断集中在 [diagnostics/](diagnostics/README.md)，包括场景预览、资产接触和夹持检查；这些检查不进入正式 GPT 任务流程。
