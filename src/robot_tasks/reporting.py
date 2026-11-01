"""Render recorded evidence into a report in the same run directory.

This module never scores a task, starts a simulator, or calls a model. result.json
remains authoritative; reports can be regenerated for old or interrupted runs.
"""

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import statistics
from urllib.parse import quote


def _cell(value) -> str:
    if value is None:
        return "未记录"
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, (dict, list, tuple)):
        value = json.dumps(value, ensure_ascii=False, allow_nan=False)
    return str(value).replace("|", "\\|").replace("\r", "").replace("\n", "<br>")


def _table(headers, rows) -> list[str]:
    return ["| " + " | ".join(map(_cell, headers)) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
            *("| " + " | ".join(map(_cell, row)) + " |" for row in rows), ""]


def _number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _flatten(data: dict, prefix=""):
    for name, value in data.items():
        key = f"{prefix}.{name}" if prefix else name
        if isinstance(value, dict):
            yield from _flatten(value, key)
        else:
            yield key, value


def collect_run_evidence(directory: Path) -> dict:
    """Stream logs; distinguish measured action tracking from task-object error."""
    warnings = []

    def read(name):
        path = directory / name
        if not path.exists():
            return {}
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                raise ValueError("expected an object")
            return value
        except (OSError, ValueError) as exc:
            warnings.append(f"{name} 无法读取：{exc}")
            return {}

    result, manifest, process = read("result.json"), read("manifest.json"), read("process.json")
    if not result:
        warnings.append("没有完整的 result.json，不能判定任务结果。")
    errors = defaultdict(lambda: defaultdict(list))
    rejections, calls, latencies = Counter(), 0, []
    depth_queries = 0
    provider = {}
    log = directory / "episode.jsonl"
    if log.exists():
        with log.open(encoding="utf-8") as stream:
            for number, line in enumerate(stream, 1):
                try:
                    event = json.loads(line)
                except ValueError:
                    warnings.append(f"episode.jsonl 第 {number} 行不完整，未纳入日志汇总。")
                    continue
                if event.get("event") == "policy_configuration":
                    metadata = event.get("configuration", {}).get("provider", {})
                    provider = {key: metadata.get(key) for key in ("model", "reasoning_effort")}
                elif event.get("event") == "model_exchange":
                    calls += 1
                    latency = event.get("exchange", {}).get("latency_s")
                    if _number(latency):
                        latencies.append(latency)
                elif event.get("event") == "action_rejected":
                    feedback = event.get("observation", {}).get("feedback", {})
                    rejections[feedback.get("metrics", {}).get("reason_code", "unspecified")] += 1
                elif event.get("event") == "depth_query":
                    depth_queries += 1
                elif event.get("event") == "transition":
                    feedback = event.get("observation", {}).get("feedback", {})
                    for arm, values in feedback.get("metrics", {}).get("tcp", {}).items():
                        for field in ("position_error_m", "orientation_error_rad"):
                            value = values.get(field)
                            if _number(value):
                                errors[arm][field].append(value)
    tracking = {arm: {field: {"samples": len(values), "mean": statistics.mean(values), "max": max(values)}
                      for field, values in fields.items()} for arm, fields in errors.items()}
    return {"result": result, "manifest": manifest, "process": process, "provider": provider, "scene": read("scene.json"),
            "tracking": tracking, "rejections": dict(rejections), "logged_model_exchanges": calls,
            "depth_query_decisions": depth_queries,
            "api_latency_total_s": sum(latencies) if latencies else None,
            "api_latency_samples": len(latencies), "warnings": warnings}


def write_run_report(directory: Path) -> Path:
    """Create/refresh report.md without changing raw logs or success criteria."""
    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError(f"Run directory does not exist: {directory}")
    data = collect_run_evidence(directory)
    result, manifest, process = data["result"], data["manifest"], data["process"]
    evaluated = result.get("task_evaluated") is True
    outcome = ("成功" if result.get("success") is True else "失败" if result.get("success") is False
               else "未确定") if evaluated else "未评估（工程检查、演练或异常回合不计为任务成功／失败）"
    lines = ["# 任务测评报告", "", "根据本回合已有记录生成；不重新评分。单回合结果不代表任务成功率。", ""]
    if result.get("continuation"):
        lines += ["**本次为重建末态续跑，不是独立的新回合，也不是无中断的物理仿真。**",
                  "保留前次模型历史、累计调用和评分过程；PhysX 接触缓存未恢复。录像仅包含本次续跑。",
                  f"来源：`{result['continuation']['source_run']}`。此前调用 {result.get('previous_model_calls')} 次，"
                  f"本段新增 {result.get('new_model_calls')} 次；其他动作数、耗时和录像为本段数据。", ""]
    lines += _table(("项目", "结果"), [
        ("任务", result.get("task_id", manifest.get("task_id", directory.name))),
        ("运行模式", result.get("mode", manifest.get("kind", manifest.get("mode")))),
        ("运行状态", result.get("status")), ("任务评价", outcome),
        ("评价有效性", result.get("evaluation_validity")),
        ("失败分类", result.get("failure_category")),
        ("工程检查通过", result.get("passed")), ("终止原因", result.get("termination_reason")),
        ("模型", data["provider"].get("model")), ("推理强度", data["provider"].get("reasoning_effort")),
        ("随机种子", manifest.get("seed")),
        ("IK 求解器", data["scene"].get("ik_solver")),
        ("运动规划信息范围", data["scene"].get("motion_planner")),
        ("停止协议", result.get("stop_protocol")),
        ("相机输入模态", manifest.get("observation_modalities")),
    ])
    if manifest.get("depth", {}).get("enabled"):
        lines += ["本回合启用 RGB＋深度。深度为与 RGB 对齐的理想仿真光轴距离（米），"
                  "不模拟奥比硬件噪声、盲区或双目匹配误差，不应与仅 RGB 结果混合统计。"
                  "原始 float32 深度保存在 depth/，无效像素为 NaN；PNG 仅作固定刻度的可视化。"
                  "保存频率与决策观测一致，不是逐物理步深度录像。像素查询不推进物理，但计入模型调用预算。", ""]
    if process:
        lines += _table(("进程检查", "结果"), [("退出码", process.get("returncode")),
                          ("用户中断", process.get("interrupted")),
                          ("强制结束", process.get("forced_termination")),
                          ("工作进程已退出", process.get("cleanup_complete")),
                          ("监控超时", process.get("timed_out")), ("进程与结果联合校验通过", process.get("passed"))])
        expected_check_failure = (process.get("returncode") == 2 and result.get("status") == "completed"
                                  and result.get("passed") is False)
        if (process.get("returncode") not in (None, 0) and not expected_check_failure) or process.get("timed_out"):
            lines += ["进程退出异常或超时；请结合 process.json 与 process.log 复核，不能作为正常完成的回合。", ""]
    if result.get("error"):
        lines += [f"运行异常：{_cell(result['error'])}", ""]
    if data["warnings"]:
        lines += ["记录缺项：", "", *(f"- {_cell(w)}" for w in data["warnings"]), ""]
    metrics = result.get("assessment", {}).get("metrics", {})
    if metrics.get("scoring_protocol") in {"recoverable_progress_v1", "native_l2_validity_v2"}:
        lines += ["## 任务目标与过程约束", "",
                  "目标状态、必要过程约束和工程参考值分别记录；完整任务结果以上方任务评价为准。"
                  "过程约束栏表示截至当前是否记录违规，不代表尚未执行的步骤已经完成。"
                  "未经标定的力／速度超标仅作过程记录，不代表已测得材料损坏。", ""]
        lines += _table(("指标", "结果"), [
            ("评分协议", metrics["scoring_protocol"]),
            ("目标状态满足（仍需满足稳定时间）", metrics.get("goal_complete")),
            ("必要过程约束满足", metrics.get("process_constraints_satisfied")),
            ("违反的必要约束", metrics.get("constraint_violations", [])),
            ("过程记录（不直接判失败）", metrics.get("quality_warnings", [])),
        ])
    if metrics.get("stages"):
        lines += ["## 分阶段结果", "", "阶段状态与整体成功分开记录；后续失败不会抹去此前已记录的关键进展。", ""]
        lines += _table(("阶段", "达到要求"), metrics["stages"].items())
    if result.get("evaluation_validity") == "invalid":
        lines += [("本回合测量无效或需要人工复核，不计入模型成功／失败分母。"
                   f"原因：{_cell(result.get('assessment', {}).get('reason'))}"), ""]
    protocol = data["scene"].get("household", {}).get("test_protocol")
    if protocol:
        lines += ["## 测试信息边界", ""]
        lines += _table(("项目", "设定"), protocol.items())
        limitations = metrics.get("limitations") or data["scene"].get("household", {}).get("limitations")
        if limitations:
            lines += [f"资产与适配限制：{_cell(limitations)}", ""]
    if result.get("evaluation_protocol") == "retry_with_history":
        lines += ["## 失败复位与重试", "",
                  "本次采用保留模型历史的复位重试，所有尝试共享调用、物理步和现实时间预算。"
                  "任务评价表示预算内最终结果，不能当作独立单次尝试的成功率。", ""]
        lines += _table(("指标", "数值"), [
            ("完成复位次数", result.get("reset_count")), ("失败尝试数", result.get("failed_attempts")),
            ("首次尝试成功", result.get("first_attempt_success")),
            ("成功发生于第几次尝试", result.get("success_attempt")),
        ])
        lines += _table(("尝试", "状态", "模型决策起止", "原因"), [
            (a.get("attempt"), a.get("status"),
             f"{a.get('decision_start')}—{a.get('decision_end')}", a.get("assessment", {}).get("reason"))
            for a in result.get("attempts", [])])
        lines += ["录像连续保留失败和复位后的过程；episode.jsonl 的 attempt_end / attempt_reset "
                  "记录对应的全局 tick 边界。仿真时间统计所有尝试与复位后的稳定过程，局部场景时间在复位时归零。", ""]
    lines += ["## 执行统计", ""]
    lines += _table(("指标", "数值"), [
        ("模型调用次数", result.get("model_calls")), ("已记录模型交互", data["logged_model_exchanges"]),
        ("深度查询决策数（已包含在模型调用中）", data["depth_query_decisions"]),
        ("执行动作数", result.get("executed_actions")), ("动作拒绝数", result.get("rejected_actions")),
        ("无效模型动作数（计入调用预算）", result.get("invalid_model_actions")),
        ("被拒绝的停止请求", result.get("rejected_stop_requests")),
        ("可重试的生成异常次数", result.get("generation_failures")),
        ("实际发起的模型重试次数", result.get("model_retry_attempts")),
        ("抓取核验拦截次数", result.get("grasp_check_rejections")),
        ("RGB-D 抬升核验通过次数（不代表任务成功）", result.get("grasp_visual_verifications")),
        ("抓取核验协议", result.get("grasp_check_protocol")),
        ("动作超时数", result.get("action_timeouts")), ("错误宣告完成", result.get("false_completion")),
        ("现实耗时（秒）", result.get("wall_time_s")), ("仿真时间（秒）", result.get("simulation_time_s")),
        ("有时延记录的 API 耗时合计（秒）", data["api_latency_total_s"]),
        ("API 时延记录条数", data["api_latency_samples"]),
    ])
    lines += ["API 耗时仅汇总已记录的调用；未返回／未记录时延的请求不补记为零。录像按仿真时间播放。", "",
              "## 动作跟踪误差", "",
              "统计动作结束时反馈中的实测 TCP 与请求目标之差，包含动作超时。单位为米／弧度；这不是物体放置误差，也不是 IK 数学求解误差。", ""]
    rows = [(arm, field, stats["samples"], stats["mean"], stats["max"])
            for arm, fields in data["tracking"].items() for field, stats in fields.items()]
    if rows:
        lines += _table(("机械臂", "误差指标", "样本数", "均值", "最大值"), rows)
    else:
        lines += ["未记录可汇总的动作结束 TCP 误差；不按零误差处理。", ""]
    if data["rejections"]:
        lines += ["## 动作拒绝原因", ""]
        lines += _table(("原因代码", "次数"), data["rejections"].items())
    assessment = result.get("assessment", {})
    lines += ["## 任务指标与判定", ""]
    if assessment:
        lines += [f"评分器状态：{_cell(assessment.get('status'))}；{_cell(assessment.get('reason'))}", ""]
        metrics = list(_flatten(assessment.get("metrics", {})))
        if "lift_confirmed" in assessment.get("metrics", {}):
            lines += ["抓放指标说明：lift_confirmed 表示本回合曾经满足持续抬升要求；lift.* 是末尾滑动窗口的状态。放下后 lift.passed 为否是正常现象，最终放置看 placement.*，整体结论看任务评价。", ""]
        if metrics:
            lines += _table(("指标（原始字段／单位）", "测量值"), metrics)
        lines += ["评分参数见 config.toml。物体误差、接触、稳定性等指标由各任务评分器定义；本报告只呈现其记录，不改变成功条件。", ""]
    else:
        # Older engineering checks expose their own case tables, not task scores.
        checks = {key: result[key] for key in ("pose_checks", "rejection_checks", "cases", "reset_checks", "tolerances") if key in result}
        if checks:
            lines += ["工程检查明细（不作为模型任务成功率）：", "", "```json",
                      json.dumps(checks, ensure_ascii=False, indent=2, allow_nan=False), "```", ""]
        else:
            lines += ["本回合没有任务评分明细，参见原始 result.json。", ""]
    lines += ["## 录像与原始记录", ""]
    paths = [path for name in ("result.json", "episode.jsonl", "motion.jsonl", "manifest.json", "instruction.txt",
                               "config.toml", "robot_profile.toml", "camera_profile.toml", "scene.json",
                               "process.json", "process.log", "frames", "depth", "analysis")
             if (path := directory / name).exists()]
    paths += sorted(directory.glob("*.mp4")) + sorted(directory.glob("video/**/*.mp4"))
    for path in paths:
        relative = path.relative_to(directory).as_posix()
        lines.append(f"- [{_cell(relative)}](<{quote(relative, safe='/')}>)")
    lines.append("")
    destination = directory / "report.md"
    temporary = directory / "report.md.tmp"
    temporary.write_text("\n".join(lines), encoding="utf-8")
    temporary.replace(destination)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+", help="Existing run directories; no simulation or API call")
    args = parser.parse_args()
    for directory in args.runs:
        print(write_run_report(directory))


if __name__ == "__main__":
    main()
