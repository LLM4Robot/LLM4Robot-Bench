"""Debugger/worker entry for the registered task; no duplicate controller or score."""

from ..execution import execute_task
from ..registry import resolve_task


def run_gpt_pick_place_check(config_path, output_root, *, gui=False, run_directory=None, model_path=None):
    spec, config = resolve_task("GPT-PICK-PLACE", config_path)

    def summarize(result):
        result.update(status="completed" if result["status"] == "finished" else result["status"],
                      passed=result["status"] == "finished" and result["success"] is True,
                      task_success=result["success"],
                      scope="Single GPT-controlled pick/place episode; not a benchmark success-rate estimate")
        return result

    return execute_task(spec, config, output_root, gui=gui, model_config=model_path,
                        run_directory=run_directory, summarize=summarize)
