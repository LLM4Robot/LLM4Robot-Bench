"""One model decision off-thread; simulator/UI ownership stays on the caller."""

import threading
import time


class PolicyRequestCleanupError(RuntimeError):
    """An interrupted request could not be joined; never reuse this episode."""


def request_action(policy, context, *, idle=None, check_budget=lambda: None,
                   poll_s=.05, cleanup_timeout_s=15.):
    done = threading.Event()
    outcome = []

    def generate():
        try:
            outcome.append((True, policy.act(context), None))
        except BaseException as exc:
            outcome.append((False, exc, exc.__traceback__))
        finally:
            done.set()

    check_budget()
    worker = threading.Thread(target=generate, name="r5-model-request", daemon=True)
    worker.start()
    try:
        while not done.wait(poll_s):
            check_budget()
            if idle is not None:
                idle()
        check_budget()  # Discard responses arriving after the episode deadline.
    except BaseException as exc:
        cancel = getattr(policy, "cancel_pending", None)
        if cancel is not None:
            try:
                cancel()
            except Exception:
                pass  # Join below still verifies cleanup rather than assuming it.
        deadline = time.monotonic() + cleanup_timeout_s
        while not done.wait(poll_s) and time.monotonic() < deadline:
            if idle is not None:
                try:
                    idle()
                except BaseException:
                    pass  # Window may already be closed; still reap the request.
        if not done.is_set():
            raise PolicyRequestCleanupError(
                "Model request thread did not stop after cancellation; episode cannot continue"
            ) from exc
        worker.join()
        raise
    worker.join()
    accepted, value, traceback = outcome[0]
    if not accepted:
        raise value.with_traceback(traceback)
    return value
