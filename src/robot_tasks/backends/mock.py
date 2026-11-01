"""Flow checks only. No physics, robot motion, task success, or real camera."""

from ..config import TaskConfig
from ..types import Action, ArmState, Assessment, EvaluationState, Feedback, PolicyInput, RGBFrame


class MockBackend:
    def open(self, config: TaskConfig) -> None:
        self.config = config
        self.t = 0.0
        self.closed = False

    def reset(self, seed: int) -> None:
        self.t = 0.0

    def step(self, duration_s: float) -> None:
        self.t += duration_s

    def pump_events(self) -> None:
        pass

    def capture_rgb(self) -> tuple[RGBFrame, ...]:
        # Tiny black placeholders deliberately carry no task visuals.
        return tuple(RGBFrame(name, self.t, 2, 2, bytes(12)) for name in self.config.cameras)

    def time_s(self) -> float:
        return self.t

    def evaluation_state(self) -> EvaluationState:
        return EvaluationState({"mock_only": True})

    def close(self) -> None:
        self.closed = True


class MockRobot:
    def initialize(self, backend, config: TaskConfig) -> None:
        self.arms = config.command_arms

    def reset(self, seed: int) -> None:
        pass

    def apply(self, action: Action) -> None:
        if action.commands:
            raise ValueError("MockRobot only accepts empty placeholder actions")

    def state(self) -> dict[str, ArmState]:
        return {arm: ArmState() for arm in self.arms}

    def feedback(self) -> Feedback:
        return Feedback("mock_only", "Placeholder; no robot moved")


class MockTask:
    def build_scene(self, backend, config: TaskConfig) -> None:
        pass

    def reset(self, backend, seed: int) -> None:
        pass

    def evaluate(self, state: EvaluationState, feedback: Feedback) -> Assessment:
        return Assessment("not_evaluated", "Flow check only; no task physics or scoring")


class NoopPolicy:
    def __init__(self, duration_s: float):
        self.duration_s = duration_s

    def reset(self, seed: int) -> None:
        pass

    def act(self, context: PolicyInput) -> Action:
        return Action((), self.duration_s)
