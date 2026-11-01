"""Settle the configured joint home without a scripted opening movement."""

from ..types import Action


def preparation_action(config):
    """Hold reset drive targets before the first model observation.

    R5Adapter.reset assigns the profile's six joint home values. The maintained
    legacy simulation profiles start at zero. Latest video tasks use per-task
    video-visible joint homes and gripper openings. Neither arm is moved to a
    scripted approach pose before GPT takes over. Gripper opening is preserved.
    """
    return Action((), config.control_dt_s)
