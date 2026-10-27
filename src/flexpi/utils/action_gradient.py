"""Optimizer-step schedule for action-to-visual gradient routing."""

from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class ActionGradientSchedule:
    """Scale only the gradient crossing visual K/V into action attention.

    Progress counts completed optimizer updates, not microbatches. The horizon
    is recorded in training state so resuming cannot silently move the ramp.
    Equal start/end scales specify a constant control.
    """

    total_steps: int
    start_scale: float = 1.0
    end_scale: float = 1.0
    ramp_start: float = 0.0
    ramp_end: float = 1.0

    def __post_init__(self):
        if self.total_steps <= 0:
            raise ValueError("action_gradient total_steps must be positive")
        for name in ("start_scale", "end_scale", "ramp_start", "ramp_end"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"action_gradient {name} must be finite and in [0, 1]")
        if self.ramp_start >= self.ramp_end:
            raise ValueError("action_gradient requires ramp_start < ramp_end")

    def value(self, completed_steps: int) -> float:
        """Return the scale for the next update at the given training position."""
        if completed_steps < 0:
            raise ValueError("completed_steps must be nonnegative")
        progress = completed_steps / self.total_steps
        fraction = min(1.0, max(0.0, (progress - self.ramp_start) /
                                (self.ramp_end - self.ramp_start)))
        return self.start_scale + fraction * (self.end_scale - self.start_scale)

    def state_dict(self) -> dict:
        return asdict(self)

    def validate_resume(self, saved_state: dict | None):
        """Reject a continuation that would change the experimental treatment."""
        if saved_state != self.state_dict():
            raise ValueError(
                "Action gradient schedule/horizon differs from saved training state. "
                "Resume with the original schedule and max_steps; use pretrained_ckpt "
                "for a deliberately new experiment."
            )
