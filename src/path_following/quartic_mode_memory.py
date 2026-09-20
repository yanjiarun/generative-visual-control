import numpy as np

from .create_path_quartic_obstacle import (
    BRANCH_X, DECISION_X, compute_quartic_obstacle_field,
)


class QuarticModeMemory:
    """Lock one learned mode and let the controller reuse its latent noise."""

    def __init__(self, config):
        self.enabled = bool(config.get("mode_persistence", False))
        self.samples = max(1, int(config.get("mode_samples", 8)))
        self.hint = config.get("mode_hint")
        self.lock_x = float(config.get("mode_lock_x", -0.44))
        self.min_angle = np.deg2rad(float(config.get("mode_lock_angle_deg", 2.0)))
        self.mode = None
        self.locked = False

    @staticmethod
    def _score(action, position):
        """Positive when an action is closer to the upper analytic field."""
        x = (float(position[0]) - 576.0) / 200.0
        y = (float(position[1]) - 275.0) / 200.0
        v = np.asarray(action, dtype=float)[:2]
        angle = np.arctan2(v[1], v[0])
        headings = []
        for mode in (1, -1):
            ux, uy, _ = compute_quartic_obstacle_field(
                x, y, mode, gradients=False)
            headings.append(np.arctan2(float(uy), float(ux)))
        errors = [abs(np.arctan2(np.sin(angle - h), np.cos(angle - h)))
                  for h in headings]
        return errors[1] - errors[0]

    @staticmethod
    def _x(position):
        return (float(position[0]) - 576.0) / 200.0

    def reset_if_finished(self, position):
        if self._x(position) > BRANCH_X[1]:
            self.mode, self.locked = None, False
            return True
        return False

    def candidate_count(self, position):
        x = self._x(position)
        ready = max(DECISION_X[0], self.lock_x) <= x < DECISION_X[1]
        hinted = self.hint in (1, -1)
        return self.samples if self.enabled and hinted and not self.locked and ready else 1

    def select(self, actions, position):
        """Return the candidate index; the caller retains its latent noise."""
        actions = np.asarray(actions, dtype=float)
        if actions.ndim == 1:
            actions = actions[None]
        if not self.enabled:
            return 0

        x = self._x(position)
        if self.reset_if_finished(position):
            return 0
        in_decision = DECISION_X[0] <= x < DECISION_X[1]
        scores = np.asarray([self._score(a, position) for a in actions])
        if self.locked:
            if self.mode is None and abs(scores[0]) >= self.min_angle:
                self.mode = int(np.sign(scores[0]))
            return 0
        if not in_decision:
            return 0
        if self.hint not in (1, -1):
            if x < self.lock_x or abs(scores[0]) < self.min_angle:
                return 0
            self.mode, self.locked = int(np.sign(scores[0])), True
            return 0
        if x >= self.lock_x:
            mode = int(self.hint)
            index = int(np.argmax(mode * scores))
            clear = mode * scores[index] >= self.min_angle
            if clear:
                self.mode, self.locked = mode, True
            return index
        return 0
