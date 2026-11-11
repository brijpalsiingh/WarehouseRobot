"""Step-by-step replay of an extracted warehouse policy.

The controller is independent of the GUI: Pygame only reads its state and
invokes Start, Pause, Next Step and Stop. The actual warehouse movement is
performed by WarehouseMDP.step(), including stochastic slips and rewards.

AI assistance: This module was developed with ChatGPT assistance in
November 2026. Include the shared prompt link in code/report before submission.
"""

from copy import deepcopy
from dataclasses import dataclass
from math import isfinite
from random import Random
from typing import Optional, Tuple

from warehouse_gridworld_domain_random import ACTIONS


@dataclass(frozen=True)
class MovementRecord:
    step: int
    previous_state: Tuple[int, int, int, int]
    selected_action: str
    actual_action: str
    next_state: Tuple[int, int, int, int]
    reward: float
    cumulative_reward: float
    event: str
    slipped: bool
    explored: bool = False


class PolicyPlayback:
    """Controls an episode without modifying the world used for planning."""

    def __init__(self, world, policy, step_delay_ms=500, on_movement=None,
                 epsilon=0.0, exploration_seed=0):
        if not isinstance(step_delay_ms, (int, float)) or int(step_delay_ms) <= 0:
            raise ValueError("step_delay_ms must be greater than zero")
        if not isinstance(epsilon, (int, float)) or not isfinite(epsilon) or not 0 <= epsilon <= 1:
            raise ValueError("epsilon must be between 0 and 1")
        self.world = deepcopy(world)
        self.world.reset()
        self.policy = policy
        self.step_delay_ms = int(step_delay_ms)
        self.status = "ready"
        self.elapsed_ms = 0
        self.history = []
        self.last_movement: Optional[MovementRecord] = None
        self.on_movement = on_movement
        self.epsilon = float(epsilon)
        self.exploration_seed = exploration_seed
        self.exploration_rng = Random(exploration_seed)
        # Initial dock placement counts as the first visit. Re-entering any
        # cell increases its count, but staying put after a blocked move does not.
        self.visit_counts = {self.world.dock_pos: 1}

    @property
    def phase(self):
        if self.world.delivered:
            return 2
        if self.world.has_package:
            return 1
        return 0

    def _reset_episode(self):
        self.world.reset()
        self.exploration_rng.seed(self.exploration_seed)
        self.elapsed_ms = 0
        self.history.clear()
        self.last_movement = None
        self.visit_counts = {self.world.dock_pos: 1}

    def start(self):
        """Start a fresh episode or resume a paused one."""
        if self.status in ("ready", "stopped", "success", "timeout"):
            self._reset_episode()
        self.status = "running"
        self.elapsed_ms = 0

    def pause(self):
        if self.status == "running":
            self.status = "paused"
            self.elapsed_ms = 0

    def next_step(self):
        """Perform exactly one policy action, leaving playback paused."""
        if self.status in ("stopped", "success", "timeout"):
            self._reset_episode()
        self.status = "paused"
        self.elapsed_ms = 0
        self._advance()

    def stop(self):
        """Stop the current demonstration and return the robot to dock."""
        self._reset_episode()
        self.status = "stopped"

    def update(self, elapsed_ms):
        """Called once per GUI frame; advances at most one movement per tick."""
        if self.status != "running":
            return
        self.elapsed_ms += max(0, elapsed_ms)
        if self.elapsed_ms >= self.step_delay_ms:
            self.elapsed_ms = 0
            self._advance()

    def _advance(self):
        if self.world.done:
            return
        before = self.world.get_state()
        greedy_action = self.policy.get(before)
        if greedy_action is None:
            raise RuntimeError(f"No policy action for active state {before}")

        # An optional post-training demonstration uses epsilon-greedy action
        # selection without changing any Q-values. Default epsilon=0 keeps
        # genuine evaluations entirely greedy.
        explored = (self.epsilon > 0 and
                    self.exploration_rng.random() < self.epsilon)
        selected_action = (self.exploration_rng.choice(ACTIONS) if explored
                           else greedy_action)

        result = self.world.step(selected_action)
        record = MovementRecord(
            step=self.world.steps,
            previous_state=before,
            selected_action=selected_action,
            actual_action=result.info.get("executed_action", selected_action),
            next_state=result.state,
            reward=result.reward,
            cumulative_reward=self.world.score,
            event=result.info.get("event", "move"),
            slipped=bool(result.info.get("slipped", False)),
            explored=explored,
        )
        self.history.append(record)
        self.last_movement = record
        if result.state[:2] != before[:2]:
            destination = result.state[:2]
            self.visit_counts[destination] = self.visit_counts.get(destination, 0) + 1
        if self.on_movement is not None:
            self.on_movement(record)

        if result.done:
            self.status = (
                "success" if result.info.get("event") == "return_to_dock" else "timeout"
            )
