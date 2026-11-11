"""Stochastic MDP extension for the instructor-provided warehouse environment.

AI assistance: ChatGPT helped develop this module on Nov 2, 2026. Before
submission, include the relevant shared-prompt link in this comment and in
the PDF report, as required by the assignment.
"""

from dataclasses import dataclass
from typing import Tuple

from warehouse_gridworld_domain_random import (
    ACTIONS, ACTION_TO_DELTA, CONGESTION, CONGESTION_PENALTY,
    DELIVERY_REWARD, DOCK_REWARD, INVALID_MOVE_PENALTY,
    PICKUP_REWARD, SHELF, STEP_PENALTY, StepResult, WarehouseGridWorld,
)

State = Tuple[int, int, int, int]
PERPENDICULAR = {
    "up": ("left", "right"),
    "down": ("left", "right"),
    "left": ("up", "down"),
    "right": ("up", "down"),
}


@dataclass(frozen=True)
class Transition:
    probability: float
    next_state: State
    reward: float
    terminated: bool
    actual_action: str


class WarehouseMDP(WarehouseGridWorld):
    """Adds a known probability model and matching stochastic step sampler."""

    def __init__(self, seed=42, max_steps=120, p=0.8):
        if not 0.0 <= p <= 1.0:
            raise ValueError("p must be a probability between 0 and 1")
        self.p = float(p)
        super().__init__(seed=seed, max_steps=max_steps)
        self._state_set = set(self.valid_states())

    @staticmethod
    def action_probabilities(action, p):
        if action not in ACTION_TO_DELTA:
            raise ValueError(f"Unknown action: {action}")
        if not 0.0 <= p <= 1.0:
            raise ValueError("p must be a probability between 0 and 1")
        side_a, side_b = PERPENDICULAR[action]
        return {action: p, side_a: (1 - p) / 2, side_b: (1 - p) / 2}

    def valid_states(self):
        """Only passable cells and the three possible task-progress phases."""
        phases = ((0, 0), (1, 0), (1, 1))
        return [
            (r, c, has_package, delivered)
            for r in range(self.size) for c in range(self.size)
            if self.grid[r, c] != SHELF
            for has_package, delivered in phases
        ]

    def is_success_state(self, state):
        return (state[0], state[1]) == self.dock_pos and state[3] == 1

    def deterministic_outcome(self, state, actual_action):
        """Single movement and reward, matching the instructor's step logic."""
        if actual_action not in ACTION_TO_DELTA:
            raise ValueError(f"Unknown action: {actual_action}")
        row, col, has_package, delivered = state
        dr, dc = ACTION_TO_DELTA[actual_action]
        nr, nc = row + dr, col + dc
        reward, event = float(STEP_PENALTY), "move"
        if not self.in_bounds(nr, nc) or not self.passable(nr, nc):
            nr, nc = row, col
            reward += INVALID_MOVE_PENALTY
            event = "invalid_move"

        # The supplied framework penalizes a step ending on congestion,
        # including a blocked movement while already on that cell.
        congestion = int(self.grid[nr, nc]) == CONGESTION
        if congestion:
            reward += CONGESTION_PENALTY
        if (nr, nc) == self.pickup_pos and not has_package:
            has_package, reward, event = 1, reward + PICKUP_REWARD, "pickup"
        if (nr, nc) == self.packing_pos and has_package and not delivered:
            delivered, reward, event = 1, reward + DELIVERY_REWARD, "delivery"
        finished = (nr, nc) == self.dock_pos and bool(delivered)
        if finished:
            reward, event = reward + DOCK_REWARD, "return_to_dock"

        next_state = (nr, nc, int(has_package), int(delivered))
        return next_state, reward, finished, event, congestion

    def transition_outcomes(self, state, action):
        """Known P(s'|s,a), including reward and terminal information.

        Distinct movement outcomes can lead to the same state at boundaries.
        Leaving them separate retains the exact expected reward calculation.
        """
        if state not in self._state_set:
            raise ValueError(f"Invalid state: {state}")
        if action not in ACTIONS:
            raise ValueError(f"Unknown action: {action}")
        if self.is_success_state(state):
            return [Transition(1.0, state, 0.0, True, action)]

        outcomes = []
        for actual_action, probability in self.action_probabilities(action, self.p).items():
            if probability == 0:
                continue
            next_state, reward, done, _, _ = self.deterministic_outcome(state, actual_action)
            outcomes.append(Transition(probability, next_state, reward, done, actual_action))
        return outcomes

    def step(self, action):
        """Sample one actual movement; the Q-Learning agent only sees this step."""
        if self.done:
            return StepResult(self.get_state(), 0.0, True, {"message": "Episode already finished."})
        probabilities = self.action_probabilities(action, self.p)
        draw, cumulative = float(self.np_random.random()), 0.0
        actual_action = next(reversed(probabilities))
        for candidate, probability in probabilities.items():
            cumulative += probability
            if draw < cumulative:
                actual_action = candidate
                break

        state, reward, terminated, event, congestion = self.deterministic_outcome(
            self.get_state(), actual_action
        )
        self.robot_pos = state[:2]
        self.has_package, self.delivered = bool(state[2]), bool(state[3])
        self.steps += 1
        self.score += reward
        self.done = terminated
        info = {
            "event": event,
            "requested_action": action,
            "executed_action": actual_action,
            "slipped": action != actual_action,
        }
        if congestion:
            info["congestion"] = True
        if self.steps >= self.max_steps and not self.done:
            self.done = True
            info.update(event="max_steps", truncated=True)
        return StepResult(state, reward, self.done, info)


if __name__ == "__main__":
    world = WarehouseMDP(seed=99, p=0.8)
    print("State:", world.get_state(), "Valid states:", len(world.valid_states()))
    for outcome in world.transition_outcomes(world.get_state(), "up"):
        print(outcome)
