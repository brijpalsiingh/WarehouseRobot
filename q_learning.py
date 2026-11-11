"""Part 2: Model-free tabular Q-Learning for the warehouse robot.

Unlike Value Iteration, training uses only observations returned by step().
The transition probability table and deterministic_outcome() are never read.

AI assistance: ChatGPT contributed to this module in November 2026.
Include the shared prompt link in the assignment's source/report citations.
"""

import argparse
import random
from copy import deepcopy
from dataclasses import dataclass
from math import isfinite
from typing import Dict, List, Optional

from warehouse_gridworld_domain_random import ACTIONS
from warehouse_mdp import State, WarehouseMDP


# The two documented experiments also appear as selectable GUI presets.
# Keep one shared definition so GUI demonstrations and batch experiments agree.
Q_CONFIGURATIONS = {
    "A": dict(alpha=0.20, gamma=0.99, epsilon_start=1.0,
              epsilon_end=0.05, epsilon_decay_fraction=0.80),
    "B": dict(alpha=0.10, gamma=0.99, epsilon_start=1.0,
              epsilon_end=0.10, epsilon_decay_fraction=0.95),
    # Optional third experiment; A and B remain the assignment baselines.
    "C": dict(alpha=0.30, gamma=0.99, epsilon_start=1.0,
              epsilon_end=0.01, epsilon_decay_fraction=0.60),
}


@dataclass
class QLearningResult:
    q_values: Dict[State, Dict[str, float]]
    policy: Dict[State, Optional[str]]
    gamma: float
    alpha: float
    episodes: int
    epsilon_start: float
    epsilon_end: float
    epsilon_decay_episodes: int
    rewards: List[float]
    steps: List[int]
    successes: List[bool]

    @property
    def training_success_rate(self):
        return sum(self.successes) / len(self.successes)

    @property
    def recent_success_rate(self):
        recent = self.successes[-min(500, len(self.successes)):]
        return sum(recent) / len(recent)


def epsilon_for_episode(index, initial, final, decay_episodes):
    """Linearly reduce exploration without dropping below the final rate."""
    if decay_episodes <= 0:
        raise ValueError("decay_episodes must be positive")
    fraction = min(1.0, index / decay_episodes)
    return initial + (final - initial) * fraction


def greedy_action(action_values, rng=None):
    """Choose the largest Q; random ties during training, stable after training."""
    best = max(action_values.values())
    choices = [action for action in ACTIONS if action_values[action] == best]
    return choices[0] if rng is None else rng.choice(choices)


def train_q_learning(world, episodes=20000, alpha=0.2, gamma=0.99,
                     epsilon_start=1.0, epsilon_end=0.05,
                     epsilon_decay_fraction=0.8, training_seed=2026,
                     progress_callback=None):
    """Learn Q(s,a) from sampled experience using epsilon-greedy exploration.

    The source world and its seeded station locations are not modified.
    Terminal/truncated transitions have no bootstrap term. The resulting
    deterministic greedy policy can be replayed by PolicyPlayback.
    """
    if not isinstance(episodes, int) or episodes <= 0:
        raise ValueError("episodes must be a positive integer")
    if not isfinite(alpha) or not 0 < alpha <= 1:
        raise ValueError("alpha must satisfy 0 < alpha <= 1")
    if not isfinite(gamma) or not 0 <= gamma < 1:
        raise ValueError("gamma must satisfy 0 <= gamma < 1")
    if not all(isfinite(value) and 0 <= value <= 1 for value in
               (epsilon_start, epsilon_end, epsilon_decay_fraction)):
        raise ValueError("epsilon values/fraction must be between 0 and 1")
    if epsilon_start < epsilon_end or epsilon_decay_fraction <= 0:
        raise ValueError("epsilon_start must be >= epsilon_end; decay fraction > 0")

    experiment = deepcopy(world)
    experiment.reseed(training_seed, regenerate_layout=False)
    randomizer = random.Random(training_seed)
    q = {state: {action: 0.0 for action in ACTIONS} for state in experiment.valid_states()}
    rewards, steps, successes = [], [], []
    decay_episodes = max(1, int(episodes * epsilon_decay_fraction))
    progress_interval = max(1, episodes // 20)

    for episode in range(episodes):
        state = experiment.reset()
        epsilon = epsilon_for_episode(episode, epsilon_start,
                                      epsilon_end, decay_episodes)
        while not experiment.done:
            if randomizer.random() < epsilon:
                action = randomizer.choice(ACTIONS)
            else:
                action = greedy_action(q[state], randomizer)

            observed = experiment.step(action)
            target = observed.reward
            if not observed.done:
                target += gamma * max(q[observed.state].values())
            q[state][action] += alpha * (target - q[state][action])
            state = observed.state

        rewards.append(experiment.score)
        steps.append(experiment.steps)
        successes.append(observed.info.get("event") == "return_to_dock")
        if progress_callback is not None and (
            (episode + 1) % progress_interval == 0 or episode + 1 == episodes
        ):
            progress_callback(episode + 1, episodes)

    policy = {
        state: None if experiment.is_success_state(state) else greedy_action(actions)
        for state, actions in q.items()
    }
    return QLearningResult(
        q_values=q, policy=policy, gamma=gamma, alpha=alpha,
        episodes=episodes, epsilon_start=epsilon_start,
        epsilon_end=epsilon_end, epsilon_decay_episodes=decay_episodes,
        rewards=rewards, steps=steps, successes=successes,
    )


def main():
    parser = argparse.ArgumentParser(description="Warehouse tabular Q-Learning")
    parser.add_argument("--seed", type=int, default=42, help="Warehouse layout seed")
    parser.add_argument("--p", type=float, default=0.8, help="Intended motion probability")
    parser.add_argument("--episodes", type=int, default=20000,
                        help="Q-Learning training episodes")
    parser.add_argument("--alpha", type=float, default=0.2, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--training-seed", type=int, default=2026)
    parser.add_argument("--step-delay", type=int, default=500,
                        help="Milliseconds between displayed moves")
    parser.add_argument("--no-gui", action="store_true")
    parser.add_argument("--export-png", action="store_true",
                        help="Save a three-stage learned-policy image")
    args, _unknown = parser.parse_known_args()  # Accept Spyder's runfile options.

    world = WarehouseMDP(seed=args.seed, p=args.p)
    result = train_q_learning(
        world, episodes=args.episodes, alpha=args.alpha, gamma=args.gamma,
        epsilon_start=args.epsilon_start, epsilon_end=args.epsilon_end,
        training_seed=args.training_seed,
    )
    # Reuse the same evaluation episodes as Value Iteration for fair comparison.
    from value_iteration import evaluate_policy
    evaluation = evaluate_policy(world, result.policy, episodes=100)

    print("\n--- COSC 4117 Assignment 2 / Part 2: Q-Learning ---")
    print(f"Seed={args.seed}, p={args.p:.2f}, training episodes={result.episodes}")
    print(f"alpha={result.alpha:.2f}, gamma={result.gamma:.2f}, "
          f"epsilon {result.epsilon_start:.2f}->{result.epsilon_end:.2f}")
    print(f"Training: total success={result.training_success_rate:.1%}, "
          f"last 500 success={result.recent_success_rate:.1%}")
    print(f"Greedy evaluation (100 episodes): reward={evaluation.average_reward:.2f}, "
          f"success={evaluation.success_rate:.1%}, steps={evaluation.average_steps:.1f}")
    print("Start state:", world.reset(), "greedy action:",
          result.policy[world.get_state()])

    if args.export_png:
        from simple_warehouse_gui import save_policy_png
        image_path = save_policy_png(
            world, result,
            f"q_learning_policy_seed{args.seed}_p{args.p:.2f}.png",
            algorithm="Q-Learning",
        )
        print("Saved Q-Learning policy image:", image_path)

    if not args.no_gui:
        from simple_warehouse_gui import show_window
        show_window(world, {"Q-Learning": result},
                    step_delay_ms=args.step_delay)


if __name__ == "__main__":
    main()
