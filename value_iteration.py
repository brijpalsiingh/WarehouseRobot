"""Part 1: Value Iteration for the COSC 4117 warehouse assignment.

Implements synchronous Bellman optimality updates using the known MDP model,
extracts a greedy policy, and evaluates that policy in the stochastic world.

AI assistance: ChatGPT helped prepare this module on November 5, 2026.
Include the shared ChatGPT prompt links in your submission comments/report.
"""

import argparse
from copy import deepcopy
from dataclasses import dataclass
from math import isfinite
from typing import Callable, Dict, List, Optional, Tuple

from warehouse_gridworld_domain_random import ACTIONS, SHELF
from warehouse_mdp import State, WarehouseMDP


@dataclass
class ValueIterationResult:
    values: Dict[State, float]
    policy: Dict[State, Optional[str]]
    iterations: int
    final_delta: float
    delta_history: List[float]
    gamma: float
    tolerance: float


@dataclass
class EvaluationResult:
    average_reward: float
    success_rate: float
    average_steps: float
    successful_episodes: int
    episodes: int


def expected_action_value(world, state, action, values, gamma, transitions=None):
    """Q(s,a) = sum P(s'|s,a) [R + gamma * V(s')].

    Terminal transitions have no future continuation value. `transitions`
    can be supplied from a precomputed cache to speed up Value Iteration.
    """
    possible = transitions if transitions is not None else world.transition_outcomes(state, action)
    return sum(
        outcome.probability * (
            outcome.reward + (0.0 if outcome.terminated else gamma * values[outcome.next_state])
        )
        for outcome in possible
    )


def format_iteration_tables(world, values, iteration, delta, gamma=None):
    """Show each of the three task stages as a 10x10 numerical value table.

    This follows the instructor's printed-iteration example, but the warehouse
    needs three tables because its state also records pickup/delivery progress.
    Shelves have no valid state and therefore appear as #### rather than zero.
    """
    phases = (
        ("BEFORE PICKUP (has_package=0, delivered=0)", 0, 0),
        ("CARRYING PACKAGE (has_package=1, delivered=0)", 1, 0),
        ("AFTER DELIVERY (has_package=1, delivered=1)", 1, 1),
    )
    gamma_label = f" | gamma={gamma:g}" if gamma is not None else ""
    lines = [
        f"Iteration {iteration}{gamma_label} | "
        f"Maximum change (delta): {delta:.9g}"
    ]
    for title, has_package, delivered in phases:
        lines.append(title)
        for row in range(world.size):
            cells = []
            for col in range(world.size):
                if int(world.grid[row, col]) == SHELF:
                    cells.append("    ####")
                else:
                    state = (row, col, has_package, delivered)
                    cells.append(f"{values[state]:8.2f}")
            lines.append(" ".join(cells))
        lines.append("")
    return "\n".join(lines)


def solve_value_iteration(world, gamma=0.99, tolerance=1e-6,
                          max_iterations=10000,
                          on_iteration: Optional[Callable] = None):
    """Compute V* and pi* from a known finite discounted MDP.

    Synchronous updates use the previous complete value table for every
    state in an iteration. This is the textbook Bellman update V_{k+1}.
    """
    if not 0.0 <= gamma < 1.0:
        raise ValueError("gamma must satisfy 0 <= gamma < 1")
    if not isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be positive and finite")
    if not isinstance(max_iterations, int) or max_iterations <= 0:
        raise ValueError("max_iterations must be a positive integer")

    states = world.valid_states()
    model = {
        (state, action): world.transition_outcomes(state, action)
        for state in states for action in ACTIONS
    }
    values = {state: 0.0 for state in states}
    delta_history = []

    for iteration in range(1, max_iterations + 1):
        updated_values = {}
        max_delta = 0.0

        for state in states:
            if world.is_success_state(state):
                best_value = 0.0
            else:
                best_value = max(
                    expected_action_value(world, state, action, values, gamma,
                                          transitions=model[(state, action)])
                    for action in ACTIONS
                )
            updated_values[state] = best_value
            max_delta = max(max_delta, abs(best_value - values[state]))

        values = updated_values
        delta_history.append(max_delta)
        if on_iteration is not None:
            # Send an independent snapshot so a GUI or logger can inspect the
            # current iteration without modifying the algorithm's value table.
            on_iteration(iteration, values.copy(), max_delta)
        if max_delta < tolerance:
            break
    else:
        raise RuntimeError(
            f"Value Iteration did not converge after {max_iterations} iterations "
            f"(last delta={max_delta:.8g})"
        )

    # Stable action tie-breaking follows ACTIONS in the instructor framework.
    policy = {}
    for state in states:
        if world.is_success_state(state):
            policy[state] = None
        else:
            policy[state] = max(
                ACTIONS,
                key=lambda action: expected_action_value(
                    world, state, action, values, gamma,
                    transitions=model[(state, action)]
                ),
            )

    return ValueIterationResult(
        values=values, policy=policy, iterations=iteration,
        final_delta=max_delta, delta_history=delta_history,
        gamma=gamma, tolerance=tolerance,
    )


def evaluate_policy(world, policy, episodes=100, motion_seed=2026):
    """Test the fixed greedy policy without exploration over many episodes.

    Reseeding the motion RNG does not regenerate or change station locations.
    A success means completing pickup, delivery, and return before max_steps.
    """
    if episodes <= 0:
        raise ValueError("episodes must be positive")

    # Use a separate copy: evaluation must not alter the original layout seed,
    # motion RNG, or robot state used by subsequent visualizations.
    evaluation_world = deepcopy(world)
    evaluation_world.reseed(motion_seed, regenerate_layout=False)
    rewards = []
    steps = []
    successes = 0

    for _ in range(episodes):
        evaluation_world.reset()
        while not evaluation_world.done:
            action = policy[evaluation_world.get_state()]
            if action is None:
                raise RuntimeError("A nonterminal episode reached a terminal policy state")
            result = evaluation_world.step(action)
        rewards.append(evaluation_world.score)
        steps.append(evaluation_world.steps)
        successes += int(result.info.get("event") == "return_to_dock")

    return EvaluationResult(
        average_reward=sum(rewards) / episodes,
        success_rate=successes / episodes,
        average_steps=sum(steps) / episodes,
        successful_episodes=successes,
        episodes=episodes,
    )


def main():
    parser = argparse.ArgumentParser(description="Warehouse Value Iteration")
    parser.add_argument("--seed", type=int, default=42, help="Fixed warehouse layout seed")
    parser.add_argument("--p", type=float, default=0.8, help="Intended movement probability")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--step-delay", type=int, default=500,
                        help="Milliseconds between robot movements in the GUI")
    parser.add_argument("--no-gui", action="store_true", help="Print results without a policy window")
    parser.add_argument("--export-png", action="store_true",
                        help="Save a three-panel policy image (requires matplotlib)")
    parser.add_argument("--show-iterations", action="store_true",
                        help="Print the value matrices for each Bellman iteration")
    parser.add_argument("--iteration-every", type=int, default=1,
                        help="With --show-iterations, print every Nth iteration (default: 1)")
    args, _unknown = parser.parse_known_args()  # Works with Spyder's runfile options.

    if args.iteration_every <= 0:
        parser.error("--iteration-every must be a positive integer")

    world = WarehouseMDP(seed=args.seed, p=args.p)

    def print_iteration(number, current_values, delta):
        # Always include the first and final iteration, even when displaying
        # only every Nth update to keep a long console trace manageable.
        if (number == 1 or number % args.iteration_every == 0
                or delta < args.tolerance):
            print(format_iteration_tables(
                world, current_values, number, delta, gamma=args.gamma
            ))
            print("-" * 60)

    result = solve_value_iteration(
        world, gamma=args.gamma, tolerance=args.tolerance,
        on_iteration=print_iteration if args.show_iterations else None,
    )
    evaluation = evaluate_policy(world, result.policy, episodes=args.episodes)

    print("\n--- COSC 4117 Assignment 2 / Part 1: Value Iteration ---")
    print(f"Warehouse seed: {args.seed}, pickup: {world.pickup_pos}, "
          f"packing: {world.packing_pos}, dock: {world.dock_pos}")
    print(f"Valid states: {len(result.values)}; actions: {len(ACTIONS)}")
    print(f"p={args.p:.2f}; gamma={result.gamma}; threshold={result.tolerance:g}")
    print(f"Converged after {result.iterations} iterations; final delta={result.final_delta:.9g}")
    start = world.reset()
    print(f"Start state={start}; V(start)={result.values[start]:.3f}; "
          f"best action={result.policy[start]}")
    print(f"Greedy evaluation ({evaluation.episodes} episodes): "
          f"average reward={evaluation.average_reward:.2f}; "
          f"success={evaluation.success_rate:.1%}; "
          f"average steps={evaluation.average_steps:.1f}")
    if args.export_png:
        from simple_warehouse_gui import save_policy_png
        filename = save_policy_png(world, result,
                                   f"value_iteration_policy_seed{args.seed}_p{args.p:.2f}.png")
        print("Saved policy image:", filename)
    if not args.no_gui:
        from simple_warehouse_gui import show_window
        show_window(world, {"Value Iteration": result},
                    step_delay_ms=args.step_delay)


if __name__ == "__main__":
    main()
