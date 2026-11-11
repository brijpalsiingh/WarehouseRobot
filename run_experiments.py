"""Required Assignment 2 experiments, separate from GUI playback.

Runs p=1.00, 0.90, 0.80 with two fixed Q-Learning configurations and multiple
training seeds on the same warehouse layout. Output tables/plots contain only
measured runs, never placeholder or invented results.

AI assistance: developed with ChatGPT in November 2026. Add the shared
conversation link here and in the final report before submission.
"""

import argparse
import csv
from pathlib import Path
from statistics import mean, stdev

from q_learning import Q_CONFIGURATIONS, train_q_learning
from simple_warehouse_gui import save_policy_png
from value_iteration import evaluate_policy, expected_action_value, solve_value_iteration
from warehouse_gridworld_domain_random import ACTIONS, CONGESTION, SHELF
from warehouse_mdp import WarehouseMDP


PROBABILITIES = (1.00, 0.90, 0.80)  # Exactly as required by Assignment 2.
CONFIGURATIONS = Q_CONFIGURATIONS
MAIN_CONFIG = "A"  # Fixed in advance, not selected using test-set results.


def save_csv(path, rows, fieldnames):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def congestion_neighbors(world):
    """Passable cells orthogonally adjacent to a congestion cell."""
    neighbors = set()
    for row in range(world.size):
        for col in range(world.size):
            if int(world.grid[row, col]) != CONGESTION:
                continue
            for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                r, c = row + dr, col + dc
                if (0 <= r < world.size and 0 <= c < world.size and
                        int(world.grid[r, c]) != SHELF):
                    neighbors.add((r, c))
    return neighbors


def policy_change_rows(world, vi_by_p):
    """List the exact state/action changes from the deterministic baseline."""
    near = congestion_neighbors(world)
    rows = []
    for state in world.valid_states():
        a1 = vi_by_p[1.0].policy[state]
        a9 = vi_by_p[0.9].policy[state]
        a8 = vi_by_p[0.8].policy[state]
        if a1 == a9 == a8:
            continue
        # Report action-value margin; a change caused only by a tie can then
        # be distinguished from a genuinely preferable new action.
        margins = {}
        for p in PROBABILITIES:
            w = WarehouseMDP(seed=world.seed_value, p=p)
            v = vi_by_p[p]
            action_values = sorted(
                (expected_action_value(w, state, action, v.values, v.gamma)
                 for action in ACTIONS), reverse=True,
            )
            margins[p] = action_values[0] - action_values[1]
        rows.append({
            "state": str(state), "row": state[0], "col": state[1],
            "has_package": state[2], "delivered": state[3],
            "next_to_congestion": (state[0], state[1]) in near,
            "action_p1.00": a1, "action_p0.90": a9, "action_p0.80": a8,
            "margin_p1.00": round(margins[1.0], 6),
            "margin_p0.90": round(margins[0.9], 6),
            "margin_p0.80": round(margins[0.8], 6),
        })
    return rows


def summarize_run_details(rows):
    """Aggregate independent evaluations across seeds, including spread."""
    groups = {}
    for row in rows:
        key = (row["p"], row["algorithm"], row["configuration"])
        groups.setdefault(key, []).append(row)
    summary = []
    for (p, algorithm, configuration), samples in sorted(groups.items(), reverse=True):
        rewards = [float(sample["average_reward"]) for sample in samples]
        success = [float(sample["success_rate"]) for sample in samples]
        steps = [float(sample["average_steps"]) for sample in samples]
        summary.append({
            "p": p, "algorithm": algorithm, "configuration": configuration,
            "seeds": ",".join(str(s["training_seed"]) for s in samples),
            "runs": len(samples),
            "average_reward": round(mean(rewards), 4),
            "reward_sd": round(stdev(rewards), 4) if len(rewards) > 1 else 0.0,
            "success_rate": round(mean(success), 6),
            "success_sd": round(stdev(success), 6) if len(success) > 1 else 0.0,
            "average_steps": round(mean(steps), 4),
        })
    return summary


def plot_performance_charts(output_dir, summary):
    """Regenerate the three-series p charts from measured summary rows."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def points(algorithm, configuration, field):
        return [next(float(row[field]) for row in summary
                     if row["p"] == p and row["algorithm"] == algorithm
                     and row["configuration"] == configuration)
                for p in PROBABILITIES]

    for field, ylabel, filename in (
        ("average_reward", "Mean cumulative reward", "average_reward_vs_p.png"),
        ("success_rate", "Task success rate", "success_rate_vs_p.png"),
    ):
        fig, ax = plt.subplots(figsize=(7.3, 4.3))
        ax.plot(PROBABILITIES, points("Value Iteration", "model-based", field),
                "o-", label="Value Iteration")
        ax.plot(PROBABILITIES, points("Q-Learning", "A", field),
                "s-", label="Q-Learning (config A)")
        ax.plot(PROBABILITIES, points("Q-Learning", "B", field),
                "^-", label="Q-Learning (config B)")
        ax.plot(PROBABILITIES, points("Q-Learning", "C", field),
                "D-", label="Q-Learning (config C)")
        ax.set_xticks(PROBABILITIES)
        ax.set_xlim(1.02, 0.78)
        ax.set_xlabel("Intended-action probability p")
        ax.set_ylabel(ylabel)
        if field == "success_rate":
            ax.set_ylim(-0.03, 1.03)
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=150)
        plt.close(fig)


def plot_experiments(output_dir, summary, vi_by_p, q_histories):
    """Save VI/Q convergence evidence and all three-policy p comparisons."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_performance_charts(output_dir, summary)

    fig, ax = plt.subplots(figsize=(7.3, 4.3))
    for config, histories in q_histories.items():
        if not histories:
            continue
        episodes = len(histories[0])
        width = max(1, episodes // 40)
        xs, ys = [], []
        for start in range(0, episodes, width):
            end = min(episodes, start + width)
            xs.append(end)
            ys.append(mean(mean(history[start:end]) for history in histories))
        ax.plot(xs, ys, label=f"Q-Learning config {config}")
    ax.set_xlabel("Training episode (p=0.80)")
    ax.set_ylabel("Mean training reward (binned across seeds)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "q_learning_convergence.png", dpi=150)
    plt.close(fig)

    vi = vi_by_p[0.8]
    fig, ax = plt.subplots(figsize=(7.3, 4.3))
    ax.semilogy(range(1, vi.iterations + 1),
                [max(d, 1e-12) for d in vi.delta_history])
    ax.axhline(vi.tolerance, linestyle="--", color="firebrick",
               label=f"Threshold {vi.tolerance:g}")
    ax.set_xlabel("Value Iteration number (p=0.80)")
    ax.set_ylabel("Maximum Bellman value change (delta)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "value_iteration_convergence.png", dpi=150)
    plt.close(fig)


def run_experiments(layout_seed=42, training_seeds=(101, 202, 303),
                    episodes=20000, evaluation_episodes=100,
                    output_dir="experiment_results"):
    """Conduct all required experiments and create evidence files."""
    training_seeds = tuple(training_seeds)
    if len(training_seeds) < 2 or len(set(training_seeds)) != len(training_seeds):
        raise ValueError("Use at least two distinct training seeds")
    if episodes <= 0 or evaluation_episodes <= 0:
        raise ValueError("Training/evaluation episodes must be positive")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    vi_by_p = {}
    q_histories = {name: [] for name in CONFIGURATIONS}
    reference_world = None
    for p in PROBABILITIES:
        world = WarehouseMDP(seed=layout_seed, p=p)
        if reference_world is None:
            reference_world = world
        vi = solve_value_iteration(world, gamma=0.99, tolerance=1e-6)
        vi_by_p[p] = vi
        print(f"p={p:.2f}: VI converged in {vi.iterations} iterations")
        save_policy_png(
            world, vi, output_dir / f"value_iteration_policy_p{p:.2f}.png",
            algorithm="Value Iteration",
        )

        for training_seed in training_seeds:
            # Same evaluation motion seed for both policies at this p.
            evaluation_seed = 9000 + training_seed
            vi_eval = evaluate_policy(world, vi.policy,
                                      episodes=evaluation_episodes,
                                      motion_seed=evaluation_seed)
            rows.append({
                "p": p, "algorithm": "Value Iteration",
                "configuration": "model-based", "training_seed": training_seed,
                "evaluation_seed": evaluation_seed,
                "average_reward": vi_eval.average_reward,
                "success_rate": vi_eval.success_rate,
                "average_steps": vi_eval.average_steps,
                "planning_iterations": vi.iterations,
                "final_delta": vi.final_delta,
            })
            for config_name, config in CONFIGURATIONS.items():
                print(f"  Training Q config {config_name}, seed {training_seed}, "
                      f"p={p:.2f}, {episodes:,} episodes...")
                learned = train_q_learning(
                    world, episodes=episodes, training_seed=training_seed,
                    **config,
                )
                measured = evaluate_policy(
                    world, learned.policy, episodes=evaluation_episodes,
                    motion_seed=evaluation_seed,
                )
                rows.append({
                    "p": p, "algorithm": "Q-Learning",
                    "configuration": config_name,
                    "training_seed": training_seed,
                    "evaluation_seed": evaluation_seed,
                    "average_reward": measured.average_reward,
                    "success_rate": measured.success_rate,
                    "average_steps": measured.average_steps,
                    "planning_iterations": "",
                    "final_delta": "",
                })
                if p == 0.8:
                    q_histories[config_name].append(learned.rewards)
                if config_name == MAIN_CONFIG and training_seed == training_seeds[0]:
                    # Representative fixed-seed learned policy; never choose
                    # a favorable seed after observing evaluation results.
                    save_policy_png(
                        world, learned,
                        output_dir / f"q_learning_policy_p{p:.2f}.png",
                        algorithm=f"Q-Learning config {MAIN_CONFIG} / seed {training_seed}",
                    )

    save_csv(output_dir / "individual_runs.csv", rows, [
        "p", "algorithm", "configuration", "training_seed", "evaluation_seed",
        "average_reward", "success_rate", "average_steps",
        "planning_iterations", "final_delta",
    ])
    summary = summarize_run_details(rows)
    save_csv(output_dir / "summary.csv", summary, [
        "p", "algorithm", "configuration", "seeds", "runs",
        "average_reward", "reward_sd", "success_rate", "success_sd",
        "average_steps",
    ])
    changes = policy_change_rows(reference_world, vi_by_p)
    save_csv(output_dir / "policy_changes.csv", changes, [
        "state", "row", "col", "has_package", "delivered",
        "next_to_congestion", "action_p1.00", "action_p0.90",
        "action_p0.80", "margin_p1.00", "margin_p0.90", "margin_p0.80",
    ])
    plot_experiments(output_dir, summary, vi_by_p, q_histories)

    print("\nAll experiments completed:")
    for row in summary:
        print(f"p={row['p']:.2f} {row['algorithm']:<16} "
              f"{row['configuration']:<11} "
              f"reward={row['average_reward']:+.1f} "
              f"success={row['success_rate']:.1%}")
    # Prefer a near-congestion change that is a strict preference at both ends.
    examples = [row for row in changes if row["next_to_congestion"]
                and row["action_p1.00"] != row["action_p0.80"]
                and row["margin_p1.00"] > 0.001
                and row["margin_p0.80"] > 0.001]
    if examples:
        sample = examples[0]
        print("Example policy change near congestion:", sample)
    else:
        print("No strict near-congestion policy change found; inspect policy_changes.csv")
    print("Evidence files saved in:", output_dir.resolve())
    return rows, summary, changes


def main():
    parser = argparse.ArgumentParser(description="Required warehouse experiments")
    parser.add_argument("--layout-seed", type=int, default=42)
    parser.add_argument("--training-seeds", default="101,202,303",
                        help="Comma-separated distinct random seeds")
    parser.add_argument("--episodes", type=int, default=20000,
                        help="Q-Learning episodes per configuration and seed")
    parser.add_argument("--evaluation-episodes", type=int, default=100)
    parser.add_argument("--output-dir", default="experiment_results")
    args, _ = parser.parse_known_args()
    try:
        seeds = tuple(int(token.strip()) for token in args.training_seeds.split(","))
    except ValueError:
        parser.error("--training-seeds must be comma-separated integers")
    run_experiments(layout_seed=args.layout_seed, training_seeds=seeds,
                    episodes=args.episodes,
                    evaluation_episodes=args.evaluation_episodes,
                    output_dir=args.output_dir)


if __name__ == "__main__":
    main()
