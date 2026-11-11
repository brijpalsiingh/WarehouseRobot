"""Three-column Pygame viewer with visible policy comparison and settings.

The main launcher waits for Apply Settings & Train; changing a field never
starts work automatically. Demonstration results appear only after playback.
The report's multi-seed experiments remain a separate explicit script.

AI assistance: developed with ChatGPT in November 2026. Add the shared
conversation link to this comment and the final report before submission.
"""

import csv
from pathlib import Path

from policy_playback import PolicyPlayback
from warehouse_gridworld_domain_random import (
    CONGESTION, CONGESTION_COLOR, DOCK, DOCK_COLOR, EMPTY_COLOR,
    PACKING, PACKING_COLOR, PICKUP, PICKUP_COLOR, SHELF, SHELF_COLOR,
)


PHASES = ((0, 0, "Find pickup"), (1, 0, "Deliver package"),
          (1, 1, "Return to dock"))
ARROWS = {"up": "↑", "down": "↓", "left": "←", "right": "→"}
CELL_COLORS = {
    SHELF: SHELF_COLOR,
    CONGESTION: CONGESTION_COLOR,
    PICKUP: PICKUP_COLOR,
    PACKING: PACKING_COLOR,
    DOCK: DOCK_COLOR,
}


def epsilon_decay_samples(training_result, sample_count=72):
    """Return points from the epsilon schedule actually used by Q training.

    These points are a scheduled exploration *probability*, not an observed
    random-action frequency. Episode numbers are one-based for display.
    """
    from q_learning import epsilon_for_episode

    if training_result is None or not hasattr(training_result, "epsilon_start"):
        return []
    episodes = training_result.episodes
    if episodes <= 0:
        return []
    last_index = episodes - 1
    sample_count = max(2, int(sample_count))
    indexes = sorted({round(i * last_index / (sample_count - 1))
                      for i in range(sample_count)})
    return [(index + 1, epsilon_for_episode(
                index, training_result.epsilon_start,
                training_result.epsilon_end,
                training_result.epsilon_decay_episodes))
            for index in indexes]


def binned_training_rewards(training_result, max_bins=80):
    """Actual training reward per episode, averaged in adjacent time bins."""
    if training_result is None or not getattr(training_result, "rewards", None):
        return []
    values = training_result.rewards
    bin_size = max(1, (len(values) + max_bins - 1) // max_bins)
    return [(end, sum(values[start:end]) / (end - start))
            for start in range(0, len(values), bin_size)
            for end in [min(start + bin_size, len(values))]]


def binned_training_success_rate(training_result, max_bins=80):
    """Percentage of recorded successful deliveries within each training bin.

    Do not confuse this training metric with a post-training greedy evaluation.
    """
    if training_result is None or not getattr(training_result, "successes", None):
        return []
    values = training_result.successes
    bin_size = max(1, (len(values) + max_bins - 1) // max_bins)
    return [(end, 100.0 * sum(values[start:end]) / (end - start))
            for start in range(0, len(values), bin_size)
            for end in [min(start + bin_size, len(values))]]


def load_benchmark_summary(path=None):
    """Read saved multi-seed evaluations; never relabel them as GUI runs."""
    if path is None:
        project = Path(__file__).resolve().parent
        three_config = project / "experiment_results_with_C" / "summary.csv"
        old_baseline = project / "experiment_results" / "summary.csv"
        path = three_config if three_config.is_file() else old_baseline
    else:
        path = Path(path)
    try:
        with path.open(newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
        if not all({"p", "algorithm", "configuration", "average_reward",
                    "success_rate"}.issubset(row) for row in rows):
            return []
        for row in rows:
            for name in ("p", "average_reward", "success_rate"):
                float(row[name])
        return rows
    except (OSError, csv.Error, ValueError, TypeError):
        return []


def save_policy_png(world, result, destination="warehouse_policy.png",
                    algorithm="Value Iteration"):
    """Export a compact three-phase policy figure for the PDF report."""
    import matplotlib.pyplot as plt
    import numpy as np

    colors = np.zeros((world.size, world.size, 3), dtype=float)
    for r in range(world.size):
        for c in range(world.size):
            colors[r, c] = [v / 255 for v in CELL_COLORS.get(
                int(world.grid[r, c]), EMPTY_COLOR
            )]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), dpi=140)
    for ax, (has_package, delivered, phase_name) in zip(axes, PHASES):
        ax.imshow(colors, interpolation="nearest")
        for r in range(world.size):
            for c in range(world.size):
                if int(world.grid[r, c]) == SHELF:
                    continue
                action = result.policy.get((r, c, has_package, delivered))
                if action:
                    ax.text(c, r, ARROWS[action], ha="center", va="center",
                            fontsize=13, color="#1b263b", weight="bold")
        ax.set_title(phase_name)
        ax.set_xticks(range(world.size))
        ax.set_yticks(range(world.size))
        ax.grid(color="#9ca3af", linewidth=0.35)
    fig.suptitle(f"{algorithm} policy (p={world.p:.2f})")
    fig.tight_layout()
    destination = Path(destination)
    fig.savefig(destination, bbox_inches="tight")
    plt.close(fig)
    return destination


def show_window(world, results=None, step_delay_ms=500, enable_settings=False,
                training_episodes=20000):
    """Run the optional viewer and its small, validated experiment controls.

    When invoked by run_warehouse.py, results is empty and the user explicitly
    selects Apply Settings & Train. Calling from the individual algorithms
    with precomputed results still works, but editing settings is disabled.
    """
    import pygame
    from q_learning import Q_CONFIGURATIONS, train_q_learning
    from value_iteration import solve_value_iteration
    from warehouse_mdp import WarehouseMDP

    results = dict(results or {})
    pygame.init()
    # Draw at one logical resolution, then scale it to the available screen.
    # This keeps map cells, text and buttons aligned on small and large displays.
    canvas_size = (1450, 850)
    desktop_sizes = pygame.display.get_desktop_sizes()
    desktop_w, desktop_h = desktop_sizes[0] if desktop_sizes else (1280, 800)
    initial_size = (
        min(desktop_w, max(480, min(1300, int(desktop_w * 0.92)))),
        min(desktop_h, max(360, min(950, int(desktop_h * 0.88)))),
    )
    window = pygame.display.set_mode(initial_size, pygame.RESIZABLE)
    screen = pygame.Surface(canvas_size)
    pygame.display.set_caption("Warehouse Robot - Value Iteration and Q-Learning")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("arial", 20)
    small = pygame.font.SysFont("arial", 16)
    tiny = pygame.font.SysFont("arial", 12)
    title_font = pygame.font.SysFont("arial", 25, bold=True)
    comparison_x, comparison_width = 18, 285
    map_x, map_y, cell_size = 329, 38, 54
    sidebar_x = 890
    sidebar_width = 542
    field_width = 265
    field_gap = 12

    def viewport():
        window_w, window_h = window.get_size()
        scale = min(window_w / canvas_size[0], window_h / canvas_size[1])
        width = max(1, round(canvas_size[0] * scale))
        height = max(1, round(canvas_size[1] * scale))
        return ((window_w - width) // 2, (window_h - height) // 2,
                width, height)

    def logical_mouse_position(point):
        x_offset, y_offset, width, height = viewport()
        px, py = point
        if not (x_offset <= px < x_offset + width and
                y_offset <= py < y_offset + height):
            return None
        return (int((px - x_offset) * canvas_size[0] / width),
                int((py - y_offset) * canvas_size[1] / height))

    def present():
        x_offset, y_offset, width, height = viewport()
        window.fill((229, 234, 240))
        if (width, height) == canvas_size:
            window.blit(screen, (x_offset, y_offset))
        else:
            scaled = pygame.transform.smoothscale(screen, (width, height))
            window.blit(scaled, (x_offset, y_offset))
        pygame.display.flip()

    # Both algorithms use the same discount factor. Q-Learning exploration
    # follows the internal A/B configurations rather than GUI epsilon fields.
    SEED_FIELD = "Grid & Q seed"
    MAX_SEED = 999999
    choices = {
        "Motion p": (1.0, 0.9, 0.8),
        "Q config": ("A", "B", "C"),
        "Episodes": (5000, 10000, 20000),
        "Gamma": (0.0, 0.5, 0.99),
    }
    selected_settings = {
        "Motion p": float(world.p),
        SEED_FIELD: int(world.seed_value),
        "Q config": "A",
        "Episodes": int(training_episodes),
        "Gamma": 0.99,
    }
    for name, value in selected_settings.items():
        if name in choices and value not in choices[name]:
            raise ValueError(f"Unsupported GUI {name}: {value}; choose from {choices[name]}")
    if not 0 <= selected_settings[SEED_FIELD] <= MAX_SEED:
        raise ValueError(f"Grid seed must be between 0 and {MAX_SEED}")

    # Logical coordinates are constant even when the physical window resizes.
    setting_fields = {
        "Motion p": pygame.Rect(sidebar_x, 89, field_width, 38),
        SEED_FIELD: pygame.Rect(sidebar_x + field_width + field_gap, 89, field_width, 38),
        "Gamma": pygame.Rect(sidebar_x, 181, field_width, 38),
        "Q config": pygame.Rect(sidebar_x, 281, field_width, 38),
        "Episodes": pygame.Rect(sidebar_x + field_width + field_gap,
                                281, field_width, 38),
    }
    buttons = {
        "Learning Convergence": pygame.Rect(comparison_x + 5, 310,
                                            comparison_width - 10, 39),
        "Reward vs P": pygame.Rect(comparison_x + 5, 355,
                                   comparison_width - 10, 39),
        "Success Rate": pygame.Rect(comparison_x + 5, 400,
                                    comparison_width - 10, 39),
        "Q Training Evidence": pygame.Rect(comparison_x + 5, 445,
                                           comparison_width - 10, 39),
        "Back to Robot View": pygame.Rect(comparison_x + 5, 490,
                                          comparison_width - 10, 39),
        "Apply Settings & Train": pygame.Rect(sidebar_x, 393, sidebar_width, 42),
        "Value Iteration": pygame.Rect(sidebar_x, 473, 168, 41),
        "Q Explore": pygame.Rect(sidebar_x + 181, 473, 168, 41),
        "Q Greedy": pygame.Rect(sidebar_x + 362, 473, 180, 41),
        "Pause": pygame.Rect(sidebar_x, 525, 173, 38),
        "Next Step": pygame.Rect(sidebar_x + 185, 525, 173, 38),
        "Stop": pygame.Rect(sidebar_x + 370, 525, 172, 38),
        "Show Policy": pygame.Rect(sidebar_x, 576, sidebar_width, 38),
    }
    selected = None
    selected_run = None
    playback = None
    analysis_view = None
    show_policy = False
    selected_phase = 0
    training = False
    dropdown_open = None
    seed_active = False
    seed_select_all = False
    seed_text = str(selected_settings[SEED_FIELD])
    seed_error = ""
    settings_status = ("Choose settings and click Apply Settings & Train"
                       if enable_settings else "Precomputed policy viewer; controls are read-only")
    # Results stay Not run until the user actually starts a demonstration.
    completed_runs = {"Value Iteration": None, "Q Explore": None,
                      "Q Greedy": None}
    q_variants = {}
    saved_benchmark = load_benchmark_summary()

    def write(message, x, y, color=(35, 45, 60), text_font=None):
        text_font = text_font or small
        screen.blit(text_font.render(str(message), True, color), (x, y))

    def chart_axes(rect, x_ticks, y_ticks, x_label, y_label):
        """Draw numeric chart axes and return the inner plotting rectangle."""
        left, right = rect.left + 79, rect.right - 23
        top, bottom = rect.top + 23, rect.bottom - 65
        for i, y_label_text in enumerate(y_ticks):
            y = round(top + i * (bottom - top) / (len(y_ticks) - 1))
            pygame.draw.line(screen, (226, 232, 239), (left, y), (right, y), 1)
            write(y_label_text, left - 55, y - 8, text_font=tiny)
        for i, x_label_text in enumerate(x_ticks):
            x = round(left + i * (right - left) / (len(x_ticks) - 1))
            pygame.draw.line(screen, (145, 156, 170), (x, bottom),
                             (x, bottom + 5), 1)
            glyph = tiny.render(x_label_text, True, (35, 45, 60))
            screen.blit(glyph, glyph.get_rect(center=(x, bottom + 17)))
        pygame.draw.line(screen, (79, 92, 109), (left, top), (left, bottom), 2)
        pygame.draw.line(screen, (79, 92, 109), (left, bottom), (right, bottom), 2)
        glyph = small.render(x_label, True, (35, 45, 60))
        screen.blit(glyph, glyph.get_rect(center=((left + right) // 2,
                                                  rect.bottom - 19)))
        vertical = pygame.transform.rotate(small.render(y_label, True,
                                                        (35, 45, 60)), 90)
        screen.blit(vertical, vertical.get_rect(center=(rect.left + 13,
                                                        (top + bottom) // 2)))
        return left, right, top, bottom

    def active_episode():
        return playback is not None and playback.status in ("running", "paused")

    def invalidate_results():
        nonlocal selected, selected_run, playback, selected_phase, settings_status
        results.clear()
        q_variants.clear()
        completed_runs.update({name: None for name in completed_runs})
        selected = None
        selected_run = None
        playback = None
        selected_phase = 0
        settings_status = "Settings changed: click Apply Settings & Train"

    def can_edit_settings():
        return enable_settings and not training and not active_episode()

    def display_value(name, value):
        if name == "Q config":
            cfg = Q_CONFIGURATIONS[value]
            return (f"{value}: a={cfg['alpha']:.2f}, "
                    f"eps={cfg['epsilon_end']:.2f}")
        if name in ("Motion p", "Gamma"):
            return f"{value:.2f}"
        if name == "Episodes":
            return f"{value:,}"
        return str(value)

    def dropdown_options(name):
        field = setting_fields[name]
        return [(choice, pygame.Rect(field.x, field.bottom + 3 + index * 37,
                                     field.width, 37))
                for index, choice in enumerate(choices[name])]

    def set_setting(name, value):
        if not can_edit_settings():
            return
        if selected_settings[name] != value:
            selected_settings[name] = value
            invalidate_results()

    def commit_seed():
        nonlocal world, settings_status
        nonlocal seed_active, seed_text, seed_error, seed_select_all
        if seed_error:
            return False
        if not seed_text or not seed_text.isascii() or not seed_text.isdecimal():
            seed_error = "Enter 0–999999"
            return False
        entered = int(seed_text)
        if not 0 <= entered <= MAX_SEED:
            seed_error = "Enter 0–999999"
            return False
        changed = entered != selected_settings[SEED_FIELD]
        set_setting(SEED_FIELD, entered)
        if changed:
            # Preview the newly seeded stations immediately, without training.
            # Applying will compute both policies for this same new layout.
            world = WarehouseMDP(seed=entered, p=world.p)
            settings_status = "New grid shown; click Apply Settings & Train"
        seed_text = str(entered)
        seed_error = ""
        seed_active = False
        seed_select_all = False
        return True

    def show_training_progress(message):
        # Training is a deliberately explicit operation. Keep its progress
        # visible and the Pygame window responsive to its operating system.
        pygame.draw.rect(screen, (247, 249, 252),
                         pygame.Rect(sidebar_x, 621, sidebar_width, 29))
        write(message, sidebar_x, 625, color=(24, 103, 77))
        present()
        pygame.event.pump()

    def apply_settings():
        nonlocal world, selected, selected_run, playback, training, settings_status
        if not enable_settings or training or active_episode():
            return
        if seed_active and not commit_seed():
            return
        training = True
        results.clear()
        completed_runs.update({name: None for name in completed_runs})
        selected = None
        selected_run = None
        playback = None
        try:
            gamma = selected_settings["Gamma"]
            target_world = WarehouseMDP(
                seed=selected_settings[SEED_FIELD],
                p=selected_settings["Motion p"],
            )
            show_training_progress(f"Calculating Value Iteration (gamma={gamma:g})...")
            vi = solve_value_iteration(target_world, gamma=gamma, tolerance=1e-6)

            config_name = selected_settings["Q config"]
            episodes = selected_settings["Episodes"]
            seed = selected_settings[SEED_FIELD]

            # Both A and B use the same seed, layout, gamma and episode budget,
            # so the convergence chart can compare their actual learning runs.
            q_variants.clear()
            for variant_name, variant_config in Q_CONFIGURATIONS.items():
                def progress(completed, total, name=variant_name):
                    show_training_progress(
                        f"Training Q config {name}: {completed:,}/{total:,} episodes"
                    )

                q_variants[variant_name] = train_q_learning(
                    target_world, episodes=episodes, training_seed=seed,
                    progress_callback=progress,
                    **{**variant_config, "gamma": gamma},
                )
            q = q_variants[config_name]
            world = target_world
            results.update({"Value Iteration": vi, "Q-Learning": q})
            settings_status = (
                f"Ready: VI {vi.iterations} sweeps; Q A/B/C trained "
                f"({episodes:,} episodes each, gamma={gamma:g})"
            )
            print(f"Trained GUI settings: grid seed={world.seed_value}, p={world.p:.2f}, "
                  f"gamma={gamma:g}, Q A/B/C trained, "
                  f"Q config={config_name}, seed={seed}, "
                  f"episodes={episodes}; "
                  f"VI iterations={vi.iterations}")
        except Exception as error:
            q_variants.clear()
            results.clear()
            settings_status = f"Training failed: {str(error)[:53]}"
            print("Training error:", repr(error))
        finally:
            training = False

    def start_model(name):
        nonlocal selected, selected_run, playback, selected_phase
        model_name = "Value Iteration" if name == "Value Iteration" else "Q-Learning"
        if model_name not in results or active_episode() or training:
            return
        selected = model_name
        selected_run = name
        demo_epsilon = 0.30 if name == "Q Explore" else 0.0
        playback = PolicyPlayback(world, results[model_name].policy,
                                  step_delay_ms=step_delay_ms,
                                  epsilon=demo_epsilon,
                                  exploration_seed=world.seed_value + 1009)
        playback.start()
        completed_runs[name] = None
        selected_phase = 0

    def stop_model():
        nonlocal playback
        if playback is None:
            return
        if selected_run is not None and completed_runs[selected_run] is None and playback.history:
            completed_runs[selected_run] = (
                "Stopped", playback.world.steps, playback.world.score,
                sum(move.slipped for move in playback.history))
        playback.stop()

    def remember_completion():
        if (playback is not None and selected_run is not None and
                completed_runs[selected_run] is None and
                playback.status in ("success", "timeout")):
            completed_runs[selected_run] = (
                playback.status.title(), playback.world.steps,
                playback.world.score,
                sum(move.slipped for move in playback.history),
            )

    running = True
    while running:
        elapsed_ms = clock.tick(30)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.VIDEORESIZE:
                # A user can drag the window edges or maximize it. The
                # logical controls remain the same size and scale together.
                requested = (max(480, event.w), max(360, event.h))
                if requested != window.get_size():
                    window = pygame.display.set_mode(requested, pygame.RESIZABLE)
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                logical_pos = logical_mouse_position(event.pos)
                if logical_pos is None:
                    dropdown_open = None
                    if seed_active and not commit_seed():
                        continue
                    continue
                if dropdown_open is not None and can_edit_settings() and analysis_view is None:
                    selected_option = next((choice for choice, rect in
                                            dropdown_options(dropdown_open)
                                            if rect.collidepoint(logical_pos)), None)
                    if selected_option is not None:
                        set_setting(dropdown_open, selected_option)
                        dropdown_open = None
                        continue
                field = (next((name for name, rect in setting_fields.items()
                               if rect.collidepoint(logical_pos)), None)
                         if analysis_view is None else None)
                if field is not None:
                    if can_edit_settings():
                        if field == SEED_FIELD:
                            dropdown_open = None
                            seed_active = True
                            seed_select_all = True
                            seed_error = ""
                        else:
                            if seed_active and not commit_seed():
                                continue
                            dropdown_open = None if dropdown_open == field else field
                    else:
                        settings_status = "Stop playback before changing settings"
                    continue
                if seed_active and not commit_seed():
                    continue
                dropdown_open = None
                for name, rect in buttons.items():
                    if not rect.collidepoint(logical_pos):
                        continue
                    if name in ("Learning Convergence", "Reward vs P",
                                "Success Rate", "Q Training Evidence"):
                        if active_episode() and playback.status == "running":
                            playback.pause()
                        analysis_view = name
                    elif name == "Back to Robot View":
                        analysis_view = None
                    elif analysis_view is not None:
                        continue
                    elif name == "Apply Settings & Train":
                        apply_settings()
                    elif name in ("Value Iteration", "Q Explore", "Q Greedy"):
                        start_model(name)
                    elif name == "Pause" and active_episode():
                        if playback.status == "running":
                            playback.pause()
                        else:
                            playback.start()
                    elif name == "Next Step" and playback is not None:
                        if playback.status in ("running", "paused"):
                            playback.next_step()
                    elif name == "Stop":
                        stop_model()
                    elif name == "Show Policy":
                        show_policy = not show_policy
                    break
            elif event.type == pygame.KEYDOWN:
                if seed_active:
                    if event.key == pygame.K_ESCAPE:
                        seed_text = str(selected_settings[SEED_FIELD])
                        seed_active = False
                        seed_select_all = False
                        seed_error = ""
                    elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER,
                                       pygame.K_TAB):
                        commit_seed()
                    elif event.key in (pygame.K_BACKSPACE, pygame.K_DELETE):
                        seed_text = "" if (seed_select_all or
                                          event.key == pygame.K_DELETE) else seed_text[:-1]
                        seed_select_all = False
                        seed_error = ""
                    elif (event.key == pygame.K_a and
                          getattr(event, "mod", 0) & pygame.KMOD_CTRL):
                        seed_select_all = True
                    else:
                        typed = getattr(event, "unicode", "")
                        if typed and typed.isascii() and typed.isdecimal():
                            updated = typed if seed_select_all else seed_text + typed
                            if len(updated) <= 8:
                                seed_text = updated
                                seed_error = ("Enter 0–999999" if int(updated) > MAX_SEED
                                              else "")
                                seed_select_all = False
                            else:
                                seed_error = "Enter 0–999999"
                        elif typed and typed.isprintable():
                            seed_error = "Digits only"
                    continue
                if event.key == pygame.K_ESCAPE:
                    if dropdown_open is not None:
                        dropdown_open = None
                    else:
                        running = False
                elif event.key == pygame.K_p:
                    show_policy = not show_policy
                elif event.key == pygame.K_SPACE and active_episode():
                    if playback.status == "running":
                        playback.pause()
                    else:
                        playback.start()
                elif event.key == pygame.K_n and active_episode():
                    playback.next_step()
                elif event.key == pygame.K_s:
                    pygame.image.save(window, "warehouse_screenshot.png")
                elif event.key in (pygame.K_1, pygame.K_2, pygame.K_3):
                    selected_phase = (pygame.K_1, pygame.K_2,
                                      pygame.K_3).index(event.key)

        if active_episode():
            playback.update(elapsed_ms)
            selected_phase = playback.phase
        remember_completion()

        screen.fill((247, 249, 252))
        write("Warehouse Robot", map_x, 8, text_font=title_font)
        for r in range(world.size):
            for c in range(world.size):
                cell_type = int(world.grid[r, c])
                rect = pygame.Rect(map_x + c * cell_size,
                                   map_y + r * cell_size, cell_size, cell_size)
                pygame.draw.rect(screen, CELL_COLORS.get(cell_type, EMPTY_COLOR), rect)
                if (playback is not None and playback.status != "stopped" and
                        (r, c) in playback.visit_counts):
                    pygame.draw.rect(screen, (245, 162, 52), rect, 4)
                pygame.draw.rect(screen, (190, 196, 204), rect, 1)
                label = {PICKUP: "P", PACKING: "D", DOCK: "C",
                         CONGESTION: "!"}.get(cell_type)
                if label:
                    glyph = small.render(label, True, (35, 45, 55))
                    screen.blit(glyph, glyph.get_rect(center=rect.center))
                if (show_policy and selected is not None and selected in results
                        and cell_type != SHELF):
                    has_package, delivered, _ = PHASES[selected_phase]
                    action = results[selected].policy.get((r, c, has_package, delivered))
                    if action:
                        glyph = font.render(ARROWS[action], True, (30, 52, 77))
                        screen.blit(glyph, glyph.get_rect(center=rect.center))

        if playback is not None and playback.status != "stopped":
            r, c = playback.world.robot_pos
            center = (map_x + c * cell_size + cell_size // 2,
                      map_y + r * cell_size + cell_size // 2)
            pygame.draw.circle(screen, (110, 45, 175), center, 17)
            pygame.draw.circle(screen, (255, 255, 255), center, 17, 2)

        # Left: keep separate real runs for VI, Q exploration, and Q greedy.
        # Exploratory demonstration results must never overwrite the strictly
        # greedy evaluation results required for the assignment.
        pygame.draw.rect(screen, (235, 240, 247),
                         (comparison_x - 6, 0, comparison_width + 12, 850))
        write("Policy Comparison", comparison_x, 8, text_font=title_font)
        write("One robot run per policy / mode", comparison_x, 44,
              color=(83, 97, 112))
        comparison_headers = (("Metric", comparison_x + 8),
                              ("VI", comparison_x + 101),
                              ("Explore", comparison_x + 153),
                              ("Greedy", comparison_x + 222))
        pygame.draw.rect(screen, (207, 220, 234),
                         (comparison_x, 76, comparison_width, 31), border_radius=4)
        for header, header_x in comparison_headers:
            write(header, header_x, 83,
                  text_font=tiny if header in ("Explore", "Greedy") else small)

        def actual_metric(name, metric):
            record = completed_runs[name]
            if record is None:
                if selected_run == name and active_episode():
                    if metric == "Status":
                        return "Running"
                    if metric == "Steps":
                        return str(playback.world.steps)
                    if metric == "Reward":
                        return f"{playback.world.score:+.0f}"
                    if metric == "Slips":
                        return str(sum(move.slipped for move in playback.history))
                model_name = "Value Iteration" if name == "Value Iteration" else "Q-Learning"
                return ("Pending" if model_name not in results else "Not run") \
                    if metric == "Status" else "—"
            status, steps, score, slips = record
            return {"Status": status, "Steps": str(steps),
                    "Reward": f"{score:+.0f}", "Slips": str(slips)}[metric]

        for index, metric in enumerate(("Status", "Steps", "Reward", "Slips")):
            row_y = 118 + 36 * index
            if index % 2 == 0:
                pygame.draw.rect(screen, (246, 248, 251),
                                 (comparison_x, row_y - 5, comparison_width, 32))
            write(metric, comparison_x + 8, row_y, text_font=small)
            for algorithm, text_x in (("Value Iteration", comparison_x + 101),
                                      ("Q Explore", comparison_x + 153),
                                      ("Q Greedy", comparison_x + 222)):
                write(actual_metric(algorithm, metric), text_x, row_y,
                      text_font=tiny if metric == "Status" else small)
        write("Reward = total earned; slips = wrong moves", comparison_x, 268,
              text_font=tiny)
        vi_result = results.get("Value Iteration")
        q_result = results.get("Q-Learning")
        vi_delta = getattr(vi_result, "final_delta", None)
        q_episodes = getattr(q_result, "episodes", None)
        q_rate = getattr(q_result, "recent_success_rate", None)

        # Keep the day-to-day comparison view compact. Both learning plots and
        # detailed explanations live in a dedicated, easy-to-capture view.
        write("CHARTS", comparison_x + 8, 291, text_font=small)
        write("Training snapshot", comparison_x + 6, 541, text_font=font)
        write(f"VI: {getattr(vi_result, 'iterations', '—')} Bellman sweeps",
              comparison_x + 8, 577)
        write(f"Q: {q_episodes:,} training episodes" if q_episodes is not None
              else "Q: not trained", comparison_x + 8, 603)
        write(f"Recent Q success: {q_rate:.0%}" if q_rate is not None
              else "Recent Q success: —", comparison_x + 8, 629)
        write(f"Gamma shared: {selected_settings['Gamma']:.2f}",
              comparison_x + 8, 662, color=(80, 99, 118), text_font=small)

        # Both modes use the trained Q policy; only the action-selection
        # epsilon differs. Show the exact mode rather than a generic label.
        pygame.draw.rect(screen, (218, 231, 244),
                         (comparison_x, 696, comparison_width, 77),
                         border_radius=6)
        write("Playback Mode:", comparison_x + 10, 705,
              color=(57, 76, 98), text_font=small)
        if selected_run == "Q Explore":
            mode_text = "Q Exploration (ε = 0.30)"
        elif selected_run == "Q Greedy":
            mode_text = "Greedy Evaluation (ε = 0)"
        elif selected_run == "Value Iteration":
            mode_text = "VI Policy (ε not used)"
        else:
            mode_text = "Greedy Evaluation (ε = 0)"
        write(mode_text, comparison_x + 10, 732,
              color=(30, 70, 113), text_font=font)
        random_draws = (sum(move.explored for move in playback.history)
                        if playback is not None else 0)
        write(f"Random-action draws: {random_draws}", comparison_x + 10, 756,
              color=(57, 76, 98), text_font=tiny)

        # Right: parameters are grouped by the algorithm that uses them.
        write("Parameters & Controls", sidebar_x, 7, text_font=title_font)
        write("Shared environment (both policies)", sidebar_x, 43,
              text_font=font)
        for divider_y in (135, 235, 384):
            pygame.draw.line(screen, (211, 220, 229),
                             (sidebar_x, divider_y),
                             (sidebar_x + sidebar_width, divider_y), 1)
        write("SHARED GAMMA  •  VI uses a known model", sidebar_x, 142,
              text_font=font)
        write("Threshold: 1e-6 (fixed)", sidebar_x + 276,
              176, text_font=small)
        write("Q-LEARNING  •  learns from sampled episodes", sidebar_x, 242,
              text_font=font)
        q_config = Q_CONFIGURATIONS[selected_settings["Q config"]]
        write(f"Config {selected_settings['Q config']}: "
              f"alpha={q_config['alpha']:.2f}, "
              f"epsilon {q_config['epsilon_start']:.2f} -> "
              f"{q_config['epsilon_end']:.2f} over "
              f"{q_config['epsilon_decay_fraction']:.0%} of episodes",
              sidebar_x, 338, text_font=small)
        write("POLICY PLAYBACK (VI | Q explore 30% | Q greedy 0%)", sidebar_x, 446,
              text_font=font)

        for name, rect in setting_fields.items():
            label_y = rect.y - 17
            write(name, rect.x, label_y)
            field_color = ((230, 242, 255) if (dropdown_open == name or
                           (name == SEED_FIELD and seed_active)) else
                           (255, 255, 255) if can_edit_settings() else
                           (226, 231, 237))
            pygame.draw.rect(screen, field_color, rect, border_radius=6)
            pygame.draw.rect(screen, (178, 41, 53) if name == SEED_FIELD and seed_error
                             else (135, 148, 165), rect, width=1, border_radius=6)
            shown_value = (seed_text if name == SEED_FIELD and seed_active else
                           display_value(name, selected_settings[name]))
            if name == SEED_FIELD and seed_active:
                shown_value += "|"
            write(shown_value, rect.x + 12, rect.y + 10,
                  color=(29, 53, 80), text_font=font)
            if name == SEED_FIELD:
                write("type", rect.right - 44, rect.y + 12,
                      color=(84, 103, 125), text_font=tiny)
                if seed_error:
                    write(seed_error, rect.x + 119, label_y + 2,
                          color=(178, 41, 53), text_font=tiny)
            else:
                write("▼", rect.right - 27, rect.y + 10,
                      color=(29, 53, 80), text_font=small)

        for name, rect in buttons.items():
            if name == "Apply Settings & Train":
                enabled = enable_settings and not active_episode() and not training
            elif name in ("Value Iteration", "Q Explore", "Q Greedy"):
                model_name = "Value Iteration" if name == "Value Iteration" else "Q-Learning"
                enabled = model_name in results and not active_episode() and not training
            elif name in ("Pause", "Next Step"):
                enabled = active_episode()
            else:
                enabled = True
            background = ((42, 111, 179) if enabled else (151, 160, 170))
            if name == analysis_view and analysis_view is not None:
                background = (25, 115, 93)
            if name == "Back to Robot View" and analysis_view is None:
                background = (151, 160, 170)
            pygame.draw.rect(screen, background, rect, border_radius=6)
            label = name
            if name == "Pause" and playback is not None and playback.status == "paused":
                label = "Resume"
            if name == "Show Policy":
                label = "Hide Policy Arrows" if show_policy else "Show Policy Arrows"
            if name == "Success Rate":
                label = "Success vs P"
            glyph = small.render(label, True, (255, 255, 255))
            screen.blit(glyph, glyph.get_rect(center=rect.center))

        write(settings_status[:83], sidebar_x, 626, color=(31, 111, 83),
              text_font=tiny)
        if "Value Iteration" in results:
            parameters_text = (f"Applied: seed={world.seed_value}  |  "
                               f"p={world.p:.2f}  |  "
                               f"gamma={results['Value Iteration'].gamma:g}")
        else:
            parameters_text = (f"Selected: seed={selected_settings[SEED_FIELD]}  |  "
                               f"p={selected_settings['Motion p']:.2f}  |  "
                               f"gamma={selected_settings['Gamma']:g} (not trained)")
        write(parameters_text, sidebar_x, 647, text_font=tiny)
        write(f"Policy: {selected_run or 'None'}   |   Phase: {PHASES[selected_phase][2]}",
              sidebar_x, 669, text_font=small)
        if playback is not None and playback.status != "stopped":
            write(f"Status: {playback.status.title()}", sidebar_x, 691)
            write(f"Steps: {playback.world.steps}/{world.max_steps}  |  "
                  f"Reward: {playback.world.score:+.0f}", sidebar_x + 205, 691)
        elif playback is not None:
            write("Status: Stopped", sidebar_x, 691)
        else:
            write("Status: Awaiting training" if not results else "Status: Ready",
                  sidebar_x, 691)

        write("Recent movements", sidebar_x, 719, text_font=font)
        if playback is not None and playback.history:
            for i, move in enumerate(playback.history[-3:]):
                slip = " SLIP" if move.slipped else ""
                explore = " EXPLORE" if move.explored else ""
                msg = (f"#{move.step} {move.selected_action} -> "
                       f"{move.actual_action}{explore}{slip}, r={move.reward:+.0f}")
                write(msg, sidebar_x + 7, 752 + i * 27, text_font=small)
        write("P = pickup   D = packing   C = dock   ! = congestion", map_x, 631)
        write("Black = shelves   Purple = robot   Orange = visited", map_x, 663)
        write("Controls: Space pause | N step | P arrows | S screenshot", map_x, 711)
        write("1/2/3 = task phase | Esc = close", map_x, 742)
        write("Rewards: pickup +25 | delivery +40 | dock +100", map_x, 788)
        write("Costs: step -1 | invalid -3 | congestion -10", map_x, 818)

        if analysis_view is not None:
            # One chart per landscape row, or one wide batch-comparison chart.
            pygame.draw.rect(screen, (247, 249, 252),
                             (map_x - 11, 0, canvas_size[0] - map_x + 11, 850))
            write(analysis_view, map_x + 8, 10, text_font=title_font)

            def draw_plot(rect, series, xlim, ylim, x_ticks, y_ticks,
                          x_label, y_label):
                """Draw an axis-labeled graph with real series on a shared scale."""
                pygame.draw.rect(screen, (255, 255, 255), rect, border_radius=5)
                pygame.draw.rect(screen, (176, 190, 205), rect, 1,
                                 border_radius=5)
                left, right, top, bottom = chart_axes(
                    rect, x_ticks, y_ticks, x_label, y_label)
                x0, x1 = xlim
                y0, y1 = ylim
                for points, color, marker in series:
                    pixels = [(left + round((x - x0) / max(1e-9, x1 - x0) *
                                            (right - left)),
                               bottom - round((y - y0) / max(1e-9, y1 - y0) *
                                              (bottom - top)))
                              for x, y in points]
                    if len(pixels) > 1:
                        pygame.draw.lines(screen, color, False, pixels, 3)
                    if marker:
                        for cx, cy in pixels:
                            if marker == "square":
                                pygame.draw.rect(screen, color, (cx - 5, cy - 5, 10, 10))
                            elif marker == "triangle":
                                pygame.draw.polygon(screen, color,
                                                    ((cx, cy - 6), (cx - 6, cy + 5),
                                                     (cx + 6, cy + 5)))
                            else:
                                pygame.draw.circle(screen, color, (cx, cy), 5)

            def legend(labels, y):
                for i, (label, color) in enumerate(labels):
                    x = map_x + 26 + i * 305
                    pygame.draw.line(screen, color, (x, y + 8), (x + 29, y + 8), 4)
                    write(label, x + 38, y, text_font=small)

            blue, orange, green, purple = ((38, 111, 177), (219, 119, 49),
                                           (39, 135, 106), (131, 78, 183))
            q_colors = {"A": orange, "B": green, "C": purple}
            if analysis_view == "Learning Convergence":
                write(f"Current run: seed={world.seed_value}, p={world.p:.2f}, "
                      f"shared gamma={selected_settings['Gamma']:.2f}",
                      map_x + 8, 49, color=(78, 91, 108), text_font=small)
                pygame.draw.rect(screen, (236, 242, 248),
                                 (map_x, 89, 1103, 327), border_radius=8)
                pygame.draw.rect(screen, (236, 242, 248),
                                 (map_x, 431, 1103, 349), border_radius=8)
                write("Value Iteration — Bellman convergence", map_x + 18, 99,
                      text_font=font)
                write(f"Iterations: {getattr(vi_result, 'iterations', '—')}  |  "
                      f"Threshold: 1e-6  |  "
                      f"Final delta: {vi_delta:.2e}" if vi_delta is not None
                      else "Run Apply Settings & Train to calculate Bellman values",
                      map_x + 18, 127, text_font=small)
                vi_plot = pygame.Rect(map_x + 18, 150, 1064, 256)
                deltas = getattr(vi_result, "delta_history", None)
                if deltas:
                    import math
                    logs = [math.log10(max(d, 1e-12)) for d in deltas]
                    high, low = max(logs), min(logs)
                    if high == low:
                        high += 0.5
                        low -= 0.5
                    draw_plot(vi_plot, [([(i + 1, v) for i, v in enumerate(logs)],
                                        blue, None)],
                              (1, max(2, len(logs))), (low, high),
                              [f"{1 + round(i * (len(logs) - 1) / 4):,}"
                               for i in range(5)],
                              [f"{high - i * (high - low) / 4:.1f}" for i in range(5)],
                              "Bellman iteration", "log10(max delta)")
                else:
                    draw_plot(vi_plot, [], (0, 1), (0, 1), ["—"] * 5, ["—"] * 5,
                              "Bellman iteration", "log10(max delta)")

                write("Q-Learning — A vs B vs C learning reward", map_x + 18, 441,
                      text_font=font)
                write("Both use the same environment, gamma and training episodes; "
                      "vertical values are binned mean episode rewards",
                      map_x + 18, 467, text_font=small)
                legend(tuple((f"{name}: alpha={cfg['alpha']:.2f}, "
                              f"eps->{cfg['epsilon_end']:.2f}", q_colors[name])
                             for name, cfg in Q_CONFIGURATIONS.items()), 488)
                q_plot = pygame.Rect(map_x + 18, 514, 1064, 256)
                lines = [(binned_training_rewards(q_variants.get(name)), color, None)
                         for name, color in q_colors.items()]
                # A standalone Q-Learning script can still show its own curve.
                if not any(pts for pts, _, _ in lines) and q_result is not None:
                    lines = [(binned_training_rewards(q_result), orange, None)]
                values = [y for pts, _, _ in lines for _, y in pts]
                if values:
                    ymin, ymax = min(values), max(values)
                    padding = max(1, (ymax - ymin) * 0.08)
                    ymin -= padding
                    ymax += padding
                    n = max(x for pts, _, _ in lines for x, _ in pts)
                    draw_plot(q_plot, lines, (1, max(2, n)), (ymin, ymax),
                              [f"{1 + round(i * (n - 1) / 4):,}" for i in range(5)],
                              [f"{ymax - i * (ymax - ymin) / 4:.0f}" for i in range(5)],
                              "Training episode", "Mean reward per bin")
                else:
                    draw_plot(q_plot, [], (0, 1), (0, 1), ["—"] * 5, ["—"] * 5,
                              "Training episode", "Mean reward per bin")

            elif analysis_view == "Q Training Evidence":
                # Both charts are derived from training records and schedules,
                # not from fabricated or separately saved benchmark results.
                write("Measured success; linear epsilon decay over A:80%, B:95%, C:60% of episodes",
                      map_x + 8, 44, color=(78, 91, 108), text_font=small)
                legend(tuple((f"{name}: alpha={cfg['alpha']:.2f}, "
                              f"eps->{cfg['epsilon_end']:.2f}", q_colors[name])
                             for name, cfg in Q_CONFIGURATIONS.items()), 69)

                write("Success Rate vs. Training Episodes", map_x + 18, 104,
                      text_font=font)
                success_plot = pygame.Rect(map_x + 18, 132, 1064, 251)
                success_lines = [
                    (binned_training_success_rate(q_variants.get(name)), color, None)
                    for name, color in q_colors.items()
                ]
                if not any(pts for pts, _, _ in success_lines) and q_result:
                    success_lines = [(binned_training_success_rate(q_result),
                                      orange, None)]
                n = max((x for pts, _, _ in success_lines for x, _ in pts),
                        default=1)
                draw_plot(success_plot, success_lines, (1, max(2, n)), (0, 100),
                          [f"{1 + round(i * (n - 1) / 4):,}" for i in range(5)],
                          ["100%", "75%", "50%", "25%", "0%"],
                          "Training episode", "Successful deliveries (%)")

                write("Epsilon vs. Training Episodes", map_x + 18, 401,
                      text_font=font)
                epsilon_plot = pygame.Rect(map_x + 18, 429, 1064, 232)
                epsilon_lines = [
                    (epsilon_decay_samples(q_variants.get(name)), color, None)
                    for name, color in q_colors.items()
                ]
                if not any(pts for pts, _, _ in epsilon_lines) and q_result:
                    epsilon_lines = [(epsilon_decay_samples(q_result), orange, None)]
                n = max((x for pts, _, _ in epsilon_lines for x, _ in pts),
                        default=1)
                draw_plot(epsilon_plot, epsilon_lines, (1, max(2, n)), (0, 1),
                          [f"{1 + round(i * (n - 1) / 4):,}" for i in range(5)],
                          ["1.00", "0.75", "0.50", "0.25", "0.00"],
                          "Training episode", "Epsilon (0 to 1)")

                # Q Explore and Q Greedy are distinct post-training robot
                # demonstrations. Only Q Greedy (epsilon=0) satisfies the
                # assignment's no-exploration evaluation requirement.
                pygame.draw.rect(screen, (230, 239, 249),
                                 (map_x + 18, 681, 1064, 142), border_radius=7)
                write("Greedy Policy Evaluation Results (epsilon = 0)",
                      map_x + 34, 690, text_font=font)
                greedy_run = completed_runs["Q Greedy"]
                exploration_run = completed_runs["Q Explore"]
                if greedy_run is None:
                    greedy_status = ("In progress — finish the run" if
                                     selected_run == "Q Greedy" and active_episode()
                                     else "Not run — select Q Greedy in Robot View")
                else:
                    status, steps, reward, slips = greedy_run
                    greedy_status = (f"{status} | Steps: {steps}/{world.max_steps} | "
                                     f"Cumulative reward: {reward:+.0f} | Slips: {slips}")
                write("Q Greedy (eps=0): " + greedy_status, map_x + 34, 725,
                      color=(29, 86, 76))
                if exploration_run is None:
                    explore_status = "Not run (optional comparison)"
                else:
                    status, steps, reward, slips = exploration_run
                    explore_status = (f"{status} | Steps: {steps}/{world.max_steps} | "
                                      f"Cumulative reward: {reward:+.0f} | Slips: {slips}")
                write("Q Explore (eps=.30): " + explore_status,
                      map_x + 34, 753, text_font=small)
                write("Both demonstrations use the same trained policy; only exploration changes.",
                      map_x + 34, 781, text_font=small)
                write(f"Motion remains stochastic at p={world.p:.2f}; "
                      "epsilon curves show training schedules, not sampled moves.",
                      map_x + 34, 803, text_font=small)

            else:
                field = ("average_reward" if analysis_view == "Reward vs P"
                         else "success_rate")
                ylabel = ("Mean cumulative reward" if field == "average_reward"
                          else "Task success rate (%)")
                write("Saved batch experiment (3 training seeds, fixed layout).",
                      map_x + 8, 52, color=(78, 91, 108))
                write("This is not recalculated from the current GUI controls; "
                      "run run_experiments.py to refresh the CSV.",
                      map_x + 8, 77, color=(78, 91, 108), text_font=small)
                pygame.draw.rect(screen, (236, 242, 248),
                                 (map_x, 111, 1103, 670), border_radius=8)
                labels = (("Value Iteration", blue),
                          ("Q-A (a=0.20)", orange),
                          ("Q-B (a=0.10)", green),
                          ("Q-C (a=0.30)", purple))
                legend(labels, 137)
                batch_lines = []
                for algorithm, config, color, marker in (
                        ("Value Iteration", "model-based", blue, "circle"),
                        ("Q-Learning", "A", orange, "square"),
                        ("Q-Learning", "B", green, "triangle"),
                        ("Q-Learning", "C", purple, "circle")):
                    pts = sorted((float(r["p"]), float(r[field]) *
                                  (100 if field == "success_rate" else 1))
                                 for r in saved_benchmark
                                 if r.get("algorithm") == algorithm and
                                 r.get("configuration") == config)
                    if pts:
                        batch_lines.append((pts, color, marker))
                batch_plot = pygame.Rect(map_x + 18, 176, 1064, 551)
                values = [y for pts, _, _ in batch_lines for _, y in pts]
                if values:
                    if field == "success_rate":
                        ymin, ymax = 0.0, 100.0
                    else:
                        ymin, ymax = min(values) - 10, max(values) + 10
                    draw_plot(batch_plot, batch_lines, (0.8, 1.0), (ymin, ymax),
                              [f"{0.8 + i * 0.05:.2f}" for i in range(5)],
                              [f"{ymax - i * (ymax - ymin) / 4:.0f}"
                               for i in range(5)],
                              "Intended-motion probability p", ylabel)
                    if field == "success_rate" and len(set(values)) == 1:
                        write("Equal rates overlap: all plotted methods have "
                              f"{values[0]:.0f}% measured success.",
                              map_x + 24, 745, text_font=small)
                else:
                    draw_plot(batch_plot, [], (0.8, 1.0), (0, 1), ["—"] * 5,
                              ["—"] * 5, "Intended-motion probability p", ylabel)
                    write("No saved experiment data. Run python run_experiments.py",
                          map_x + 135, 430, text_font=font)

        # Paint expanded dropdowns last so their options stay in front of the
        # controls and cannot be hidden behind a button or another field.
        if dropdown_open is not None and can_edit_settings() and analysis_view is None:
            for choice, option_rect in dropdown_options(dropdown_open):
                is_selected = selected_settings[dropdown_open] == choice
                pygame.draw.rect(screen, (222, 236, 254) if is_selected else
                                 (255, 255, 255), option_rect)
                pygame.draw.rect(screen, (122, 143, 166), option_rect, width=1)
                write(display_value(dropdown_open, choice),
                      option_rect.x + 12, option_rect.y + 9, text_font=font)
                if is_selected:
                    write("✓", option_rect.right - 28, option_rect.y + 10,
                          color=(25, 103, 78))

        present()

    pygame.quit()
