"""Open the COSC 4117 Warehouse Robot GUI in Spyder or a terminal.

This is intentionally a GUI-only entry point. Spyder can retain old script
arguments such as --no-gui; those arguments must never silently disable the
window. The experiment and professor-style console scripts are separate.

AI assistance: developed with ChatGPT in November 2026. Add the shared
conversation link to this comment and the final report before submission.
"""

from warehouse_mdp import WarehouseMDP


def main():
    """Always open the window; choose and apply experiment settings inside."""
    try:
        import pygame  # Check dependency early, before constructing the grid.
    except ImportError as exc:
        raise SystemExit(
            "Pygame is missing in this Python environment. "
            "Open Anaconda Prompt and run: python -m pip install pygame"
        ) from exc

    from simple_warehouse_gui import show_window

    print("Opening Warehouse Robot GUI...")
    print("Select motion p, shared gamma, Q configuration, seed and episodes.")
    print("The left comparison panel updates when each policy is actually played.")
    print("Chart views: Learning Convergence, Q Training Evidence, Reward vs P, Success vs P.")
    print("Then click 'Apply Settings & Train'.")

    warehouse = WarehouseMDP(seed=42, p=0.80)
    try:
        show_window(
            warehouse,
            results={},
            step_delay_ms=500,
            enable_settings=True,
            training_episodes=20000,
        )
    except pygame.error as exc:
        print("Pygame could not create or operate the window:", exc)
        print("Try launching this script from Anaconda Prompt on Windows.")
        raise


if __name__ == "__main__":
    main()
