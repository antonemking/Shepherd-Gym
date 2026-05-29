"""L2 asset — value iteration on a TEACHING gridworld derived from Shepherd-v0.

IMPORTANT: Shepherd-v0 itself has NO discrete/gridworld mode (the real env is a
continuous-action boid sim). This is a deliberately simplified, *derived* MDP for
hand-working Bellman backups and drawing a value heatmap. Its pieces map onto the
real env's reward terms so the intuition transfers:

  * one grid cell per metre of the real `world` (default 20x20);
  * the GOAL cell is the discretised `pen_center`; reaching it pays `r_success`
    (+10, exactly the env's terminal bonus) and ends the episode;
  * every step costs `r_time` (-0.01, the env's per-step time penalty);
  * a STRESS cost is charged for sitting near the fence — grounded in the env's
    boundary-avoidance force (`w_bounds`, active within 1.5 m of a wall in
    `_step_flock`) and the isolation term that punishes stragglers. Pressing the
    flock along the rails is stressful; the value map should learn to avoid it.
    The weight reuses the env's welfare weight `r_arousal` (0.5) for honesty.

We think of the grid state as "where the flock currently is"; the dog's job is to
deliver it to the fold. Transitions are deterministic 4-connected moves (the clean
case you can check by hand). We run value iteration to convergence and render V*(s)
as a heatmap over the pasture, with the greedy policy as arrows.

Run:
    python course/assets/l2_value_iteration.py
Outputs:
    course/assets/out/l2_value_heatmap.png
    course/assets/out/l2_value_table.txt   (a 5x5 corner you can check by hand)

Deps: numpy, matplotlib.
"""
from __future__ import annotations
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from shepherd_gym import ShepherdConfig

OUT = pathlib.Path(__file__).resolve().parent / "out"

ACTIONS = {"N": (0, 1), "E": (1, 0), "S": (0, -1), "W": (-1, 0)}
ARROW = {"N": (0, 0.35), "E": (0.35, 0), "S": (0, -0.35), "W": (-0.35, 0)}


def build_mdp(cfg: ShepherdConfig):
    """Return (N, goal_cell, stress[N,N]) for the derived gridworld."""
    N = int(round(cfg.world))                       # 1 cell / metre
    gx, gy = (int(round(cfg.pen_center[0])), int(round(cfg.pen_center[1])))
    goal = (min(gx, N - 1), min(gy, N - 1))

    # stress cost per cell: high within ~1.5 m of any wall (env's bounds threshold),
    # tapering to 0 in the open middle. Scaled by the env welfare weight r_arousal.
    xs = np.arange(N)
    dx = np.minimum(xs, N - 1 - xs)                 # cells from nearest vertical wall
    edge_dist = np.minimum.outer(dx, dx)            # Chebyshev-ish distance to wall
    near = np.clip(1.5 - edge_dist, 0.0, 1.5) / 1.5  # 1 at the rail -> 0 by 1.5 m in
    stress = cfg.r_arousal * near                   # welfare cost map
    return N, goal, stress


def value_iteration(N, goal, stress, gamma=0.99, r_time=0.01, r_success=10.0,
                    tol=1e-6, max_iter=5000):
    """Classic synchronous value iteration. Deterministic transitions.
    V(goal)=0 anchor: the +r_success is paid on the transition INTO the goal.
    """
    V = np.zeros((N, N))
    for it in range(max_iter):
        Vnew = V.copy()
        delta = 0.0
        for x in range(N):
            for y in range(N):
                if (x, y) == goal:
                    continue                        # terminal/absorbing: V=0
                best = -1e18
                for dx, dy in ACTIONS.values():
                    nx, ny = x + dx, y + dy
                    if not (0 <= nx < N and 0 <= ny < N):
                        nx, ny = x, y               # bump the fence -> stay put
                    # reward for this transition: time + stress at the cell we land in,
                    # plus the terminal bonus if we step into the fold.
                    r = -r_time - stress[nx, ny]
                    if (nx, ny) == goal:
                        r += r_success
                        q = r                       # goal is terminal -> no bootstrap
                    else:
                        q = r + gamma * V[nx, ny]
                    best = max(best, q)
                Vnew[x, y] = best
                delta = max(delta, abs(Vnew[x, y] - V[x, y]))
        V = Vnew
        if delta < tol:
            break
    # greedy policy
    pi = np.empty((N, N), dtype=object)
    for x in range(N):
        for y in range(N):
            if (x, y) == goal:
                pi[x, y] = "*"
                continue
            best, arg = -1e18, "N"
            for name, (dx, dy) in ACTIONS.items():
                nx, ny = x + dx, y + dy
                if not (0 <= nx < N and 0 <= ny < N):
                    nx, ny = x, y
                r = -r_time - stress[nx, ny]
                q = (r + r_success) if (nx, ny) == goal else (r + gamma * V[nx, ny])
                if q > best:
                    best, arg = q, name
            pi[x, y] = arg
    return V, pi, it + 1


def render(V, pi, goal, stress, cfg, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    N = V.shape[0]
    fig, ax = plt.subplots(figsize=(7.2, 6.6))
    # imshow expects [row=y, col=x]; transpose so x is horizontal, y vertical, origin lower
    im = ax.imshow(V.T, origin="lower", extent=[0, N, 0, N], cmap="viridis", aspect="equal")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("V*(state) — expected discounted return to the fold")

    # stress band outline (where welfare cost bites)
    ax.contour(np.arange(N) + 0.5, np.arange(N) + 0.5, stress.T,
               levels=[0.01], colors="#E04634", linewidths=1.2, linestyles="--")

    # greedy policy arrows
    for x in range(N):
        for y in range(N):
            if (x, y) == goal:
                continue
            a = pi[x, y]
            if a in ARROW:
                dx, dy = ARROW[a]
                ax.arrow(x + 0.5, y + 0.5, dx, dy, head_width=0.16,
                         head_length=0.16, fc="white", ec="white", alpha=0.7,
                         length_includes_head=True)

    # the fold
    ax.add_patch(plt.Circle((goal[0] + 0.5, goal[1] + 0.5), cfg.pen_radius,
                            fill=False, color="#FFD75A", lw=2.5))
    ax.text(goal[0] + 0.5, goal[1] + 0.5, "FOLD", color="#FFD75A",
            ha="center", va="center", fontsize=9, fontweight="bold")

    ax.set_title("L2 — Value of every pasture cell (γ=0.99)\n"
                 "value rises toward the fold; the red-dashed fence band is dented by stress",
                 fontsize=10)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    print(f"wrote {path}")


def dump_corner(V, pi, goal, cfg, path, k=5):
    """A small hand-checkable table near the fold."""
    N = V.shape[0]
    x0, y0 = max(goal[0] - k + 1, 0), max(goal[1] - k + 1, 0)
    lines = [f"# Derived gridworld value table (γ=0.99, r_success=10, r_time=0.01, "
             f"stress=r_arousal·near-wall={cfg.r_arousal})",
             f"# goal/FOLD cell = {goal}.  Showing a {k}x{k} block, rows y high->low.", ""]
    for y in range(min(y0 + k, N) - 1, y0 - 1, -1):
        row = []
        for x in range(x0, min(x0 + k, N)):
            tag = "FOLD" if (x, y) == goal else pi[x, y]
            row.append(f"({x:2d},{y:2d}) V={V[x, y]:6.2f} {tag:>4}")
        lines.append("  ".join(row))
    OUT.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    print(f"wrote {path}")


if __name__ == "__main__":
    cfg = ShepherdConfig()
    N, goal, stress = build_mdp(cfg)
    V, pi, iters = value_iteration(N, goal, stress, gamma=0.99,
                                   r_time=cfg.r_time, r_success=cfg.r_success)
    print(f"value iteration converged in {iters} sweeps on a {N}x{N} grid; "
          f"goal={goal}, V range [{V.min():.2f}, {V.max():.2f}]")
    render(V, pi, goal, stress, cfg, OUT / "l2_value_heatmap.png")
    dump_corner(V, pi, goal, cfg, OUT / "l2_value_table.txt")
