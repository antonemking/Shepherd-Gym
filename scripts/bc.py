"""Behavior-clone the flank (Strömbom) shepherd into the policy network.

Generates (actor_obs, flank_action) pairs by rolling out the heuristic, then
regresses the actor's mean output to the expert action. This warm-starts PPO past
the exploration wall — the agent starts knowing *how* to herd, and fine-tuning then
makes it gentler. Saves weights to out/bc_actor.pt.

    .venv/bin/python scripts/bc.py
"""
import sys, pathlib
import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from shepherd_gym.env import ShepherdEnv, ShepherdConfig
from shepherd_gym.baselines import FlankShepherd
from train import Actor, evaluate

OUT = ROOT / "out"


def collect(n_episodes=150, seed0=0):
    cfg = ShepherdConfig(); pol = FlankShepherd()
    O, A = [], []
    for k in range(n_episodes):
        env = ShepherdEnv(cfg); env.reset(seed=seed0 + k)
        for _ in range(cfg.max_steps):
            a = np.asarray(pol(env), dtype=np.float32)   # expert action for the CURRENT state
            O.append(env._obs().copy())                  # actor's partial view of the same state
            A.append(a)
            _, _, term, trunc, _ = env.step(a)
            if term or trunc:
                break
    return np.asarray(O, np.float32), np.asarray(A, np.float32)


def bc_train(O, A, obs_dim, act_dim, epochs=10, bs=512, lr=1e-3):
    actor = Actor(obs_dim, act_dim)
    opt = torch.optim.Adam(actor.net.parameters(), lr=lr)
    O, A = torch.tensor(O), torch.tensor(A)
    n = len(O); idx = np.arange(n)
    for ep in range(epochs):
        np.random.shuffle(idx); tot = 0.0
        for s in range(0, n, bs):
            mb = idx[s:s + bs]
            loss = ((actor.net(O[mb]) - A[mb]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(mb)
        print(f"  BC epoch {ep + 1:>2}: mse={tot / n:.4f}")
    return actor


def main():
    probe = ShepherdEnv(ShepherdConfig())
    print("Collecting flank demonstrations...")
    O, A = collect()
    print(f"  {len(O)} (obs, action) pairs")
    actor = bc_train(O, A, probe.obs_dim, probe.action_dim)
    OUT.mkdir(exist_ok=True)
    torch.save(actor.state_dict(), OUT / "bc_actor.pt")
    m = evaluate(actor, episodes=24)
    print(f"\nBC actor (mean-action) eval: success={m['success']*100:.0f}%  "
          f"penned/N={m['penned']*100:.0f}%  arousal={m['arousal']:.3f}  "
          f"steps={m['steps'] if not np.isnan(m['steps']) else '—'}")
    print(f"saved → {OUT}/bc_actor.pt")
    print("(if penned/N is high, the warm start works — fine-tune with "
          "`train.py --init out/bc_actor.pt --r_arousal <w>`)")


if __name__ == "__main__":
    main()
