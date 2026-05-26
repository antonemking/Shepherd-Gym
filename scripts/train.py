"""PPO with an ASYMMETRIC actor-critic for Shepherd-v0.

  actor  π(a | obs)        — obs is the partial, realistically-sensable view
                             (positions/velocities/geometry + noisy ear-angle).
  critic V(privileged)     — privileged state adds the TRUE latent arousal (sim-only),
                             which cuts value-estimate variance during training.
  reward                   — computed on the true latent (integrated stress dose).

Single-file, CleanRL-style. Logs a learning curve (return / success / arousal),
saves checkpoints for checkpoint-ghosting footage, and renders the final policy.

    .venv/bin/python scripts/train.py --steps 400000
"""
import sys, os, time, json, argparse, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import torch.nn as nn

from shepherd_gym.env import ShepherdEnv, ShepherdConfig
from shepherd_gym.vec_env import VecShepherdEnv
from shepherd_gym import render as R

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
CKPT = OUT / "checkpoints"


def layer_init(m, std=np.sqrt(2)):
    if isinstance(m, nn.Linear):
        torch.nn.init.orthogonal_(m.weight, std)
        torch.nn.init.constant_(m.bias, 0.0)
    return m


class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden)), nn.Tanh(),
            layer_init(nn.Linear(hidden, hidden)), nn.Tanh(),
            layer_init(nn.Linear(hidden, act_dim), std=0.01),
        )
        self.log_std = nn.Parameter(torch.full((act_dim,), -0.5))

    def dist(self, obs):
        mean = self.net(obs)
        return torch.distributions.Normal(mean, self.log_std.exp().expand_as(mean))


class Critic(nn.Module):
    def __init__(self, state_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(state_dim, hidden)), nn.Tanh(),
            layer_init(nn.Linear(hidden, hidden)), nn.Tanh(),
            layer_init(nn.Linear(hidden, 1), std=1.0),
        )

    def forward(self, state):
        return self.net(state).squeeze(-1)


def make_cfg(r_arousal=0.5):
    return ShepherdConfig(r_arousal=r_arousal)


@torch.no_grad()
def evaluate(actor, episodes=12, seed0=9000, r_arousal=0.5):
    """Deterministic (mean-action) eval — success, steps, integrated welfare cost.
    (r_arousal doesn't change the measured arousal — eval just reports it.)"""
    cfg = make_cfg(r_arousal)
    succ, steps, ar, pen = 0, [], [], []
    for k in range(episodes):
        env = ShepherdEnv(cfg)
        obs, _ = env.reset(seed=seed0 + k)
        a_sum, t = 0.0, 0
        for _ in range(cfg.max_steps):
            mean = actor.net(torch.tensor(obs, dtype=torch.float32))
            obs, r, term, trunc, info = env.step(mean.numpy())
            a_sum += info["mean_arousal"]; t += 1
            if term or trunc:
                break
        won = bool(env.penned.all())
        succ += won
        if won: steps.append(t)
        ar.append(a_sum / max(t, 1)); pen.append(float(env.penned.mean()))
    return {
        "success": succ / episodes,
        "steps": float(np.mean(steps)) if steps else float("nan"),
        "arousal": float(np.mean(ar)),
        "penned": float(np.mean(pen)),
    }


def train(total_steps, n_envs=128, horizon=256, seed=0, r_arousal=0.5):
    torch.manual_seed(seed); np.random.seed(seed)
    tag = f"ra{r_arousal:g}"
    ckpt_dir = CKPT / tag
    OUT.mkdir(exist_ok=True); ckpt_dir.mkdir(parents=True, exist_ok=True)
    cfg = make_cfg(r_arousal)
    print(f"== run tag '{tag}'  (r_arousal={r_arousal}) ==")
    vec = VecShepherdEnv(cfg, num_envs=n_envs, seed=seed)
    obs_dim, state_dim, act_dim = vec.obs_dim, vec.state_dim, vec.action_dim
    print(f"obs_dim={obs_dim} state_dim={state_dim} act_dim={act_dim} (asymmetric, vectorised x{n_envs})")
    cur_obs = vec.reset()
    cur_state = vec.privileged_state()

    actor, critic = Actor(obs_dim, act_dim), Critic(state_dim)
    opt = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=3e-4, eps=1e-5)
    gamma, lam, clip, epochs, mb = 0.99, 0.95, 0.2, 4, 8
    ent_coef, vf_coef = 0.0, 0.5

    batch = n_envs * horizon
    n_updates = total_steps // batch
    eval_every = max(1, n_updates // 12)
    history, global_step, t0 = [], 0, time.time()

    for update in range(1, n_updates + 1):
        o_buf = np.zeros((horizon, n_envs, obs_dim), np.float32)
        s_buf = np.zeros((horizon, n_envs, state_dim), np.float32)
        a_buf = np.zeros((horizon, n_envs, act_dim), np.float32)
        lp_buf = np.zeros((horizon, n_envs), np.float32)
        r_buf = np.zeros((horizon, n_envs), np.float32)
        d_buf = np.zeros((horizon, n_envs), np.float32)
        v_buf = np.zeros((horizon, n_envs), np.float32)

        for t in range(horizon):
            o_buf[t] = cur_obs; s_buf[t] = cur_state
            with torch.no_grad():
                dist = actor.dist(torch.tensor(cur_obs))
                act = dist.sample()
                lp = dist.log_prob(act).sum(-1)
                val = critic(torch.tensor(cur_state))
            a_np = act.numpy()
            a_buf[t] = a_np; lp_buf[t] = lp.numpy(); v_buf[t] = val.numpy()
            nobs, rew, done, info = vec.step(a_np)        # one batched pass, auto-resets
            r_buf[t] = rew; d_buf[t] = done.astype(np.float32)
            cur_obs = nobs; cur_state = info["privileged_state"]
            global_step += n_envs

        # GAE
        with torch.no_grad():
            last_v = critic(torch.tensor(cur_state)).numpy()
        adv = np.zeros((horizon, n_envs), np.float32); last = np.zeros(n_envs, np.float32)
        for t in reversed(range(horizon)):
            nonterm = 1.0 - d_buf[t]
            nextv = last_v if t == horizon - 1 else v_buf[t + 1]
            delta = r_buf[t] + gamma * nextv * nonterm - v_buf[t]
            last = delta + gamma * lam * nonterm * last
            adv[t] = last
        ret = adv + v_buf

        # flatten
        b_o = torch.tensor(o_buf.reshape(-1, obs_dim))
        b_s = torch.tensor(s_buf.reshape(-1, state_dim))
        b_a = torch.tensor(a_buf.reshape(-1, act_dim))
        b_lp = torch.tensor(lp_buf.reshape(-1))
        b_adv = torch.tensor(adv.reshape(-1)); b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)
        b_ret = torch.tensor(ret.reshape(-1))
        idx = np.arange(batch)
        for _ in range(epochs):
            np.random.shuffle(idx)
            for start in range(0, batch, batch // mb):
                mbi = idx[start:start + batch // mb]
                dist = actor.dist(b_o[mbi])
                newlp = dist.log_prob(b_a[mbi]).sum(-1)
                ratio = (newlp - b_lp[mbi]).exp()
                a_mb = b_adv[mbi]
                pg = torch.max(-a_mb * ratio, -a_mb * torch.clamp(ratio, 1 - clip, 1 + clip)).mean()
                v = critic(b_s[mbi])
                vloss = 0.5 * ((v - b_ret[mbi]) ** 2).mean()
                ent = dist.entropy().sum(-1).mean()
                loss = pg + vf_coef * vloss - ent_coef * ent
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(list(actor.parameters()) + list(critic.parameters()), 0.5)
                opt.step()

        if update % eval_every == 0 or update == n_updates:
            m = evaluate(actor, r_arousal=r_arousal)
            m["step"] = global_step
            history.append(m)
            sps = int(global_step / (time.time() - t0))
            print(f"[{global_step:>8}] success={m['success']*100:5.1f}%  penned/N={m['penned']*100:4.0f}%  "
                  f"arousal={m['arousal']:.3f}  steps={m['steps'] if not np.isnan(m['steps']) else '—'}  ({sps}/s)")
            torch.save(actor.state_dict(), ckpt_dir / f"actor_{global_step}.pt")

    json.dump(history, open(OUT / f"train_history_{tag}.json", "w"), indent=2)
    _plot(history, tag)
    _render_final(actor, r_arousal, tag)
    print(f"\nDone ({tag}). {n_updates} updates, {global_step} env-steps. Curve+gif in {OUT}/")
    return actor, history


def _plot(history, tag="ra0.5"):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    xs = [h["step"] for h in history]
    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.plot(xs, [h["success"] for h in history], color="#3C6E3C", lw=2, marker="o", label="success rate")
    ax1.plot(xs, [h["penned"] for h in history], color="#78E68C", lw=1.5, ls="--", label="penned/N")
    ax1.set_xlabel("env steps"); ax1.set_ylabel("fraction"); ax1.set_ylim(0, 1.05)
    ax2 = ax1.twinx()
    ax2.plot(xs, [h["arousal"] for h in history], color="#E04634", lw=2, marker="s", label="mean arousal")
    ax2.set_ylabel("mean arousal (welfare ↓)", color="#E04634")
    ax1.set_title(f"Training ({tag}) — does it learn to herd, and gently?")
    ax1.legend(loc="center right"); fig.tight_layout()
    fig.savefig(OUT / f"learning_curve_{tag}.png", dpi=120)


@torch.no_grad()
def _render_final(actor, r_arousal=0.5, tag="ra0.5"):
    cfg = make_cfg(r_arousal); env = ShepherdEnv(cfg)
    rnd = R.Renderer(cfg.world)

    class Pol:
        def __call__(self, env):
            mean = actor.net(torch.tensor(env._obs(), dtype=torch.float32))
            return mean.numpy()
    frames, info = R.record_episode(env, Pol(), renderer=rnd, seed=7, label=f"ppo ({tag})")
    R.save_gif(frames, str(OUT / f"trained_{tag}.gif")); R.save_mp4(frames, str(OUT / f"trained_{tag}.mp4"))
    print(f"  trained policy: penned {info['penned']}/{info['n_sheep']}, arousal {info['mean_arousal']:.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2000000)
    ap.add_argument("--envs", type=int, default=128)
    ap.add_argument("--horizon", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--r_arousal", type=float, default=0.5, help="welfare penalty weight (0 = pure penning)")
    a = ap.parse_args()
    train(a.steps, n_envs=a.envs, horizon=a.horizon, seed=a.seed, r_arousal=a.r_arousal)
