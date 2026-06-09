"""
CLI entry point for RL training.

Usage:
    python scripts/train.py --config configs/train_ppo_noisy.toml
    python scripts/train.py --config configs/train_ppo_noisy.toml --resume results/run_1/model_final.zip
"""

from __future__ import annotations

import argparse
import sys
import tomllib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.trainer import RLTrainer
from gym_envs.sim_config import SimConfig

_ROOT = Path(__file__).resolve().parent.parent


def load_config(path: Path) -> dict:
    with path.open("rb") as fh:
        return tomllib.load(fh)


def build_sim_config(cfg: dict) -> SimConfig:
    section = cfg.get("sim_config", {})
    return SimConfig(
        pend_noise_sigma=float(section.get("pend_noise_sigma", 0.0)),
        motor_noise_sigma=float(section.get("motor_noise_sigma", 0.0)),
        sensor_latency_steps=int(section.get("sensor_latency_steps", 0)),
        dt_jitter_sigma=float(section.get("dt_jitter_sigma", 0.0)),
    )


def build_trainer(cfg: dict, sim_config: SimConfig) -> tuple[RLTrainer, dict]:
    t = cfg.get("trainer", {})
    trainer = RLTrainer(
        agent_type=str(t.get("agent_type", "PPO")),
        task=str(t.get("task", "equilibrium")),
        max_steps=int(t["max_steps"]) if "max_steps" in t else None,
        seed=int(t["seed"]) if "seed" in t else None,
        render_mode=None,
        sim_config=sim_config,
        starting_offset=float(t.get("starting_offset", 0.4)),
    )
    train_kwargs = {
        "total_timesteps": int(t.get("total_timesteps", 100_000)),
        "eval_freq":       int(t.get("eval_freq", 10_000)),
        "n_eval_episodes": int(t.get("n_eval_episodes", 5)),
    }
    return trainer, train_kwargs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an RL agent on the Furuta pendulum sim")
    parser.add_argument("--config", required=True, help="Path to training TOML config")
    parser.add_argument("--resume", default=None, help="Path to a saved model to resume from")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (_ROOT / config_path).resolve()

    cfg        = load_config(config_path)
    sim_config = build_sim_config(cfg)
    trainer, train_kwargs = build_trainer(cfg, sim_config)

    print(f"Config:     {config_path.name}")
    print(f"Algorithm:  {trainer.agent_type}")
    print(f"Task:       {trainer.task}")
    print(f"Timesteps:  {train_kwargs['total_timesteps']:,}")
    print(f"SimConfig:  noise_pend={sim_config.pend_noise_sigma}  "
          f"latency={sim_config.sensor_latency_steps}steps  "
          f"jitter={sim_config.dt_jitter_sigma}µs")

    trainer.train(**train_kwargs, resume_from=args.resume)


if __name__ == "__main__":
    main()
