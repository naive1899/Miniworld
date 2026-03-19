#!/usr/bin/env python3
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gymnasium as gym
import miniworld
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from envs.wrappers import PerturbationWrapper, EpisodeInfoWrapper
from models.feature_extractor import NatureCNN
from utils.callbacks import PerturbationSchedulerCallback, TensorboardCallback


def make_env(rank, domain_rand=True, perturb_prob=0.0):
    def _init():
        env = gym.make(
            'MiniWorld-Maze-v0',
            num_rows=4,
            num_cols=4,
            room_size=4,
            domain_rand=domain_rand,
            render_mode=None,
        )
        # Только один wrapper - он уже включает EpisodeInfo
        env = PerturbationWrapper(env, perturb_prob=perturb_prob)
        env.reset(seed=rank)
        return env
    return _init


def train(num_envs=4, total_timesteps=500_000, save_dir="./models/"):
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"🚀 Создание {num_envs} окружений...")
    
    # Используем DummyVecEnv вместо SubprocVecEnv для надёжности
    env = DummyVecEnv([
        make_env(i, perturb_prob=0.0)
        for i in range(num_envs)
    ])
    
    # Policy с NatureCNN
    policy_kwargs = dict(
        features_extractor_class=NatureCNN,
        features_extractor_kwargs=dict(features_dim=512),
        net_arch=[dict(pi=[256, 128], vf=[256, 128])],
    )
    
    print("🧠 Инициализация PPO...")
    model = PPO(
        "CnnPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log="./tensorboard/",
    )
    
    callbacks = [
        CheckpointCallback(save_freq=50_000, save_path=save_dir, name_prefix="ppo_maze"),
        PerturbationSchedulerCallback(start_step=50_000, end_step=200_000, final_prob=0.8),
        TensorboardCallback(),
    ]
    
    print("▶️  Начало обучения...")
    print(f"   Timesteps: {total_timesteps:,}")
    print(f"   Environments: {num_envs}")
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=False,
        )
    except KeyboardInterrupt:
        print("\n⏹️  Прервано")
    
    final_path = os.path.join(save_dir, "ppo_maze_final")
    model.save(final_path)
    print(f"\n✅ Сохранено: {final_path}")
    
    env.close()
    return model


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--envs', type=int, default=4)
    parser.add_argument('--steps', type=int, default=500_000)
    parser.add_argument('--save-dir', type=str, default='./models/')
    args = parser.parse_args()
    
    train(args.envs, args.steps, args.save_dir)