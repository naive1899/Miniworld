#!/usr/bin/env python3
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import gymnasium as gym
import miniworld
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

from envs.wrappers import PerturbationWrapper
from models.feature_extractor import NatureCNN
from utils.callbacks import PerturbationSchedulerCallback, TensorboardCallback


def make_env(rank, mode='baseline', seed=0):
    """Создание окружения под конкретный режим обучения."""
    def _init():
        # Базовые параметры
        domain_rand = (mode != 'baseline')  # Только для baseline серые стены
        
        env = gym.make(
            'MiniWorld-Maze-v0',
            num_rows=3,
            num_cols=3,
            room_size=4,
            domain_rand=domain_rand,
            render_mode=None,
        )
        
        # Настройка помех в зависимости от режима
        if mode == 'baseline':
            # Без помех
            env = PerturbationWrapper(env, perturb_prob=0.0)
        elif mode == 'static':
            # Сразу высокий уровень помех (naive DR)
            env = PerturbationWrapper(env, perturb_prob=0.8)
        elif mode == 'progressive':
            # Начинаем с 0, будем наращивать через callback
            env = PerturbationWrapper(env, perturb_prob=0.0)
        
        env.reset(seed=rank + seed * 100)
        return env
    return _init


def train(mode='baseline', seed=0, num_envs=4, total_timesteps=200_000, save_dir="./models/"):
    """
    Обучение одной конфигурации.
    
    Args:
        mode: 'baseline', 'static', или 'progressive'
        seed: random seed (0, 1, 2 для повторяемости)
    """
    # Создаем папку с учетом режима и сида
    save_path = os.path.join(save_dir, f"{mode}_seed{seed}")
    os.makedirs(save_path, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"🚀 Режим: {mode.upper()} | Seed: {seed}")
    print(f"{'='*60}")
    
    # Создаем окружения
    env = DummyVecEnv([make_env(i, mode, seed) for i in range(num_envs)])
    
    # Callbacks
    callbacks = [
        CheckpointCallback(
            save_freq=50_000, 
            save_path=save_path,
            name_prefix="ppo"
        ),
        TensorboardCallback(),
    ]
    
    # Для progressive добавляем scheduler
    if mode == 'progressive':
        callbacks.append(
            PerturbationSchedulerCallback(
                start_step=10_000,  # Быстрее для тестов, для диплома можно 50k
                end_step=150_000,
                final_prob=0.8
            )
        )
    
    # Модель
    policy_kwargs = dict(
        features_extractor_class=NatureCNN,
        features_extractor_kwargs=dict(features_dim=512),
        net_arch=dict(pi=[256, 128], vf=[256, 128]),
    )
    
    model = PPO(
        "CnnPolicy",
        env,
        learning_rate=2e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        ent_coef=0.05,
        vf_coef=0.5,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=f"./tensorboard/{mode}_seed{seed}/",
        seed=seed,
    )
    
    # Обучение
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=False,
        )
    except KeyboardInterrupt:
        print("\n⏹️  Прервано")
    
    # Сохранение
    final_path = os.path.join(save_path, "final")
    model.save(final_path)
    print(f"✅ Сохранено: {final_path}")
    
    env.close()
    return final_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='baseline', 
                       choices=['baseline', 'static', 'progressive'],
                       help='Режим обучения')
    parser.add_argument('--seed', type=int, default=0,
                       help='Random seed (0, 1, 2)')
    parser.add_argument('--envs', type=int, default=4)
    parser.add_argument('--steps', type=int, default=200_000)
    parser.add_argument('--save-dir', type=str, default='./models/')
    
    args = parser.parse_args()
    
    train(args.mode, args.seed, args.envs, args.steps, args.save_dir)