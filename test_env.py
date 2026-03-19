#!/usr/bin/env python3
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_imports():
    print("Тест импортов...")
    try:
        import gymnasium as gym
        import miniworld
        import stable_baselines3
        import torch
        import numpy as np
        import pygame
        from PIL import Image
        print("✅ Все библиотеки")
        return True
    except ImportError as e:
        print(f"❌ {e}")
        return False


def test_custom():
    print("\nТест модулей...")
    try:
        from envs.wrappers import PerturbationWrapper
        from perturbations.visual_perturbations import PerturbationManager
        from models.feature_extractor import NatureCNN
        print("✅ Модули")
        return True
    except ImportError as e:
        print(f"❌ {e}")
        return False


def test_env():
    print("\nТест среды...")
    try:
        import gymnasium as gym
        import miniworld
        
        env = gym.make('MiniWorld-Maze-v0', render_mode=None)
        obs, info = env.reset(seed=42)
        obs, reward, done, truncated, info = env.step(0)
        env.close()
        print(f"✅ Среда работает, obs: {obs.shape}")
        return True
    except Exception as e:
        print(f"❌ {e}")
        return False


def test_perturbations():
    print("\nТест помех...")
    try:
        from perturbations.visual_perturbations import PerturbationManager
        import numpy as np
        
        mgr = PerturbationManager()
        img = np.random.randint(0, 255, (60, 80, 3), dtype=np.uint8)
        
        for pert in mgr.get_available():
            mgr.set(pert, 0.5)
            result = mgr.apply(img)
            assert result.shape == img.shape
        
        print(f"✅ Помехи: {mgr.get_available()}")
        return True
    except Exception as e:
        print(f"❌ {e}")
        return False


def main():
    print("="*50)
    print("ТЕСТИРОВАНИЕ")
    print("="*50)
    
    tests = [test_imports, test_custom, test_env, test_perturbations]
    results = [t() for t in tests]
    
    print("\n" + "="*50)
    if all(results):
        print("✅ ВСЕ ТЕСТЫ ПРОЙДЕНЫ")
        print("Запускай: python train.py --envs 2 --steps 10000")
    else:
        print("❌ ЕСТЬ ОШИБКИ")
    print("="*50)


if __name__ == "__main__":
    main()