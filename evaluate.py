#!/usr/bin/env python3
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
import argparse
import gymnasium as gym
import miniworld
import numpy as np
from PIL import Image
from tqdm import tqdm
from stable_baselines3 import PPO
import matplotlib.pyplot as plt

from perturbations.visual_perturbations import PerturbationManager


def evaluate(model_path, perturb_type, severity, num_rows=3, num_cols=3,
             num_episodes=50, seed=42, save_gif=False):
    """Оценка на одной помехе (без wrapper, помехи применяем вручную)."""
    
    # Создаем чистое окружение (без PerturbationWrapper!)
    env = gym.make('MiniWorld-Maze-v0', num_rows=num_rows, num_cols=num_cols, 
                   render_mode='rgb_array' if save_gif else None)
    
    # Создаем менеджер помех отдельно
    perturb = PerturbationManager()
    perturb.set(perturb_type, severity)
    
    # Загружаем модель (она ожидает обертку, но мы применим помехи вручную)
    model = PPO.load(model_path, env=env)
    
    rewards = []
    lengths = []
    successes = []
    
    for ep in tqdm(range(num_episodes), desc=f"{perturb_type}_sev{severity}"):
        obs, info = env.reset(seed=seed + ep)
        
        # Применяем помеху к начальному наблюдению
        if perturb_type != 'none':
            obs = perturb.apply(obs)
        
        frames = [] if (save_gif and ep == 0) else None
        if save_gif and ep == 0:
            frames.append(Image.fromarray(obs))
        
        done = False
        total_reward = 0.0
        steps = 0
        
        while not done and steps < 1000:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1
            
            # Применяем помеху к каждому кадру!
            if perturb_type != 'none':
                obs = perturb.apply(obs)
            
            if frames is not None:
                frames.append(Image.fromarray(obs))
        
        rewards.append(total_reward)
        lengths.append(steps)
        successes.append(1 if total_reward > 0 else 0)
        
        if save_gif and ep == 0 and len(frames) > 1:
            gif_path = f"traj_{perturb_type}_{severity}.gif"
            frames[0].save(gif_path, save_all=True, append_images=frames[1:], 
                          duration=100, loop=0)
            print(f"\n💾 Saved: {gif_path}")
    
    env.close()
    
    return {
        'mean_reward': float(np.mean(rewards)),
        'std_reward': float(np.std(rewards)),
        'success_rate': float(np.mean(successes)),
        'mean_length': float(np.mean(lengths)),
    }


def full_evaluation(model_path, num_rows=3, num_cols=3, num_episodes=50, 
                   output="robustness_results.json"):
    """Полная оценка на всех помехах."""
    print(f"🔍 Оценка: {model_path}")
    print(f"   Размер: {num_rows}×{num_cols}, Эпизодов: {num_episodes}")
    
    perturbations = ['none', 'gaussian', 'blur', 'color', 'occlusion', 'fog']
    severities = [0.3, 0.5, 0.7]
    
    results = {'model': model_path, 'perturbations': {}, 'summary': {}}
    
    for pert in perturbations:
        results['perturbations'][pert] = {}
        for sev in severities:
            print(f"\n  Тест: {pert} (severity={sev})")
            save_gif = (pert == 'none' and sev == 0.5)
            metrics = evaluate(model_path, pert, sev, num_rows, num_cols, 
                             num_episodes, save_gif=save_gif)
            results['perturbations'][pert][f'sev_{sev}'] = metrics
            print(f"     Success: {metrics['success_rate']:.2f}")
    
    # Summary: берем severity 0.5 для сравнения
    clean = results['perturbations']['none']['sev_0.5']['success_rate']
    pert_avg = np.mean([
        results['perturbations'][p]['sev_0.5']['success_rate'] 
        for p in perturbations if p != 'none'
    ])
    
    results['summary'] = {
        'clean_success_rate': clean,
        'avg_perturbed_success': pert_avg,
        'robustness_gap': clean - pert_avg,
        'relative_robustness': pert_avg / clean if clean > 0 else 0,
    }
    
    with open(output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*60)
    print("📊 ИТОГОВЫЕ РЕЗУЛЬТАТЫ:")
    print(f"  Чистая среда:     {clean:.3f} ({clean*100:.1f}%)")
    print(f"  Среднее с помехами: {pert_avg:.3f} ({pert_avg*100:.1f}%)")
    print(f"  Робастность (gap):  {clean - pert_avg:.3f}")
    print(f"  Сохранено: {output}")
    print("="*60)
    
    # График
    fig, ax = plt.subplots(figsize=(10, 6))
    success_rates = [results['perturbations'][p]['sev_0.5']['success_rate'] 
                     for p in perturbations]
    colors = ['green' if p == 'none' else 'red' for p in perturbations]
    
    bars = ax.bar(perturbations, success_rates, color=colors, alpha=0.7)
    ax.set_ylabel('Success Rate', fontsize=12)
    ax.set_xlabel('Тип помехи', fontsize=12)
    ax.set_title(f'Robustness: {os.path.basename(model_path)}\n'
                 f'Clean: {clean:.2f} | Avg Perturbed: {pert_avg:.2f}', 
                 fontsize=14)
    ax.axhline(y=clean, color='green', linestyle='--', alpha=0.5, label='Clean')
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    chart_path = output.replace('.json', '.png')
    plt.savefig(chart_path, dpi=150)
    print(f"📊 График: {chart_path}")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', type=str, help='Путь к модели')
    parser.add_argument('--rows', type=int, default=3, help='Строки лабиринта')
    parser.add_argument('--cols', type=int, default=3, help='Колонки лабиринта')
    parser.add_argument('--episodes', type=int, default=50, help='Число эпизодов')
    parser.add_argument('--output', type=str, default='robustness_results.json')
    args = parser.parse_args()
    
    full_evaluation(args.model_path, args.rows, args.cols, args.episodes, args.output)