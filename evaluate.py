#!/usr/bin/env python3
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
import argparse

import gymnasium as gym
import miniworld  # <-- ДОБАВИТЬ ЭТО
import numpy as np
from PIL import Image
from tqdm import tqdm
from stable_baselines3 import PPO
import matplotlib.pyplot as plt

from envs.wrappers import PerturbationWrapper


def evaluate(model_path, perturb_type, severity, num_episodes=50, seed=42, save_gif=False):
    """Оценка на одной помехе."""
    env = gym.make('MiniWorld-Maze-v0', num_rows=4, num_cols=4, render_mode='rgb_array' if save_gif else None)
    env = PerturbationWrapper(env, perturb_prob=1.0)
    env.perturb.set(perturb_type, severity)
    
    model = PPO.load(model_path, env=env)
    
    rewards = []
    lengths = []
    successes = []
    
    for ep in tqdm(range(num_episodes), desc=f"{perturb_type}"):
        obs, info = env.reset(seed=seed + ep)
        
        frames = [] if (save_gif and ep == 0) else None
        
        done = False
        total_reward = 0.0
        steps = 0
        
        while not done and steps < 1000:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1
            
            if frames is not None:
                frame = env.render()
                if frame is not None:
                    frames.append(Image.fromarray(frame))
        
        rewards.append(total_reward)
        lengths.append(steps)
        successes.append(1 if total_reward > 0 else 0)
        
        if save_gif and ep == 0 and frames and len(frames) > 1:
            gif_path = f"traj_{perturb_type}_{severity}.gif"
            frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=100, loop=0)
            print(f"\n💾 Saved: {gif_path}")
    
    env.close()
    
    return {
        'mean_reward': float(np.mean(rewards)),
        'std_reward': float(np.std(rewards)),
        'success_rate': float(np.mean(successes)),
        'mean_length': float(np.mean(lengths)),
    }


def full_evaluation(model_path, num_episodes=50, output="robustness_results.json"):
    """Полная оценка."""
    print(f"🔍 Оценка: {model_path}")
    
    perturbations = ['none', 'gaussian', 'blur', 'color', 'occlusion', 'fog']
    severities = [0.3, 0.5, 0.7]
    
    results = {'model': model_path, 'perturbations': {}, 'summary': {}}
    
    for pert in perturbations:
        results['perturbations'][pert] = {}
        for sev in severities:
            save_gif = (pert == 'none' and sev == 0.5)
            metrics = evaluate(model_path, pert, sev, num_episodes, save_gif=save_gif)
            results['perturbations'][pert][f'sev_{sev}'] = metrics
    
    # Summary
    clean = results['perturbations']['none']['sev_0.5']['success_rate']
    pert_avg = np.mean([results['perturbations'][p]['sev_0.5']['success_rate'] for p in perturbations if p != 'none'])
    
    results['summary'] = {
        'clean_success_rate': clean,
        'avg_perturbed_success': pert_avg,
        'robustness_gap': clean - pert_avg,
        'relative_robustness': pert_avg / clean if clean > 0 else 0,
    }
    
    with open(output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*50)
    print("📊 РЕЗУЛЬТАТЫ:")
    print(f"  Чистая: {clean:.3f}")
    print(f"  С помехами: {pert_avg:.3f}")
    print(f"  Разрыв: {clean - pert_avg:.3f}")
    print(f"  Сохранено: {output}")
    print("="*50)
    
    

    # График успеха по помехам
    fig, ax = plt.subplots()
    perturbations = ['none', 'gaussian', 'blur', 'color', 'occlusion', 'fog']
    success_rates = [results['perturbations'][p]['sev_0.5']['success_rate'] for p in perturbations]
    ax.bar(perturbations, success_rates)
    ax.set_ylabel('Success Rate')
    ax.set_title('Robustness Evaluation')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('robustness_chart.png')
    print("📊 График сохранён: robustness_chart.png")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', type=str)
    parser.add_argument('--episodes', type=int, default=50)
    parser.add_argument('--output', type=str, default='robustness_results.json')
    args = parser.parse_args()
    
    full_evaluation(args.model_path, args.episodes, args.output)