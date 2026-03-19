#!/usr/bin/env python3
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse

import gymnasium as gym
import miniworld
import numpy as np
import pygame
from pygame.locals import *

from envs.wrappers import PerturbationWrapper


class ManualController:
    def __init__(self, num_rows=4, num_cols=4, domain_rand=False, perturbation='none', severity=0.5):
        pygame.init()
        
        self.screen = pygame.display.set_mode((400, 500))
        pygame.display.set_caption("MiniWorld Control - W/A/S/D")
        self.font = pygame.font.SysFont(None, 28)
        self.small_font = pygame.font.SysFont(None, 20)
        
        print(f"Создание среды {num_rows}x{num_cols}...")
        
        self.env = gym.make(
            'MiniWorld-Maze-v0',
            num_rows=num_rows,
            num_cols=num_cols,
            room_size=4,
            render_mode='human',
            domain_rand=domain_rand,
        )
        
        self.use_perturbation = perturbation != 'none'
        if self.use_perturbation:
            self.env = PerturbationWrapper(self.env, perturb_prob=1.0)
            self.env.perturb.set(perturbation, severity)
            print(f"Помеха: {perturbation} ({severity})")
        
        self.obs, self.info = self.env.reset(seed=42)
        self.total_reward = 0.0
        self.steps = 0
        self.episode_num = 1
        
        print("Готово! Управление: W/A/S/D, R=reset, Q=quit")
        
    def get_top_down(self):
        try:
            return self.env.render(mode='top_down', width=400, height=300)
        except:
            return None
    
    def draw_panel(self):
        self.screen.fill((30, 30, 30))
        
        y = 10
        texts = [
            f"Episode: {self.episode_num}",
            f"Step: {self.steps}",
            f"Reward: {self.total_reward:.2f}",
        ]
        
        if self.use_perturbation:
            texts.append(f"Pert: {self.env.perturb.current}")
            texts.append(f"Severity: {self.env.perturb.severity:.2f}")
        
        for text in texts:
            surface = self.small_font.render(text, True, (255, 255, 255))
            self.screen.blit(surface, (10, y))
            y += 25
        
        y += 20
        controls = ["Controls:", "W - forward", "S - backward", "A - turn left", "D - turn right", "R - reset", "Q - quit"]
        for ctrl in controls:
            color = (255, 255, 0) if ctrl.endswith(':') else (200, 200, 200)
            surface = self.small_font.render(ctrl, True, color)
            self.screen.blit(surface, (10, y))
            y += 22
        
        # Top-down view
        top_down = self.get_top_down()
        if top_down is not None:
            top_down = np.transpose(top_down, (1, 0, 2))
            surface = pygame.surfarray.make_surface(top_down)
            surface = pygame.transform.scale(surface, (380, 280))
            self.screen.blit(surface, (10, 210))
        
        pygame.display.flip()
    
    def run(self):
        running = True
        clock = pygame.time.Clock()
        
        self.env.render()
        
        while running:
            action = None
            
            for event in pygame.event.get():
                if event.type == QUIT:
                    running = False
                elif event.type == KEYDOWN:
                    if event.key == K_q or event.key == K_ESCAPE:
                        running = False
                    elif event.key == K_r:
                        self.reset()
                    elif event.key == K_w:
                        action = self.env.unwrapped.actions.move_forward
                    elif event.key == K_s:
                        action = self.env.unwrapped.actions.move_back
                    elif event.key == K_a:
                        action = self.env.unwrapped.actions.turn_left
                    elif event.key == K_d:
                        action = self.env.unwrapped.actions.turn_right
                    elif event.key == K_p and self.use_perturbation:
                        self.cycle_perturbation()
                    elif event.key == K_EQUALS or event.key == K_PLUS:
                        self.change_severity(0.1)
                    elif event.key == K_MINUS:
                        self.change_severity(-0.1)
            
            if action is not None:
                self.step(action)
            
            self.env.render()
            self.draw_panel()
            clock.tick(30)
        
        self.close()
    
    def step(self, action):
        self.obs, reward, terminated, truncated, self.info = self.env.step(action)
        self.total_reward += reward
        self.steps += 1
        print(f"Step {self.steps}: reward={reward:+.2f}, total={self.total_reward:.2f}")
        
        if terminated or truncated:
            success = self.total_reward > 0
            print(f"\n{'✅ SUCCESS' if success else '❌ FAILED'}!")
            self.reset()
    
    def reset(self):
        self.obs, self.info = self.env.reset()
        self.total_reward = 0.0
        self.steps = 0
        self.episode_num += 1
        print(f"\n🔄 Episode {self.episode_num}")
    
    def cycle_perturbation(self):
        if self.use_perturbation:
            available = self.env.perturb.get_available()
            curr = self.env.perturb.current
            idx = available.index(curr)
            next_p = available[(idx + 1) % len(available)]
            self.env.perturb.set(next_p)
            print(f"Помеха: {next_p}")
    
    def change_severity(self, delta):
        if self.use_perturbation:
            new_sev = np.clip(self.env.perturb.severity + delta, 0.0, 1.0)
            self.env.perturb.set(self.env.perturb.current, new_sev)
            print(f"Severity: {new_sev:.2f}")
    
    def close(self):
        self.env.close()
        pygame.quit()
        print("👋 Завершено")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rows', type=int, default=4)
    parser.add_argument('--cols', type=int, default=4)
    parser.add_argument('--domain-rand', action='store_true')
    parser.add_argument('--perturbation', type=str, default='none')
    parser.add_argument('--severity', type=float, default=0.5)
    args = parser.parse_args()
    
    ctrl = ManualController(args.rows, args.cols, args.domain_rand, args.perturbation, args.severity)
    try:
        ctrl.run()
    except KeyboardInterrupt:
        print("\nПрервано")
    finally:
        ctrl.close()


if __name__ == '__main__':
    main()