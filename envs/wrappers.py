import gymnasium as gym
import numpy as np

from perturbations.visual_perturbations import PerturbationManager


class PerturbationWrapper(gym.Wrapper):
    """Добавляет визуальные помехи."""
    
    def __init__(self, env: gym.Env, perturb_prob: float = 0.7):
        super().__init__(env)
        self.perturb = PerturbationManager()
        self.perturb_prob = perturb_prob
        self.episode_reward = 0.0
        self.episode_steps = 0
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        
        self.episode_reward = 0.0
        self.episode_steps = 0
        
        if np.random.random() < self.perturb_prob:
            self.perturb.randomize()
        else:
            self.perturb.set('none')
            
        obs = self.perturb.apply(obs)
        info['perturbation_type'] = self.perturb.current
        info['perturbation_severity'] = self.perturb.severity
        
        return obs, info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        self.episode_reward += reward
        self.episode_steps += 1
        
        obs = self.perturb.apply(obs)
        info['perturbation_type'] = self.perturb.current
        info['perturbation_severity'] = self.perturb.severity
        
        # SB3 формат для episode info
        if terminated or truncated:
            info['episode'] = {
                'r': self.episode_reward,
                'l': self.episode_steps,
            }
            
        return obs, reward, terminated, truncated, info


# Оставляем для совместимости, но он больше не нужен
EpisodeInfoWrapper = PerturbationWrapper