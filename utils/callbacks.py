import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class PerturbationSchedulerCallback(BaseCallback):
    """Прогрессивное введение помех."""
    
    def __init__(self, start_step=50_000, end_step=200_000, final_prob=0.8, verbose=0):
        super().__init__(verbose)
        self.start_step = start_step
        self.end_step = end_step
        self.final_prob = final_prob
        
    def _on_step(self) -> bool:
        current_step = self.num_timesteps
        
        if current_step < self.start_step:
            prob = 0.0
        elif current_step > self.end_step:
            prob = self.final_prob
        else:
            progress = (current_step - self.start_step) / (self.end_step - self.start_step)
            prob = progress * self.final_prob
        
        try:
            venv = self.training_env
            if hasattr(venv, 'venv'):
                venv = venv.venv
            
            if hasattr(venv, 'set_attr'):
                venv.set_attr('perturb_prob', prob)
        except Exception as e:
            if self.verbose > 0:
                print(f"Could not set perturb_prob: {e}")
        
        self.logger.record("train/perturbation_prob", prob)
        return True


class TensorboardCallback(BaseCallback):
    """Логирование метрик."""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        
    def _on_step(self) -> bool:
        for info in self.locals.get('infos', []):
            if 'episode' in info:
                # SB3 использует 'r' и 'l'
                self.episode_rewards.append(info['episode']['r'])
                self.episode_lengths.append(info['episode']['l'])
                
                if len(self.episode_rewards) >= 100:
                    self.logger.record("rollout/ep_rew_mean", np.mean(self.episode_rewards))
                    self.logger.record("rollout/ep_len_mean", np.mean(self.episode_lengths))
                    self.episode_rewards = []
                    self.episode_lengths = []
            
            if 'perturbation_type' in info:
                self.logger.record("perturbation/current", info['perturbation_type'])
                self.logger.record("perturbation/severity", info['perturbation_severity'])
        
        return True