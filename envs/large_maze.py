"""
Расширенный лабиринт MiniWorld с поддержкой вида сверху.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from miniworld.envs.maze import Maze
from miniworld.params import DEFAULT_PARAMS


class LargeMaze(Maze):
    """
    Расширенный лабиринт с возможностью настройки размера и сложности.
    
    Parameters
    ----------
    num_rows : int
        Количество строк комнат (default: 4)
    num_cols : int
        Количество колонок комнат (default: 4)
    room_size : int
        Размер комнаты (default: 4)
    render_mode : str | None
        Режим рендера: 'human', 'rgb_array', или None
    domain_rand : bool
        Включить рандомизацию домена
    """
    
    def __init__(
        self,
        num_rows: int = 4,
        num_cols: int = 4,
        room_size: int = 4,
        render_mode: str | None = None,
        domain_rand: bool = False,
        **kwargs
    ):
        # Увеличиваем размер мира для большего лабиринта
        params = DEFAULT_PARAMS.copy()
        params.set('world_size', max(num_rows, num_cols) * room_size * 1.5)
        
        super().__init__(
            num_rows=num_rows,
            num_cols=num_cols,
            room_size=room_size,
            domain_rand=domain_rand,
            params=params,
            **kwargs
        )
        
        self.render_mode = render_mode
        self.top_view_size = 256
        
    def reset(self, seed: int | None = None, options: dict | None = None):
        """Reset environment and return observation with top-down view."""
        obs, info = super().reset(seed=seed, options=options)
        
        # Добавляем top-down view если нужно
        if self.render_mode == "rgb_array":
            info['top_down_view'] = self._render_top_down()
            
        return obs, info
    
    def step(self, action):
        """Step environment and update top-down view."""
        obs, reward, terminated, truncated, info = super().step(action)
        
        if self.render_mode == "rgb_array":
            info['top_down_view'] = self._render_top_down()
            
        return obs, reward, terminated, truncated, info
    
    def _render_top_down(self):
        """Рендер вида сверху."""
        return self.render(
            mode='top_down', 
            width=self.top_view_size, 
            height=self.top_view_size
        )


# Регистрация среды в Gymnasium
gym.register(
    id='MiniWorld-LargeMaze-v0',
    entry_point='envs.large_maze:LargeMaze',
    max_episode_steps=1000,
)