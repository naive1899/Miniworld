import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class NatureCNN(BaseFeaturesExtractor):
    """
    Стандартная Nature CNN для MiniWorld.
    Надёжная архитектура как в DQN.
    """
    
    def __init__(self, observation_space, features_dim: int = 512):
        super().__init__(observation_space, features_dim)
        
        n_input_channels = observation_space.shape[0]
        
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        # Вычисляем размер выхода
        with torch.no_grad():
            sample = torch.zeros(1, *observation_space.shape)
            n_flatten = self.cnn(sample).shape[1]
        
        print(f"NatureCNN: input {observation_space.shape} -> flatten {n_flatten} -> features {features_dim}")
        
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )
        
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x = observations.float() / 255.0
        return self.linear(self.cnn(x))