import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn as nn
import torch.optim as optim

from network_training_utensils.module.actuator_net import ActuatorNet
from network_training_utensils.data_loader.data_loader_prototype import MotorDataLoader

class PPO:
    def __init__(self,
                 net : ActuatorNet,
                 storage : MotorDataLoader,
                 num_learning_epochs=1,
                 num_mini_batches=1,
                 clip_param=0.2,
                 gamma=0.998,
                 lam=0.95,
                 value_loss_coef=1.0,
                 learning_rate=1e-3,
                 max_grad_norm=1.0,
                 use_clipped_value_loss=True,
                 shuffle=True,
                 schedule='fixed',
                 desired_kl=0.01,
                 json_file=None,
                 device='cpu'):
        self.net = net
        self._storage = storage
        self.storage = None # 进行单独初始化
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.clip_param = clip_param
        self.gamma = gamma
        self.lam = lam
        self.value_loss_coef = value_loss_coef
        self.learning_rate = learning_rate
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.shuffle = shuffle
        self.schedule = schedule
        self.desired_kl = desired_kl
        self.json_file = json_file
        self.device = device

        self._init_storage()

    def _init_storage(self):
        self.storage = self._storage(json_file=self.json_file,
                                     batch_size=self.num_mini_batches,
                                     shuffle=self.shuffle)
        
    def test_mode(self):
        self.net.eval()

    def train_mode(self):
        self.net.train()

    


if __name__ == "__main__":
    print('Successfully import packages!')