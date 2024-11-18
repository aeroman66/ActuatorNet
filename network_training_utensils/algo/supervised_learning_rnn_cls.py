import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time

from network_training_utensils.module.actuator_net_rnn import ActuatorNetRNN
from network_training_utensils.storage.data_loader_rnn import MiniBatchGenerator, RNNDataLoader

class SupervisedLearning:
    def __init__(self,
                 net : ActuatorNetRNN,
                 storage : MiniBatchGenerator,
                 num_classes=1000,
                 batch_size=32,
                 num_learning_epochs=10,
                 num_testing_epochs=10,
                 clip_param=0.2,
                 learning_rate=1e-3,
                 weight_decay=0.001,
                 max_grad_norm=1.0,
                 shuffle=False,
                 file_path=None,
                 device='cpu'):
        self.net = net
        self.storage = storage # 进行单独初始化
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.num_learning_epochs = num_learning_epochs
        self.num_testing_epochs = num_testing_epochs
        self.clip_param = clip_param
        self.max_grad_norm = max_grad_norm
        self.shuffle = shuffle
        self.file_path = file_path
        self.device = device

        # 网络更新相关
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        # self.criterion = nn.HuberLoss()  
        self.criterion_train = nn.CrossEntropyLoss()
        self.criterion_test = nn.MSELoss()
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        # attributes
        self.dof_pos_batch = None
        self.dof_vel_batch = None
        self.dof_tor_batch = None
        self.tar_dof_pos_batch = None

        self.train_batch_gen = None
        self.test_batch_gen = None
        # next(self.batch_gen)
        
    def test_mode(self):
        self.net.eval()

    def train_mode(self):
        self.net.train()

    def act(self, obs):
        return self.net.act_inference(obs)
    
    def update(self):
        # 1. 获取 batch
        mini_batch = next(self.train_batch_gen)
        # print('required one batch')
        mini_batch = torch.Tensor(np.array(mini_batch)).to(self.device)
        self.dof_pos_batch, self.dof_vel_batch, self.dof_tor_batch, self.tar_dof_pos_batch = mini_batch.split(1, dim=0)
        self.dof_pos_batch, self.dof_vel_batch, self.dof_tor_batch, self.tar_dof_pos_batch = (
            self.dof_pos_batch.squeeze(0),
            self.dof_vel_batch.squeeze(0),
            self.dof_tor_batch.squeeze(0),
            self.tar_dof_pos_batch.squeeze(0),
        )
        obs = self._construct_observation(self.dof_pos_batch, self.dof_vel_batch, self.tar_dof_pos_batch)
        # print(obs.shape)
        logits = self.net(obs) # Shape: [batch_size, seq_len, num_classes]
        # print(f"action_inferenced: {action_inferenced.shape}")

        # Reshape dof_tor_batch to handle all values
        flat_tor = self.dof_tor_batch.reshape(-1)
        # Calculate distances for all values
        distances = torch.abs(flat_tor.unsqueeze(-1) - torch.linspace(-15, 15, self.num_classes))
        # Get closest indices
        closest_indices = torch.argmin(distances, dim=-1)
        # Create one-hot vectors
        target_one_hot = torch.zeros(flat_tor.shape[0], len(self.net.discrete_values), device=self.device)
        target_one_hot.scatter_(1, closest_indices.unsqueeze(-1), 1.0)
        # Reshape back to match logits
        target_one_hot = target_one_hot.reshape(self.dof_tor_batch.shape + (-1,))
        # print(f"target_one_hot:{target_one_hot.shape}")

        # action_inferenced = action_inferenced.squeeze(2)
        # print(f"action_inferenced: {action_inferenced}")

        # 2. 计算 loss
        # print(f"dof_tor_batch: {self.dof_tor_batch.shape}")
        # print(f"logits: {logits.shape}")
        loss = self.criterion_train(logits, target_one_hot)
        # print(f"loss: {loss.item()}")

        # 3. 更新网络参数
        self.optimizer.zero_grad()
        # print('have you reached here?')
        loss.backward() # RNN 中这里第二次反向传播竟然有问题; 太容易出问题啦，分类神经网络竟然也这里出问题
        nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
        self.optimizer.step()

        # self.net.reset()

        return loss.item()
    
    def test_update(self):
        # Get a batch from the test data generator
        mini_batch = next(self.test_batch_gen)
        mini_batch = torch.Tensor(np.array(mini_batch)).to(self.device)
        self.dof_pos_batch, self.dof_vel_batch, self.dof_tor_batch, self.tar_dof_pos_batch = mini_batch.split(1, dim=0)
        self.dof_pos_batch, self.dof_vel_batch, self.dof_tor_batch, self.tar_dof_pos_batch = (
            self.dof_pos_batch.squeeze(0),
            self.dof_vel_batch.squeeze(0),
            self.dof_tor_batch.squeeze(0),
            self.tar_dof_pos_batch.squeeze(0),
        )
        
        # Construct the observation
        obs = self._construct_observation(self.dof_pos_batch, self.dof_vel_batch, self.tar_dof_pos_batch)
        
        # Get the model's prediction
        with torch.no_grad():
            action_inferenced = self.net.act_inference(obs)
            # print(f"action_inferenced: {action_inferenced.shape}")
            action_inferenced = action_inferenced.squeeze(-1)
        
        # Calculate the loss
        # label = self.dof_tor_batch[:,-1]
        loss = self.criterion_test(action_inferenced, self.dof_tor_batch)
        # print(f"loss: {loss.item()}")
        
        return loss.item()
    
    def _construct_observation(self, dof_pos_batch, dof_vel_batch, tar_dof_pos_batch):
        obs = torch.stack([dof_pos_batch - tar_dof_pos_batch, dof_vel_batch], dim=2)
        return obs


if __name__ == "__main__":
    """
    因为目前数据不多，网络过于复杂可能出现过拟合的情况：选择降低参数数目，目前为 [64, 32, 16]
    """
    file_path = 'data_sets/merged_motor_data_ultimate.json'
    algo = SupervisedLearning(net=ActuatorNetRNN(input_size=30, output_size=1), storage=MiniBatchGenerator(file_path=file_path,loader=JsonConfigDataLoader, history_length=15, mini_batch_size=32), num_learning_epochs=10, file_path=file_path)
    with algo.storage.loaders as algo.storage.loaded_loaders:
        losslist = []
        for iter in range(10):
            print(f'iter: {iter + 1}')
            loss_single = algo.update() # 最后一个凑不齐 batch_size 的 batch 会让代码报错
            losslist.append(loss_single)
        plt.plot(losslist)
        plt.show()

    print('Successfully update the network!')