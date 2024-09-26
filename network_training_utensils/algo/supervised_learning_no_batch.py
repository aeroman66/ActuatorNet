import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from network_training_utensils.module.actuator_net import ActuatorNet
from network_training_utensils.storage.data_loader_prototype import MotorDataLoader

class SupervisedLearning:
    def __init__(self,
                 net : ActuatorNet,
                 storage : MotorDataLoader,
                 num_learning_epochs=1,
                 batch_size=20,
                 clip_param=0.2,
                 value_loss_coef=1.0,
                 learning_rate=1e-3,
                 max_grad_norm=1.0,
                 use_clipped_value_loss=True,
                 shuffle=False,
                 json_file=None,
                 device='cpu'):
        self.net = net
        self._storage = storage
        self.storage = None # 进行单独初始化
        self.num_learning_epochs = num_learning_epochs
        self.batch_size = batch_size
        self.clip_param = clip_param
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.shuffle = shuffle
        self.json_file = json_file
        self.device = device

        # 网络更新相关
        self.learning_rate = learning_rate
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.learning_rate)

        # attributes
        self.dof_pos_batch = None
        self.dof_vel_batch = None
        self.dof_tor_batch = None

        self._init_storage()

    def _init_storage(self):
        self.storage = self._storage(json_file=self.json_file,
                                     batch_size=self.batch_size,
                                     shuffle=self.shuffle)
        print(f"storage: {self.storage}")
        
    def test_mode(self):
        self.net.eval()

    def train_mode(self):
        self.net.train()

    def act(self, obs):
        return self.net.act_inference(obs)
    
    def update(self):
        losslist = []
        # 1. 获取 batch
        for  batch in self.storage.dataloader:
            # print(batch)
            self.dof_pos_batch, self.dof_vel_batch, self.dof_tor_batch = batch.split(1, dim=1)
            obs = self._construct_observation(self.dof_pos_batch, self.dof_vel_batch)
            action_inferenced = self.net.act_inference(obs)
            label = self.dof_tor_batch[-1]

            # 2. 计算 loss
            loss = self.criterion(action_inferenced, label)

            # 3. 更新网络参数
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
            self.optimizer.step()

            losslist.append(loss.item())
        return losslist

    def _construct_observation(self, dof_pos_batch, dof_vel_batch):
        obs = torch.cat((dof_pos_batch, dof_vel_batch), dim=0).squeeze(dim=1)
        # print(obs.shape)
        return obs


if __name__ == "__main__":
    algo = SupervisedLearning(net=ActuatorNet(input_size=40, output_size=1), storage=MotorDataLoader, json_file='data_sets/motor_data.json')
    losslist = algo.update() # 最后一个凑不齐 batch_size 的 batch 会让代码报错
    plt.plot(losslist)
    plt.show()

    print('Successfully update the network!')