import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from network_training_utensils.algo.supervised_learning import SupervisedLearning
from network_training_utensils.module.actuator_net import ActuatorNet
from network_training_utensils.storage.data_loader_prototype import MotorDataLoader

import  torch
from torch.utils.tensorboard import SummaryWriter
from collections import deque
import time

from network_training_utensils.storage.data_loader_dyna import MiniBatchGenerator, JsonConfigDataLoader
from network_training_utensils.algo.supervised_learning import SupervisedLearning
from network_training_utensils.module.actuator_net import ActuatorNet
from scripts import cfg

class Runner:
    def  __init__(self,
                  algo: SupervisedLearning,
                  loader: JsonConfigDataLoader,
                  generator: MiniBatchGenerator,
                  net: ActuatorNet,
                  cfg,
                  log_dir=None,
                  device='cpu'):
        self.cfg = cfg
        self.log_dir = log_dir
        self.device = device

        self.net = net(
            input_size=self.cfg.net.half_input_size * 2,
            output_size=self.cfg.net.output_size,
            )
        # 感觉还是应该定义成实例变量，因为需要保证在整个类的生命周期内都活着
        self.loader = loader(
            file_path=self.cfg.file_path,
            history_length=self.cfg.algo.history_length,
        )
        self.generator = generator(
            file_path=self.cfg.file_path,
            loader=self.loader,
            history_length=self.cfg.algo.history_length,
            mini_batch_size=self.cfg.algo.batch_size,
        )
        self.algo = algo(
            net=self.net,
            storage=self.generator,
            num_learning_epochs=self.cfg.algo.num_learning_epochs,
            history_length=self.cfg.algo.history_length,
            batch_size=self.cfg.algo.batch_size,
            clip_param=self.cfg.algo.clip_param,
            learning_rate=self.cfg.algo.learning_rate,
            max_grad_norm=self.cfg.algo.max_grad_norm,
            shuffle=self.cfg.algo.shuffle,
            json_file=self.cfg.file_path,
            device=self.cfg.device
        )

if __name__ == "__main__":
    runner = Runner(
        algo=SupervisedLearning,
        loader=JsonConfigDataLoader,
        generator=MiniBatchGenerator,
        net=ActuatorNet,
        cfg=cfg,
    )

    print("Initialization completed!")