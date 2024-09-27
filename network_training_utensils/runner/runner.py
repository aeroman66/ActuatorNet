import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import  torch
from torch.utils.tensorboard import SummaryWriter
from collections import deque
import time

from network_training_utensils.storage.data_loader_real import MiniBatchGenerator, JsonConfigDataLoader
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
                  save_dir=None,
                  log_dir=None,
                  device='cpu'):
        self.cfg = cfg
        self.save_dir = save_dir
        self.log_dir = log_dir
        self.device = device
        self.current_learning_iteration = 0
        self.tot_timesteps = 0
        self.tot_time = 0

        self.ifload = False

        self.writer = SummaryWriter(log_dir=self.log_dir)
        self.net = net(
            input_size=self.cfg.net.half_input_size * 2,
            output_size=self.cfg.net.output_size,
            )
        # 感觉还是应该定义成实例变量，因为需要保证在整个类的生命周期内都活着
        # self.loader = loader(
        #     file_path=self.cfg.file_path,
        #     history_length=self.cfg.algo.history_length,
        # )
        self.generator = generator(
            file_path=self.cfg.file_path,
            loader=loader,
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
            weight_decay=self.cfg.algo.weight_decay,
            max_grad_norm=self.cfg.algo.max_grad_norm,
            shuffle=self.cfg.algo.shuffle,
            file_path=self.cfg.file_path,
            device=self.cfg.device
        )
    # ************************train**************************
    def learn(self, init_at_random_ep_len : bool = False):
        self.train_mode()
        ep_info = []

        start_iter = self.current_learning_iteration
        tot_iter = start_iter + self.cfg.algo.num_learning_epochs
        for iter in range(start_iter, tot_iter):
            start = time.time()
            mean_loss = self.algo.update()
            end = time.time()
            time_consumed = end - start

            self.writer.add_scalar('loss', mean_loss, iter)

            ep_info_dict = {
                    "iter": iter,
                    "loss": mean_loss,
                    "time": time_consumed
                }
            ep_info.append(ep_info_dict)
            print(f"Epoch: {iter + 1}/{tot_iter}")
            print(f"Loss: {mean_loss:.4f}")
            print(f"Time: {time_consumed:.2f}s")
            print("-" * 30)
            
            self.current_learning_iteration = iter

            if not (iter + 1) % self.cfg.runner.save_interval:
                print(f"Saving model at {iter + 1}")
                self.save_model(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), f"{self.save_dir}/model_{iter+1}.pth"))
                # self.save_model(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))
            ep_info.clear()

        self.writer.close()

            
    # ***********************save & load**************************
    def save_model(self, file_path):
        saved_dict = {
            "model_state_dict": self.algo.net.state_dict(),
            "optimizer_state_dict": self.algo.optimizer.state_dict(),
            "iter": self.current_learning_iteration,
            # "infos": infos,
        }
        file_dir = os.path.dirname(file_path)
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)
        os.chmod(file_dir, 0o777)
        torch.save(saved_dict, file_path)

    def load_model(self, file_path, load_optimizer=True):
        try:
            loaded_dict = torch.load(file_path, map_location=torch.device('cpu'))
            print('File loaded:)')    
        except FileNotFoundError:
            print(f"File '{file_path}' missing:(")

        try:
            self.algo.net.load_state_dict(loaded_dict["model_state_dict"])
        except:
            print('Somehow fail load:(')

        if load_optimizer:
            self.algo.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
        self.current_learning_iteration = loaded_dict["iter"]
        self.ifload = True
        return

    # *************************modes******************************
    def train_mode(self):
        self.net.train()

    def eval_mode(self):
        self.net.eval()

if __name__ == "__main__":
    runner = Runner(
        algo=SupervisedLearning,
        loader=JsonConfigDataLoader,
        generator=MiniBatchGenerator,
        net=ActuatorNet,
        cfg=cfg,
        save_dir=cfg.runner.save_dir,
        log_dir=cfg.runner.log_dir,
    )

    with runner.algo.storage.loaders as runner.algo.storage.loaded_loaders:
        runner.learn()

    print("Training completed!")