import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import  torch
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import time

from network_training_utensils.storage.data_loader_rnn import MiniBatchGenerator, RNNDataLoader
from network_training_utensils.algo.supervised_learning_rnn import SupervisedLearning
from network_training_utensils.module.actuator_net_rnn import ActuatorNetRNN
from scripts import cfg

class Runner:
    def  __init__(self,
                  algo: SupervisedLearning,
                  loader: RNNDataLoader,
                  generator: MiniBatchGenerator,
                  net: ActuatorNetRNN,
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
        self.num_test = 0

        self.ifload = False

        self.writer = SummaryWriter(log_dir=self.log_dir)
        self.net = net(
            input_size=self.cfg.net.input_size,
            output_size=self.cfg.net.output_size,
            hidden_size=self.cfg.net.hidden_size,
            num_layers=self.cfg.net.num_layers,
            )
        self.generator = generator(
            file_path=self.cfg.file_path,
            loader=loader,
            sequence_length=self.cfg.algo.sequence_length,
            mini_batch_size=self.cfg.algo.batch_size,
        )
        self.algo = algo(
            net=self.net,
            storage=self.generator,
            num_learning_epochs=self.cfg.algo.num_learning_epochs,
            num_testing_epochs=self.cfg.algo.num_testing_epochs,
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
    def learn(self, id: int = None, init_at_random_ep_len : bool = False):
        self.train_mode()
        ep_info = []
        mean_losses_test = []
        mean_losses_train = []

        for epoch in range(self.cfg.algo.num_learning_epochs):
            self.algo.train_batch_gen = self.algo.storage.data_gen(dataset='train', id=id)
            with self.algo.storage.loaders_splited as self.algo.storage.loaded_loaders:
                iter = 0
                tot_loss = 0
                start = time.time()
                while True:
                    try:
                        loss = self.algo.update()
                        tot_loss += loss
                        iter += 1
                    except RuntimeError as info:
                        print(f"errare u hereor info: {info}")
                        # print('data ran out')
                        break
                end = time.time()
                time_consumed = end - start
            print(f"iter: {iter}")
            mean_loss = tot_loss / iter
            self.writer.add_scalar('loss', mean_loss, epoch)
            ep_info_dict = {
                    "epoch": epoch + 1,
                    "loss": mean_loss,
                    "time": time_consumed,
                }
            ep_info.append(ep_info_dict)
            print(f"Epoch: {epoch + 1}/{self.cfg.algo.num_learning_epochs}")
            print(f"Loss: {mean_loss:.4f}")
            print(f"Time: {time_consumed:.2f}s")
            print("-" * 30)

            # 对模型进行测试，以确定是否需要早停
            mean_loss_test = self.test(id=id)
            mean_losses_test.append(mean_loss_test)
            mean_losses_train.append(mean_loss)
            self.plot_loss(test_loss=mean_losses_test, train_loss=mean_losses_train)

            # 检查是否需要保存当前模型参数
            if not (epoch + 1) % self.cfg.runner.save_interval:
                print(f"Saving model at {epoch + 1}")
                self.save_model(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), f"{self.save_dir}/model_{epoch+1}_id{id}.pt"))
            ep_info.clear()


        self.writer.close()
        return mean_loss
    
    def test(self, id: int = None):
        '''
        input id to perform this operation on the specific motor.
        Without an input id, the operation will be performed on all motors.
        '''
        self.current_learning_iteration = 0
        self.algo.test_batch_gen = self.algo.storage.data_gen(dataset='test', id=id)
        with self.algo.storage.loaders_splited as self.algo.storage.loaded_loaders:
            self.algo.test_mode()
            total_loss = 0
            num_batches = 0
            self.algo.test_batch_gen = self.algo.storage.data_gen(dataset='test', id=id)

            start_iter = self.current_learning_iteration
            tot_iter = start_iter + self.cfg.algo.num_testing_epochs
            for iter in range(start_iter, tot_iter):
                try:
                    loss = self.algo.test_update()
                    total_loss += loss
                    num_batches += 1
                    # print(f"Test batch {num_batches}, Loss: {loss:.4f}")
                except RuntimeError:
                    break

                self.writer.add_scalar(f'motor_{id}_loss', loss, self.current_learning_iteration)
                self.current_learning_iteration += 1
        return total_loss / num_batches

            
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

    def load_model(self, file_path, load_optimizer=False):
        try:
            loaded_dict = torch.load(file_path, map_location=torch.device('cpu'), weights_only=True)
            print('File loaded:-)')    
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
        '''
        在训练模式中网络会把所有 latent 值全部记录下来
        所需要的内存会比较大
        '''
        self.net.train()

    def eval_mode(self):
        '''
        网络只会给出最后的结果
        对内存消耗没那么大
        '''
        self.net.eval()

    # *************************utiles******************************
    def plot_loss(self, test_loss, train_loss):
        '''
        绘制 test 和 train 的 loss 曲线
        '''
        plt.figure(figsize=(18, 6))
        # plt.plot(dof_tor_values)
        plt.plot(test_loss)
        plt.plot(train_loss)
        plt.legend(['test loss', 'train loss'])
        plt.title('test and train loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.grid(True)
        plt.ylim(-0.5, 6.0)

        plt.savefig(f'img/test & mean loss.png')
        plt.close()

if __name__ == "__main__":
    runner = Runner(
        algo=SupervisedLearning,
        loader=RNNDataLoader,
        generator=MiniBatchGenerator,
        net=ActuatorNetRNN,
        cfg=cfg,
        save_dir=cfg.runner.save_dir,
        log_dir=cfg.runner.log_dir,
    )

    with runner.algo.storage.loaders as runner.algo.storage.loaded_loaders:
        runner.learn()

    print("Training completed!")