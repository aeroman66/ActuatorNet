import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt

from network_training_utensils.storage.data_loader_real import MiniBatchGenerator, JsonConfigDataLoader
from network_training_utensils.algo.supervised_learning import SupervisedLearning
from network_training_utensils.module.actuator_net import ActuatorNet
from network_training_utensils.runner.runner import Runner
from scripts import cfg

if __name__ == '__main__':
    runner = Runner(
        algo=SupervisedLearning,
        loader=JsonConfigDataLoader,
        generator=MiniBatchGenerator,
        net=ActuatorNet,
        cfg=cfg,
        save_dir=cfg.runner.save_dir,
        log_dir=cfg.runner.log_dir,
    )

    runner.load_model(cfg.policy_path)
    # print('mean_loss:', runner.test())

    losses = []
    for id in range(12):
        loss = runner.test(id)
        losses.append(loss)

    plt.figure(figsize=(18, 6))
    # plt.plot(dof_tor_values)
    plt.plot(losses)
    plt.title('mean loss of each motor')
    plt.xlabel('motor_id')
    plt.ylabel('loss')
    plt.grid(True)

    plt.savefig(f'img/test_loss.png')
    plt.close()

    print("Testing completed!")