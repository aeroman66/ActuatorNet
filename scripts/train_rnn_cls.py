import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from network_training_utensils.storage.data_loader_rnn import MiniBatchGenerator, RNNDataLoader
from network_training_utensils.algo.supervised_learning_rnn_cls import SupervisedLearning
from network_training_utensils.module.actuator_net_rnn_cls import ActuatorNetRNNDIS
from network_training_utensils.runner.runner_rnn_cls import Runner
from scripts import cfg

if __name__ == '__main__':
    runner = Runner(
        algo=SupervisedLearning,
        loader=RNNDataLoader,
        generator=MiniBatchGenerator,
        net=ActuatorNetRNNDIS,
        cfg=cfg,
        save_dir=cfg.runner.save_dir,
        log_dir=cfg.runner.log_dir,
    )

    runner.learn()
    # runner.test()

    print("Training completed!")