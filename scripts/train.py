import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

    # with runner.algo.storage.loaders as runner.algo.storage.loaded_loaders:
    runner.learn(id=2)
    # runner.test()

    print("Training completed!")