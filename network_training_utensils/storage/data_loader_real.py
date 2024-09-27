import json
import time
import random
import numpy as np
from typing import List, Dict, Any
from dynaconf import Dynaconf

from queue import Queue
from threading import Thread


"""
{
  "motor_0": {
    "dof_pos": [
        0.0, 
        0.0, 
        0.0, 
        0.0, 
        0.0, 
        0.0],
    "dof_vel": [
        0.0, 
        0.0, 
        0.0, 
        0.0, 
        0.0, 
        0.0],
    "dof_tor": [
        0.0, 
        0.0, 
        0.0, 
        0.0, 
        0.0, 
        0.0],
    },
  "motor_1": {
    "dof_pos": [
        0.0, 
        0.0, 
        0.0, 
        0.0, 
        0.0, 
        0.0],
    "dof_vel": [
        0.0, 
        0.0, 
        0.0, 
        0.0, 
        0.0, 
        0.0],
    "dof_tor": [
        0.0, 
        0.0, 
        0.0, 
        0.0, 
        0.0, 
        0.0],
    },
    ...
}
"""

class JsonConfigDataLoader:
    """
    这个类产生的不能被称为 batch
    因为比如 20 条历史数据构成一份我们提供给神经网络的输入，只能供进行一次训练。
    一个 batch 应该需要提供多次训练
    """
    attrs = ('dof_pos', 'dof_vel', 'dof_tor', 'tar_dof_pos')
    num_motors = 96
    _shared_data = None

    def __init__(self, motor_id: int, file_path: str, history_length: int = 10, drop_last: bool = True):
        self.motor_id = motor_id
        self.file_path = file_path
        self.history_length = history_length
        self.drop_last = drop_last
        self.cfg = Dynaconf(settings_files=[file_path])
        self.file = None
        self.data = None
        self.current_index = 0

        if JsonConfigDataLoader._shared_data is None:
            with open(file_path, 'r') as file:
                JsonConfigDataLoader._shared_data = json.load(file)
            JsonConfigDataLoader.num_motors = len(JsonConfigDataLoader._shared_data.keys())
            print(f"Number of motors: {JsonConfigDataLoader.num_motors}")

        # 将数据转换为NumPy数组
        start = time.time()
        self.data = {
            'dof_pos': np.array(JsonConfigDataLoader._shared_data[f'motor_{self.motor_id}']['dof_pos']),
            'dof_vel': np.array(JsonConfigDataLoader._shared_data[f'motor_{self.motor_id}']['dof_vel']),
            'dof_tor': np.array(JsonConfigDataLoader._shared_data[f'motor_{self.motor_id}']['dof_tor']),
            'tar_dof_pos': np.array(JsonConfigDataLoader._shared_data[f'motor_{self.motor_id}']['tar_dof_pos'])
        }
        print(self.data.keys())
        end = time.time()
        print(f"Data loading time: {end - start} seconds")

        self.indices = list(range(len(self.data['dof_pos'])))
        self.shuffle_indices() # 不放在 __init__ 中是因为我们希望每次迭代都进行 shuffle。最后放哪还得研究
    
    def shuffle_indices(self):
        random.shuffle(self.indices)

    def __enter__(self):
        """加载文件比较慢，执行的越少效率越高
        但这好像不是真正耗时的地方？！
        """
        self.file = open(self.file_path, 'r')
        # self.data = json.load(self.file)
        # print("you've excuted me!!!")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file:
            self.file.close()

    def __iter__(self):
        self.current_index = 0
        return self

    def __next__(self) -> List[Dict[str, Any]]:
        """
        + 1 是为了满足我们要取下一时刻的 tor 值而不让 index out of range
        """
        if self.drop_last:
            if self.current_index + 1 >= len(self.data['dof_pos']) or self.current_index + self.history_length + 1 > len(self.data['dof_pos']):
                raise StopIteration
        else:
            if self.current_index + 1 >= len(self.data['dof_pos']):
                raise StopIteration

        index = self.indices[self.current_index]
        while index + self.history_length + 1 > len(self.data['dof_pos']):
            self.current_index += 1
            index = self.indices[self.current_index]
        start = index
        end = start + self.history_length
        
        # item = self.data[f'motor_{self.motor_id}']
        item = self.data
        # data = [[] for _ in range(len(self.attrs))]
        # for i in range(index, min(index + self.history_length, len(self.cfg.motor_0.dof_pos))):
        #     data[0].append(item["dof_pos"][i])
        #     data[1].append(item["dof_vel"][i])
        #     data[2].append(item["dof_tor"][i + 1]) # 当前 pos_error 决定是下一时刻 tor 的值，所以是 i + 1
        #     data[3].append(item["tar_dof_pos"][i])
        data = [
            item['dof_pos'][start : end],
            item['dof_vel'][start : end],
            item['dof_tor'][start : end],
            item['tar_dof_pos'][start : end]
        ] # 换用 numpy 让单步训练时间从 1.85 s 下降到 1.4 s 左右

        self.current_index += 1
        return data
    
class LoaderManager:
    def __init__(self, loaders):
        self.loaders = loaders
        self.loaded_loaders = []

    def __enter__(self):
        for loader in self.loaders:
            loaded = loader.__enter__()
            self.loaded_loaders.append(loaded)
        return self.loaded_loaders

    def __exit__(self, exc_type=None, exc_val=None, exc_tb=None):
        for loader in self.loaders:
            loader.__exit__(exc_type, exc_val, exc_tb)

# 实现 batch 的生成
class MiniBatchGenerator:
    """
    通过 push 上面的类产生多条数据，来制作 mini_batch
    目前有个问题，这样获得的 mini_batch 不是随机的，而是按时序来的
    """
    def __init__(self, file_path: str, loader: JsonConfigDataLoader, history_length: int = 10, mini_batch_size: int = 32, drop_last: bool = True):
        self.file_path = file_path
        self.history_length = history_length
        self.loader = loader
        self.loaders = None
        self.mini_batch_size = mini_batch_size
        self.drop_last = drop_last
        # self.prefetcher = DataPrefetcher(self.loader, num_prefetch=2)
        self.loaded_loaders = None
        self.motor_iterators = None

        self._init_loader_list()
        print("you've reached here!")

    def _init_loader_list(self):
        loader_list = []
        for id in range(self.loader.num_motors):
            loader_list.append(self.loader(id, self.file_path, self.history_length, self.drop_last))
            print((f"loader {id} has been initialized!"))
        self.loaders = LoaderManager(loader_list)

    def __iter__(self):
        return self

    def data_gen(self, num_learning_epochs: int):
        for _ in range(num_learning_epochs):
            mini_batch = [[] for _ in range(len(self.loader.attrs))]
            for _ in range(self.mini_batch_size):
                motor_id = random.randint(0, self.loader.num_motors - 1)
                try:
                    data = next(self.loaded_loaders[motor_id])
                    for idx, attr in enumerate(data):
                        # attr = [motor_id] # to verify the sampling
                        mini_batch[idx].append(attr)
                except StopIteration:
                    if not mini_batch or (self.drop_last and len(mini_batch[0]) < self.mini_batch_size):
                        raise StopIteration
                    break
            
            if self.drop_last and len(mini_batch[0]) < self.mini_batch_size:
                raise StopIteration
            
            yield mini_batch

# Usage example
if __name__ == "__main__":
    # file_path = 'data_sets/smooth/go1_dataset_x0.25_smooth.json'
    file_path = 'data_sets/merged_motor_data_ultimate.json'
    mini_batch_gen = MiniBatchGenerator(file_path=file_path,loader=JsonConfigDataLoader, history_length=15, mini_batch_size=32)

    print(f"Number of motors: {mini_batch_gen.loader.num_motors}")
    with mini_batch_gen.loaders as mini_batch_gen.loaded_loaders:
        batch_gen = mini_batch_gen.data_gen(100)
        for idx in range(100):
            mini_batch = next(batch_gen)
            print(f"Mini-batch {idx + 1}:")
            print(f"  Batch Size: {len(mini_batch[0])}")
            for attr, item in zip(mini_batch_gen.loader.attrs, mini_batch):
                print(f"    {attr} length: {len(item)}")
                print(f"    {attr} : {item}")
            print()