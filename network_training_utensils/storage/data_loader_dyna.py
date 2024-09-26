import json
import random
from typing import List, Dict, Any
from dynaconf import Dynaconf

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
    def __init__(self, file_path: str, history_length: int = 10, drop_last: bool = True):
        self.file_path = file_path
        self.history_length = history_length
        self.drop_last = drop_last
        self.cfg = Dynaconf(settings_files=[file_path])
        self.file = None
        self.data = None
        self.current_index = 0
        self.indices = list(range(len(self.cfg.motor_0.dof_pos)))
        self.attrs = ('dof_pos', 'dof_vel', 'dof_tor', 'tar_dof_pos')
        self.shuffle_indices() # 不放在 __init__ 中是因为我们希望每次迭代都进行 shuffle。最后放哪还得研究
    
    def shuffle_indices(self):
        random.shuffle(self.indices)

    def __enter__(self):
        self.file = open(self.file_path, 'r')
        self.data = json.load(self.file)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file:
            self.file.close()

    def __iter__(self):
        self.current_index = 0
        return self

    def __next__(self) -> List[Dict[str, Any]]:
        if self.drop_last:
            if self.current_index >= len(self.cfg.motor_0.dof_pos) or self.current_index + self.history_length > len(self.cfg.motor_0.dof_pos):
                raise StopIteration
        else:
            if self.current_index >= len(self.cfg.motor_0.dof_pos):
                raise StopIteration

        index = self.indices[self.current_index]
        while index + self.history_length > len(self.cfg.motor_0.dof_pos):
            self.current_index += 1
            index = self.indices[self.current_index]
        item = self.data['motor_0']
        data = [[] for _ in range(len(self.attrs))]
        for i in range(index, min(index + self.history_length, len(self.cfg.motor_0.dof_pos))):
            data[0].append(item["dof_pos"][i])
            data[1].append(item["dof_vel"][i])
            data[2].append(item["dof_tor"][i])
            data[3].append(item["tar_dof_pos"][i])

        # self.current_index += self.batch_size
        self.current_index += 1
        # print(self.current_index)
        return data

# 实现 batch 的生成
class MiniBatchGenerator:
    """
    通过 push 上面的类产生多条数据，来制作 mini_batch
    目前有个问题，这样获得的 mini_batch 不是随机的，而是按时序来的
    """
    def __init__(self, file_path: str, history_length: int = 10, mini_batch_size: int = 32, drop_last: bool = True):
        self.loader = JsonConfigDataLoader(file_path, history_length)
        self.mini_batch_size = mini_batch_size
        self.drop_last = drop_last

    def __iter__(self):
        return self

    def __next__(self) -> List[List[Dict[str, Any]]]:
        mini_batch = [[] for _ in range(len(self.loader.attrs))]
        with self.loader as loader:
            for _ in range(self.mini_batch_size):
                try:
                    data = next(loader)
                    for idx, attr in enumerate(data):
                        mini_batch[idx].append(attr)
                except StopIteration:
                    if not mini_batch or (self.drop_last and len(mini_batch[0]) < self.mini_batch_size):
                        raise StopIteration
                    break
        
        if self.drop_last and len(mini_batch[0]) < self.mini_batch_size:
            raise StopIteration
        
        return mini_batch

# Usage example
if __name__ == "__main__":
    file_path = 'data_sets/merged_motor_data.json'
    mini_batch_gen = MiniBatchGenerator(file_path=file_path, history_length=10, mini_batch_size=2)
    # for idx, mini_batch in enumerate(mini_batch_gen):
    #     pass
    # print(f"mini_batch_gen.loader.indices: {mini_batch_gen.loader.indices}")
    for idx, mini_batch in enumerate(mini_batch_gen):
        print(f"Mini-batch {idx + 1}:")
        print(f"  Batch Size: {len(mini_batch[0])}")
        for attr, item in zip(mini_batch_gen.loader.attrs, mini_batch):
            print(f"    {attr} length: {len(item)}")
            print(f"    {attr} : {item}")
        print()