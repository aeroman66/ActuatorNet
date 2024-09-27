import json
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
# class DataPrefetcher:
#     def __init__(self, loader, num_prefetch=1):
#         self.loader = loader
#         self.queue = Queue(maxsize=num_prefetch)
#         self.thread = None

#     def __enter__(self):
#         self.thread = Thread(target=self._prefetch_data, daemon=True)
#         self.thread.start()
#         return self

#     def __exit__(self, exc_type, exc_val, exc_tb):
#         if self.thread:
#             self.thread.join()

#     def _prefetch_data(self):
#         with self.loader as loader:
#             for data in loader:
#                 self.queue.put(data)
#         self.queue.put(None)  # 结束信号

#     def __iter__(self):
#         return self

#     def __next__(self):
#         data = self.queue.get()
#         if data is None:
#             raise StopIteration
#         return data

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

        # print("will you stuck?")
        with open(file_path, 'r') as file:
            data = json.load(file)
        # 将数据转换为NumPy数组
        self.data = {
            'dof_pos': np.array(data['motor_0']['dof_pos']),
            'dof_vel': np.array(data['motor_0']['dof_vel']),
            'dof_tor': np.array(data['motor_0']['dof_tor']),
            'tar_dof_pos': np.array(data['motor_0']['tar_dof_pos'])
        }
        # self.data = {}
        # for attr in self.attrs:
        #     temp_file = f"{file_path}_{attr}.npy"
        #     np.save(temp_file, np.array(data['motor_0'][attr]))
        #     self.data[attr] = np.load(temp_file, mmap_mode='r')
    
    def shuffle_indices(self):
        random.shuffle(self.indices)

    def __enter__(self):
        """加载文件比较慢，执行的越少效率越高
        但这好像不是真正耗时的地方？！
        """
        self.file = open(self.file_path, 'r')
        self.data = json.load(self.file)
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
            if self.current_index + 1 >= len(self.cfg.motor_0.dof_pos) or self.current_index + self.history_length + 1 > len(self.cfg.motor_0.dof_pos):
                raise StopIteration
        else:
            if self.current_index + 1 >= len(self.cfg.motor_0.dof_pos):
                raise StopIteration

        index = self.indices[self.current_index]
        while index + self.history_length + 1 > len(self.cfg.motor_0.dof_pos):
            self.current_index += 1
            index = self.indices[self.current_index]
        start = index
        end = start + self.history_length
        
        item = self.data['motor_0']
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
    def __init__(self, file_path: str, loader: JsonConfigDataLoader, history_length: int = 10, mini_batch_size: int = 32, drop_last: bool = True):
        self.loader = loader
        self.mini_batch_size = mini_batch_size
        self.drop_last = drop_last
        # self.prefetcher = DataPrefetcher(self.loader, num_prefetch=2)
        self.loaded = None

    def __iter__(self):
        return self

    # def __next__(self) -> List[List[Dict[str, Any]]]:
    #     mini_batch = [[] for _ in range(len(self.loader.attrs))]
    #     # with self.loader as load:
    #     for _ in range(self.mini_batch_size):
    #         try:
    #             data = next(self.loaded)
    #             for idx, attr in enumerate(data):
    #                 mini_batch[idx].append(attr)
    #         except StopIteration:
    #             if not mini_batch or (self.drop_last and len(mini_batch[0]) < self.mini_batch_size):
    #                 raise StopIteration
    #             break
        
    #     if self.drop_last and len(mini_batch[0]) < self.mini_batch_size:
    #         raise StopIteration
        
    #     return mini_batch

    def data_gen(self, num_learning_epochs: int):
        for _ in range(num_learning_epochs):
            mini_batch = [[] for _ in range(len(self.loader.attrs))]
            # with self.loader as load:
            for _ in range(self.mini_batch_size):
                try:
                    data = next(self.loaded)
                    for idx, attr in enumerate(data):
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
    file_path = 'data_sets/merged_motor_data.json'
    loader = JsonConfigDataLoader(file_path=file_path, history_length=10)
    mini_batch_gen = MiniBatchGenerator(file_path=file_path,loader=loader, history_length=10, mini_batch_size=2)

    # with mini_batch_gen.loader as mini_batch_gen.loaded:
    #     for idx, mini_batch in enumerate(mini_batch_gen):
    #         print(f"Mini-batch {idx + 1}:")
    #         print(f"  Batch Size: {len(mini_batch[0])}")
    #         for attr, item in zip(mini_batch_gen.loader.attrs, mini_batch):
    #             print(f"    {attr} length: {len(item)}")
    #             print(f"    {attr} : {item}")
    #         print()
    # with mini_batch_gen.loader as mini_batch_gen.loaded:
    #     batch_gen = mini_batch_gen.data_gen()
    #     for idx, mini_batch in enumerate(batch_gen):
    #         print(f"Mini-batch {idx + 1}:")
    #         print(f"  Batch Size: {len(mini_batch[0])}")
    #         for attr, item in zip(mini_batch_gen.loader.attrs, mini_batch):
    #             print(f"    {attr} length: {len(item)}")
    #             print(f"    {attr} : {item}")
    #         print()
    with mini_batch_gen.loader as mini_batch_gen.loaded:
        batch_gen = mini_batch_gen.data_gen(10)
        for idx in range(10):
            mini_batch = next(batch_gen)
            print(f"Mini-batch {idx + 1}:")
            print(f"  Batch Size: {len(mini_batch[0])}")
            for attr, item in zip(mini_batch_gen.loader.attrs, mini_batch):
                print(f"    {attr} length: {len(item)}")
                print(f"    {attr} : {item}")
            print()