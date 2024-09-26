import json
from typing import List, Dict, Any
from dynaconf import Dynaconf

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
        self.attrs = ('dof_pos', 'dof_vel', 'dof_tor')

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
            if self.current_index >= len(self.cfg.motor_id0.dof_pos) or self.current_index + self.history_length > len(self.cfg.motor_id0.dof_pos):
                raise StopIteration
        else:
            if self.current_index >= len(self.cfg.motor_id0.dof_pos):
                raise StopIteration

        item = self.data['motor_id0']
        data = [[] for _ in range(3)]
        for i in range(self.current_index, min(self.current_index + self.history_length, len(self.cfg.motor_id0.dof_pos))):
            data[0].append(item["dof_pos"][i])
            data[1].append(item["dof_vel"][i])
            data[2].append(item["dof_tor"][i])

        # self.current_index += self.batch_size
        self.current_index += 1
        # print(self.current_index)
        return data

# 实现 batch 的生成
class MiniBatchGenerator:
    """
    通过 push 上面的类产生多条数据，来制作 mini_batch
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
        #     for _ in range(self.mini_batch_size):
        #         try:
        #             batch = next(loader)
        #             mini_batch.append(batch)
        #         except StopIteration:
        #             if not mini_batch or (self.drop_last and len(mini_batch) < self.mini_batch_size):
        #                 raise StopIteration
        #             break
        
        # if self.drop_last and len(mini_batch) < self.mini_batch_size:
        #     raise StopIteration
        
        return mini_batch

# Usage example
if __name__ == "__main__":
    config_path = 'data_sets/motor_data.json'
    mini_batch_gen = MiniBatchGenerator(config_path, history_length=14, mini_batch_size=5)
    
    for idx, mini_batch in enumerate(mini_batch_gen):
        print(f"Mini-batch {idx + 1}:")
        print(f"  Batch Size: {len(mini_batch[0])}")
        for attr, item in zip(mini_batch_gen.loader.attrs, mini_batch):
            print(f"    {attr} length: {len(item)}")
            print(f"    {attr} : {item}")
        print()


# # 使用示例
# if __name__ == "__main__":
#     config_path = 'data_sets/motor_data.json'
#     with JsonConfigDataLoader(config_path, history_length=26, drop_last=True) as loader:
#         cnt = 0
#         for batch in loader:
#             print(f"New batch:{cnt}")
#             for attr, item in zip(('dof_pos', 'dof_vel', 'dof_tor'), batch):
#                 print(f"{attr} : {item}", len(item))
#             cnt += 1
#             print()