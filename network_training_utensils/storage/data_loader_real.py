import json
import time
import random
import numpy as np
from typing import List, Dict, Any
from dynaconf import Dynaconf

"""
JSON 数据格式
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
        # print(self.data.keys())
        end = time.time()
        # print(f"Data loading time: {end - start} seconds")

        self.indices = list(range(len(self.data['dof_pos'])))
    
    def shuffle_indices(self):
        random.shuffle(self.indices)

    def __enter__(self):
        """加载文件比较慢，执行的越少效率越高
        但这好像不是真正耗时的地方？！
        """
        self.file = open(self.file_path, 'r')
        # self.data = json.load(self.file)
        self.shuffle_indices() # 不放在 __init__ 中是因为我们希望每次迭代都进行 shuffle。最后放哪还得研究
        # print('data has been shuffled')
        self.current_index = 0
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
                # print("error 4")
                raise StopIteration
        else:
            if self.current_index + 1 >= len(self.data['dof_pos']):
                # print("error 3")
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
            item['dof_tor'][start + 1: end + 1], # 这里不写 + 1 损失更小的原因可能是网络直接学了一个类似kp kd 的控制器用上一时刻的数据直接算最后一位的 tor 值，而没有真正的去做预测工作
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
def auto_initialize(func):
    """自动实现生成器的初始化
    用上装饰器啦，太优雅啦！
    """
    def wrapper(*args, **kwargs):
        gen = func(*args, **kwargs)
        next(gen)  # 自动执行初始化
        return gen
    return wrapper

class MiniBatchGenerator:
    """
    通过 push 上面的类产生多条数据，来制作 mini_batch
    目前有个问题，这样获得的 mini_batch 不是随机的，而是按时序来的
    """
    def __init__(self, file_path: str, loader: JsonConfigDataLoader, history_length: int = 10, mini_batch_size: int = 32, drop_last: bool = True):
        self.file_path = file_path
        self.history_length = history_length
        self.mini_batch_size = mini_batch_size
        self.drop_last = drop_last

        self.consumed_set = set()

        self.loader = loader
        self.loaders = None
        self.loaders_splited = None
        self.loaded_loaders = None
        self.motor_iterators = None
        self.train_loaded_loaders = None
        self.val_loaded_loaders = None
        self.test_loaded_loaders = None

        self._init_loader_list()
        # print("you've reached here!")

    def _init_loader_list(self):
        loader_list = []
        for id in range(self.loader.num_motors):
            loader_list.append(self.loader(id, self.file_path, self.history_length, self.drop_last))
            # print((f"loader {id} has been initialized!"))
        self.loaders = LoaderManager(loader_list)
        self._split_datasets()

    def _split_datasets(self):
        train_loaders, val_loaders, test_loaders = [], [], []
        for loader in self.loaders.loaders:
            data_size = len(loader.data['dof_pos'])
            indices = np.arange(data_size)
            # np.random.shuffle(indices) # 这个打乱顺序好像有问题啊
            
            train_size = int(0.6 * data_size)
            val_size = int(0.2 * data_size)
            
            train_indices = indices[:train_size]
            val_indices = indices[train_size:train_size+val_size]
            test_indices = indices[train_size+val_size:]
            
            train_loader = self._create_subset_loader(loader, train_indices)
            val_loader = self._create_subset_loader(loader, val_indices)
            test_loader = self._create_subset_loader(loader, test_indices)
            
            train_loaders.append(train_loader)
            val_loaders.append(val_loader)
            test_loaders.append(test_loader)
        
        self.train_loaded_loaders = LoaderManager(train_loaders)
        self.val_loaded_loaders = LoaderManager(val_loaders)
        self.test_loaded_loaders = LoaderManager(test_loaders)

    def _create_subset_loader(self, original_loader, indices):
        subset_loader = self.loader(original_loader.motor_id, original_loader.file_path, original_loader.history_length, original_loader.drop_last)
        subset_loader.data = {key: value[indices] for key, value in original_loader.data.items()}
        subset_loader.indices = list(range(len(indices)))
        return subset_loader

    def __iter__(self):
        return self

    @auto_initialize
    def data_gen(self, dataset: str ='train'):
        """‘预热’或‘初始化’，让生成器函数第一次调用时执行部分函数体
        """
        self.consumed_set = set()
        # print("reset consumed_set!")
        if dataset == 'train':
            self.loaders_splited = self.train_loaded_loaders
            # print("train_loaded_loaders has been initialized!")
        elif dataset == 'val':
            self.loaders_splited = self.val_loaded_loaders
        elif dataset == 'test':
            self.loaders_splited = self.test_loaded_loaders
        else:
            raise ValueError("dataset must be 'train', 'val', or 'test'")
        
        yield None
        
        while True:
            mini_batch = [[] for _ in range(len(self.loader.attrs))]
            while len(mini_batch[0]) < self.mini_batch_size:
                # print(f"Num_motors: {len(self.loaders_splited.loaders)}")
                motor_id = random.randint(0, len(self.loaders_splited.loaders) - 1)
                while (len(self.consumed_set) < len(self.loaders_splited.loaders)) and (motor_id in self.consumed_set):
                    motor_id = random.randint(0, len(self.loaders_splited.loaders) - 1)
                try: 
                    data = next(self.loaded_loaders[motor_id]) # 不能再使用总的 num_motor，要使用 train 中的 num_motor
                    for idx, attr in enumerate(data):
                        # attr = [motor_id] # to verify the sampling
                        mini_batch[idx].append(attr)
                except StopIteration:
                    self.consumed_set.add(motor_id)
                    # In our circumstances, we need to handle StopIteration diffierently depending on their cause.
                    if not mini_batch:
                        raise StopIteration('Not mini_batch?')
                    if len(self.consumed_set) == len(self.loaders_splited.loaders):
                        raise StopIteration('Data has ran out!')
                    if self.drop_last and len(mini_batch[0]) < self.mini_batch_size:
                        # print('consumed_set', len(self.consumed_set))
                        continue
                    break
            
            if self.drop_last and len(mini_batch[0]) < self.mini_batch_size:
                print("error 1")
                raise StopIteration
            
            yield mini_batch

# Usage example
if __name__ == "__main__":
    file_path = 'data_sets/merged_motor_data_ultimate.json'
    mini_batch_gen = MiniBatchGenerator(file_path=file_path,loader=JsonConfigDataLoader, history_length=5, mini_batch_size=2)

    print(f"Number of motors: {mini_batch_gen.loader.num_motors}")
    batch_gen = mini_batch_gen.data_gen(100, 'train')
    with mini_batch_gen.loaders_splited as mini_batch_gen.loaded_loaders:
        for idx in range(100):
            mini_batch = next(batch_gen)
            print(f"Mini-batch {idx + 1}:")
            print(f"  Batch Size: {len(mini_batch[0])}")
            for attr, item in zip(mini_batch_gen.loader.attrs, mini_batch):
                print(f"    {attr} length: {len(item)}")
                print(f"    {attr} : {item}")
            print()