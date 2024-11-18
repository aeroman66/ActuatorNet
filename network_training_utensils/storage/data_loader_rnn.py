import json
import random
import numpy as np
from typing import List, Dict, Any

class RNNDataLoader:
    attrs = ('dof_pos', 'dof_vel', 'dof_tor', 'tar_dof_pos')
    num_motors = 12
    _shared_data = None

    def __init__(self, motor_id: int, file_path: str, sequence_length: int = 10, 
                 split_mode: str = 'sequential', drop_last: bool = True):
        self.motor_id = motor_id
        self.file_path = file_path
        self.sequence_length = sequence_length
        self.drop_last = drop_last
        self.valid_starts = []

        if split_mode not in ['sequential', 'random']:
            raise ValueError("split_mode must be 'sequential' or 'random'")
        self.split_mode = split_mode
        
        if RNNDataLoader._shared_data is None:
            with open(file_path, 'r') as file:
                RNNDataLoader._shared_data = json.load(file)
            RNNDataLoader.num_motors = len(RNNDataLoader._shared_data.keys())

        self.data = {
            'dof_pos': np.array(RNNDataLoader._shared_data[f'motor_{self.motor_id}']['dof_pos']),
            'dof_vel': np.array(RNNDataLoader._shared_data[f'motor_{self.motor_id}']['dof_vel']),
            'dof_tor': np.array(RNNDataLoader._shared_data[f'motor_{self.motor_id}']['dof_tor']),
            'tar_dof_pos': np.array(RNNDataLoader._shared_data[f'motor_{self.motor_id}']['tar_dof_pos'])
        }
        
        self._create_sequences()
        
    def _create_sequences(self):
        data_length = len(self.data['dof_pos'])
        # print(f"Data length: {data_length}")
        self.sequences = []

        if self.split_mode == 'sequential': # 比较简单，对原序列进行顺序切片
            for i in range(0, data_length - self.sequence_length, self.sequence_length):
                sequence = {
                    'dof_pos': self.data['dof_pos'][i:i+self.sequence_length],
                    'dof_vel': self.data['dof_vel'][i:i+self.sequence_length],
                    'dof_tor': self.data['dof_tor'][i+1:i+self.sequence_length+1],
                    'tar_dof_pos': self.data['tar_dof_pos'][i:i+self.sequence_length]
                }
                self.sequences.append(sequence)
        
        elif self.split_mode == 'random':
            self.valid_starts = list(range(0, data_length - self.sequence_length, self.sequence_length))
            self.shuffle_indices(self.valid_starts)
            for start in self.valid_starts:
                sequence = {
                    'dof_pos': self.data['dof_pos'][start:start+self.sequence_length],
                    'dof_vel': self.data['dof_vel'][start:start+self.sequence_length],
                    'dof_tor': self.data['dof_tor'][start+1:start+self.sequence_length+1],
                    'tar_dof_pos': self.data['tar_dof_pos'][start:start+self.sequence_length]
                }
                self.sequences.append(sequence)

    def shuffle_indices(self):
        random.shuffle(self.valid_starts)

    def print_sequences(self):
        for sequence in self.sequences:
            print(sequence)

    def __enter__(self):
        self.shuffle_indices()
        self.current_index = 0
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def __iter__(self):
        self.current_index = 0
        return self

    def __next__(self):
        if self.current_index >= (len(self.sequences) - 1):
            raise StopIteration
        sequence = self.sequences[self.current_index]
        self.current_index += 1

        data = [
            sequence['dof_pos'],
            sequence['dof_vel'],
            sequence['dof_tor'],
            sequence['tar_dof_pos']
        ]

        return data
        # return sequence

    def __repr__(self):
        return f"RNNDataLoader(motor_id={self.motor_id}, num_sequence={len(self.sequences)})"


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

def auto_initialize(func):
    def wrapper(*args, **kwargs):
        gen = func(*args, **kwargs)
        next(gen)
        return gen
    return wrapper

class MiniBatchGenerator:
    def __init__(self, file_path: str, loader: RNNDataLoader, sequence_length: int = 10, 
                 mini_batch_size: int = 32, drop_last: bool = True):
        self.file_path = file_path
        self.sequence_length = sequence_length
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

    def _init_loader_list(self):
        loader_list = []
        loader_list.append(self.loader(motor_id=0, file_path=self.file_path, sequence_length=self.sequence_length, split_mode='sequential', drop_last=self.drop_last))
        for id in range(1, self.loader.num_motors):
            loader_list.append(self.loader(motor_id=id, file_path=self.file_path, sequence_length=self.sequence_length, split_mode='sequential', drop_last=self.drop_last))
        self.loaders = LoaderManager(loader_list)
        self._split_datasets()

    def _split_datasets(self):
        train_loaders, val_loaders, test_loaders = [], [], []
        for loader in self.loaders.loaders:
            data_size = len(loader.sequences)
            indices = np.arange(data_size)
            
            train_size = int(0.6 * data_size)
            val_size = int(0.1 * data_size)
            
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
        subset_loader = self.loader(original_loader.motor_id, original_loader.file_path, 
                                  original_loader.sequence_length, original_loader.split_mode)
        subset_loader.sequences = [original_loader.sequences[i] for i in indices]
        return subset_loader

    @auto_initialize
    def data_gen(self, dataset: str ='train', id: int = None):
        self.consumed_set = set()
        
        if dataset == 'train':
            self.loaders_splited = self.train_loaded_loaders
        elif dataset == 'val':
            self.loaders_splited = self.val_loaded_loaders
        elif dataset == 'test':
            self.loaders_splited = self.test_loaded_loaders
        else:
            raise ValueError("dataset must be 'train', 'val', or 'test'")
        
        id_list_origin = list(range(RNNDataLoader.num_motors))
        id_filtered = [x for x in id_list_origin if x % 3 != 2]

        # print(f"length of idlist: {len(id_filtered)}")
        # RNNDataLoader.num_motors = len(id_filtered)

        yield None

        if id is not None:
            max_multiple = int(len(self.loaders_splited.loaders) / 12 - 1)
            id_list = [id + 3 * multiple for multiple in range(max_multiple + 1)]
        
        while True:
            '''
            在这里面要做的，就是把前面那些 sequence 字典抽取出来，然后组成 mini_batch。
            与原先的 loader 还不一样, batch 里的每个sample 都得给一个隐藏状态
            '''
            if id is None:
                mini_batch = [[] for _ in range(len(self.loader.attrs))]
                while len(mini_batch[0]) < self.mini_batch_size:
                    motor_id = random.choice(id_filtered)
                    try: 
                        data = next(self.loaded_loaders[motor_id])
                        for idx, attr in enumerate(data):
                            mini_batch[idx].append(attr)
                    except StopIteration:
                        id_filtered.remove(motor_id)
                        if not mini_batch:
                            raise StopIteration('Not mini_batch?')
                        if not id_filtered:
                            raise StopIteration('Data has ran out!')
                        if self.drop_last and len(mini_batch[0]) < self.mini_batch_size:
                            continue
                        break
                
                if self.drop_last and len(mini_batch[0]) < self.mini_batch_size:
                    print("No enough data for one more batch.")
                    raise StopIteration
                
            else:
                mini_batch = [[] for _ in range(len(self.loader.attrs))]
                while len(mini_batch[0]) < self.mini_batch_size:
                    motor_id = random.choice(id_list)
                    try: 
                        data = next(self.loaded_loaders[motor_id])
                        for idx, attr in enumerate(data):
                            mini_batch[idx].append(attr)
                    except StopIteration:
                        id_list.remove(motor_id)
                        if not mini_batch:
                            raise StopIteration('Not mini_batch?')
                        if not id_list:
                            raise StopIteration('Data has ran out!')
                        if self.drop_last and len(mini_batch[0]) < self.mini_batch_size:
                            continue
                        break
                
                if self.drop_last and len(mini_batch[0]) < self.mini_batch_size:
                    raise StopIteration
                
            yield mini_batch

if __name__ == "__main__":
    mini_batch_gen = MiniBatchGenerator(
        file_path='data_sets/merged_motor_data_ultimate_pro_max_plus.json',
        loader=RNNDataLoader,
        sequence_length=15,
        mini_batch_size=32
    )

    batch_gen = mini_batch_gen.data_gen('train')
    with mini_batch_gen.loaders_splited as mini_batch_gen.loaded_loaders:
        print(RNNDataLoader.num_motors)
        for idx in range(100):
            mini_batch = next(batch_gen)
            print(f"Mini-batch {idx + 1}:")
            print(f"  Batch Size: {len(mini_batch[0])}")
            for attr, item in zip(mini_batch_gen.loader.attrs, mini_batch):
                print(f"    {attr}: ")
                print(f"    Num_sequence: {len(item)}")
                print(f"    Shape of each sequence: {len(item[0])}")
            print()

