import json
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

"""
JSON 文件格式：
[
    {
        "motor_id": 0,
        "dof_pos": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "dof_vel": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    },
    {
        "motor_id": 1,
        "dof_pos": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "dof_vel": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    },
    ...
]
"""

class MotorDataset(Dataset):
    def __init__(self, json_file):
        with open(json_file, 'r') as f:
            self.data = json.load(f)
        
    def __len__(self):
        return len(self.data[0]['dof_pos'])
    
    def __getitem__(self, idx):
        motor_data = []
        motor = self.data[0]
        # for motor in self.data:
        #     motor_data.extend([motor['dof_pos'][idx], motor['dof_vel'][idx]])
        motor_data.extend([motor['dof_pos'][idx], motor['dof_vel'][idx], motor['dof_tor'][idx]])
        return torch.tensor(motor_data, dtype=torch.float32)
    
class MotorDataLoader:
    def __init__(self,
                 json_file,
                 batch_size=32,
                 shuffle=False,
                 drop_last=False):
        self.dataset = MotorDataset(json_file)
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last) # 这里不进行打乱是因为我们想保留数据的时序信息

# 使用示例
def main():
    json_file = 'data_sets/motor_data.json'
    # 创建 DataLoader

    dataloader = MotorDataLoader(json_file)

    # 使用数据加载器
    for batch in dataloader.dataloader:
        # 处理每个批次的数据
        print(batch.shape)  # 打印批次的形状
        print(batch)  # 打印批次的数据
        # 在这里进行您的模型训练或其他操作

if __name__ == "__main__":
    main()

