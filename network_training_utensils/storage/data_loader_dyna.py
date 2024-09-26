import json
from typing import List, Dict, Any
from dynaconf import Dynaconf

class JsonConfigBatchLoader:
    def __init__(self, file_path: str, batch_size: int = 10, drop_last: bool = True):
        self.file_path = file_path
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.cfg = Dynaconf(settings_files=[file_path])
        self.file = None
        self.data = None
        self.current_index = 0

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
            if self.current_index >= len(self.cfg.motor_id0.dof_pos) or self.current_index + self.batch_size > len(self.cfg.motor_id0.dof_pos):
                raise StopIteration
        else:
            if self.current_index >= len(self.cfg.motor_id0.dof_pos):
                raise StopIteration

        item = self.data['motor_id0']
        batch = [[] for _ in range(3)]
        for i in range(self.current_index, min(self.current_index + self.batch_size, len(self.cfg.motor_id0.dof_pos))):
            batch[0].append(item["dof_pos"][i])
            batch[1].append(item["dof_vel"][i])
            batch[2].append(item["dof_tor"][i])

        # self.current_index += self.batch_size
        self.current_index += 1
        # print(self.current_index)
        return batch

# 使用示例
if __name__ == "__main__":
    config_path = 'data_sets/motor_data.json'
    with JsonConfigBatchLoader(config_path, batch_size=26, drop_last=True) as loader:
        cnt = 0
        for batch in loader:
            print(f"New batch:{cnt}")
            for attr, item in zip(('dof_pos', 'dof_vel', 'dof_tor'), batch):
                print(f"{attr} : {item}", len(item))
            cnt += 1
            print()