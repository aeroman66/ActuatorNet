import json
import random

def generate_motor_data(num_motors=12, array_length=600):
    data = {}
    for motor_id in range(num_motors):
        motor_data = {
            "dof_pos": [round(random.uniform(-1, 1), 4) for _ in range(array_length)],
            "dof_vel": [round(random.uniform(-0.5, 0.5), 4) for _ in range(array_length)],
            "dof_tor": [round(random.uniform(-1, 1), 4) for _ in range(array_length)],
        }
        data[f"motor_{motor_id}"] = motor_data
    return data

def merge_motors(data):
    merged = {
        "dof_pos": [],
        "dof_vel": [],
        "dof_tor": [],
        "tar_dof_pos": [],
    }
    
    for motor in data.values():
        merged["dof_pos"].extend(motor["dof_pos"])
        merged["dof_vel"].extend(motor["dof_vel"])
        merged["dof_tor"].extend(motor["dof_tor"])
        merged["tar_dof_pos"].extend(motor["dof_pos"])
    
    return {"motor_0": merged}


if __name__ == "__main__":
    # 生成数据
    file_ori = 'data_sets/go1_dataset_x0.25.json'
    json_file = 'merged_motor_data.json'

    with open(file_ori, 'r') as f:
        data = json.load(f)

    merged_data = merge_motors(data)
    with open(json_file, 'w') as f:
        json.dump(merged_data, f, indent=2)

    # # 将数据写入 JSON 文件
    # with open(json_file, 'w') as f:
    #     json.dump(motor_data, f, indent=2) # dump 的时候一定要指定 indent，不然整个 json 文件全部在一行上，可读性非常差。

    print("JSON file 'motor_data.json' has been created.")