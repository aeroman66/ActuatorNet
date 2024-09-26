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
        data[f"motor_id{motor_id}"] = motor_data
    return data


if __name__ == "__main__":
    # 生成数据
    motor_data = generate_motor_data()
    json_file = 'motor_data.json'

    # 将数据写入 JSON 文件
    with open(json_file, 'w') as f:
        json.dump(motor_data, f, indent=2) # dump 的时候一定要指定 indent，不然整个 json 文件全部在一行上，可读性非常差。

    print("JSON file 'motor_data.json' has been created.")