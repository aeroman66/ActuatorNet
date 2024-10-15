import os
import json
import random
import numpy as np
import matplotlib.pyplot as plt

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

def merge_motors(input_file, output_file):
    # 读取输入文件
    with open(input_file, 'r') as f:
        data = json.load(f)

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
    
    merged_data = {"motor_0": merged}

    # 将合并后的数据写入输出文件
    with open(output_file, 'w') as f:
        json.dump(merged_data, f, indent=2)

    print(f"Merged data has been saved to {output_file}")

def smooth_json_data(input_file, output_file, window_size=5):
    # 读取 JSON 文件
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # 对每个电机的每个属性进行平滑处理
    for motor in data.values():
        for attr in ['dof_pos', 'dof_vel', 'dof_tor', 'tar_dof_pos']:
            values = np.array(motor[attr])
            smoothed = np.convolve(values, np.ones(window_size)/window_size, mode='same')
            motor[attr] = smoothed.tolist()
    
    # 将平滑后的数据写入新的 JSON 文件
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)

def plot_json_data(json_file, output_file=None):
    # 读取 JSON 文件
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # 获取第一个电机的数据作为示例
    motor_data = data['motor_0']
    
    # 创建子图
    fig, axs = plt.subplots(4, 1, figsize=(12, 15))
    
    # 绘制 dof_pos
    axs[0].plot(motor_data['dof_pos'])
    axs[0].set_title('DOF Position')
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('Position')
    
    # 绘制 dof_vel
    axs[1].plot(motor_data['dof_vel'])
    axs[1].set_title('DOF Velocity')
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('Velocity')
    
    # 绘制 dof_tor
    axs[2].plot(motor_data['dof_tor'])
    axs[2].set_title('DOF Torque')
    axs[2].set_xlabel('Time')
    axs[2].set_ylabel('Torque')

    # 绘制 tar_dof_pos
    axs[3].plot(motor_data['tar_dof_pos'])
    axs[3].set_title('Target DOF Position')
    axs[3].set_xlabel('Time')
    axs[3].set_ylabel('Position')
    
    # 调整子图布局
    plt.tight_layout()
    
    # 保存图片或显示
    if output_file:
        plt.savefig(output_file)
        print(f"Plot saved to {output_file}")
    else:
        plt.show()

def plot_all_motors_data(json_file, output_file=None):
    # 读取 JSON 文件
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # 获取电机数量
    num_motors = len(data)
    
    # 创建子图
    fig, axs = plt.subplots(4, 1, figsize=(12, 20))
    
    attributes = ['dof_pos', 'dof_vel', 'dof_tor', 'tar_dof_pos']
    titles = ['DOF Position', 'DOF Velocity', 'DOF Torque', 'Target DOF Position']
    
    for i, (attr, title) in enumerate(zip(attributes, titles)):
        for motor_id, motor_data in data.items():
            axs[i].plot(motor_data[attr], label=motor_id)
        axs[i].set_title(title)
        axs[i].set_xlabel('Time')
        axs[i].set_ylabel(attr)
        axs[i].legend()
    
    # 调整子图布局
    plt.tight_layout()
    
    # 保存图片或显示
    if output_file:
        plt.savefig(output_file)
        print(f"Plot saved to {output_file}")
    else:
        plt.show()

def trim_json_data(input_file, output_file, start_index):
    # 读取 JSON 文件
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # 对每个电机的每个属性进行裁剪
    for motor in data.values():
        for attr in ['dof_pos', 'dof_vel', 'dof_tor', 'tar_dof_pos']:
            motor[attr] = motor[attr][start_index:]
    
    # 将裁剪后的数据写入新的 JSON 文件
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)

def operation_func(input_file, output_file):
    trim_json_data(input_file, output_file, start_index=200)
    plot_json_data(output_file, output_file[:-5]+'.jpg')

def process_all_files_in_directory(input_dir, output_dir, operation_func):
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 遍历输入目录中的所有文件
    for filename in os.listdir(input_dir):
        if filename.endswith('.json'):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, f"processed_{filename}")
            
            # 对每个文件执行操作
            operation_func(input_path, output_path)

def merge_json_files(input_dir, output_file):
    merged_data = {}
    motor_count = 0

    for filename in os.listdir(input_dir):
        if filename.endswith('.json'):
            file_path = os.path.join(input_dir, filename)
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            for motor_data in data.values():
                merged_data[f"motor_{motor_count}"] = motor_data
                motor_count += 1

    with open(output_file, 'w') as f:
        json.dump(merged_data, f, indent=2)

    print(f"Merged {motor_count} motors into {output_file}")

if __name__ == "__main__":
    input_path = 'data_sets/many_data'
    output_path = 'data_sets/exploited'
    process_all_files_in_directory(input_path, output_path, operation_func)
    merge_json_files(output_path, 'data_sets/merged_motor_data_ultimate_pro_plus.json')