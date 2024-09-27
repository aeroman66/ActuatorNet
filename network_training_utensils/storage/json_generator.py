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

if __name__ == "__main__":
    # # 生成数据
    # file_ori = 'data_sets/go1_dataset_x0.25.json'
    # json_file = 'merged_motor_data.json'

    # with open(file_ori, 'r') as f:
    #     data = json.load(f)

    # merged_data = merge_motors(data)
    # with open(json_file, 'w') as f:
    #     json.dump(merged_data, f, indent=2)

    # # # 将数据写入 JSON 文件
    # # with open(json_file, 'w') as f:
    # #     json.dump(motor_data, f, indent=2) # dump 的时候一定要指定 indent，不然整个 json 文件全部在一行上，可读性非常差。

    # print("JSON file 'motor_data.json' has been created.")

    # file_path_ori = "data_sets/merged_motor_data.json"
    # output_file = "data_sets/merged_motor_data_jpg.jpg"
    file_path_ori = "data_sets/origin/go1_dataset_x0.25.json"
    file_path_trim = "data_sets/trim/go1_dataset_x0.25_trim.json"
    file_path_smooth = "data_sets/smooth/go1_dataset_x0.25_smooth.json"
    file_path_merged = "data_sets/merged_motor_data_exploited.json"
    output_img_file_ori = "data_sets/img/go1_dataset_x0.25_jpg.jpg"
    output_img_file_trim = "data_sets/img/go1_dataset_x0.25_trim_jpg.jpg"
    output_img_file_smooth = "data_sets/img/go1_dataset_x0.25_smooth_jpg.jpg"
    output_img_file_smooth_all = "data_sets/img/go1_dataset_x0.25_smooth_all_jpg.jpg"
    output_img_file_merged = "data_sets/img/go1_dataset_x0.25_merged_jpg.jpg"

    # 绘制数据
    plot_json_data(file_path_ori, output_img_file_ori)
    # 裁剪数据
    trim_json_data(file_path_ori, file_path_trim, 250)
    # 绘制数据
    plot_json_data(file_path_trim, output_img_file_trim)
    # 平滑数据
    smooth_json_data(file_path_trim, file_path_smooth)
    # 绘制数据
    plot_json_data(file_path_smooth, output_img_file_smooth)
    plot_all_motors_data(file_path_smooth, output_img_file_smooth_all)

    # # 合并数据
    # merge_motors(file_path_smooth, file_path_merged)
    # # 绘制数据
    # plot_json_data(file_path_merged, output_img_file_merged)