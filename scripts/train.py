import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # 不清楚为什么我的根目录需要自己添加

from network_training_utensils.storage.json_generator import generate_motor_data
from scripts import *

def main():
    # 生成数据
    motor_data = generate_motor_data()
    json_file = 'motor_data.json'
    print('Successfully excute the script!')

if __name__ == "__main__":
    main()