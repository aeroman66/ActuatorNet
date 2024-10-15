import torch
import torch.nn as nn

class ActuatorNet(nn.Module):
    def __init__(self,
                 input_size,
                 output_size,
                 hidden_size=[128, 64, 32, 16],
                 activation='elu',
                 init_noise_std=0.01):
        super().__init__()
        activation = get_activation(activation)

        # ********net**********
        net = []
        net.append(nn.Linear(input_size, hidden_size[0]))
        net.append(activation)
        for i in range(len(hidden_size) - 1):
            net.append(nn.Linear(hidden_size[i], hidden_size[i + 1]))
            net.append(activation)
        net.append(nn.Linear(hidden_size[-1], output_size))
        self.net = nn.Sequential(*net)
        self.net.float()

        print(f"actuator net: {self.net}")

        # ********input_noise**********
        self.input_noise = nn.Parameter(init_noise_std * torch.ones(input_size))

    def init_weights(self, sequential, scales):
        """可在这选择实现定制化的参数初始化方法
        """
        raise NotImplementedError
    
    def reset(self, dones=None):
        """重置网络参数
        """
        pass

    def forward(self):
        raise NotImplementedError
    
    def act_inference(self, obs):
        obs = torch.Tensor(obs)
        return self.net(obs)
       

# *************Utensils*************
def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.CReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None