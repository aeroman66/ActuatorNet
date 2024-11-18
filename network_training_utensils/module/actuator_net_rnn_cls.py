import torch
import torch.nn as nn

class ActuatorNetRNNDIS(nn.Module):
    def __init__(self,
                 input_size=2,
                 output_size=1,
                 num_classes=1000,  # 改为分类数量
                 hidden_size=64,
                 num_layers=2,
                 init_noise_std=0.01):
        super().__init__()
        self.output_size = output_size
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        
        # 修改输出层为分类层
        self.output_layer = nn.Sequential(
            # nn.Linear(hidden_size, 1000),
            # nn.Linear(1000, num_classes),
            nn.Linear(hidden_size, num_classes),
            nn.LogSoftmax(dim=-1)  # 使用 LogSoftmax 获得概率分布
        )
        
        self.input_noise = nn.Parameter(init_noise_std * torch.ones(input_size))
        self.hidden = None
        
        # 生成对应的离散值标签
        self.discrete_values = torch.linspace(-15, 15, num_classes)
        
        print(f"actuator net: LSTM(input_size={input_size}, hidden_size={hidden_size}, num_layers={num_layers}, num_classes={num_classes})")

    def forward(self, x):
        lstm_out, new_hidden = self.lstm(x, self.hidden)
        self.hidden = tuple(h.detach() for h in new_hidden)
        logits = self.output_layer(lstm_out)  # 输出概率分布
        return logits
    
    def reset(self, dones=None):
        self.hidden = None
        
    def act_inference(self, obs):
        if not isinstance(obs, torch.Tensor):
            obs = torch.Tensor(obs)
        if len(obs.shape) == 2:
            obs = obs.unsqueeze(0)
            
        logits = self.forward(obs)
        probabilities = torch.exp(logits)  # Convert log probabilities back to probabilities
        
        # Get top k probabilities and indices
        top_probs, top_indices = torch.topk(probabilities, k=self.output_size, dim=-1)
        # Convert indices to actual values
        top_values = self.discrete_values[top_indices]
        
        return top_values