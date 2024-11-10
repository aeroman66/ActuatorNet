import torch
import torch.nn as nn

class ActuatorNetRNN(nn.Module):
    def __init__(self,
                 input_size=2,
                 output_size=1,
                 hidden_size=64, # h_t 的维度
                 num_layers=2, # 定义了深度 LSTM
                 init_noise_std=0.01):
        super().__init__()
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        
        # Output layer
        self.output_layer = nn.Linear(hidden_size, output_size)
        
        # Input noise parameter
        self.input_noise = nn.Parameter(init_noise_std * torch.ones(input_size))
        
        # Hidden state
        self.hidden = None
        
        print(f"actuator net: LSTM(input_size={input_size}, hidden_size={hidden_size}, num_layers={num_layers})")

    def forward(self, x):
        # x shape should be (batch_size, sequence_length, input_size)
        
        # lstm_out, self.hidden = self.lstm(x, self.hidden)
        # output = self.output_layer(lstm_out)  # Take the last output
        # return output

        lstm_out, new_hidden = self.lstm(x, self.hidden)
        self.hidden = tuple(h.detach() for h in new_hidden)  # 分离隐藏状态的梯度
        output = self.output_layer(lstm_out)
        return output

    
    def reset(self, dones=None):
        # Reset hidden states
        self.hidden = None
        
    def act_inference(self, obs):
        # Ensure input is tensor and has correct shape (batch_size, sequence_length, input_features)
        if not isinstance(obs, torch.Tensor):
            obs = torch.Tensor(obs)
        if len(obs.shape) == 2:
            obs = obs.unsqueeze(0)  # Add batch dimension if needed
        return self.forward(obs)

       

# # *************Utensils*************
# def get_activation(act_name):
#     if act_name == "elu":
#         return nn.ELU()
#     elif act_name == "selu":
#         return nn.SELU()
#     elif act_name == "relu":
#         return nn.ReLU()
#     elif act_name == "crelu":
#         return nn.CReLU()
#     elif act_name == "lrelu":
#         return nn.LeakyReLU()
#     elif act_name == "tanh":
#         return nn.Tanh()
#     elif act_name == "sigmoid":
#         return nn.Sigmoid()
#     else:
#         print("invalid activation function!")
#         return None