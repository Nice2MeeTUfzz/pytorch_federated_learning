import torch
import numpy as np
from utils.fed_utils import init_model

np.random.seed(123)
model = init_model(model_name="LeNet", num_class=10, image_channel=1)
model.load_state_dict(torch.load('../model_state.pth'))


def pytorch_count_params(model):
    """count number trainable parameters in a pytorch model"""
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params


highest_range = np.finfo('float16').max
shares_dict = {}
num_servers = 2
all_servers = []
for server_index in range(num_servers):
    all_servers.append({})
# 初始化存储每个参数份额的字典
shares_dict = {param_name: np.zeros((num_servers,) + param_tensor.shape, dtype=np.float64) for param_name, param_tensor
               in model.state_dict().items()}

# 模型的每个参数
for param_name, param_tensor in model.state_dict().items():
    # 将参数 tensor 转换为 numpy 数组
    original_param = param_tensor.numpy()

    # 为每个服务器生成随机份额，除了最后一个
    for server_index in range(num_servers - 1):
        share = np.random.uniform(
            low=-highest_range, high=highest_range, size=original_param.shape).astype(np.float64)
        shares_dict[param_name][server_index] = share

    # 计算最后一个服务器的份额
    share_sum_except_last = np.sum(shares_dict[param_name][:num_servers - 1], axis=0)
    last_share = original_param - share_sum_except_last
    shares_dict[param_name][num_servers - 1] = last_share

# all_servers中是每个server分到的份数
for server_index in range(num_servers):
    for layer_index in shares_dict:
        all_servers[server_index][layer_index] = shares_dict[layer_index][server_index]

assert all_servers[0].keys() == all_servers[1].keys()

combined_shares = {}

for layer_index, layer_share in all_servers[0].items():
    # 检查第二个服务器是否有对应的层索引
    if layer_index in all_servers[1]:
        # 将两个服务器的权重份额相加
        combined_shares[layer_index] = all_servers[0][layer_index] + all_servers[1][layer_index]
    else:
        raise KeyError(f"Layer index {layer_index} not found on both servers")

print(combined_shares)
