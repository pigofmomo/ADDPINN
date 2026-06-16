"""Global configuration defaults for ADD-PINN experiments.
ADD-PINN 实验的全局默认配置。
"""

import random
import numpy as np
import torch

lr_decay = ["step", 1000, 0.9]

random_seed = 0

# Gradient-analysis controls. / 梯度分析控制项。
min_iteration = 10
max_iteration = 50
probe_grad_period = 5
ratio_threshold = 0.5
window_length = 3

static_frequency = False
frequency = 1

# Pretraining iterations. / 预训练迭代次数。
pretrain_iterations = 1000
# Adaptive subdomain update schedule. / 自适应子区域更新计划。
adjust_subdomain_start_iteration = 10000
adjust_subdomain_end_iteration = 15000
adjust_subdomain_period = 2000

# Graph-optimization settings. / 图优化设置。
boundary_length_tolerance = 10
max_iter_times_weighted_balance = 10
lambda1 = 0.001

max_iter_times_weighted_min = 5
boundary_length_tolerance2 = 5



def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)