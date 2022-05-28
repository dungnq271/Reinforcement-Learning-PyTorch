import torch
from torch.distributions import Categorical
import numpy as np

a = np.random.randint(5, size=(2, 2, 1))
print(np.repeat(a, 3, axis=2).shape)
