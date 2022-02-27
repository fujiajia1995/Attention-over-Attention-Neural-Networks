import torch
import numpy as np
from config import config
"""
x = torch.tensor([1, 2, 3])
y = torch.tensor([2, 3, 4])

a = list([x,y])
print(a)
print(x.shape)
b = torch.stack([x,y,y])
print(b.shape)

a = torch.randn(2,3,5)
b = torch.randn(2,3,6)
print(torch.cat((a,b),2).shape)


a = torch.randn((2,3,4))
b = torch.randn((2,4,3))
print(torch.matmul(a,b).shape)
"""
"""
random_list = np.random.permutation(config.data_size)
print(min(random_list))
test = torch.nn.Embedding(num_embeddings=3, embedding_dim=4)
print(test.weight.size())
a = [float(0) for i in range(10)]
a[1] = 1
print(a)

import numpy as np
print(np.zeros(10))
"""

import datetime
print(datetime.datetime.now())
print(10 % 5)
