import torch
from torch import nn
from model import DenseNetMoE 


# Initialize the DenseNetMoE model
dim = 64
growth_rate = 32
num_dense_blocks = 2
num_layers_per_block = 2
num_experts = 4
model = DenseNetMoE(dim, growth_rate, num_dense_blocks, num_layers_per_block, num_experts)

# Prepare input tensor of shape (batch_size, sequence_length, dim)
batch_size = 128
sequence_length = 100
input_tensor = torch.randn(batch_size, sequence_length, dim)

# Pass input tensor through the model
output_tensor = model(input_tensor)

# The shape of the output tensor will be (batch_size, sequence_length, dim)
print(output_tensor.shape)
