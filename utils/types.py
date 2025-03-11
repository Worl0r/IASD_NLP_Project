from typing import Any

import torch
import torch.optim.optimizer
import torch.utils.data
from torch.nn import GELU, SELU, ReLU, Sigmoid
from torch.optim.adam import Adam

# Alias for Tensors
Tensor = torch.Tensor

# Alias for input data in the DataLoader (e.g., time series data)
# time_series_data: {'encoder_cont': Tensor, 'decoder_cont': Tensor, etc.}
TimeSeriesData = dict[str, Tensor]

# Alias for model configuration
Config = dict[str, Any]

# Alias for linear layer parameters (e.g., input and output dimensions)
LinearLayerParams = tuple[int, int]

# Alias for optimizer and learning rate scheduler
Optimizer = Adam | Any

# Scheduler
Scheduler = torch.optim.lr_scheduler._LRScheduler | Any

# Alias for PyTorch Dataset
Dataset = torch.utils.data.Dataset

# Alias for training or validation metrics (e.g., loss, accuracy, etc.)
Metrics = dict[str, float]

# Alias for activation functions
ActivationFunction = type[Sigmoid] | type[SELU] | type[GELU] | type[ReLU]

# Optional boolean
OptBool = Tensor | None

# Device
TorchDevice = torch.device
