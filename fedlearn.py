
import torch

import hashlib

from typing import Dict, List, Optional, Tuple, Any
from collections import OrderedDict
import numpy as np
import numpy.typing as npt
from functools import reduce
from io import BytesIO
from dataclasses import dataclass
from typing import cast

NDArray = npt.NDArray[Any]
NDArrayInt = npt.NDArray[np.int_]
NDArrayFloat = npt.NDArray[np.float_]
NDArrays = List[NDArray]

def add_percentage_gaussian_noise_to_model(model, device, percentage=0.2):
    with torch.no_grad():
        for param in model.parameters():
            # Calculate the standard deviation as a percentage of the parameter's own standard deviation
            std_dev = param.data.std().item()
            noise_std = percentage * std_dev
            # Generate and add the Gaussian noise
            param.add_(torch.randn(param.size(), device=device) * noise_std)

def add_constant_gaussian_noise_to_model(model, device, std=0.01):
    with torch.no_grad():
        for param in model.parameters():
            param.add_(torch.randn(param.size()).to(device) * std)

def compress_parameters(parameters: List[np.ndarray], compression_rate) -> List[np.ndarray]:
    # Flatten all parameters to a single array
    all_params = np.concatenate([p.flatten() for p in parameters])
    
    # Determine the threshold value for the top 'compression_rate' fraction of parameters
    if compression_rate > 0:
        threshold = np.quantile(np.abs(all_params), 1 - compression_rate)
    else:
        # If compression_rate is 0, we keep no parameters (everything is set to 0)
        threshold = np.inf
    
    # Apply the threshold to each parameter matrix
    compressed_parameters = []
    for param in parameters:
        # Use np.where to keep values above the threshold, else set to 0
        compressed_param = np.where(np.abs(param) >= threshold, param, 0)
        compressed_parameters.append(compressed_param)
    
    return compressed_parameters

def sha256_hash(data):
    hash_object = hashlib.sha256()
    hash_object.update(data)
    
    return hash_object.hexdigest()

@dataclass
class Parameters:
    """Model parameters."""

    tensors: List[bytes]
    tensor_type: str

def ndarray_to_bytes(ndarray: NDArray) -> bytes:
    """Serialize NumPy ndarray to bytes."""
    bytes_io = BytesIO()
    # WARNING: NEVER set allow_pickle to true.
    # Reason: loading pickled data can execute arbitrary code
    # Source: https://numpy.org/doc/stable/reference/generated/numpy.save.html
    np.save(bytes_io, ndarray, allow_pickle=False)
    return bytes_io.getvalue()

def bytes_to_ndarray(tensor: bytes) -> NDArray:
    """Deserialize NumPy ndarray from bytes."""
    bytes_io = BytesIO(tensor)
    # WARNING: NEVER set allow_pickle to true.
    # Reason: loading pickled data can execute arbitrary code
    # Source: https://numpy.org/doc/stable/reference/generated/numpy.load.html
    ndarray_deserialized = np.load(bytes_io, allow_pickle=False)
    return cast(NDArray, ndarray_deserialized)

def ndarrays_to_parameters(ndarrays: NDArrays) -> Parameters:
    """Convert NumPy ndarrays to parameters object."""
    tensors = [ndarray_to_bytes(ndarray) for ndarray in ndarrays]
    return Parameters(tensors=tensors, tensor_type="numpy.ndarray")

def parameters_to_ndarrays(parameters: Parameters) -> NDArrays:
    """Convert parameters object to NumPy ndarrays."""
    return [bytes_to_ndarray(tensor) for tensor in parameters.tensors]

def get_parameters(net) -> List[np.ndarray]:
        return [val.cpu().numpy() for _, val in net.state_dict().items()]

def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v.astype(float)) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)

def fedavg_aggregate(results: List[Tuple[NDArrays, int]]) -> NDArrays:
    """Compute weighted average."""
    # Calculate the total number of examples used during training
    num_examples_total = sum([num_examples for _, num_examples in results])

    # Create a list of weights, each multiplied by the related number of examples
    weighted_weights = [
        [layer * num_examples for layer in weights] for weights, num_examples in results
    ]

    # Compute average weights of each layer
    weights_prime: NDArrays = [
        reduce(np.add, layer_updates) / num_examples_total
        for layer_updates in zip(*weighted_weights)
    ]
    return parameters_to_ndarrays(ndarrays_to_parameters(weights_prime))