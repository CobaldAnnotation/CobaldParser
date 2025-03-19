from torch import nn


def get_activation_fn(activation_name: str) -> callable:
    act2fn = {
        "relu": nn.ReLU(),
        "leaky_relu": nn.LeakyReLU(),
        "sigmoid": nn.Sigmoid(),
        "tanh": nn.Tanh()
    }

    if activation_name not in act2fn:
        raise ValueError(f"activation must be one of {act2fn.keys()}, got {activation_name}.")
    return act2fn[activation_name]