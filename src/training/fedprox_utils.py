"""FedProx regularization utilities for zk0."""

import torch


def compute_fedprox_proximal_loss(trainable_params, global_params, fedprox_mu):
    """Compute FedProx proximal regularization loss.

    Args:
        trainable_params: List of torch tensors (current model parameters)
        global_params: List of numpy arrays (global model parameters from server)
        fedprox_mu: FedProx regularization coefficient

    Returns:
        torch.Tensor: Proximal loss tensor (for consistent backprop with main_loss)
    """
    if global_params is None or fedprox_mu <= 0 or not trainable_params:
        return torch.tensor(
            0.0,
            device=trainable_params[0].device if trainable_params else "cpu",
            dtype=torch.float32,
        )

    proximal_loss = torch.tensor(
        0.0, device=trainable_params[0].device, dtype=trainable_params[0].dtype
    )
    for param, global_param in zip(trainable_params, global_params):
        # Convert global param to same device/dtype as current param
        global_param_tensor = torch.from_numpy(global_param).to(
            param.device, dtype=param.dtype
        )
        param_diff = torch.sum((param - global_param_tensor) ** 2)
        proximal_loss += param_diff

    return (fedprox_mu / 2.0) * proximal_loss