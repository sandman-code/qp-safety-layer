import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# -----------------------------
# 1. Simple neural network
# -----------------------------
class MLP(nn.Module):
    def __init__(self, in_dim=1, hidden_dim=32, out_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


# -----------------------------
# 2. Flatten parameter gradients
# -----------------------------
def output_gradient(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """
    Compute grad_theta f_theta(x) as a single flat vector.

    Args:
        model: neural network
        x: shape (1, input_dim)

    Returns:
        grad vector of shape (num_params,)
    """
    model.zero_grad(set_to_none=True)

    y = model(x)  # shape (1, 1)
    # Assume scalar output for simplicity
    y.backward()

    grads = []
    for param in model.parameters():
        if param.grad is None:
            grads.append(torch.zeros_like(param).reshape(-1))
        else:
            grads.append(param.grad.reshape(-1))
    return torch.cat(grads)


# -----------------------------
# 3. NTK between two inputs
# -----------------------------
def ntk_value(model: nn.Module, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """
    k_NTK(x1, x2) = <grad_theta f(x1), grad_theta f(x2)>
    """
    g1 = output_gradient(model, x1)
    g2 = output_gradient(model, x2)
    return torch.dot(g1, g2)


# -----------------------------
# 4. NTK Gram matrix for a set
# -----------------------------
def ntk_matrix(model: nn.Module, X1: torch.Tensor, X2: torch.Tensor) -> torch.Tensor:
    """
    Build kernel matrix K where K[i, j] = k_NTK(X1[i], X2[j])

    Args:
        X1: shape (N, input_dim)
        X2: shape (M, input_dim)

    Returns:
        K: shape (N, M)
    """
    grads1 = [output_gradient(model, X1[i : i + 1]) for i in range(X1.shape[0])]
    grads2 = [output_gradient(model, X2[j : j + 1]) for j in range(X2.shape[0])]

    G1 = torch.stack(grads1)  # (N, P)
    G2 = torch.stack(grads2)  # (M, P)

    return G1 @ G2.T


# -----------------------------
# 5. GP-style posterior variance from NTK
# -----------------------------
def ntk_posterior_variance(
    model: nn.Module,
    X_train: torch.Tensor,
    X_test: torch.Tensor,
    noise_std: float = 1e-2,
) -> torch.Tensor:
    """
    Compute posterior predictive variance using NTK as the GP kernel.

    Var(test) = diag(K_tt - K_tT (K_TT + sigma^2 I)^-1 K_Tt)

    Args:
        model: trained neural net
        X_train: shape (N, input_dim)
        X_test: shape (M, input_dim)
        noise_std: observation noise

    Returns:
        variances: shape (M,)
    """
    K_TT = ntk_matrix(model, X_train, X_train)  # (N, N)
    K_tT = ntk_matrix(model, X_test, X_train)  # (M, N)
    K_tt = ntk_matrix(model, X_test, X_test)  # (M, M)

    N = X_train.shape[0]
    noise = (noise_std**2) * torch.eye(N, dtype=K_TT.dtype)

    K_inv = torch.linalg.inv(K_TT + noise)
    posterior_cov = K_tt - K_tT @ K_inv @ K_tT.T

    return torch.diagonal(posterior_cov)


# -----------------------------
# 6. Example usage
# -----------------------------
if __name__ == "__main__":
    device = (
        torch.accelerator.current_accelerator().type
        if torch.accelerator.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    torch.manual_seed(0)

    # Training data
    X_train = torch.tensor([[-2.0], [-1.0], [0.0], [1.0], [2.0]])
    y_train = torch.sin(X_train)

    # Test points
    X_test = torch.linspace(-4, 4, steps=25).unsqueeze(1)
    y_test = torch.sin(X_test)

    # Create and train model a little
    model = MLP(in_dim=1, hidden_dim=32, out_dim=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = nn.MSELoss()

    for step in range(500):
        optimizer.zero_grad()
        pred = model(X_train)
        loss = loss_fn(pred, y_train)
        loss.backward()
        optimizer.step()

    # Example: kernel between two points
    x_a = torch.tensor([[0.5]])
    x_b = torch.tensor([[0.6]])
    k_ab = ntk_value(model, x_a, x_b)
    print("NTK(0.5, 0.6) =", k_ab.item())

    # Uncertainty over test points
    var_test = ntk_posterior_variance(model, X_train, X_test, noise_std=1e-2)

    print("\nTest point variances:")
    for x, v in zip(X_test.squeeze(), var_test):
        print(f"x={x.item(): .2f}, var={v.item():.6f}")
