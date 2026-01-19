import torch
import torch.nn as nn

def sgd_linear(X, y):
    num_samples, num_features = X.shape
    num_outputs = y.shape[1] if y.dim() > 1 else 1
    model = nn.Linear(in_features=num_features, out_features=num_outputs)
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    for epoch in range(10):
        y_pred = model(X)
        loss = criterion(y_pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Iterative Result: Weight={model.weight.item():.4f},Bias={model.bias.item():.4f}")

def sgd_linear_manual(X, y):
    lr = 0.01
    n_features = X.shape[1]
    # pytorch will keep computational graph history to allow for gradient calculation
    theta = torch.randn((n_features, 1), requires_grad=True)

    for epoch in range(10):
        y_pred = X.matmul(theta)
        loss = torch.mean((y_pred - y)**2)
        # uses the .grad attribute on each tensor
        loss.backward()
        # do not track the update computation
        with torch.no_grad():
            theta -= lr * theta_grad
            # zero the gradient for the next step
            theta.grad.zero_()

    # detach creates a view detached from computational graph no longer
    # requires grad
    return theta.detach().flatten()


def normal_equation_linear(X, y):
    # add column of 1s to X for theta_0 intercept
    m = X.shape[0]
    X_b = torch.cat([torch.ones((m,1)), X], dim=1)

    # ensure y column vector
    if y.dim() == 1:
        y = y.view(-1,1)

    # X^T * X
    XTX = torch.t(X_b) @ X_b
    # pinv handles non-invertable matrices and is numerically stable
    XTX_inv = torch.linalg.pinv(XTX)

    # (XTX^-1) * X^T * y
    theta = XTV_inv @ torch.t(X_b) @ y
    return torch.round(theta.flatten(), decimals=4)

