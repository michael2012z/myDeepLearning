import torch
import torch.nn as nn
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

# Step 1: Reproducibility and data generation (true coefficients)
torch.manual_seed(42)
n = 300
X = torch.linspace(-2.0, 2.0, n).view(-1, 1).float()

# True coefficients for y = a x^5 + b x^4 + c x^3 + d x^2 + e x + f
a_true, b_true, c_true, d_true, e_true, f_true = 0.5, -1.0, 0.8, -0.5, 2.0, 1.0

# Generate noisy data
noise = 0.3 * torch.randn_like(X)
y = (
    a_true * X**5
    + b_true * X**4
    + c_true * X**3
    + d_true * X**2
    + e_true * X
    + f_true
    + noise
)

# Step 2: Build polynomial features [x, x^2, x^3, x^4, x^5]
X_poly = torch.cat([X**i for i in range(1, 6)], dim=1)  # shape [n, 5]

# Step 3: Define the polynomial regression model
class Poly5Reg(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(5, 1)  # 5 input features: x^1..x^5

    def forward(self, x):
        return self.lin(x)

# Step 4: Training helper (returns trained model and final loss)
def train(num_epochs, lr=0.01):
    model = Poly5Reg()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for _ in range(num_epochs):
        y_pred = model(X_poly)
        loss = criterion(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model, float(loss.item())

# Step 5: Train four models for different epochs
epoch_list = [10, 50, 200, 800]
models, losses = [], []
for ep in epoch_list:
    m, L = train(ep)
    models.append(m)
    losses.append(L)

# Step 6: Smooth grid for plotting
x_grid = torch.linspace(-2.0, 2.0, 400).view(-1, 1)
xg_poly = torch.cat([x_grid**i for i in range(1, 6)], dim=1).float()

# Step 7: Plot results (blue data, red fit)
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
axes = axes.flatten()

for i, ax in enumerate(axes):
    ax.scatter(X.numpy(), y.numpy(), s=10, color='blue', label='data')

    with torch.no_grad():
        y_fit = models[i](xg_poly)
    ax.plot(x_grid.numpy(), y_fit.numpy(), color='red', linewidth=2, label='fit')

    w = models[i].lin.weight.detach().view(-1).numpy()
    f_hat = models[i].lin.bias.detach().item()
    title = f"{epoch_list[i]} epochs | loss={losses[i]:.4f}"
    ax.set_title(title)
    ax.legend()

plt.tight_layout()
plt.show()

# Step 8: Print true vs estimated coefficients for the most-trained model
w_final = models[-1].lin.weight.detach().view(-1).numpy()
f_hat = float(models[-1].lin.bias.detach().item())
print("True coefficients:")
print(f"a={a_true}, b={b_true}, c={c_true}, d={d_true}, e={e_true}, f={f_true}")
print("Estimated (800 epochs):")
print(
    "a=%.3f, b=%.3f, c=%.3f, d=%.3f, e=%.3f, f=%.3f"
    % (w_final[4], w_final[3], w_final[2], w_final[1], w_final[0], f_hat)
)
