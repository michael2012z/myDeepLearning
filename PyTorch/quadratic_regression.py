import torch
import torch.nn as nn
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

# Step 1: Reproducibility and data generation (true: a=1.5, b=-3.0, c=2.0)
torch.manual_seed(42)
n = 200
X = torch.linspace(-5.0, 5.0, n).view(-1, 1).float()  # shape [n, 1]
a_true, b_true, c_true = 1.5, -3.0, 2.0
noise = 0.5 * torch.randn_like(X)                     # moderate noise
y = a_true * X**2 + b_true * X + c_true + noise       # shape [n, 1]

# Step 2: Build polynomial features [x, x^2]
X_poly = torch.cat([X, X**2], dim=1).float()          # shape [n, 2]

# Step 3: Define a simple quadratic regressor (linear over [x, x^2])
class QuadReg(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(2, 1)   # weights map [x, x^2] -> y

    def forward(self, x):
        return self.lin(x)

# Step 4: Training helper (returns trained model and final loss)
def train(num_epochs, lr=0.02):
    model = QuadReg()
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

# Step 6: Prepare a smooth curve for plotting predictions
# (Using the same X works too, but a dense grid makes the line silky-smooth)
x_grid = torch.linspace(-5.0, 5.0, 400).view(-1, 1)
xg_poly = torch.cat([x_grid, x_grid**2], dim=1).float()

# Step 7: Plot four subplots showing improvement with more epochs
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
axes = axes.flatten()

for i, ax in enumerate(axes):
    # scatter the noisy observations
    ax.scatter(X.numpy(), y.numpy(), s=10, color='blue', label="data")

    # model prediction on the smooth grid
    with torch.no_grad():
        y_fit = models[i](xg_poly)

    ax.plot(x_grid.numpy(), y_fit.numpy(), color='red', label=f"fit @ {epoch_list[i]} epochs")

    # extract learned coefficients: weight = [w_x, w_x2], bias = c
    w = models[i].lin.weight.detach().view(-1).numpy()
    c_hat = models[i].lin.bias.detach().item()
    a_hat, b_hat = float(w[1]), float(w[0])  # order matches [x, x^2] input

    title = (f"epochs={epoch_list[i]} | loss={losses[i]:.3f}\n"
             f"estimated: a={a_hat:.2f}, b={b_hat:.2f}, c={c_hat:.2f}")
    ax.set_title(title)
    ax.legend()

plt.tight_layout()
plt.show()

# Step 8: Print the true vs. estimated parameters from the most-trained model
w_final = models[-1].lin.weight.detach().view(-1).numpy()
a_hat, b_hat = float(w_final[1]), float(w_final[0])
c_hat = float(models[-1].lin.bias.detach().item())
print("True params:     a=%.2f, b=%.2f, c=%.2f" % (a_true, b_true, c_true))
print("Estimated (800): a=%.2f, b=%.2f, c=%.2f" % (a_hat, b_hat, c_hat))
