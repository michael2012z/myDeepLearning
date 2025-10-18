import torch
import torch.nn as nn
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

# ===== Config =====
DEGREE = 6                 # <-- set any integer >= 1
EPOCHS_LIST = [10, 50, 200, 800]
N_SAMPLES = 300
X_MIN, X_MAX = -2.0, 2.0
NOISE_STD = 0.3
LR = 0.01
SEED = 42

# Step 1: Reproducibility and synthetic data
torch.manual_seed(SEED)
X = torch.linspace(X_MIN, X_MAX, N_SAMPLES).view(-1, 1).float()

# True coefficients for y = sum_{k=0..DEGREE} coef[k] * x^k
# coef[0] is constant term; coef[1] multiplies x^1; ...; coef[DEGREE] multiplies x^DEGREE
true_coefs = torch.randn(DEGREE + 1)  # random but reproducible due to seed

with torch.no_grad():
    y_clean = sum(true_coefs[k] * (X ** k) for k in range(DEGREE + 1))
    noise = NOISE_STD * torch.randn_like(X)
    y = y_clean + noise

# Step 2: Build polynomial features [x^1, x^2, ..., x^DEGREE]
def poly_features(x, degree: int):
    feats = [x ** k for k in range(1, degree + 1)]
    return torch.cat(feats, dim=1)

X_poly = poly_features(X, DEGREE).float()

# Step 3: Define the polynomial regressor (linear over polynomial features)
class PolyReg(nn.Module):
    def __init__(self, degree: int):
        super().__init__()
        self.lin = nn.Linear(degree, 1)  # weights map [x^1..x^degree] -> y

    def forward(self, feats):
        return self.lin(feats)

# Step 4: Training helper (returns trained model and final loss)
def train(num_epochs, lr=LR):
    model = PolyReg(DEGREE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(num_epochs):
        y_pred = model(X_poly)
        loss = criterion(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model, float(loss.item())

# Step 5: Train models for different epoch counts
models, losses = [], []
for ep in EPOCHS_LIST:
    m, L = train(ep)
    models.append(m)
    losses.append(L)

# Step 6: Smooth grid for plotting
x_grid = torch.linspace(X_MIN, X_MAX, 500).view(-1, 1)
xg_poly = poly_features(x_grid, DEGREE).float()

# Step 7: Plot results (blue data, red fit)
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
axes = axes.flatten()

for i, ax in enumerate(axes):
    # data as blue dots
    ax.scatter(X.numpy(), y.numpy(), s=10, color='blue', label='data')

    # fitted curve as red line
    with torch.no_grad():
        y_fit = models[i](xg_poly)
    ax.plot(x_grid.numpy(), y_fit.numpy(), color='red', linewidth=2, label='fit')

    # extract learned coefficients: weights correspond to x^1..x^DEGREE; bias is constant term
    w = models[i].lin.weight.detach().view(-1).numpy()      # length = DEGREE
    bias_hat = models[i].lin.bias.detach().item()

    title = f"{EPOCHS_LIST[i]} epochs | loss={losses[i]:.4f}"
    ax.set_title(title)
    ax.legend()

plt.tight_layout()
plt.show()

# Step 8: Print true vs. estimated coefficients from most-trained model
w_final = models[-1].lin.weight.detach().view(-1).numpy()
bias_hat = float(models[-1].lin.bias.detach().item())

print("=== True coefficients (coef[k] * x^k) ===")
for k in range(DEGREE, -1, -1):
    print(f"k={k}: {float(true_coefs[k]): .4f}")

print("\n=== Estimated coefficients at {} epochs ===".format(EPOCHS_LIST[-1]))
# Bias (constant term)
print(f"k=0: {bias_hat: .4f}")
# Weights map to k=1..DEGREE in order
for k in range(1, DEGREE + 1):
    print(f"k={k}: {w_final[k-1]: .4f}")
