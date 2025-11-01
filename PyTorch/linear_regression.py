import torch
import torch.nn as nn
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

# Step 1: Generate synthetic data: y = 2x + 1 + noise
X = torch.linspace(0, 10, 200).unsqueeze(1)
y = 2 * X + 1 + 0.5 * torch.randn(X.size())

# Step 2: Define a simple linear regression model
class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)

# Step 3: Define a training helper function
def train_model(num_epochs):
    model = LinearModel()
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    for epoch in range(num_epochs):
        y_pred = model(X)
        loss = criterion(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model, loss.item()

# Step 4: Train models for different epoch counts
epoch_list = [10, 20, 40, 80]
models = []
losses = []

for e in epoch_list:
    model, loss = train_model(e)
    models.append(model)
    losses.append(loss)

# Step 5: Plot results
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes = axes.flatten()

for i, ax in enumerate(axes):
    ax.scatter(X, y, color='blue', s=10, label='Data')
    ax.plot(X, models[i](X).detach(), color='red', label=f'Fitted (epoch={epoch_list[i]})')
    w, b = models[i].linear.weight.item(), models[i].linear.bias.item()
    ax.set_title(f"{epoch_list[i]} Epochs\nLoss={losses[i]:.3f}, w={w:.2f}, b={b:.2f}")
    ax.legend()

plt.tight_layout()
plt.show()
