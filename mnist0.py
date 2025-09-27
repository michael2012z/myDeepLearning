import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T

# Always use CPU for simplicity
device = torch.device("cpu")

# Load MNIST (download if needed)
transform = T.ToTensor()
train_set = torchvision.datasets.MNIST(root="./data", train=True,
                                       transform=transform, download=True)
test_set  = torchvision.datasets.MNIST(root="./data", train=False,
                                       transform=transform, download=True)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
test_loader  = torch.utils.data.DataLoader(test_set,  batch_size=64, shuffle=False)

# Define a small CNN
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)   # -> [16,28,28]
        self.pool  = nn.MaxPool2d(2, 2)               # -> [16,14,14]
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)  # -> [32,14,14]
        self.fc1   = nn.Linear(32 * 7 * 7, 128)
        self.fc2   = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))          # -> [32,7,7]
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)                            # logits

model = Net().to(device)

# Loss & Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(3):  # 3 epochs for quick demo
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)

    avg_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch {epoch+1}: train loss = {avg_loss:.4f}")

# Evaluation
model.eval()
correct = total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images.to(device))
        preds = outputs.argmax(1)
        total += labels.size(0)
        correct += (preds.cpu() == labels).sum().item()

print(f"Test accuracy: {100 * correct / total:.2f}%")
