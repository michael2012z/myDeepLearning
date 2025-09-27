import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

# ToTensor() converts a PIL image (0–255) to a FloatTensor [0,1]
transform = transforms.ToTensor()

# Download MNIST dataset
train_dataset = torchvision.datasets.MNIST(
    root="./data", train=True, transform=transform, download=True
)
test_dataset = torchvision.datasets.MNIST(
    root="./data", train=False, transform=transform, download=True
)

print("Train set size :", len(train_dataset))
print("Test set size  :", len(test_dataset))
print("Sample shape   :", train_dataset[0][0].shape)  # (1,28,28)

# Create a DataLoader to batch the data
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=32, shuffle=True
)

# Peek at a single batch to visualize the data
images, labels = next(iter(train_loader))
print("Batch images shape :", images.shape)  # [32,1,28,28]
print("Batch labels shape :", labels.shape)  # [32]

# Visualize a batch of examples
def show_examples(images, labels):
    plt.figure(figsize=(20, 10))
    for i in range(4):
        for j in range(8):
            index = i * 8 + j
            plt.subplot(4, 8, index + 1)
            plt.imshow(images[index][0], cmap="gray")
            plt.title(f"{labels[index].item()}")
            plt.axis("off")
    plt.show()

show_examples(images, labels)
