import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ======================
# IMAGE TRANSFORMS
# ======================

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

train_data = datasets.ImageFolder(
    "chest_xray/train",
    transform=transform
)

train_loader = DataLoader(
    train_data,
    batch_size=16,
    shuffle=True
)

model = nn.Sequential(

    nn.Conv2d(3, 16, 3),
    nn.ReLU(),
    nn.MaxPool2d(2),

    nn.Conv2d(16, 32, 3),
    nn.ReLU(),
    nn.MaxPool2d(2),

    nn.Flatten(),

    nn.Linear(32 * 30 * 30, 64),

    nn.ReLU(),

    nn.Linear(64, 2)
)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)

epochs = 10

for epoch in range(epochs):

    running_loss = 0

    for images, labels in train_loader:

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {running_loss}")


torch.save(
    model.state_dict(),
    "pneumonia_model.pth"
)

print("MODEL SAVED")