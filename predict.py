import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

print("STARTING...")


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

print("LOADING MODEL...")

model.load_state_dict(
    torch.load("pneumonia_model.pth")
)

model.eval()



transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor()
])

print("LOADING IMAGE...")

img = Image.open("test.jpg.jpeg").convert("RGB")

img = transform(img)

img = img.unsqueeze(0)

print("PREDICTING...")


with torch.no_grad():

    output = model(img)

    prediction = torch.argmax(output, 1)

if prediction.item() == 0:
    print("NORMAL")
else:
    print("PNEUMONIA DETECTED")