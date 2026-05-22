from flask import Flask, render_template, request
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

app = Flask(__name__)

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

# Load trained model
model.load_state_dict(
    torch.load("pneumonia_model.pth")
)

model.eval()



transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor()
])



@app.route("/", methods=["GET", "POST"])

def home():

    prediction = ""

    if request.method == "POST":

        file = request.files["file"]

        img = Image.open(file).convert("RGB")

        img = transform(img)

        img = img.unsqueeze(0)

        with torch.no_grad():

            output = model(img)

            probs = torch.softmax(output, dim=1)

            confidence, pred = torch.max(probs, 1)

        if pred.item() == 0:
            prediction = f"NORMAL ({confidence.item()*100:.2f}%)"
        else:
            prediction = f"PNEUMONIA DETECTED ({confidence.item()*100:.2f}%)"

    return render_template(
        "index.html",
        prediction=prediction
    )



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
