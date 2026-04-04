import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2

class CourtKeypointRegressor(nn.Module):
    def __init__(self, num_keypoints=12):
        super(CourtKeypointRegressor, self).__init__()
        # Use a pre-trained ResNet-18
        self.resnet = models.resnet18(pretrained=True)
        # Replace the final fully connected layer to output 24 values (12 keypoints * 2 (x,y))
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_keypoints * 2)

    def forward(self, x):
        return self.resnet(x)

def detect_court_keypoints(frame, model_path=None):
    """
    Detects 12 court keypoints using a ResNet-based regressor.
    """
    print("Detecting court keypoints using ResNet...")
    model = CourtKeypointRegressor()

    if model_path:
        try:
            model.load_state_dict(torch.load(model_path))
            model.eval()
        except Exception as e:
            print(f"Could not load court model from {model_path}: {e}")
            print("Returning empty keypoints.")
            return []
    else:
        print("No model path provided. Returning empty keypoints.")
        return []

    # Preprocess the frame for ResNet
    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_frame)

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    input_tensor = preprocess(pil_img).unsqueeze(0) # Add batch dimension

    with torch.no_grad():
        output = model(input_tensor)

    # Output is 24 flattened coordinates
    # We need to un-normalize these back to the original image dimensions
    h, w = frame.shape[:2]
    # Assuming the network is trained to output normalized coordinates between 0 and 1
    # or scaled coordinates directly. Let's assume normalized 0-1 for this implementation.
    # In a real scenario, this would depend on how the model was trained.
    coords = output.view(12, 2).numpy()

    # Scale back to original frame size
    keypoints = []
    for (x, y) in coords:
        # Prevent completely out of bounds values if the network is poorly initialized
        kx = int(np.clip(x * w, 0, w))
        ky = int(np.clip(y * h, 0, h))
        keypoints.append((kx, ky))

    return keypoints

def train_model(train_loader, num_epochs=10, learning_rate=1e-3, save_path="models/court_detector/resnet_keypoints.pt"):
    """
    Skeleton training script for the keypoint regressor.
    """
    model = CourtKeypointRegressor()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")
