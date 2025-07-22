import torch
import logging

from DogCatDataset import DogCatDataset
from SafeTransform import SafeTransform


def predict_image(image_path, model, device):
    try:
        # Load ảnh an toàn
        dataset_temp = DogCatDataset.__new__(DogCatDataset)
        image = dataset_temp._load_image_safely(image_path)

        transform = SafeTransform()
        image = transform(image)
        image = image.unsqueeze(0).to(device)

        model.eval()
        with torch.no_grad():
            output = model(image)
            probabilities = torch.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

        if predicted.item() == 0:
            return "Đây là mèo!", confidence.item()
        else:
            return "Đây là chó!", confidence.item()

    except Exception as e:
        logging.error(f"Error during prediction: {str(e)}")
        print("Không thể dự đoán ảnh này.")
        return "Error", 0.0
