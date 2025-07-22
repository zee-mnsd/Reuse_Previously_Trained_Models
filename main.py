import torch
import torch.nn as nn
from torchvision.models import resnet50

from DogCatClassifier import DogCatClassifier
from predict_image import predict_image

# Khởi tạo model (cấu trúc giống như lúc train)
model = DogCatClassifier()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load file .pth và sửa tiền tố
state_dict = torch.load('model.pth', map_location=device)
new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
model.load_state_dict(new_state_dict)
model.to(device)
model.eval()

# Sử dụng
result, confidence = predict_image('cho.jpg', model, device)
print(f"Kết quả: {result}, Độ tin cậy: {confidence:.4f}")
