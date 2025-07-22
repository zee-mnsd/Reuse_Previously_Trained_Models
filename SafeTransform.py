import torch
import logging
import torchvision.transforms as transforms


class SafeTransform:
    """
    Preprocessing transform với error handling

        Transform = "dịch thuật" ảnh từ dạng con người hiểu sang dạng AI hiểu

        Logic:
            Ảnh gốc (.jpg file)
                ↓ transform ToPILImage, ToTensor, Normalize(Chuẩn hóa theo ImageNet)
            Tensor (3, 224, 224) với giá trị [-2, +2]

        Hàm:
            - init : định nghĩa transform
            - call : thực hiện transform
    """

    def __init__(self):
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __call__(self, image):
        try:
            return self.transform(image)
        except Exception as e:
            logging.error(f"Transform error: {str(e)}")
            # Trả về tensor mặc định
            return torch.zeros(3, 224, 224)
