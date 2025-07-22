import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image


class DogCatDataset(Dataset):
    def __init__(self, root_folder, transform=None):
        self.image_paths = []
        self.labels = []
        self.transform = transform
        self.skipped_files = 0
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']

        for sub_folder in os.listdir(root_folder):
            sub_folder_path = os.path.join(root_folder, sub_folder)

            if os.path.isdir(sub_folder_path):
                if "dog" in sub_folder.lower():
                    label = 1
                elif "cat" in sub_folder.lower():
                    label = 0
                else:
                    continue # Bỏ vòng for bên dưới!

                for file_name in os.listdir(sub_folder_path):
                    if any(file_name.lower().endswith(ext) for ext in image_extensions):
                        image_path = os.path.join(sub_folder_path, file_name)

                        # Kiểm tra tính hợp lệ của file ảnh trong metadata
                        if self.is_valid_image(image_path):
                            self.image_paths.append(image_path)
                            self.labels.append(label)
                        else:
                            self.skipped_files += 1
                            logging.warning(f"Skipping invalid image: {image_path}")
        logging.info(f"Total valid images loaded: {len(self.image_paths)}")
        logging.info(f"Total skipped files: {self.skipped_files}")

    def is_valid_image(self, image_path):
        """Kiểm tra tính hợp lệ của file ảnh"""
        try:
            # Kiểm tra với OpenCV
            img = cv2.imread(image_path)
            if img is None:
                return False
            # Kiểm tra kích thước hợp lệ
            if img.shape[0] < 32 or img.shape[1] < 32:
                return False
            # Kiểm tra với PIL để catch lỗi JPEG hỏng
            with Image.open(image_path) as img_pil:
                img_pil.verify()
            # Kiểm tra với PIL để đảm bảo có thể load được
            with Image.open(image_path) as img_pil:
                img_pil.load()
            return True

        except (OSError, IOError, cv2.error) as e:
            logging.warning(f"Skipping invalid image: {image_path} - Error: {str(e)}")
            return False

        except Exception as e:
            logging.warning(f"Skipping invalid image: {image_path} - Error: {str(e)}")
            return False

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            # Đọc ảnh với error handling
            image = self._load_image_safely(self.image_paths[idx])
            if self.transform:
                image = self.transform(image)
            label = self.labels[idx]
            return image, label

        except Exception as e:
            logging.error(f"Error loading image: {self.image_paths[idx]} - Error: {str(e)}")
            # Trả về ảnh mặc định hoặc tensor rỗng
            if self.transform:
                default_image = np.zeros((3, 224, 224))
                image = self.transform(default_image)
            else:
                image = torch.zeros(3, 224, 224)
            label = self.labels[idx]
            return image, label

    def _load_image_safely(self, image_path):
        """Đọc ảnh một cách an toàn với nhiều phương pháp fallback"""
        # Thử đọc với OpenCV
        try:
            image = cv2.imread(image_path)
            if image is not None:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (224, 224))
                return image
        except Exception as e:
            pass  # Nếu lỗi xảy ra, tiếp tục với PIL

        # Thử đọc với PIL
        try:
            with Image.open(image_path) as image:
                image = image.convert('RGB')
                image = image.resize((224, 224))
                image = np.array(image)
                return image
        except Exception as e:
            logging.error(f"Error loading image: {image_path} - Error: {str(e)}")
            return np.zeros((224, 224, 3), dtype=np.uint8)
