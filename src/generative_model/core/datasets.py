
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import os
from torchvision import transforms

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def get_target_transform():
    def target_transform(vector_field):
        return vector_field[2:]  # Extract only the action components (ux, uy)
    return target_transform

class VectorFieldDataset(Dataset):
    def __init__(self, vector_field_file, img_dir, resize_shape=(256, 256), crop_shape=(224, 224), transform=None, target_transform=get_target_transform(), image_size=128):
        self.vector_field_data = pd.read_csv(vector_field_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.image_size = image_size
        
        self.preprocess = transforms.Compose([
            transforms.Resize(resize_shape) if resize_shape else transforms.Lambda(lambda x: x),
            transforms.CenterCrop(crop_shape) if crop_shape else transforms.Lambda(lambda x: x),
        ])

        # Preload all images and vector field data into memory
        self.images_tensor = []
        self.vector_fields_tensor = []
        
        print("Starting data preloading...")
        for idx in range(len(self.vector_field_data)):
            row = self.vector_field_data.iloc[idx]
            frame_id = int(row['frame_id'])
            
            # Load and process image
            img_name = f"rectified_color_panorama_{frame_id:04d}.jpg"
            img_path = os.path.join(self.img_dir, img_name)
            image = Image.open(img_path).convert('RGB')
            image = self.preprocess(image)
            
            # Apply transformation
            if self.transform:
                image_tensor = self.transform(image)
            else:
                # Default transformation: PIL image -> Tensor
                image_tensor = torch.FloatTensor(np.array(image)).permute(2, 0, 1) / 255.0
            
            self.images_tensor.append(image_tensor)
            
            # Load vector field data
            vector_field = torch.tensor([row['x'], row['y'], row['ux'], row['uy']], dtype=torch.float32)
            if self.target_transform:
                vector_field = self.target_transform(vector_field)
            
            self.vector_fields_tensor.append(vector_field)
        
        print(f"Successfully preloaded {len(self.images_tensor)} images and {len(self.vector_fields_tensor)} vector fields")

    def __len__(self):
        return len(self.vector_field_data)

    def __getitem__(self, idx):
        # Directly return preloaded tensors
        return torch.stack([self.images_tensor[idx]]), self.vector_fields_tensor[idx]

class VectorFieldWithGradientsDataset(Dataset):
    def __init__(self, vector_field_file, img_dir, resize_shape=(256, 256), crop_shape=(224, 224), transform=None, target_transform=get_target_transform(), image_size=128):
        self.vector_field_data = pd.read_csv(vector_field_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.image_size = image_size
        
        self.preprocess = transforms.Compose([
            transforms.Resize(resize_shape) if resize_shape else transforms.Lambda(lambda x: x),
            transforms.CenterCrop(crop_shape) if crop_shape else transforms.Lambda(lambda x: x),
        ])

        # Preload all images and vector field data into memory
        self.images_tensor = []
        self.vector_fields_with_gradients_tensor = []
        
        print("Starting data preloading...")
        for idx in range(len(self.vector_field_data)):
            row = self.vector_field_data.iloc[idx]
            frame_id = int(row['frame_id'])
            
            # Load and process image
            img_name = f"rectified_color_panorama_{frame_id:04d}.jpg"
            img_path = os.path.join(self.img_dir, img_name)
            image = Image.open(img_path).convert('RGB')
            image = self.preprocess(image)
            
            # Apply transformation
            if self.transform:
                image_tensor = self.transform(image)
            else:
                # Default transformation: PIL image -> Tensor
                image_tensor = torch.FloatTensor(np.array(image)).permute(2, 0, 1) / 255.0
            
            self.images_tensor.append(image_tensor)
            
            # Load vector field with gradients data
            ux = row['ux']
            uy = row['uy']
            grad_ux_dx = row['grad_ux_dx']
            grad_ux_dy = row['grad_ux_dy']
            grad_uy_dx = row['grad_uy_dx']
            grad_uy_dy = row['grad_uy_dy']
            
            # Calculate expression: delta_uy * ux - delta_ux * uy
            # where delta_ux = [grad_ux_dx, grad_ux_dy], delta_uy = [grad_uy_dx, grad_uy_dy]
            expr_x = grad_uy_dx * ux - grad_ux_dx * uy
            expr_y = grad_uy_dy * ux - grad_ux_dy * uy
            
            # Create 4-dimensional output: [ux, uy, expr_x, expr_y]
            vector_fields_with_gradients = torch.tensor([ux, uy, expr_x, expr_y], dtype=torch.float32)
            
            self.vector_fields_with_gradients_tensor.append(vector_fields_with_gradients)
        
        print(f"Successfully preloaded {len(self.images_tensor)} images, {len(self.vector_fields_with_gradients_tensor)} vector fields with gradients")

    def __len__(self):
        return len(self.vector_field_data)

    def __getitem__(self, idx):
        # Directly return preloaded tensors
        return torch.stack([self.images_tensor[idx]]), self.vector_fields_with_gradients_tensor[idx]

class SparseVectorFieldWithGradientsDataset(Dataset):
    def __init__(self, vector_field_file, img_dir, step_size=10, resize_shape=(256, 256), crop_shape=(224, 224), transform=None, target_transform=get_target_transform(), image_size=128):
        self.vector_field_data = pd.read_csv(vector_field_file)
        self.img_dir = img_dir
        self.step_size = step_size
        self.transform = transform
        self.target_transform = target_transform
        self.image_size = image_size
        
        self.preprocess = transforms.Compose([
            transforms.Resize(resize_shape) if resize_shape else transforms.Lambda(lambda x: x),
            transforms.CenterCrop(crop_shape) if crop_shape else transforms.Lambda(lambda x: x),
        ])

        # Preload all images and vector field data into memory
        self.images_tensor = []
        self.vector_fields_with_gradients_tensor = []
        
        print(f"Starting sparse data preloading with step size {step_size}...")
        for idx in range(0, len(self.vector_field_data), step_size):
            row = self.vector_field_data.iloc[idx]
            frame_id = int(row['frame_id'])
            
            # Load and process image
            img_name = f"rectified_color_panorama_{frame_id:04d}.jpg"
            img_path = os.path.join(self.img_dir, img_name)
            image = Image.open(img_path).convert('RGB')
            image = self.preprocess(image)
            
            # Apply transformation
            if self.transform:
                image_tensor = self.transform(image)
            else:
                # Default transformation: PIL image -> Tensor
                image_tensor = torch.FloatTensor(np.array(image)).permute(2, 0, 1) / 255.0
            
            self.images_tensor.append(image_tensor)
            
            # Load vector field with gradients data
            ux = row['ux']
            uy = row['uy']
            grad_ux_dx = row['grad_ux_dx']
            grad_ux_dy = row['grad_ux_dy']
            grad_uy_dx = row['grad_uy_dx']
            grad_uy_dy = row['grad_uy_dy']
            
            # Calculate expression: delta_uy * ux - delta_ux * uy
            expr_x = grad_uy_dx * ux - grad_ux_dx * uy
            expr_y = grad_uy_dy * ux - grad_ux_dy * uy
            
            # Create 4-dimensional output: [ux, uy, expr_x, expr_y]
            vector_fields_with_gradients = torch.tensor([ux, uy, expr_x, expr_y], dtype=torch.float32)
            
            self.vector_fields_with_gradients_tensor.append(vector_fields_with_gradients)
        
        print(f"Successfully preloaded {len(self.images_tensor)} sparse images, {len(self.vector_fields_with_gradients_tensor)} sparse vector fields with gradients")

    def __len__(self):
        return len(self.images_tensor)

    def __getitem__(self, idx):
        return torch.stack([self.images_tensor[idx]]), self.vector_fields_with_gradients_tensor[idx]

class RectangularMaskedDataset(Dataset):
    def __init__(self, vector_field_file, img_dir, 
                 mask_x_range=(-0.5, 0.5), mask_y_range=(-0.3, 0.3), mask_type='none',
                 resize_shape=(256, 256), crop_shape=(224, 224)):
        self.vector_field_data = pd.read_csv(vector_field_file)
        self.img_dir = img_dir
        self.mask_x_range = mask_x_range
        self.mask_y_range = mask_y_range
        self.mask_type = mask_type
        
        self.preprocess = transforms.Compose([
            transforms.Resize(resize_shape),
            transforms.CenterCrop(crop_shape),
        ])
        
        self.images_tensor = []
        self.vector_fields_tensor = []
        self.indices = []
        
        print(f"Loading data with rectangular mask (type: {mask_type})...")
        for idx in range(len(self.vector_field_data)):
            row = self.vector_field_data.iloc[idx]
            x, y = row['x'], row['y']
            
            is_in_mask = (mask_x_range[0] <= x <= mask_x_range[1] and 
                mask_y_range[0] <= y <= mask_y_range[1])

            if mask_type == 'none' and is_in_mask:
                continue
            
            frame_id = int(row['frame_id'])
            img_name = f"rectified_color_panorama_{frame_id:04d}.jpg"
            img_path = os.path.join(self.img_dir, img_name)
            image = Image.open(img_path).convert('RGB')
            image = self.preprocess(image)
            image_tensor = torch.FloatTensor(np.array(image)).permute(2, 0, 1) / 255.0
            self.images_tensor.append(image_tensor)
            
            ux, uy = row['ux'], row['uy']
            grad_ux_dx, grad_ux_dy = row['grad_ux_dx'], row['grad_ux_dy']
            grad_uy_dx, grad_uy_dy = row['grad_uy_dx'], row['grad_uy_dy']

            if mask_type == 'random' and is_in_mask:
                vector_field = 'random'
            elif mask_type == 'bimodal' and is_in_mask:
                vector_field = torch.tensor([[1, 0, 0, 0], [-1, 0, 0, 0]], dtype=torch.float32)
            else:
                expr_x = grad_uy_dx * ux - grad_ux_dx * uy
                expr_y = grad_uy_dy * ux - grad_ux_dy * uy
                vector_field = [torch.tensor([ux, uy, expr_x, expr_y], dtype=torch.float32)]
            self.vector_fields_tensor.append(vector_field)
            self.indices.append(idx)
        
        print(f"Masked region: x∈{mask_x_range}, y∈{mask_y_range}")
        print(f"Loaded {len(self.images_tensor)} samples (original: {len(self.vector_field_data)})")
    
    def __len__(self):
        return len(self.images_tensor)
    
    def __getitem__(self, idx):
        vector_field = self.vector_fields_tensor[idx]
        if vector_field == 'random':
            theta = np.random.uniform(0, 2 * np.pi)
            ux = np.cos(theta)
            uy = np.sin(theta)
            chosen_vector = torch.tensor([ux, uy, 0, 0], dtype=torch.float32)
        elif isinstance(vector_field, list):
            chosen_vector = vector_field[torch.randint(low=0, high=len(vector_field), size=(1,)).item()]
        else:
            chosen_vector = vector_field[torch.randint(low=0, high=len(vector_field), size=(1,)).item()]
        return torch.stack([self.images_tensor[idx]]), chosen_vector

