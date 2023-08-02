"cam.py" 180L, 5870B                                                                                                                                                                                                                                                                                                                                                                      180,0-1       All
# %%
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import os.path
import torch
import torch.nn as nn
import os
import torchvision
import cv2
import matplotlib.pyplot as plt
import io
from torchvision.transforms import ToTensor
from PIL import Image
from torchvision import datasets, transforms
from torch import optim
from torchvision import datasets, transforms
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, XGradCAM, EigenCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


def reshape_vit_transform(tensor, height=14, width=14):
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result


# %%
def visualize_all_vit_model(model, image, cam_type, img_size, target_classes):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
    input_tensor = transform(image).unsqueeze(0)
    # Specify the target layers
    target_layers = [model.blocks[-1].norm1]

    if cam_type == "GradCAM":
        cam = GradCAM(
            model=model,
            target_layers=target_layers,
            use_cuda=True,
            reshape_transform=reshape_vit_transform,
        )
    elif cam_type == "ScoreCAM":
        cam = [
            ScoreCAM(
                model=model,
                target_layers=target_layers,
                use_cuda=True,
                reshape_transform=reshape_vit_transform,
            )
        ]
    elif cam_type == "GradCAMPlusPlus":
        cam = GradCAMPlusPlus(
            model=model,
            target_layers=target_layers,
            use_cuda=True,
            reshape_transform=reshape_vit_transform,
        )
    elif cam_type == "XGradCAM":
        cam = XGradCAM(
            model=model,
            target_layers=target_layers,
            use_cuda=True,
            reshape_transform=reshape_vit_transform,
        )
    elif cam_type == "EigenCAM":
        cam = EigenCAM(
            model=model,
            target_layers=target_layers,
            use_cuda=True,
            reshape_transform=reshape_vit_transform,
        )

    if target_classes == None:
        # Target classes and titles
        targets = None
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]
        img_normalized = np.float32(image) / 255.0
        visualization = show_cam_on_image(img_normalized, grayscale_cam, use_rgb=True)
        visualization = visualization.tolist()
        return visualization

    else:
        visualizations = []

        for target_class in target_classes:
            targets = [ClassifierOutputTarget(target_class)]
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
            grayscale_cam = grayscale_cam[0, :]
            img_normalized = np.float32(img) / 255.0
            visualization = show_cam_on_image(
                img_normalized, grayscale_cam, use_rgb=True
            )
            visualization = visualization.tolist()
            visualizations.append(visualization)

        return visualizations

        # num_classes = len(target_classes)
        # fig, axs = plt.subplots(1, num_classes, figsize=(4 * num_classes, 8))

        # # Display each target class visualization and gb visualization in separate subplots
        # for i in range(num_classes):
        #     axs[i].imshow(visualizations[i])
        #     axs[i].axis("off")
        #     axs[i].set_title(f"Target Class:")

        # # Adjust the spacing between subplots
        # plt.tight_layout()

        # # Show the plot
        # plt.show()


# %%
def visualize_all_cnn_model(model, img_path, cam_type, img_size, target_classes):
    # Load the image
    img = Image.open(img_path)

    # Preprocess the image
    transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ]
    )
    input_tensor = transform(img).unsqueeze(0)
    img = img.resize((448, 448))

    # Specify the target layers
    target_layers = [model.layer4[-1]]

    # GradCAM instances for each target layer
    if cam_type == "GradCAM":
        cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)
    elif cam_type == "ScoreCAM":
        cam = ScoreCAM(model=model, target_layers=target_layers, use_cuda=False)
    elif cam_type == "GradCAMPlusPlus":
        cam = GradCAMPlusPlus(model=model, target_layers=target_layers, use_cuda=False)
    elif cam_type == "XGradCAM":
        cam = XGradCAM(model=model, target_layers=target_layers, use_cuda=False)
    elif cam_type == "EigenCAM":
        cam = EigenCAM(model=model, target_layers=target_layers, use_cuda=False)

    if target_classes == None:
        # Target classes and titles
        targets = None
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]
        img_normalized = np.float32(img) / 255.0
        visualization = show_cam_on_image(img_normalized, grayscale_cam, use_rgb=True)

        visualization = visualization.tolist()
        return visualization

    else:
        visualizations = []

        for target_class in target_classes:
            targets = [ClassifierOutputTarget(target_class)]
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
            grayscale_cam = grayscale_cam[0, :]
            img_normalized = np.float32(img) / 255.0
            visualization = show_cam_on_image(
                img_normalized, grayscale_cam, use_rgb=True
            )
            visualizations.append(visualization)

        return visualizations
