import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import os
import timm
import cv2
import matplotlib.ticker as ticker
import torch.nn.functional as F
import json
import torchvision
import mlflow

from torchvision.transforms import ToTensor
from pathlib import Path
from PIL import Image
from torchvision import datasets, transforms

from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image, deprocess_image
from pytorch_grad_cam import GuidedBackpropReLUModel

def visualize_cnn_model(model, img_path, cam_type, img_size):
    # Load the image
    img = Image.open(img_path)

    # Preprocess the image
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    input_tensor = transform(img).unsqueeze(0)

    # Specify the target layers
    target_layers = [model.layer4[-1]]

    # GradCAM instances for each target layer
    if cam_type == 'GradCAM':
        cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)
    elif cam_type == 'HiResCAM':
        cam = HiResCAM(model=model, target_layers=target_layers, use_cuda=False)
    elif cam_type == 'ScoreCAM':
        cam = ScoreCAM(model=model, target_layers=target_layers, use_cuda=False)
    elif cam_type == 'GradCAMPlusPlus':
        cam = GradCAMPlusPlus(model=model, target_layers=target_layers, use_cuda=False)
    elif cam_type == 'AblationCAM':
        cam = AblationCAM(model=model, target_layers=target_layers, use_cuda=False)
    elif cam_type == 'XGradCAM':
        cam = XGradCAM(model=model, target_layers=target_layers, use_cuda=False)
    elif cam_type == 'EigenCAM':
        cam = EigenCAM(model=model, target_layers=target_layers, use_cuda=False)
    elif cam_type == 'FullGrad':
        cam = FullGrad(model=model, target_layers=target_layers, use_cuda=False)

    # Target classes and titles
    target_classes = [0, 1, 2, 3, 4]
    target_class_titles = ["1++", "1+", "1", "2", "3"]

    # Compute the gradients and visualizations for each layer and target class
    visualizations = []
    for target_class in target_classes:
        targets = [ClassifierOutputTarget(target_class)]
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]
        img_normalized = np.float32(img) / 255.0
        visualization = show_cam_on_image(img_normalized, grayscale_cam, use_rgb=True)
        visualizations.append(visualization)

    # Create subplots for each layer visualization and target class
    num_classes = len(target_classes)
    fig, axs = plt.subplots(1, num_classes, figsize=(3, 10))

    # Display each layer visualization for each target class in a separate subplot
    for i in range(num_classes):
        axs[i].imshow(visualizations[i])
        axs[i].axis('off')
        axs[i].set_title(f"Target Class: {target_class_titles[i]}\n")

    # Adjust the spacing between subplots
    plt.tight_layout()

    # Show the plot
    plt.show()

###### CNN ######
logged_model = '/Users/jeesuppark/Downloads/car_test/mlruns/2/7157531a0e0047a6a718df6faa48d036/artifacts/best'
model = mlflow.pytorch.load_model(logged_model, map_location=torch.device('cpu'))
img_path = "/Users/jeesuppark/Downloads/cropped_224_1/QC_cow_segmentation_1_000504.jpg"
img_size = 224
# GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
visualize_cnn_model(model, img_path, "XGradCAM", img_size)