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

def reshape_vit_transform(tensor, height=14, width=14):
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result

def visualize_vit_model(model, img_path, cam_type, img_size):
    # Load the image
    model.eval()
    img = cv2.imread(img_path, 1)[:, :, ::-1]
    img = cv2.resize(img, (img_size, img_size))
    img = np.float32(img) / 255
    # Preprocess the image
    input_tensor = preprocess_image(img, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    # Specify the target layers
    target_layers = [model.blocks[-1].norm1]

    if cam_type == 'GradCAM':
        cams = [GradCAM(model=model, target_layers=target_layers, use_cuda=False, reshape_transform=reshape_vit_transform)]
    elif cam_type == 'HiResCAM':
        cams = [HiResCAM(model=model, target_layers=target_layers, use_cuda=False, reshape_transform=reshape_vit_transform)]
    elif cam_type == 'ScoreCAM':
        cams = [ScoreCAM(model=model, target_layers=target_layers, use_cuda=False, reshape_transform=reshape_vit_transform)]
    elif cam_type == 'GradCAMPlusPlus':
        cams = [GradCAMPlusPlus(model=model, target_layers=target_layers, use_cuda=False, reshape_transform=reshape_vit_transform)]
    elif cam_type == 'AblationCAM':
        cams = [AblationCAM(model=model, target_layers=target_layers, use_cuda=False, reshape_transform=reshape_vit_transform)]
    elif cam_type == 'XGradCAM':
        cams = [XGradCAM(model=model, target_layers=target_layers, use_cuda=False, reshape_transform=reshape_vit_transform)]
    elif cam_type == 'EigenCAM':
        cams = [EigenCAM(model=model, target_layers=target_layers, use_cuda=False, reshape_transform=reshape_vit_transform)]
    elif cam_type == 'FullGrad':
        cams = [FullGrad(model=model, target_layers=target_layers, use_cuda=False, reshape_transform=reshape_vit_transform)]
    elif cam_type == 'EigenGradCAM':
        cams = [FullGrad(model=model, target_layers=target_layers, use_cuda=False, reshape_transform=reshape_vit_transform)]
    elif cam_type == 'LayerCAM':
        cams = [FullGrad(model=model, target_layers=target_layers, use_cuda=False, reshape_transform=reshape_vit_transform)]

    # Target classes and titles
    target_classes = [0, 1, 2, 3, 4]
    target_class_titles = ["1++", "1+", "1", "2", "3"]

    # Compute the gradients and visualizations for each layer and target class
    visualizations = []
    gb_visualizations = []
    gb_model = GuidedBackpropReLUModel(model=model, use_cuda=False)

    for cam in cams:
        for target_class in target_classes:
            targets = [ClassifierOutputTarget(target_class)]
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
            grayscale_cam = grayscale_cam[0, :]
            visualization = show_cam_on_image(img, grayscale_cam, use_rgb=True)
            visualizations.append(visualization)
            
            gb = gb_model(input_tensor, target_category=None)
            cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
            cam_gb = deprocess_image(cam_mask * gb)
            gb = deprocess_image(gb)
            gb_visualizations.append(gb)
        
    num_classes = len(target_classes)
    fig, axs = plt.subplots(1, num_classes, figsize=(4 * num_classes, 8))

    # Display each target class visualization and gb visualization in separate subplots
    for i in range(num_classes):
        # Visualization subplot
        axs[i].imshow(visualizations[i])
        axs[i].axis('off')
        axs[i].set_title(f"Target Class: {target_class_titles[i]}")


    # Adjust the spacing between subplots
    plt.tight_layout()
    plt.show()

    ##### Show score value ####
    # img = Image.open(img_path)
    # input_tensor = preprocess_image(img).unsqueeze(0)
    # model.eval()
    # with torch.no_grad():
    #     output = model(input_tensor)
    # scores = torch.softmax(output, dim=1) * 100
    # score_values = scores.squeeze(0).tolist()
    # print(score_values)

###### VIT ######
logged_model = '/Users/jeesuppark/Downloads/car_test/mlruns/1/3743becc4ccc436e843e83ce1bc6240d/artifacts/best'
model = mlflow.pytorch.load_model(logged_model, map_location=torch.device('cpu'))
img_path = "/Users/jeesuppark/Downloads/cropped_224_1/QC_cow_segmentation_1_000014.jpg"
img_size = 448
# GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
visualize_vit_model(model, img_path, "GradCAM", img_size)