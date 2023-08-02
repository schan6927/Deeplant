import cv2
import matplotlib.pyplot as plt
import numpy as np
import io
from PIL import Image


def colorGraph(name):
    
    plt.figure
    file0 = name
    img = cv2.imread(file0)
    color = ('b','g','r')
    data = {}
    plt.figure()
    for i,col in enumerate(color):
        for r in range(0,201,50):
            histr = cv2.calcHist([img],[i],None,[250],[0,250])
            pixel_values_0_to_50 = histr[r:r+51]
            total_pixels = np.sum(pixel_values_0_to_50)

            if color[i] not in data.keys():
                data[color[i]]=[]
            data[color[i]].append(total_pixels)
        plt.plot(histr,color = col)
        plt.fill_between(np.arange(250), histr.flatten(), color=col, alpha=0.3)
        plt.xlim([0,256])
    
    fig = plt.gcf()
    color_buf = io.BytesIO()
    fig.savefig(color_buf,format='png')
    color_buf.seek(0)
    color = Image.open(color_buf)
    plt.close(fig)
    return color

def grayGraph(name):
    image_path = name
    gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Check if the image was loaded successfully
    if gray_image is None:
        raise ValueError("Error loading the image. Please check the image path.")

    # Data dictionary to store the histogram information
    data = {}

    # Only one color for grayscale image
    color = ('gray',)

    plt.figure()

    for i, col in enumerate(color):
        for r in range(0, 256, 50):
            histr = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
            pixel_values_r_to_r_plus_50 = histr[r:r + 51]
            total_pixels = np.sum(pixel_values_r_to_r_plus_50)
            

        if col not in data.keys():
            data[col] = []
        data[col].append(total_pixels)
        

    plt.plot(histr, color=col, label='Histogram')
    plt.fill_between(np.arange(256), histr.flatten(), color=col, alpha=0.3)

    plt.xlim([0, 256])
    fig = plt.gcf()
    gray_buf = io.BytesIO()
    fig.savefig(gray_buf,format='png')
    gray_buf.seek(0)
    gray = Image.open(gray_buf)
    plt.close(fig)
    
    return gray
