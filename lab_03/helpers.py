from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def load_breast_cancer_data():
    data = load_breast_cancer()
    X = data.data
    y = data.target.reshape(-1, 1)  # Make y a column vector
        
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test

def visualize_image_and_channels(img_np):
    plt.figure(figsize=(14, 10))
    
    # Plot the original RGB image.
    plt.subplot(2, 2, 1)
    plt.imshow(img_np)
    plt.title('Original Image (RGB)', fontsize=15)
    plt.axis('off')
    
    # Plot each channel separately
    red_cmap = mcolors.LinearSegmentedColormap.from_list('red_cmap', [(0, 0, 0), (1.0, 0, 0)])
    green_cmap = mcolors.LinearSegmentedColormap.from_list('green_cmap', ['black', (0, 1.0, 0)])
    blue_cmap = mcolors.LinearSegmentedColormap.from_list('blue_cmap', ['black', (0, 0, 1.0)])
    cmap_list = [red_cmap, green_cmap, blue_cmap]
    channel_names = ['Red Channel', 'Green Channel', 'Blue Channel']
    for i in range(3):
        plt.subplot(2, 2, i + 2)
        channel_data = img_np[..., i]
        im = plt.imshow(channel_data, cmap=cmap_list[i], vmin=0, vmax=255)
        plt.title(f'{channel_names[i]}', fontsize=15)
        plt.axis('off')
        
        cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
        cbar.set_label('Pixel Intensity (8-bit: 0-255)', fontsize=12)
    
    plt.tight_layout()
    plt.show()