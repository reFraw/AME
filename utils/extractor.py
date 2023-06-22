import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2

from tensorflow.keras import models
from tensorflow.keras import layers


def process_image(img_path, img_size, channels):
    image_array = cv2.imread(img_path)
    image_array = cv2.resize(image_array, (img_size, img_size))

    if channels == 1:
        image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)

    elif channels == 3:
        image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)

    image_array = np.expand_dims(image_array, axis=0)
    image_array = image_array.reshape(-1, img_size, img_size, channels)

    return image_array


def normalize_input(img_array):
    normalized_input = img_array.astype('float32') / 255

    return normalized_input


def extract(model_path, input_img, normalize_val, saving_path, img_name):
    # Create saving path
    save_path = os.path.join('output', saving_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    img_folder = os.path.join(save_path, img_name)
    if not os.path.isdir(img_folder):
        os.makedirs(img_folder)

    model = models.load_model(model_path)

    # Retrieving convolutional layers name
    conv_layers = list()
    for layer in model.layers:
        if 'conv' in layer.name:
            conv_layers.append(layer.name)

    if normalize_val:
        input_img = normalize_input(input_img)

    # Extract feature maps for every convolutional layers
    images_per_row = 8

    for layer in conv_layers:

        layer_output = model.get_layer(layer).output
        activation_model = models.Model(inputs=model.input, outputs=layer_output)
        activation = activation_model(input_img)

        n_features = activation.shape[-1]
        n_cols = n_features // images_per_row
        size = activation.shape[1]
        display_grid = np.zeros((size * n_cols, size * images_per_row))

        for col in range(n_cols):
            for row in range(images_per_row):
                channel_image = activation[0, :, :, col * images_per_row + row].numpy()
                channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                display_grid[col * size: (col + 1) * size, row * size: (row + 1) * size] = channel_image

        scale = 1. / size

        plt.figure(figsize=(2 * scale * display_grid.shape[1], 2 * scale * display_grid.shape[0]))

        plt.yticks(np.arange(0, size * n_cols, size))
        plt.xticks(np.arange(0, size * images_per_row, size))
        plt.grid(True)
        plt.axis('on')
        plt.imshow(display_grid, aspect='auto', cmap='viridis')

        filename_save = img_name + '_' + layer + '.png'
        image_savepath = os.path.join(img_folder, filename_save)
        plt.savefig(image_savepath)
        plt.close('all')


class AME:

    def __init__(self, image_size, channels, model_path, image_path, saving_path, entire_dataset=False,
                 normalize=False):

        self.image_size = image_size
        self.channels = channels
        self.model_path = model_path
        self.image_path = image_path
        self.saving_path = saving_path
        self.entire_dataset = entire_dataset
        self.normalize = normalize

    def manage_image(self):

        if self.entire_dataset:

            for img in os.listdir(self.image_path):
                img_path = os.path.join(self.image_path, img)
                input_img = process_image(img_path, self.image_size, self.channels)
                folder_name = img.split('/')[-1].replace('.apk.png', '')
                print(folder_name)
                extract(self.model_path, input_img, self.normalize, self.saving_path, folder_name)
        else:
            input_img = process_image(self.image_path, self.image_size, self.channels)
            folder_name = self.image_path.split('/')[-1].replace('.apk.png', '')
            print(folder_name)
            extract(self.model_path, input_img, self.normalize, self.saving_path, folder_name)
