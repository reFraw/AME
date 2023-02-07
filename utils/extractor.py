import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2

from tensorflow.keras import models
from tensorflow.keras import layers

class AME:

	def __init__(self, image_size, channels, model_path, image_path, saving_path, normalize=False):

		self.image_size = image_size
		self.channels = channels
		self.model_path = model_path
		self.image_path = image_path
		self.saving_path = saving_path
		self.normalize = normalize


	def process_image(self):

		image_array = cv2.imread(self.image_path)
		image_array = cv2.resize(image_array, (self.image_size, self.image_size))

		if self.channels == 1:
			image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)

		elif self.channels == 3:
			image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)

		image_array = np.expand_dims(image_array, axis=0)
		image_array = image_array.reshape(-1, self.image_size, self.image_size, self.channels)

		return image_array


	def normalize_input(self, image_array):

		normalized_input = image_array.astype('float32') / 255

		return normalized_input


	def extract(self):

		model = models.load_model(self.model_path)

		# Retrieving convolutional layers name
		conv_layers = list()
		for layer in model.layers:
			if 'conv' in layer.name:
				conv_layers.append(layer.name)


		# Processing input image
		input_image = self.process_image()
		if self.normalize:
			input_image = self.normalize_input(input_image)

		# Create saving path
		save_path = os.path.join('output', self.saving_path)
		if not os.path.exists(save_path):
			os.makedirs(save_path)


		# Extract feature maps for every convolutional layers
		images_per_row = 8

		for layer in conv_layers:

			layer_output = model.get_layer(layer).output
			activation_model = models.Model(inputs=model.input, outputs=layer_output)
			activation = activation_model(input_image)

			n_features = activation.shape[-1]
			n_cols = n_features // images_per_row
			size = activation.shape[1]
			display_grid = np.zeros((size * n_cols, size * images_per_row))

			for col in range(n_cols):
				for row in range(images_per_row):

					channel_image = activation[0, :, :, col*images_per_row + row].numpy()
					channel_image = np.clip(channel_image, 0, 255).astype('uint8')
					display_grid[col * size : (col + 1) * size, row * size : (row + 1) * size] = channel_image

			scale = 1. / size

			plt.figure(figsize=(2 * scale * display_grid.shape[1], 2 * scale * display_grid.shape[0]))

			plt.yticks(np.arange(0, size*n_cols, size))
			plt.xticks(np.arange(0, size*images_per_row, size))
			plt.grid(True)
			plt.axis('on')
			plt.imshow(display_grid, aspect='auto', cmap='viridis')

			savename = layer + '.png'
			image_savepath = os.path.join(save_path, savename)
			plt.savefig(image_savepath)