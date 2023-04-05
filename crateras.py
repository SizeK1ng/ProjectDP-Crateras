import os
import matplotlib.pyplot as plt
import tensorflow as tf

dataset_dir = os.path.join(os.getcwd(), 'Imagens')

dataset_train_dir = os.path.join(dataset_dir, 'train')
dataset_train_crateras_len = len(os.listdir(os.path.join(dataset_train_dir, 'crateras')))

dataset_valid_dir = os.path.join(dataset_dir, 'valid')
dataset_valid_crateras_len = len(os.listdir(os.path.join(dataset_valid_dir, 'crateras')))

print('Treino de Crateras: %s' %dataset_train_crateras_len)
print('Validação de Crateras: %s' %dataset_valid_crateras_len)
#mostra o tanto de imagens disponiveis nos diretorios especificados

image_width = 160
image_height = 160
image_color_channel = 3
imagem_color_channel_size = 255
image_shape = image_size + (image_color_channel)

