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
image_color_channel_size = 255
image_size = (image_width, image_height)
image_shape = image_size + (image_color_channel,)

batch_size = 32
epochs = 20
learning_rate = 0.0001

class_names = ['Cratera']

dataset_train = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_train_dir,
    image_size = image_size,
    batch_size = batch_size,
    shuffle = True
)

dataset_valid = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_valid_dir,
    image_size = image_size,
    batch_size = batch_size,
    shuffle = True
)

dataset_valid_cardinality = tf.data.experimental.cardinality(dataset_valid)
dataset_valid_batches = dataset_valid_cardinality // 5 

dataset_test = dataset_valid.take(dataset_valid_batches)
dataset_valid = dataset_valid.skip(dataset_valid_batches)

print('Validação Dataset Cardinality: %d' % tf.data.experimental.cardinality(dataset_valid))
print('Teste Dataset Cardinality: %s' % tf.data.experimental.cardinality(dataset_test))

def plot_dataset(dataset):

    plt.gcf().clear()
    plt.figure(figsize = (15, 15))

    for features, labels in dataset.take(1):

        for i in range(9):
        
            plt.subplot(3, 3, i + 1)
            plt.axis('off')

            plt.imshow(features[i].numpy().astype('uint8'))
            plt.tittle(class_names[labels[i]])

plot_dataset(dataset_train)

plot_dataset(dataset_valid)

plot_dataset(dataset_test)

model = tf.keras.models.Sequential([
    tf.keras.experimental.preprocessing.Rescaling(
        1. / image_color_channel_size, input_shape = image_shape 
    ),
    tf.keras.layer.Conv2D(16, 3, padding = 'same', activation = 'relu'),
    tf.keras.layer.MaxPooling2D(),
    tf.keras.layer.Conv2D(32, 3, padding = 'same', activation = 'relu'),
    tf.keras.layer.MaxPooling2D(),
    tf.keras.layer.Conv2D(64, 3, padding = 'same', activation = 'relu'),
    tf.keras.layer.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation = 'relu'),
    tf.keras.layers.Dense(1, activation = 'signoid')    
])
