"""
Using cat/dog data from Kaggle to create a very simple CNN model using TensorFlow

The output file is a model in hdf5 format
"""

import os
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime

from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator

###############################
# ------ DATA HANDLING ------ #
###############################

BATCH_SIZE = 10

base_dir = 'Data/cats_and_dogs_filtered'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

# Directory with our training cat/dog pictures
train_cats_dir = os.path.join(train_dir, 'cats')
train_dogs_dir = os.path.join(train_dir, 'dogs')

# Directory with our validation cat/dog pictures
validation_cats_dir = os.path.join(validation_dir, 'cats')
validation_dogs_dir = os.path.join(validation_dir, 'dogs')

train_cats_fnames = os.listdir(train_cats_dir)
train_dogs_fnames = os.listdir(train_dogs_dir)
val_cats_fnames = os.listdir(validation_cats_dir)
val_dogs_fnames = os.listdir(validation_dogs_dir)

num_train_images = len(train_cats_fnames) + len(train_dogs_fnames)
num_val_images = len(train_cats_fnames) + len(train_dogs_fnames)

train_datagen = ImageDataGenerator(rescale=1.0 / 255,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

train_generator = train_datagen.flow_from_directory(train_dir,
                                                    batch_size=BATCH_SIZE,
                                                    class_mode='binary',
                                                    target_size=(150, 150))

validation_generator = test_datagen.flow_from_directory(train_dir,
                                                        batch_size=BATCH_SIZE,
                                                        class_mode='binary',
                                                        target_size=(150, 150))

###############################
# ----- MODEL CREATION ------ #
###############################

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(lr=1e-4),
              metrics=['accuracy'])

history = model.fit(
    train_generator,
    steps_per_epoch=num_train_images / BATCH_SIZE,  # 2000 images = batch_size * steps
    epochs=100,
    validation_data=validation_generator,
    validation_steps=num_val_images / BATCH_SIZE,  # 1000 images = batch_size * steps
    verbose=2)

dt_string = datetime.now().strftime("%d%m%Y_%H%M%S")
tf.keras.models.save_model(model, f'Models/cat_dog_classifier_{dt_string}.hdf5')

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

num_epochs = range(len(acc))

plt.plot(num_epochs, acc, 'bo', label='Training accuracy')
plt.plot(num_epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')

plt.figure()

plt.plot(num_epochs, loss, 'bo', label='Training Loss')
plt.plot(num_epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
