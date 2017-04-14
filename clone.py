import csv
import cv2
import os
import sys
from random import randint
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split

def extract_actual_path (path, separator):
  filename = path.split(separator)[-1]
  return os.curdir + os.path.sep + 'data' + os.path.sep + 'IMG' + os.path.sep + filename

def get_lines_from_csv(csv_file):
  lines = []
  with open(os.curdir + os.path.sep + 'data' + os.path.sep + csv_file) as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
      lines.append(line)
  return lines

def get_path_separator ():
  path_separator = "/"
  if (len(sys.argv) > 1 and sys.argv[1] and sys.argv[1] == 'win') :
    path_separator = "\\"
  return path_separator

def get_augmented_data (images, angles):
  augmented_images, augmented_angles = [], []
  for image, angle in zip(images, angles):
    augmented_images.append(cv2.flip(image, 1))
    augmented_angles.append(angle*-1.0)
  return augmented_images, augmented_angles

def get_data_from_csv_rows(rows):
  images, angles = [], []
  path_separator = get_path_separator()
  for row in rows:
    steering_center = float(row[3])

    # create adjusted steering angles for the side camera images
    correction = 0.2 # this is a parameter to tune
    steering_left = steering_center + correction
    steering_right = steering_center - correction

    image_center = cv2.imread(extract_actual_path(row[0], path_separator))
    image_left = cv2.imread(extract_actual_path(row[1], path_separator))
    image_right = cv2.imread(extract_actual_path(row[2], path_separator))

    images.append(image_center)
    images.append(image_left)
    images.append(image_right)

    angles.append(steering_center)
    angles.append(steering_left)
    angles.append(steering_right)

  return images, angles

def get_data_from_batch (samples):
  path_separator = get_path_separator()
  chosen_images, chosen_angles = [], []
  for row in samples:
    steering_center = float(row[3])

    # create adjusted steering angles for the side camera images
    correction = 0.2 # this is a parameter to tune
    steering_left = steering_center + correction
    steering_right = steering_center - correction

    images = []
    image_center = cv2.imread(extract_actual_path(row[0], path_separator))
    image_left = cv2.imread(extract_actual_path(row[1], path_separator))
    image_right = cv2.imread(extract_actual_path(row[2], path_separator))
    image_center_augmented =  cv2.flip(image_center, 1)
    image_left_augmented =  cv2.flip(image_left, 1)
    image_right_augmented =  cv2.flip(image_right, 1)
    angle_center_augmented = (steering_center*-1.0)
    angle_left_augmented = (steering_left*-1.0)
    angle_right_augmented = (steering_right*-1.0)

    images.append((image_center, steering_center))
    images.append((image_left, steering_left))
    images.append((image_right, steering_right))
    images.append((image_center_augmented, angle_center_augmented))
    images.append((image_left_augmented, angle_left_augmented))
    images.append((image_right_augmented, angle_right_augmented))

    chosen_tuple = images[randint(0, len(images) - 1)]
    chosen_images.append(chosen_tuple[0])
    chosen_angles.append(chosen_tuple[1])
  return chosen_images, chosen_angles



def generator(samples, batch_size=32):
  num_samples = len(samples)
  while 1:
    for offset in range(0, num_samples, batch_size):
      batch_samples = samples[offset:offset + batch_size]

      images, angles = get_data_from_batch(batch_samples)

      X_train = np.array(images)
      y_train = np.array(angles)
      yield sklearn.utils.shuffle(X_train, y_train)

# ------------------------------------------------------------------------------
# ------------------- Start the pipeline here ---------------------------------
samples = get_lines_from_csv('driving_log.csv')
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers.convolutional import Conv2D, Cropping2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
model.add(Conv2D(24, (5, 5), activation="relu", strides=(2, 2)))
model.add(Conv2D(36, (5, 5), activation="relu", strides=(2, 2)))
model.add(Conv2D(48, (5, 5), activation="relu", strides=(2, 2)))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, steps_per_epoch = len(train_samples)//32, validation_data=validation_generator,
                    validation_steps=len(validation_samples)//32, epochs=3)

#model.fit_generator(train_generator, samples_per_epoch=len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=3)

model.save('model_generator.h5')

