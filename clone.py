import csv
import cv2
import os
import sys
import numpy as np

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
  print('file separator :', path_separator)
  return path_separator

def get_augmented_data (images, angles):
  augmented_images, augmented_angles = [], []
  for image, angle in zip(images, angles):
    augmented_images.append(cv2.flip(image, 1))
    augmented_angles.append(angle*-1.0)
  return augmented_images, augmented_angles

def get_data_from_csv_rows(rows):
  images, angles = [], []
  i = 0
  for row in rows:
    steering_center = float(row[3])

    # create adjusted steering angles for the side camera images
    correction = 0.2 # this is a parameter to tune
    steering_left = steering_center + correction
    steering_right = steering_center - correction

    image_center = cv2.imread(extract_actual_path(row[0], file_separator))
    image_left = cv2.imread(extract_actual_path(row[1], file_separator))
    image_right = cv2.imread(extract_actual_path(row[2], file_separator))

    images.append(image_center)
    images.append(image_left)
    images.append(image_right)

    angles.append(steering_center)
    angles.append(steering_left)
    angles.append(steering_right)

  return images, angles

lines = get_lines_from_csv('driving_log.csv')
file_separator = get_path_separator()
images, angles = get_data_from_csv_rows(lines)
augmented_images, augmented_angles = get_augmented_data(images, angles)
images.extend(augmented_images)
angles.extend(augmented_angles)

from sklearn.utils import shuffle

X_train = np.array(shuffle(images))
y_train = np.array(shuffle(angles))

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers.convolutional import Conv2D, Cropping2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
model.add(Conv2D(6, (5, 5), activation="relu"))
model.add(MaxPooling2D())
model.add(Conv2D(6, (5, 5), activation="relu"))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=3)

model.save('model.h5')

