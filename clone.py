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

lines = get_lines_from_csv('driving_log.csv')
file_separator = get_path_separator()

images = []
measurements = []
for line in lines:
  steering_center = float(line[3])

  # create adjusted steering measurements for the side camera images
  correction = 0.2 # this is a parameter to tune
  steering_left = steering_center + correction
  steering_right = steering_center - correction

  image_center = cv2.imread(extract_actual_path(line[0], file_separator))
  image_left = cv2.imread(extract_actual_path(line[1], file_separator))
  image_right = cv2.imread(extract_actual_path(line[2], file_separator))

  images.append(image_center)
  images.append(image_left)
  images.append(image_right)

  measurements.append(steering_center)
  measurements.append(steering_left)
  measurements.append(steering_right)

augmented_images, augmented_measurements = [], []
for image, measurement in zip(images, measurements):
  augmented_images.append(image)
  augmented_measurements.append(measurement)
  augmented_images.append(cv2.flip(image, 1))
  augmented_measurements.append(measurement*-1.0)

X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

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

