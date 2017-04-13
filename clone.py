import csv
import cv2
import os
import sys
import numpy as np

lines = []

file_separator = "/"
if (len(sys.argv) > 1 and sys.argv[1] and sys.argv[1] == 'win') :
  file_separator = "\\"

print('file separator :', file_separator)

with open(os.curdir + os.path.sep + 'data' + os.path.sep + 'driving_log.csv') as csvfile:
  reader = csv.reader(csvfile)
  for line in reader:
    lines.append(line)

images = []
measurements = []
for line in lines:
  steering_center = float(line[3])

  # create adjusted steering measurements for the side camera images
  correction = 0.2 # this is a parameter to tune
  steering_left = steering_center + correction
  steering_right = steering_center - correction

  source_path_center = line[0]
  source_path_left = line[1]
  source_path_right = line[2]

  filename_center = source_path_center.split(file_separator)[-1]
  filename_left = source_path_left.split(file_separator)[-1]
  filename_right = source_path_right.split(file_separator)[-1]

  current_path_center = os.curdir + os.path.sep + 'data' + os.path.sep + 'IMG' + os.path.sep + filename_center
  current_path_left = os.curdir + os.path.sep + 'data' + os.path.sep + 'IMG' + os.path.sep + filename_left
  current_path_right = os.curdir + os.path.sep + 'data' + os.path.sep + 'IMG' + os.path.sep + filename_right
  #with open('paths.log', 'a') as myfile:
  #  myfile.write(current_path)
  #  myfile.write('\n')
  image_center = cv2.imread(current_path_center)
  image_left = cv2.imread(current_path_left)
  image_right = cv2.imread(current_path_right)

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

