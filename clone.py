import csv
import cv2
import os
import sys
import numpy as np

lines = []

file_separator = '/'
if(len(sys.argv) > 1 and sys.argv[1]):
  file_separator = sys.argv[1]

print('file separator :', file_separator)

with open(os.curdir + os.path.sep + 'data' + os.path.sep + 'driving_log.csv') as csvfile:
  reader = csv.reader(csvfile)
  for line in reader:
    lines.append(line)

images = []
measurements = []
for line in lines:
  source_path = line[0]
  filename = source_path.split(file_separator)[-1]
  current_path = os.curdir + os.path.sep + 'data' + os.path.sep + 'IMG' + os.path.sep + filename
  #with open('paths.log', 'a') as myfile:
  #  myfile.write(current_path)
  #  myfile.write('\n')
  image = cv2.imread(current_path)
  images.append(image)
  measurement = float(line[3])
  measurements.append(measurement)

augmented_images, augmented_measurements = [], []
for image, measurement in zip(images, measurements):
  augmented_images.append(image)
  augmented_measurements.append(measurement)
  augmented_images.append(cv2.flip(image, 1))
  augmented_measurements.append(measurement*-1.0)

X_train = np.array(images)
y_train = np.array(measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
model.add(Convolution2D(6,5,5,activation="relu"))
model.add(MaxPooling2D())
model.add(Convolution2D(6,5,5,activation="relu"))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=3)

model.save('model.h5')

