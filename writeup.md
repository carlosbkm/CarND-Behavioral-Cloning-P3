#**Behavioral Cloning** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model consists of a convolution neural network based on <a href="https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf">NVIDIA model architecture</a> with 5x5 and 3x3 filter sizes and depths between 24 and 64 (model.py lines 96-107). 

The model includes RELU layers to introduce nonlinearity and the data is normalized in the model using a Keras lambda layer (code line 98). At the input of the model, I use a Cropping layer to crop height and width of the image by 50 px and 20 px respectively.

####2. Attempts to reduce overfitting in the model

A max pooling layer was introduced in the first attempts to reduce overfitting. Since there were not observable improvements, I decided to remove it from the final architecture.

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track. However, for a better generalization, further work on the overfitting problem would have been required. This could help for example to be able to address more general problems like driving on a different track than the one we trained the model for.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 109).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. 

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

My first step was to use LeNet 5 convolution neural network model similar to the one used in the Traffic Sign classifier project but this time implemented in Keras. This model was proven to work well in the past, so I took as my base to start working. However, I added to this architecture a Cropping layer and a Lambda layer for normalization.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

Since I had an architecture that was actually working, I decided to keep it as it is and start working on the data processing. 

First, I implemented data augmentation, by flipping images horizontally (lines 48 to 50). Also, I used left and right cameras, with an steering angle correction of +0.2 / -0.2. In the first implementation, I added all those images (3 cameras and its respective flipping) to the train data set, which it made it 6 times bigger. That made the training process per epoch too slow.

To improve performance, I called the train images generation from a python generator (lines 71 to 81). Also, I implemented the images augmentation method ("get_data_from_batch") in the following way:
* Take the images from center, left and right cameras.
* Flip them horizontally.
* Add them to a 6 elements array.
* Choose randomly one from that array.
* Repeat it for all the elements in a batch.

By doing this per batch required from the generator, the model is more robust and we reduce overfitting. Then, we use a "fit_generator" from keras which takes len(train_samples)/batch_size as the number of steps per epoch. This way we reduce dramatically the number of steps per epoch and in turn we get a better generalization, since the average of the loss in a batch is calculated per step. 

Once I got the data processing working as I wanted, I decided to try which a more powerful architecture, the NVIDIA model architecture suggested in the lessons. I observerd that with this model training and validation error were very close one to another, so the overfitting problem was almost gone.

After tuning the batch size to 32, the model proved to generalize good enough to make the car drive a full lap as required. Only 3 epochs were needed to get these results, what it was obtained very fast on the Amazon ECS instance.

However, to come to this result, the most important step was the data collecting. After some failed strategies, the one which yield the best results was to drive 2 laps keeping the car in the middle of the road, and then do small passes of data collection only with the driving correction where the car tended to get out of the road. 

####2. Final Model Architecture

The final model architecture (clone.py lines 96-107) consisted of a convolution neural network with the following layers and layer sizes:
* Lambda layer for normalization. Input 160x320x3. Divides the values by 255 and adjust the average by 0.5.
* Cropping layer: Input 160x320x3, cropping 50 px height, and 20 px width.
* Convolution layer: 24 filters with 5x5 kernel and 2x2 stride. Activation: RELU.
* Convolution layer: 36 filters with 5x5 kernel and 2x2 stride. Activation: RELU.
* Convolution layer: 48 filters with 5x5 kernel and 2x2 stride. Activation: RELU.
* Convolution layer: 64 filters with 3x3 kernel. Activation: RELU.
* Convolution layer: 64 filters with 3x3 kernel. Activation: RELU.
* Flatten layer.
* Fully connected. Output: 100
* Fully connected. Output: 50
* Fully connected. Output: 1

The NVIDIA architecture which it was based in can be seen in the picture below:

![alt text][image1]

####3. Creation of the Training Set & Training Process

The creation of Training Set and Training process was already explain in section 1 "Solution Design Approach".
