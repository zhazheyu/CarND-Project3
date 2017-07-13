#**Behavioral Cloning**

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/NvidiaCNN.png "Nvidia model"
[image2]: ./examples/center_image1.jpg "center image one"
[image3]: ./examples/center_image2.jpg "center image from left road side"
[image4]: ./examples/center_image3.jpg "center image from right road side"
[image5]: ./examples/center_image4.jpg "Center image"
[image6]: ./examples/center_image4_flipped.jpg "Flipped center image"
[image7]: ./examples/image.png "Image preprocessing"

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

The model employed in this project is based on the following paper by Nvidia
http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf

This model consists of 5 layers convolution networks and 3 fully connected networks.
The details are the following:

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (e.g., model.py lines 14, 16 and so on).
The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.
Default value is 0.001

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. Only center lane driving images are used, since I do not think manually guessing an offset angle for left camera and right camera is correct.

For details about how I created the training data, see the next section.

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was referred to
http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting.

To combat the overfitting, I added dropout layers into the model.

Moreover, I also tuning batch size to reduce overfitting.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. To improve the driving behavior in these cases, I collect more data from these spots, to teach networks how to deal with this situations.

Meanwhile, I also used different preprocessing method, such as cropping, blur and changing colorspace to enhance the performance of network.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py #model_nvidia) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

Note: the number in this Nvidia CNN model image is not the same as used in this project


| Layer | Input Shape | Output Shape | Param #  
|:-------:|:-------:|:-------:|:-------:|
| Lambda normalization | (75, 160, 3) | (75, 160, 3)   							| |
| Convolution2D 5x5 kernel, same padding | (75, 160, 3) | (30, 78, 24) | 1824 |
| SpatialDropout2D |	|	|	|
| Convolution2D 5x5 kernel, same padding | (30, 78, 24) | (13, 37, 36) | 21636 |
| SpatialDropout2D |	|	|	|
| Convolution2D 5x5 kernel, valid padding | (13, 37, 36) | (6, 18, 48) | 43248 |
| SpatialDropout2D |	|	|	|
| Convolution2D 3x3 kernel, valid padding | (6, 18, 48) | (6, 18, 64) | 27712 |
| SpatialDropout2D |	|	|	|
| Convolution2D 3x3 kernel, valid padding | (6, 18, 64) | (6, 18, 64) | 36928 |
| SpatialDropout2D |	|	|	|
| Flatten |	|	|	|
| Dropout |	|	|	|
| Fully connected | Input 6912, | Output 100 | 691300 |
| Fully connected | Input 100, | Output 50 | 5050 |
| Fully connected | Input 50, | Output 10 | 510 |
| Dropout |	|	|	|
| Fully connected | Input 10, | Output 1 | 11 | |

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded three laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to how to handle when vehicle is on left/right side. These images show what a recovery looks like starting from off track :

![alt text][image3]

![alt text][image4]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would generalize collected data. For example, here is an image that has then been flipped:

![alt text][image5]

![alt text][image6]
Etc ....

After the collection process, I had 12214 number of data points. I then preprocessed this data by cropping, blur and change colorspace.

![alt text][image7]


I finally randomly shuffled the data set and put 20% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was around 7 as evidenced by that the validation is not decreasing as epochs (please check the following image).  I used an adam optimizer so that manually training the learning rate wasn't necessary.
