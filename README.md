# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./data_distribution.png "Data Distribution"
[image2]: ./grayscaling.jpg "Original vs. grayscaled image"
[image3]: ./test_images/50kph.jpg "Speed Limit 50 kph"
[image4]: ./test_images/bumpy_road.jpg "Bumpy Road"
[image5]: ./test_images/gefahrenstelle.jpg "General Caution"
[image6]: ./test_images/roadworks.jpg "Roadworks"
[image7]: ./test_images/slippery.jpg "Slippery Road"
[image8]: ./test_images/snow.png "Snow/Ice"
[image9]: ./test_images/warning-pedestrian.png "Pedestrians"
[image10]: ./softmax.png "Original images and their 5 highest softmax probabilities"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/artemkaliuk/Traffic-Sign-Recognition/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how frequently each of the traffic signs are present in the corresponding data sets. It can be seen that, despite the sizes of the training, validation and test sets being different, the traffic sign images are distributed according to a similar pattern. In this visualization, the distribution data has been normalized in order to be able to compare the frequency of the corresponding traffic sign in one set to its frequency in the other one. It can be seen that some of the classes/traffic signs have a much more frequent occurance in all three data sets than the others. It can be related to a higher real-world probability of occurance as compared to the other traffic signs.

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

As a first step, images were normalized from (0,255) to (0,1) and converted to grayscale. The preprocessing function is located in cell 4. Although traffic signs do differ in color, they can be recognized and differentiated from each other by their shape. Converting to grayscale allows us to reduce the dimensionality of the input data while sustaining the acceptable accuracy in prediction. Normalization technique is applied in order to represent the images within a universal scale.

Note: in the "Traffic Sign Recognition with Multi-Scale Convolutional Networks" paper authors propose to convert the original images to YUV space with manipulation of the Y dimension following, while leaving the other two channels unchanged.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

I started with the LeNet architecture and gradually modified it. The final architecture was achieved by making the convolutional layers deeper. In order to prevent overfitting, a dropout technique was also used.
My final model consisted of the following layers:

Layer 1 - convolutional, Input: 32x32x1->28x28x6->ReLu->Pooling->14x14x12 (Output)
Layer 2 - convolutional, Input: 14x14x12->10x10x16->Pooling-> 5x5x16->625 (Flattening, Output)
Layer 3 - fully connected, Input: 625->300->Dropout->ReLu->300 (Output)
Layer 4 - fully connected, Input: 300->100->Dropout->ReLu->100 (Output)
Layer 5 - Fully connected, Input: 100->43 (Output)

 
The model is implemented in cell 5.

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

For the training, 25 Epochs were used with a batch size of 156. As an optimizer I used the Adam Optimizer presented in the LeNet lab. A learning rate of 0.0008 allowed for a decrease of the "overshooting" effects during later epochs (as compared to higher rates). For the dropout technique, a keep probability of 0.4 was used during the training session. With these hyperparameters the model was able to converge in a reasonable amount of time.

The code for model training is implemented in cells 6 and 7.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.998
* validation set accuracy of 0.967
* test set accuracy of 0.942

To achieve these results, I started by choosing the LeNet architecture (which is often used for OCR and thus, to my opinion, would be a good fit for pattern recognition required for the traffic sign classification) and followed by gradually increasing the depth of the convolutional layers. I also decided to use two fully connected layers prior to the net output (more or less a trial-and-error approach). In order to prevent overfitting, dropout technique was deployed. In order for the network to converge, I used 25 epochs; in order to prevent oscillations in the accuracy from epoch to epoch (especially at later stages of training), I decreased the learning rate to 0.0008.
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five+2 German traffic signs that I found on the web:

![alt text][image3] ![alt text][image4] ![alt text][image5] 
![alt text][image6] ![alt text][image7] ![alt text][image8]
![alt text][image9]

I have intentionally added the images "Slippery road" and "Speed Limit 50 kph" as they might be particularly difficult to classify because they were captured under an angle. As the data set might not include enough samples of such perturbations, the system might be not accurate enough in classifying such images.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:
Original image classes are: [25 18 22 27  2 30 23]
Predicted image classes are: [25 18 22 27  1 30 24]

It can be seen, that, as assumed, the "Speed Limit 50 kph" and "Slippery Road" signs were mistaken for other traffic signs.

The overall accuracy on this extra test set is thus at 0.714. This signals of underfitting. For these particular examples the best way to prevent underfitting would be to augment the training set with more images of rotated traffic signs.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook. The visualization shows the original signs and their top 5 softmax probabilities with randomly chosen corresponding images from the trainings set. It can be seen that the test images "Pedestrians", "Snow/ice", "General caution", "Roadworks" and "Bumpy road" have strong correspondences with the "ground truth" with high probabilities, whereas the images "Slippery road" and "50 kph" are classified wrongly (the 50 kph class has also a very low probability, while the class "Slippery road" was not even in the top 5 softmax probabilities for its correspondent test image).

![alt text][image10] 