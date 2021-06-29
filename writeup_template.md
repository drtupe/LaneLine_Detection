# **Self Driving Engineer** 

## Project Lane Line Detection

### Introduction

One of the fundamental tasks in computer vision for autonomous driving is lane line detection on road. As this markings were made for humans to see and follow them to maintain driving dicipline, similarly they are been used by autonomous vehicles developed today to drive along side humans. 

In this project, I have implemented Computer Vision fundamental algorithm to detect lane line markings using OpenCV and Python. The project is about detection of lane lines on the road in given images and video. I have developed a pipeline on a series of individual images and later apply the results to a video stream (which is just a series of images).

In this project, I have used Python and OpenCV to meet the objective of lane line detections in Images and videos. Following are the techniques which I used,

* Color Selection
* Canny Edge Detection
* Region of Intrest Selection
* Hough Transform

Finally, I complied all these techniques into a pipeline to process a video clip to detect the lane line in them.

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report

<img src="files/examples/sample_examples/laneLines_thirdPass.jpg">

[image1]: ./examples/grayscale.jpg "Grayscale"

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of 5 steps. First, I converted the images to grayscale, then I .... 

In order to draw a single line on the left and right lanes, I modified the draw_lines() function by ...

If you'd like to include images to show how the pipeline works, here is how to include an image: 

![alt text][image1]


### 2. Identify potential shortcomings with your current pipeline


One potential shortcoming would be what would happen when ... 

Another shortcoming could be ...


### 3. Suggest possible improvements to your pipeline

A possible improvement would be to ...

Another potential improvement could be to ...
