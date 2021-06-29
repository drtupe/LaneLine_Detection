# **Lane Line Detection** 

## **Introduction**
---
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

This is the final result expected as output from from this project.
<img src="files/examples/sample_examples/laneLines_thirdPass.jpg">

We will be using the following image through the project demonstrating results of various steps.
<img src="files/test_images/solidYellowCurve.jpg">

### Color Selection:

Color selection can be done using various color models such as RGB, HSV and HSL. RGB being a basic additive color model is based on mixing colors Red, Green and Blue varying in range from 0 to 255 to create whole range of colors. HSV is a little complex and was design to understand how human's persive vision perceives color attributes ([link](https://labs.imaginea.com/rbg-vs-hsv-for-computer-vision/)). We'll be using RGB for white lane lines and HSV for yellow lane lines detection. Following is the code to acquire white and yellow lane lines from image:

* converting given RGB image into HSV color model
HSV_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

* detecting white lines whose intensity varies from 200 to 255 for RGB colors.
White = cv2.inRange(image, (200, 200, 200), (255, 255, 255))

* detecting yellow lines using HSV converted image
* the values for rue 17, saturation 70 were obtained from trial and error basis
Yellow = cv2.inRange(HSV_image, (17, 70, 34), (30, 255, 255))

* once both white and yellow lanes are obtained, we will merge them using bitwise_or function from cv2 and set other color pixels to 0.
mask_color = cv2.bitwise_or(White, Yellow)
masked_image = np.copy(image)
masked_image[mask_color == 0] = [0, 0, 0]

Once the image has been masked and only yellow and white pixels are available in the image and other colors have been eliminated, this process has now reduced the importance of color in the image. This allows us to convert the image to a grayscale image for easy processing.

Following is the result obtained from color selection:
<img src="files/examples/color_selection.jpg\">

### Smoothing Image:

Once the masked image is converted to gray image, we can now undertake the smoothing of image. We will be using gaussian smoothing, this smoothing process reduce the level of noise in the image. This improves the result of the following edge-detection process.

gray = grayscale(masked_image)
kernal_size = 5
blur_gray = gaussian_blur(gray, kernal_size)
Following is the result obtained from smoothing of gray scale image:
<img src="files/examples/blur_gray.jpg\">

### Canny Edge Detection:

Canny edge detection is a technique which extracts useful information from image which in our case is to detect lane lines in a gray scale image. This helps us to reduce the amount of data to be processed. Canny edge detection computes the gradient across 2 direction x and y, this produces a matrix representing the difference intensity between neighbouring pixels. The algorithm used low threshold and high_threshold to detect and filter out pixels in given range. In our case we have considered low threshold = 200 and high threshold = 250 which were selected on trial and error bases to get optimal result. Pixels values between low and high threshold are included as they are connected to strong edges. This results in a binary image at output with white edge detected.

low_threshold = 200
high_threshold = 250
edges = canny(gray, low_threshold, high_threshold)

Following is the result obtained from Canny Edge Detection:
<img src="files/examples/canny_edges.jpg\">

### Region of interest area Selection:

As we can see from the output of canny edge detection process outcome, there are many unnecessary things which have been detected and processed from the image other than lanes on road. To eliminated this noise, we make use of region of interest in the image. As we all know that camera's are mounted at a fixed position on the car, we can define a fixed polygon region where lane lines are mostly expected to be present. For our example, we have selected a trapezoidal region starting from the bottom of the image which goes towards the center.

mask = np.zeros_like(edges)   
ignore_mask_color = 255 
imshape = image.shape
    # vertices = np.array([[(image.shape[0]-100,imshape[0]),(425, 300), (500, 300), (900,imshape[0])]], dtype=np.int32)
    
point_A = (imshape[1]*0.1, imshape[0])       # (50,imshape[0])
point_B = (imshape[1]*0.45, imshape[0]*0.6)   # (425, 300)
point_C = (imshape[1]*0.55, imshape[0]*0.6)   # (500, 300)
point_D = (imshape[1]*0.95, imshape[0])       # (900,imshape[0])

vertices = np.array([[point_A,point_B, point_C, point_D]], dtype=np.int32)
    
cv2.fillPoly(mask, vertices, ignore_mask_color)
masked_edges = cv2.bitwise_and(edges, mask)

During region of interest selection we first create a mask (which is initially all zeros matrix similar to our grayscale image) using cv2.fillPoly function. In cv2.fillPoly function we have to supply with vertices of the polygon which we want to select. Onces vertices are provied for the polygon then all the pixels within that region are set to 255. Once this process is completed, this mask is applied to edge detection image using bitwise_and function.

<img src="files/examples/ROI.jpg\">

### Hough Transform:

Till now we have processed the image and obtained the pixels from which a line equation can be calculated. For this purpose we will be using Hough Transform. So what exactly is a hough transform? Hough transform takes pixel's coordinate from image space which are (x, y) and then transform them to hough space which is represented by (m, b) in graphical representation. So, a line in image space is represented as a point in hough space. For more details on Hough Transform [click here](https://en.wikipedia.org/wiki/Hough_transform)

rho = 1
theta = np.pi/180
threshold = 20
min_line_length = 20
max_line_gap = 300
line_image = np.copy(image)*0 # For creating a blank to draw lines on
    
* masked edges is the output image of region of intrest
    
lines = hough_lines(masked_edges, rho, theta, threshold, min_line_length, max_line_gap)

* color_edges = np.dstack((masked_edges, masked_edges, masked_edges))

combo = weighted_img(lines, image, 0.8, 1, 0)

Here above you can see the hough_line method takes in lot of parameters to work and tune properly, (more details [here](https://docs.opencv.org/3.4/dd/d1a/group__imgproc__feature.html#ga8618180a5948286384e3b7ca02f6feeb)),
* rho: This is the distance resolution of the pixels.
    
* theta: Angle resolution of the pixels.
    
* threshold: Only lines above this threshold are selected.

* min_line_length: Line segments less than this are rejected.

* max_line_length: This is the max allowed gap between 2 points to link as a line.

With all these parameters and constrains, lines are computed with the HoughLinesP function that applies the transformation on the edges. Onces the lines are found then they are feed to draw_lines function.

def draw_lines(img, lines, color=[0, 0, 255], thickness=10):
   
left_lane, right_lane = average_intercept_slope(lines)

y1 = img.shape[0]

y2 = y1 * 0.6
    
left_line = line_points(y1, y2, left_lane)

right_line = line_points(y1, y2, right_lane)

if left_line and right_line is not None:

cv2.line(img, (left_line[0], left_line[1]), (left_line[2], left_line[3]), [255, 255, 0], thickness)

cv2.line(img, (right_line[0], right_line[1]), (right_line[2], right_line[3]), [0, 255, 255], thickness)


### Averaging and Extrapolating Lines:

So, from the result of Hough Transform, we have got multiple lines detected for a lane line. In these lines few are partially recorgnized, so to work on them we will be using averaging and extrapolating the lines to cover the full lane lines in the image.

We have to do this calculation for 2 lane lines, one for left line and other for right. So, lines from Hough Transform are grouped in left and right category based on the slopes. The left lane should have positive slope and the right lane should have a negative slope. Using the below mentioned method we can calculate average slope (m) and intercept (b) for the left and right lanes for each image.

def average_intercept_slope(lines):

left_line = []
left_len = []
right_line = []
right_len = []
  
for line in lines:
    for x1, y1, x2, y2 in line:
        if x2 == x1:
            continue
        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1
        length = np.sqrt((y2 - y1)**2 + (x2 - x1)**2)
        if m < 0:
            left_line.append((m, b))
            left_len.append(length)
        else:
            right_line.append((m, b))
            right_len.append(length)
            
Once slope and intercepts are obtained, we will then have to calculate the x co-ordinates so as to locate the pixel location on the image.

def line_points(y1, y2, line):
    if line is None:
        return None
    m, b = line
    x1 = int((y1 - b) / m)
    x2 = int((y2 - b) / m)
    y1 = int(y1)
    y2 = int(y2)
    return [x1, y1, x2, y2]
    
Once averaging and extrapolation is done, we get the points where we can draw the lines on image. For easy distinction of left and right lane, I have used different color for both the line to be represented. Following is the final result which we obtained.

<img src="files/examples/lane_lines/.jpg\">

### Conclusion:

This Project was successful as per given objectives which you can clearly see in this [video](./test_videos_output/solidWhiteRight.mp4).

As mentioned in the objective at the start of the project, the project was all about finding the lane lines on road. It was successfull in detecting those but, it does not work smoothly on [curved lane lines](./test_videos_output/challenge.mp4). To make this work smoothly on a all types of lane lines, we'll have to use perspective transformtion and also poly fitting lane lines instead of straight lines.

This project is the basis of finding lane line detection program as all the lane lines are almost straight near the car and curve appears at further distance. So this techinque is much helpful in understanding how image processing works in lane detection.

