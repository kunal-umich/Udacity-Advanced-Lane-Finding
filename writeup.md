# **Project: Advanced Lane Finding** 

## Writeup for my implementation

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.


## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

Note: All of the code is available in the notebook Project2.ipynb
      The output images are present in output_images folder
      The final output video is in the main folder and titled: project_video_output.mp4


### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.


1) I started by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

2) I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  For the code, I started by using the base code provided in the example folder and made sure that I used  a 9x6 chessboard size for the project instead of 8x6 used in the class notes.

3) Then I passed all the twenty chessboard images, converted them to grayscale and passed them to the `cv2.findChessboardCorners` function to identify the chessboard corners in the image. Then, I appended the corners returned by the funcion to the `imgpoints` list and the object points (which is the same for each image) to the `objpoints` list.

4) The `imgpoints` and  `objpoints` were then passed to the OpenCV funcion `cv2.calibrateCamera()` to get the Camera Calibration matrix `mtx` and the distortion coefficients `dist` (which contains radial and tangential coefficients for the camera).
 Code snippet below:
 
 ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints,imgpoints,gray.shape[::-1],None,None)

5) Finally,I applied this distortion correction to the test image using the `cv2.undistort()` function as shown below:
   
   undistorted = cv2.undistort(img, mtx, dist, None, mtx)

Example of undistorted image is shown below:

![Original Distorted Image](https://view5639f7e7.udacity-student-workspaces.com/view/CarND-Advanced-Lane-Lines/output_images/Original_distorted_Image.jpg)

![Undistorted Image](https://view5639f7e7.udacity-student-workspaces.com/view/CarND-Advanced-Lane-Lines/output_images/Undistorted_Image.jpg)


### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:

![Original Distorted Image](https://view5639f7e7.udacity-student-workspaces.com/view/CarND-Advanced-Lane-Lines/output_images/Original_Lane_distorted_Image.jpg)
![Undistorted Image](https://view5639f7e7.udacity-student-workspaces.com/view/CarND-Advanced-Lane-Lines/output_images/Undistorted_Lane_Image.jpg)

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

1) I defined the following functions to try out different combinations to transform the image:

   `rgb_to_HLS(img)`: Function to convert RGB image to HLS colorspace
   
   `abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255))` : Function to take gradient in x or y direction
   
   `magnitude_thresh(image, sobel_kernel=3, thresh=(0, 255))`: Function to calculate magnitude of gradient by taking both x and y gradients
   
   `dir_threshold(image, sobel_kernel=3, thresh=(0, np.pi/2))`: Function to calculate direction gradients to identify vertical lines
  
2) Finally in the `binary_image(img)` I tried out different combinations and called the above functions to apply the transformation.
   Finally I decided to only use threshold on the `s-channel` in the range (160, 250) and the Sobel-x gradient with threshold in the range    (30, 100).
   
    The code for the same is available in the 4th code cell in the Project2.ipynb notebook.
    
Here's an example of my output for this step.

![Original Undistorted Image](https://view5639f7e7.udacity-student-workspaces.com/view/CarND-Advanced-Lane-Lines/output_images/Test_image6_Undistorted.jpg)

![Resulting Binary Image](https://view5639f7e7.udacity-student-workspaces.com/view/CarND-Advanced-Lane-Lines/output_images/Test_image6_Binary_threshold.jpg)


#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.


1) First I defined a function called `perspective_transform(img)` to transform the image to bird's eye view. 
   
   In the funtion, I defined the source points from the staright line image provided for testing. I found the pixel positions using an            interactive image viewer on Windows. The points I came up with are shown below:
   
   src = np.float32([[195,720],[1135,720], [690,450],[590,450]])
   
2) After that I came up with the destination points to warp my selected source points into a rectangular region.
   The points I came up with are shown below:
   
    offset = 320
    height = img.shape[0]   ##720
    width = img.shape[1]    ##1280
    dst = np.float32([[offset,height],[width-offset,height],[width-offset,0],[offset,0]])
    
3) This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 195,  720     | 320, 720      | 
| 1135, 720     | 960, 720      |
| 690,  450     | 960, 0        |
| 590,  450     | 320, 0        |

4) After defining the source and destination points, I used the `cv2.getPerspectiveTransform(src, dst)` function to calculate the              perspective transform matrix for warping the image.
   I also calculated the Inverse transform matrix for unwarping image later for displaying the boundary lines back onto the original image.
   This was done using `cv2.getPerspectiveTransform(dst, src)` function by swapping the source and destination points.
   
   The code for the same is available in the 5th code cell in the Project2.ipynb notebook.
   
Example images for the above function is shown below:

![Original Undistorted Image](https://view5639f7e7.udacity-student-workspaces.com/view/CarND-Advanced-Lane-Lines/output_images/straight_lines1_Undistorted.jpg)

![Selected region in original Image](https://view5639f7e7.udacity-student-workspaces.com/view/CarND-Advanced-Lane-Lines/output_images/straight_lines1_selected_region.jpg)

![Warped Image](https://view5639f7e7.udacity-student-workspaces.com/view/CarND-Advanced-Lane-Lines/output_images/straight_lines1_warped.jpg)


#### Combining Perspective transform and Binary Transform:

1) I defined a function `perspective_binary_processing(img)`, in which I first called the `binary_image(img)` defined earlier to convert      the image to binary.

2) I also defined a selected region of interest (ROI) to eliminate the area with shadows in the image as it may cause problems while trying    to use a histogram to detect lane line areas.

3) After masking the image with the selected ROI, I passed the image to the `perspective_transform(img)` function to transform the image to    bird's eye view.

   The code for the same is available in the 6th and 7th code cell in the Project2.ipynb notebook.

Example images for the combined function are shown below:

![Original Undistorted Image](https://view5639f7e7.udacity-student-workspaces.com/view/CarND-Advanced-Lane-Lines/output_images/Test_image6_Undistorted.jpg)

![Resulting Binary Perspective Transformed Image](https://view5639f7e7.udacity-student-workspaces.com/view/CarND-Advanced-Lane-Lines/output_images/Test_image6_Perspective_binary_transformed.jpg)


#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

1) To identify lane-line pixels, I first used histogram method to identify the pixels in the binary image where most white pixels are          detected. For this, I defined `histo(img)` function which calculated histogram for the bottom-half of the image (as lane lines are most    likely to be vertical near to the car).
  
   The code for the same is available in the 9th code cell in the Project2.ipynb notebook.
   
2) Then, I defined a function `find_lane_pixels(binary_warped)` which will use sliding window method to follow the lane lines.
   In the function, I first called the histogram function `histo(img)` defined above to identify region containing the lane lines.
   Then I identified the 2 peaks of the histogram to separate the left and right lane lines. The peaks then serve as the starting points      for the two lane lines.
   Then, I defined the hyper-parameters for my sliding window.
   After that, I identified all the `x` and `y` positions of all the non-zero pixels (white pixels) in the image using the following code:
   
   nonzero = binary_warped.nonzero()
   
   nonzeroy = np.array(nonzero[0])
   
   nonzerox = np.array(nonzero[1])
    
   Then, I iterated through all of the windows defined by the hyper-parameters and identified the non-zero pixels in each window. The 
   windows were also recentered based on the number of pixels detected to left and right.
   
   The code for the same is available in the 10th code cell in the Project2.ipynb notebook.
   
3) A function was defined to fit polynomial through the pixels identified as left and right lane lines. 
   The function is called  `fit_polynomial(binary_warped,ym_per_pix=1,xm_per_pix=1)`.
   
   The parameters `ym_per_pix` and `xm_per_pix` are 1 by default and when the function is called for calculating the radius of curvature,      the actual value for the parameters is passed.
   
   In the `fit_polynomial` function, the left and right pixel positions are obtained by calling the `find_lane_pixels` function defined        above. Then using `np.polyfit`, we get the polynomial coefficients to fit a polynomial through the lane line pixels.
   
   The code for the same is available in the 10th code cell in the Project2.ipynb notebook.

Output image of the above step are shown below:

![Sliding window and curve fitted image](https://view5639f7e7.udacity-student-workspaces.com/view/CarND-Advanced-Lane-Lines/output_images/Test_image6_Sliding_window_polyfit.png)


#### Using previous polynomial fit to skip sliding window

1) To improve processing, we can use a targeted search around the polynomial fit obtained using a sliding window.. This helps to              limit our area of search to a confined space in the image.
    
2) To achieve this I defined a function `search_around_poly(binary_warped,left_fit,right_fit)` in which the last detected polynomial fits      for the left and right lanes are provided as input. 
   First, the the `x` and `y` positions of all the non-zero pixels (white pixels) in the new image are detected. 
   Then, the left and right lane indices of non-zero pixels are identified within the search area defined using the previous polynomial        fits.
   Using these indices our new polynomial fits are calculated.
   
   The code for the same is available in the 15th code cell in the Project2.ipynb notebook.
   
 
Output image of the above step is shown below:

![Lane detection using Targeted Search](https://view5639f7e7.udacity-student-workspaces.com/view/CarND-Advanced-Lane-Lines/output_images/Test_image3_Targeted_search.jpg)

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

#### Radius of Curvature

1) I defined a function called `radius_curvature(img)` to calculate the radius of curvature.

2) I used following global variables to convert `x` and `y` values in pixels to meters:
   
   ym_per_pix = 30/img.shape[0]   
   #Standard lane length in US is 30 meters and img.shape[0] is height of image in pixels
   
   xm_per_pix = 3.7/700           
   #Standard lane width in US is 3.7 meters and 700 is avearge distance b/w lanes in image in pixels
   
3) I called the `fit_polynomial(img,ym_per_pix,xm_per_pix)` function defined earlier and passed the actual `ym_per_pix` and `xm_per_pix`      values to calculate the polynomial fits in meters instead of pixels. 

4) I used the polynomial fit values obtained to calculate the radius of curvature at the bottom of the image, i.e, near to the car's          location using the following formula:

   R_curve = ((1+(2Ay+B)^2)^3/2)/ |2A|
   
   where, 
   
   A and B are polynomial coefficients
   
   y is the location with respect to which we are calculating the radius of curvature.
          
   
   The code for the same is available in the 13th code cell in the Project2.ipynb notebook.
   
#### Position of the vehicle with respect to center

1) I defined a function called `dist_from_center(img,left_fit,right_fit)` to calculate distance from center.

2) In the function, the most recent polynomial fits are passed as input. Using the polynomial fits, I calculated the left and right lane      `x` positions at the bottom of the image, i.e, near to the car's location.
   Then I calculated the center of the lane by averaging the `x` positions of the two lane lines as shown below:
   
   y = img.shape[0]  ##Bottom point of the image
   
   #Find the x-axis values for left and right lanes at the bottom of the image
   
   left_fitx = left_fit[0]*y**2 + left_fit[1]*y + left_fit[2]
   
   right_fitx = right_fit[0]*y**2 + right_fit[1]*y + right_fit[2]
   
   lane_center = (right_fitx + left_fitx)/2
   
   lane_center = (right_fitx + left_fitx)/2
   
3) Then, I calculated the center of the image, which is just half of the width of the image.

   center = img.shape[1]/2
   
4) As the difference between the image center from lane center is the offset of the vehicle from the lane center, it can be calculated as      follows:    
    
   dist_from_center = (center - lane_center)*xm_per_pix
   
   This gives us the distance in meters.
   
   The code for the same is available in the 14th code cell in the Project2.ipynb notebook.


#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in  the 16th code cell in the Project2.ipynb notebook.
Here is an example of my result on a test image:

![Lane boundary warped back on original image](https://view5639f7e7.udacity-student-workspaces.com/view/CarND-Advanced-Lane-Lines/output_images/Test_image6_Unwarped_and_Displayed.jpg)

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

![link to my video result](https://view5639f7e7.udacity-student-workspaces.com/view/CarND-Advanced-Lane-Lines/project_video_output.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  


### Issues/Problems Faced and Improvements:

1) I faced issues with tuning my binary image parameters and what methods to apply. The areas in the shadows were especially tricky.
   To improve robustness in areas with shadows on the image edges, I defined an ROI to eliminate those areas.
   
2) The Perspective transform function could be improved by better selection and tuning of the source and destination points.

3) I found that in the video, in some places the lane boundary outputs are jittery, especially around the end of the video.
   This might be fixed by performing some sanity checks or by taking average over the previous measurements to achieve smoother results.
   The main hindrance in implementing this was that I couldn't understand the function of various instance variables defined in the            Line class. I would love to know how this functionality can be implemented to make the pipeline more robust. 
