'''
image_processing.py
Authors: Andrew, Brian, Kris, Rawan, Daniel, Chau, Andres, Abe, Sophie

Image processing script for finding horizon edges, regression line, pitch and roll of satellite
Input: sample satellite Earth Horizon Sensor (EHS) images

https://learnopencv.com/edge-detection-using-opencv/
https://docs.opencv.org/4.0.0/d7/de1/tutorial_js_canny.html
https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10375389

'''

import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys, os
import math
import time
import json

# if RUNNING_MAYA:
    # from Nearspace.params import *

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from params import *

FOV_rad_y = math.radians(CAM_FOV_VERTICAL) # radians
# focal length = distance from cam lens to image sensor
# should this be aperature or sensor width?
# FOCAL_LENGTH = SENSOR_WIDTH / (2 * math.tan(FOV_rad / 2)) # 11.9 (mm?)
FOCAL_LENGTH = 5.8 # mm
# resolution of image in degrees per pixel
PIXEL_HEIGHT = CAM_FOV_VERTICAL / IMAGE_HEIGHT
PIXEL_WIDTH = CAM_FOV_HORIZONTAL / IMAGE_WIDTH
NUM_PIXELS_FROM_TOP_TO_CROP = 8
CROPPED_WIDTH = IMAGE_WIDTH
CROPPED_HEIGHT = IMAGE_HEIGHT - NUM_PIXELS_FROM_TOP_TO_CROP

showTwoImages = True

def processImage(image=None, degree="auto", img_name = None):
    '''
    Given a Earth Horizon Sensor (EHS) image, find the line that best fits the horizon
    and return the pitch, roll, and alpha of the satellite

    @params:
        image (24x32 pixels): numpy array of EHS pixels
        degree (optional): degree of polynomial to fit (1=linear, 2=quadratic, 3=cubic, "auto"=try all and pick best)
        img_name (file name, optional): name of image file to load
    @returns:
        roll (float): rotation about x axis. How tilted each line is. 0 = flat horizon. (degrees)
        pitch (float): angle of pointing up and down based on center of image. 0 = horizon is centered. (degrees)
            (not anymore) yaw (float): angle of side to side rotation, based on line midpoint and image center
        alpha: the percentage of the image filled by the Earth (float %)
        edges (1x4 array): top, right, bottom, left edges respectively. All values between [0-1] representing how much earth is on that edge
        midpoint (int, int): x, y coordinates of midpoint of filtered horizon line
        ends ( (int, int), (int, int) ): coordinates of the two endpoints of the filtered horizon line
    '''

    # ============ IMAGE GENERATION ================================

    if type(image) == type(None):
        image_directory = "images"
        # image_directory = os.path.join("imageProcessingProject", "src", "images")
        # image_name = "ehs_ir_tilted_4_cam2.png" # infrared photo
        # image_name = "ehs_ir_tilted_20.png" # infrared photo tilted
        if (img_name == None):
            # image_name = "ehs17_IR_second_1.png"
            image_name = "ehs_ir_tilted_11.png"
        else:
            image_name = img_name

        # Get the absolute path to the script's directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Construct the absolute path to the image
        image_path = os.path.join(script_dir, image_directory, image_name)

        # read our image
        img = cv2.imread(image_path)

        print(f'{image_path} loaded by OpenCV successfully!')
    else:
        img = image

    # Define cropping parameters (Accounting for the hull being in the way of the camera)
    top_pixels_to_remove = 0
    bottom_pixels_to_remove = 0
    left_pixels_to_remove = 0
    right_pixels_to_remove = 0

    # Crop the color image first
    img = img[top_pixels_to_remove : IMAGE_HEIGHT - bottom_pixels_to_remove,
              left_pixels_to_remove : IMAGE_WIDTH - right_pixels_to_remove]

    # Then convert to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    actual_height, actual_width = gray_img.shape

    # simulate our sensor output better by unfocusing and adding noise to our image
    smoothed_img = cv2.GaussianBlur(gray_img, (3, 3), 0)

    # add salt-pepper noise to image (try to mimic image in nearspace controls doc)
    # adding uniform noise to the pictures with actual cropped dimensions
    uni_noise = np.zeros((actual_height, actual_width), dtype=np.uint8)
    # create uniform noise (every number has equal chance)
    cv2.randu(uni_noise, 0, 255)  # low, high
    uni_noise_factor = 0.01
    uni_noise = (uni_noise * uni_noise_factor).astype(np.uint8)
    # create gaussian (normal) noise with correct dimensions
    normal_noise = np.zeros_like(smoothed_img, dtype=np.uint8)
    normal_noise_factor = 3
    cv2.randn(normal_noise, 0, normal_noise_factor)  # mean, stddev
    # add uniform + normal noise for better randomness
    noisy_img = np.clip(cv2.add(smoothed_img, uni_noise), 0, 255)
    noisy_img = np.clip(cv2.add(noisy_img, normal_noise), 0, 255)

    if img is None:
        print("Image not loaded correctly!")
        return -1, -1, 0.0, [-1, -1, -1, -1], (-1, -1), ((-1, -1), (-1, -1))

    # ============= IMAGE PROCESSING =================================

    # Apply Gaussian blur filter to reduce noise
    smoothed_img = cv2.GaussianBlur(noisy_img, (3, 3), 0)

    # Adjust the contrast and brightness
    alpha = 6  # Contrast control (1.0 is no change)
    beta = 0    # Brightness control (0 is no change)
    contrasted_img = cv2.convertScaleAbs(smoothed_img, alpha=alpha, beta=beta)

    # Apply Sobel edge detector: find intensity gradient for each pixel
    #     Look for large changes in pixel intensity in x and y direction
    # Combined X and Y Sobel Edge Detection (fxy partial derivative)
    # TODO: try different values, smoothed_img. Maybe print through cv2.imshow()?
    sobelxy = cv2.Sobel(src=smoothed_img, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)

    # Apply Canny edge detector: run sobel, suppress false edges, apply hysteresis thresholding
    #     If gradient is above threshold, those pixels are included in edge map
    #     If gradient is below, the pixels are suppressed and excluded from the final edge map
    #     Between thresholds = ‘weak’ edges, hysteresis mechanism will detect which belong in edge map
    # TODO: see if l2gradient parameter is needed
    canny_edges = cv2.Canny(smoothed_img, threshold1=50, threshold2=150)

    # Create another canny edge image using the increased contrast image "adjusted_img"
    canny_edges2 = cv2.Canny(contrasted_img, threshold1=50, threshold2=150)

    # extract edge pixels from both canny methods and choose the better one
    edge_pixels_1 = np.where(canny_edges != 0)
    edge_pixels_2 = np.where(canny_edges2 != 0)

    edge_coordinates_1 = np.array(list(zip(edge_pixels_1[1], edge_pixels_1[0]))) if len(edge_pixels_1[0]) > 0 else np.array([])
    edge_coordinates_2 = np.array(list(zip(edge_pixels_2[1], edge_pixels_2[0]))) if len(edge_pixels_2[0]) > 0 else np.array([])

    # Choose the edge detection result with more points (usually better for horizon)
    if len(edge_coordinates_1) == 0 and len(edge_coordinates_2) == 0:
        edge_coordinates = np.array([])
    elif len(edge_coordinates_1) >= len(edge_coordinates_2):
        edge_coordinates = edge_coordinates_1
        print(f"Using standard Canny edges: {len(edge_coordinates)} points")
    else:
        edge_coordinates = edge_coordinates_2
        print(f"Using contrast-boosted Canny edges: {len(edge_coordinates)} points")

    if len(edge_coordinates) == 0:
        # print("No edge pixels found!")
        if (np.sum(np.array(smoothed_img)) / (actual_width * actual_height)) > 20:
            # TODO: better way to calculate this constant?
            # if no edge is found but we're looking at earth, return alpha = 1
            return -1, -1, 1.0, [-1, -1, -1, -1], (-1, -1), ((-1, -1), (-1, -1))
        else:
            # if no edge is found return alpha = 0.0 and handle using alpha over time
            return -1, -1, 0.0, [-1, -1, -1, -1], (-1, -1), ((-1, -1), (-1, -1))

    # extract x and y coordinates of edge pixels
    x = edge_coordinates[:, 0]
    y = edge_coordinates[:, 1]
    # store the original
    x_uncut = edge_coordinates[:, 0]
    y_uncut = edge_coordinates[:, 1]

    # sort the edge pixels by x and y coordinates (while keeping their pairing)
    sorted_x = np.argsort(edge_coordinates[:, 0])
    sorted_edge_coordinates_x = edge_coordinates[sorted_x]
    x_sorted_x = sorted_edge_coordinates_x[:, 0]
    y_sorted_x = sorted_edge_coordinates_x[:, 1]

    sorted_y = np.argsort(edge_coordinates[:, 1])
    sorted_edge_coordinates_y = edge_coordinates[sorted_y]
    x_sorted_y = sorted_edge_coordinates_y[:, 0]
    y_sorted_y = sorted_edge_coordinates_y[:, 1]

    # define how large of a gap we allow between edge pixels before discarding one half
    max_pixel_gap = 4
    # check for split edge pieces (if two edges are detected)
    for i in range(1, len(x)) :
        # for each edge pixel, check if the gap between it and the previous pixel is too large
        if (abs((x_sorted_x[i] - x_sorted_x[i-1])) > max_pixel_gap):
            # print ("X Split detected")
            # take the larger horizon piece found
            x = x_sorted_x[i:] if i < len(x) / 2.0 else x_sorted_x[:i]
            y = y_sorted_x[i:] if i < len(y) / 2.0 else y_sorted_x[:i]
            break
        elif (abs((y_sorted_y[i] - y_sorted_y[i-1])) > max_pixel_gap):
            # check for gaps along y direction as well
            # print ("Y Split detected")
            # take the larger horizon piece found
            x = x_sorted_y[i:] if i < len(x) / 2.0 else x_sorted_y[:i]
            y = y_sorted_y[i:] if i < len(y) / 2.0 else y_sorted_y[:i]
            break

    # find the average brightness of horizon pixels to use as threshold to differentiate between space and earth
    #   this will allow us another method to recognize horizon vs space pixels
    # WARNING: watch out for 8bit int overflow
    total_edge_brightness = sum([ float(smoothed_img[y[i]][x[i]]) for i in range(len(y)) ])

    average_brightness = total_edge_brightness / float(len(y))

    threshold_brightness = average_brightness   # ADD offset here if neccessary
    # print("Average intensity of edge pixels: ", average_brightness)

    # find the percentage of the image that is the Earth (alpha)
    alpha = 0
    edge_alpha = [0,0,0,0] # top, right, bottom, left
    total_pixels = actual_width * actual_height
    # values_above_threshold = []
    num_pixels_above_threshold = 0

    for w in range(actual_width):
        for h in range(actual_height):
            # for every pixel, check if it is above the threshold
            if smoothed_img[h][w] > threshold_brightness:
                #check if pixel is an edge pixel and add to edge alpha array
                if w == 0:
                    edge_alpha[3] += 1    # left
                if w == actual_width - 1:
                    edge_alpha[1] += 1    # right
                if h == 0:
                    edge_alpha[0] += 1    # top
                if h == actual_height - 1:
                    edge_alpha[2] += 1    # bottom
                num_pixels_above_threshold += 1

    alpha = float(num_pixels_above_threshold) / float(total_pixels)

    edge_alpha[0] /= actual_width
    edge_alpha[1] /= actual_height
    edge_alpha[2] /= actual_width
    edge_alpha[3] /= actual_height

    # use numpy polynomial solver to find regression line
    # If degree is "auto", try multiple degrees and pick the best fit
    # Otherwise, use the specified degree only
    if degree == "auto":
        degrees_to_try = [1, 2, 3]
    else:
        degrees_to_try = [degree]

    best_coef = None
    best_degree = degree if degree != "auto" else 1
    best_error = float('inf')

    for deg in degrees_to_try:
        if len(x) > deg:  # Need more points than polynomial degree
            try:
                coef_temp = np.polyfit(x, y, deg)
                # Calculate fitting error (residual)
                y_pred = np.polyval(coef_temp, x)
                error = np.sum((y - y_pred) ** 2)
                print(f"  Degree {deg}: error = {error:.2f}")

                if error < best_error:
                    best_error = error
                    best_coef = coef_temp
                    best_degree = deg
            except np.RankWarning:
                print(f"  Degree {deg}: fitting failed (rank warning)")
                continue

    coef = best_coef
    print(f"Selected degree {best_degree} polynomial (error: {best_error:.2f})")
    print(f"Polynomial coefficients: {coef}")

    # Extract linear coefficients for roll calculation (use first-order terms)
    #my changes (below image_midpoint)
    if best_degree >= 1:
        a = coef[-2] if len(coef) >= 2 else 0  # coefficient of x term
        b = coef[-1]  # constant term
    else:
        a = 0
        b = coef[0]

    # find intersection point from center of image to horizon line
    # x_p = - (a * b) / (1 + a**2)
    # y_p = - b / (1 + a**2)
    # roll = math.degrees(math.atan(abs(x_p) / abs(y_p))) # according to article
    # roll = math.degrees(math.atan2(x_p, y_p))

    # tilt of horizon: only depends on slope. TODO: define bounds
    # positive regression line slope means positive roll
    # roll = math.degrees(-math.atan(a))
      # atan2(y, x) is more stable

    image_midpoint_x = actual_width / 2
    image_midpoint_y = actual_height / 2

    if best_degree >= 1:
        derivative_coef = np.polyder(coef)
        a = np.polyval(derivative_coef, image_midpoint_x)
    else:
        a = 0

    roll = math.degrees(math.atan2(-a, 1))

    #my changes
    #change degree algorithm to give 2
    #derivative_coef = np.polyder(coef)
    #slope_at_center = np.polyval(derivative_coef, image_midpoint_x)
    #roll = math.degrees(math.atan2(-slope_at_center, 1))
    # find shortest perpendicular distance between center of image and horizon line (pixels)
    # p = - (a * image_midpoint_x - image_midpoint_y + b) / math.sqrt(a**2 + 1) # gives us positives/negatives
    # print("Pixels from center to closest point on horizon line: ", p)

    # distance to horizon from nadir
    # c = math.sqrt((EARTH_RADIUS + HEIGHT)**2 - EARTH_RADIUS**2)
    # horizon is not on same level as nadir, as surface drops away as you look further
    # this offset is how many degrees below the center of the image the horizon will naturally appear if we're pointed straight down
    # O = math.degrees(np.arctan2(c, EARTH_RADIUS))
    # print("Offset angle: ", O)
    # angle from center of image to closest point on horizon line. NOTE: altitude + mount tilt can be included to convert from camera's frame
    # pitch_old = p * PIXEL_HEIGHT # - O
    # print("PITCH1: ", pitch_old)
    # this is equivalent to multiplying by pixel height, but uses trig instead
    # pitch = math.degrees(math.atan(p / FOCAL_LENGTH)) # - O

    # Compute the midpoint of the detected horizon line
    line_midpoint_x = np.mean(x)
    line_midpoint_y = np.mean(y)
    ends = (int(min(x)), int(np.polyval(coef, min(x)))), (int(max(x)), int(np.polyval(coef, max(x))))

    # Calculate yaw: Ratio of x-direction offset from center to total width, converted to degrees
    # Positive = right
    yaw_ratio = (line_midpoint_x - image_midpoint_x) / actual_width
    yaw = yaw_ratio * CAM_FOV_HORIZONTAL

    # Calculate pitch: Ratio of y-direction offset from center to total height, converted to degrees
    pitch_ratio = (image_midpoint_y - line_midpoint_y) / actual_height
    pitch = pitch_ratio * CAM_FOV_VERTICAL

    # Display the results!!! 3x3 grid
    # Check if this script is being run standalone
    if __name__ == "__main__":

        fig, axs = plt.subplots(3, 3, figsize=(12, 9), sharex=True, sharey=True)

        # Add image name as overall figure title
        if img_name:
            fig.canvas.manager.set_window_title(f'Image: {os.path.basename(img_name)}')
        else:
            fig.canvas.manager.set_window_title('Unable to Load Image Name')

        axs[0, 0].imshow(gray_img, cmap='gray')
        axs[0, 0].set_title('Original', fontsize=10)
        # Move stats text outside the image area using figure coordinates
        fig.text(0.02, 0.02, "Roll: {}\nPitch: {}\nYaw: {}\nAlpha: {}%\nTop Alpha: {}%\nRight Alpha: {}%\nBottom Alpha: {}%\nLeft Alpha: {}%".format(round(roll, 2), round(pitch, 2), round(yaw, 2), round(alpha*100, 2), round(edge_alpha[0]*100, 2), round(edge_alpha[1]*100, 2), round(edge_alpha[2]*100, 2), round(edge_alpha[3]*100, 2)),
                fontsize=8, verticalalignment='bottom',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.8))

        axs[0, 1].imshow(noisy_img, cmap='gray')
        axs[0, 1].set_title('Noisy', fontsize=10)

        axs[0, 2].imshow(smoothed_img, cmap='gray')
        axs[0, 2].set_title('Blurred', fontsize=10)

        axs[1, 0].imshow(canny_edges, cmap='gray')
        axs[1, 0].set_title('Canny Edge', fontsize=10)

        axs[1, 1].imshow(sobelxy, cmap='gray')
        axs[1, 1].set_title('Sobel Edge', fontsize=10)

        # axs[1, 2].imshow(contrasted_img, cmap='gray')
        # axs[1, 2].set_title('Contrast Boosted', fontsize=10)

        # axs[2, 0].imshow(canny_edges2, cmap='gray')
        # axs[2, 0].set_title('Contrast-boosted edge', fontsize=10)

        axs[2, 1].imshow(gray_img, cmap='gray')
        axs[2, 1].set_title('TODO: custom edge', fontsize=10)
        axs[2, 1].scatter(x_uncut, y_uncut, color='orange', s=3, label='Unmanaged Edge')
        axs[2, 1].legend()

        # find each point along our fitted horizon lines
        x_horizon = np.linspace(min(x), max(x), 100)  # More points for smoother curve
        # make sure all line points are within image bounds
        y_horizon = np.clip(np.polyval(coef, x_horizon), 0, actual_height)

        # Also fit a simple linear line for comparison
        linear_coef = np.polyfit(x, y, 1)
        y_linear = np.clip(np.polyval(linear_coef, x_horizon), 0, actual_height)

        axs[2, 2].imshow(gray_img, cmap='gray')
        axs[2, 2].set_title(f'Polynomial Fit (degree {best_degree})', fontsize=10)
        # overlay our fitted horizon line

        #my changes
        #axs[2, 2].plot(x_horizon, y_horizon, color='red', linewidth=2, label=f'Degree {best_degree} fit')
        axs[2, 2].plot(x_horizon, y_horizon, color='red', linewidth=2, label=f'Quadratic fit')
        axs[2, 2].plot(x_horizon, y_linear, color='orange', linewidth=1, linestyle='--', label='Linear fit')
        axs[2, 2].scatter(x, y, color='blue', s=3, label='Edge pixels')
        axs[2, 2].legend(loc='upper left', fontsize=7)

        # Add fitting statistics as text at the bottom left, avoiding legend
        axs[2, 2].text(2, actual_height-2, f'Error: {best_error:.1f} | Deg: {best_degree} | Pts: {len(x)}',
                      fontsize=7, color='white', bbox=dict(boxstyle="round,pad=0.2", facecolor='black', alpha=0.8),
                      verticalalignment='top')

        # Adjust spacing to prevent overlaps
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15, left=0.15)  # Make room for text

        if (img_name == None):
            plt.show()

        print("=" * 50)
        print("PROCESSING RESULTS:")
        print("-" * 20)
        print(f"Polynomial Fitting:")
        print(f"  Degree: {best_degree}")
        print(f"  Error: {best_error:.3f}")
        print(f"  Edge points: {len(x)}")
        print(f"  Coefficients: {[f'{c:.4f}' for c in coef]}")
        print(f"Image Analysis:")
        print(f"  Threshold: {threshold_brightness:.1f}")
        print(f"  Alpha: {alpha:.3f} ({alpha*100:.1f}%)")
        print(f"Attitude Results:")
        print(f"  ROLL: {roll:.2f}°")
        print(f"  PITCH: {pitch:.2f}°")
        print(f"  YAW: {yaw:.2f}°")
        print(f"Edge Coverage:")
        print(f"  Top: {edge_alpha[0]*100:.1f}%")
        print(f"  Right: {edge_alpha[1]*100:.1f}%")
        print(f"  Bottom: {edge_alpha[2]*100:.1f}%")
        print(f"  Left: {edge_alpha[3]*100:.1f}%")
        print(f"Horizon Line Midpoint: ({line_midpoint_x:.1f}, {line_midpoint_y:.1f})")
        print("Horizon Line Ends: ({}, {}), ({}, {})".format(ends[0][0], ends[0][1], ends[1][0], ends[1][1]))
        print("=" * 50)

    return roll, pitch, alpha, edge_alpha, (line_midpoint_x, line_midpoint_y), ends


def firmwareImageProcessing(image):
    """
    simplified C++ image processing function.
    """
    # Define cropping parameters
    top_pixels_to_remove = NUM_PIXELS_FROM_TOP_TO_CROP
    bottom_pixels_to_remove = 0
    left_pixels_to_remove = 0
    right_pixels_to_remove = 0

    # Crop the color image first
    img = image[top_pixels_to_remove : IMAGE_HEIGHT - bottom_pixels_to_remove,
              left_pixels_to_remove : IMAGE_WIDTH - right_pixels_to_remove]

    # Then convert to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    actual_height, actual_width = gray_img.shape

    # simulate our sensor output better by unfocusing and adding noise to our image
    smoothed_img_noise = cv2.GaussianBlur(gray_img, (3, 3), 0)

    # add salt-pepper noise to image
    uni_noise = np.zeros((actual_height, actual_width), dtype=np.uint8)
    cv2.randu(uni_noise, 0, 255)
    uni_noise_factor = 0.01
    uni_noise = (uni_noise * uni_noise_factor).astype(np.uint8)
    
    normal_noise = np.zeros_like(smoothed_img_noise, dtype=np.uint8)
    normal_noise_factor = 3
    cv2.randn(normal_noise, 0, normal_noise_factor)
    
    noisy_img = np.clip(cv2.add(smoothed_img_noise, uni_noise), 0, 255)
    noisy_img = np.clip(cv2.add(noisy_img, normal_noise), 0, 255)

    # ============= FIRMWARE IMAGE PROCESSING =================================
    # 2. Image Processing
    # Reduce noise with Gaussian Blur
    smoothed_img = cv2.GaussianBlur(noisy_img, (3, 3), 0)

    # Apply Canny Edge Detector
    canny_edges_mask = cv2.Canny(smoothed_img, 50, 150, 3)

    # Find coordinates of edge pixels
    y_coords, x_coords = np.where(canny_edges_mask > 0)
    num_edge_pixels = len(x_coords)

    # 3. Handle cases
    if num_edge_pixels == 0:
        # Determine if we're looking at Earth or only space
        if np.mean(smoothed_img) > 20:
            alpha = 1.0
        else:
            alpha = 0.0
        return -1, -1, alpha, [-1, -1, -1, -1], (-1, -1), ((-1, -1), (-1, -1))

    # 4. Edge Pixel Analysis
    # Find the average brightness of horizon pixels
    total_edge_brightness = np.sum(smoothed_img[y_coords, x_coords])
    average_brightness = total_edge_brightness / num_edge_pixels
    threshold_brightness = average_brightness

    # Find the percentage of the image that is the Earth (alpha)
    # num_pixels_above_threshold = np.sum(smoothed_img > threshold_brightness)
    # alpha = num_pixels_above_threshold / (CROPPED_WIDTH * CROPPED_HEIGHT)
    alpha = 0
    edge_alpha = [0,0,0,0] # top, right, bottom, left
    total_pixels = actual_width * actual_height
    # values_above_threshold = []
    num_pixels_above_threshold = 0

    for w in range(actual_width):
        for h in range(actual_height):
            # for every pixel, check if it is above the threshold
            if smoothed_img[h][w] > threshold_brightness:
                #check if pixel is an edge pixel and add to edge alpha array
                if w == 0:
                    edge_alpha[3] += 1    # left
                if w == actual_width - 1:
                    edge_alpha[1] += 1    # right
                if h == 0:
                    edge_alpha[0] += 1    # top
                if h == actual_height - 1:
                    edge_alpha[2] += 1    # bottom
                num_pixels_above_threshold += 1

    alpha = float(num_pixels_above_threshold) / float(total_pixels)

    edge_alpha[0] /= actual_width
    edge_alpha[1] /= actual_height
    edge_alpha[2] /= actual_width
    edge_alpha[3] /= actual_height

    # 5. Linear Regression
    a = 0.0  # slope
    b = 0.0  # intercept

    if num_edge_pixels >= 2:
        x_mean = np.mean(x_coords)
        denom = np.sum((x_coords - x_mean) ** 2)
        if abs(denom) > 1e-12:
            y_mean = np.mean(y_coords)
            num = np.sum((x_coords - x_mean) * (y_coords - y_mean))
            a = num / denom
            b = y_mean - a * x_mean
        else:
            # All x are (almost) identical -> horizontal line
            b = np.mean(y_coords)
    elif num_edge_pixels == 1:
        b = y_coords[0]

    # 6. Roll and Pitch Calculation
    roll = math.degrees(math.atan2(-a, 1))

    y_c = CROPPED_HEIGHT / 2.0
    line_midpoint_y = np.mean(y_coords)

    pitch_ratio = (y_c - line_midpoint_y) / CROPPED_HEIGHT
    pitch = pitch_ratio * CAM_FOV_VERTICAL

    return roll, pitch, alpha, edge_alpha, (-1, -1), ((-1, -1), (-1, -1))


if __name__ == "__main__":

    if len(sys.argv) > 1:
        # Use the filename provided as a command-line argument
        image_path = sys.argv[1]
        roll, pitch, alpha, edges, line_midpoint, ends = processImage(None, 1, image_path)
        print(roll, pitch, alpha, edges)
        plt.show()
    else:
        if showTwoImages:
            # displays both angles of the image on two separate windows

            image_num = 13 # kinda near nadir
            # image_num = 39 # 1 cam upside down
            # image_num = 87 # totally sideways
            # image_num = 729 # 1 cam seeing mostly earth

            image_path1 = f"ehs{image_num}_IR_first_1.png"
            image_path2 = f"ehs{image_num}_IR_second_1.png"

            image_path1 = "down70.png" # tilt = -40
            # 6.197228350447218e-15 -12.604166666666671 0.6302083333333334 [1.0, 0.59375, 0.0, 0.59375]

            image_path1 = "down50.png" # 50 degrees down (tilt = -20) = horizon at bottom of image
            # -0.4234828325222835 -29.93489583333333 0.7877604166666666 [1.0, 0.75, 0.0, 0.75]

            # down30 = all earth
            # swing through earth = ~10 degrees to see any space, ~20 degrees to see entire edge

            image_path1 = "down20.png" # 20 degrees down
            # max pitch ~= 50
            # -2.7861502315372853e-16 47.265625 0.9153645833333334 [0.0, 0.875, 1.0, 0.875]

            image_path1 = "rotated_hand.png"
            # 0.8967302800580964 23.203125 0.6953125 [0.0, 0.65625, 1.0, 0.65625]

            image_path2 = "up20.png" # 20 degrees up (cam tilt = 50)
            # 0.34875261160221255 10.3125 0.5794270833333334 [0.0, 0.5625, 1.0, 0.5625]

            image_path2 = "up40.png" # 40 degrees up
            # -8.069843488168754e-15 -0.5729166666666707 0.4830729166666667 [0.0, 0.46875, 1.0, 0.46875]

            image_path2 = "up60.png" # 60 degrees up (cam_tilt = 90)
            # 0.42348283252230867 -11.315104166666671 0.3854166666666667 [0.0, 0.375, 1.0, 0.34375]

            image_path2 = "up70.png"
            # 0.6476638335736823 -16.90104166666667 0.3307291666666667 [0.0, 0.3125, 1.0, 0.3125]

            image_path2 = "up95.png" # cam_tilt = 125
            # lowest pitch ~= -40
            # 2.1627230086327196e-14 -42.39583333333333 0.10026041666666667 [0.0, 0.0625, 1.0, 0.0625]

            image_path1 = os.path.join("rotation_sequence", image_path1)
            image_path2 = os.path.join("rotation_sequence", image_path2)

            # Set poly degree here
            roll1, pitch1, alpha1, edges1, midpoint1, ends1 = processImage(None, 2, image_path1)
            print(roll1, pitch1, alpha1, edges1)
            roll2, pitch2, alpha2, edges2, midpoint2, ends2 = processImage(None, 2, image_path2)
            print(roll2, pitch2, alpha2, edges2)

            plt.show()
        else:
            processImage(degree=2)


    # =========  SPEED TESTING  ==========================
    # image_path = "ehs_ir_tilted_15.png"

    # image_directory = "images"
    # # Get the absolute path to the script's directory
    # script_dir = os.path.dirname(os.path.abspath(__file__))
    # # Construct the absolute path to the image
    # image_path = os.path.join(script_dir, image_directory, image_path)

    # # read our image and convert to grayscale
    # img = cv2.imread(image_path)

    # num_runs = 1000
    # total_time = 0.0

    # for i in range(num_runs):
    #     start = time.time()

    #     result = processImage(img)

    #     end = time.time()
    #     elapsed = (end - start) * 1000  # milliseconds
    #     total_time += elapsed

    #     # print(f"Run {i + 1}: {elapsed:.2f} ms")

    # avg_time = total_time / num_runs
    # print(f"\nAverage time per run: {avg_time:.2f} ms")