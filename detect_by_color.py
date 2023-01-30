import cv2
import numpy as np

cap = cv2.VideoCapture("vid0.mp4")
ret, frame = cap.read()

bb=(163, 112, 537, 396)
# Only consider the region within the bounding box `bb`
bb_roi = frame[bb[1]:bb[1]+bb[3], bb[0]:bb[0]+bb[2]]
hsv = cv2.cvtColor(bb_roi, cv2.COLOR_BGR2HSV)

# Define the range of blue color in HSV color space
lower_blue = np.array([ 90,  12, 204])
upper_blue = np.array([110, 112, 304])

# Threshold the image to create a binary mask
mask = cv2.inRange(hsv, lower_blue, upper_blue)

# Find contours in the binary mask
contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Find the largest contour
largest_contour = max(contours, key=cv2.contourArea)

# Draw a bounding rectangle around the largest contour
x, y, w, h = cv2.boundingRect(largest_contour)
x += bb[0]
y += bb[1]
cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.imshow("Frame", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()

#def lower and upper for your image
"""
import cv2
import numpy as np

# Load the image
img = cv2.imread('C:/Users/meren/Desktop/obj_tracking/elmas.png')

# Convert the image to HSV color space
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Select a pixel with the desired blue color and print its HSV values
x=40
y=40
hsv_value = hsv[x, y]
print(hsv_value)

# Use the HSV values to define the lower and upper bounds for the blue color
lower_blue = np.array([hsv_value[0] - 10, hsv_value[1] - 50, hsv_value[2] - 50])
upper_blue = np.array([hsv_value[0] + 10, hsv_value[1] + 50, hsv_value[2] + 50])
"""