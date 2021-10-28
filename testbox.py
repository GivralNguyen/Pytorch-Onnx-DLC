# Python program to explain cv2.rectangle() method 
   
# importing cv2 
import cv2 
   
# path 
path = r'/media/HDD/ssdlite/pytorch-ssd/car.jpg'
   
# Reading an image in default mode
image = cv2.imread(path)
   
# Window name in which image is displayed
window_name = 'Image'
  
# Start coordinate, here (5, 5)
# represents the top left corner of rectangle
start_point = (192,277)
  
# Ending coordinate, here (220, 220)
# represents the bottom right corner of rectangle
end_point = (378,457)
  
# Blue color in BGR
color = (255, 0, 0)
  
# Line thickness of 2 px
thickness = 2
  
# Using cv2.rectangle() method
# Draw a rectangle with blue line borders of thickness of 2 px
image = cv2.rectangle(image, start_point, end_point, color, thickness)
  
# Displaying the image 
cv2.imwrite("testbox.jpg", image) 