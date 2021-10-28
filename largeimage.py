# Python program to explain cv2.imshow() method 
  
# importing cv2 
import cv2 
  
# path 
path = "/media/HDD/ssdlite/pytorch-ssd/I-80_Eastshore_Fwy.jpg"
  
# Reading an image in default mode
image = cv2.imread(path)
print(image.shape)
y=0
x=109
h=1152
w=1382
crop = image[y:y+h, x:x+w]
print(crop.shape)

dim = (2464, 2056)
  
# resize image
resized = cv2.resize(crop, dim, interpolation = cv2.INTER_AREA)
print(resized.shape)

filename = 'large2k.jpg'
  
# Using cv2.imwrite() method
# Saving the image
cv2.imwrite(filename, resized)
  
# List files and directories  
# in 'C:/Users / Rajnish / Desktop / GeeksforGeeks'  
print("After saving image:")  
  
print('Successfully saved')

# # Window name in which image is displayed
# window_name = 'resized'


# # Using cv2.imshow() method 
# # Displaying the image 
# cv2.imshow(window_name, resized)
  
# #waits for user to press any key 
# #(this is necessary to avoid Python kernel form crashing)
# cv2.waitKey(0) 
  
# #closing all open windows 
# cv2.destroyAllWindows() 