#we read and show an image
import pandas as pd 
import cv2
img = cv2.imread('../../../Codes/Jupyter Notebook/Pokemon/Images/296.jpg')
cv2.imshow('pikachu',img)
gray = cv2.imread('../../../Codes/Jupyter Notebook/Pokemon/Images/296.jpg',cv2.IMREAD_GRAYSCALE)
cv2.imshow('greyed_out',gray)
cv2.waitKey(0)
cv2.destroyAllWindows()
