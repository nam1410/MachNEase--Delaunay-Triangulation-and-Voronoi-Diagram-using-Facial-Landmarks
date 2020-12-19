#MachNEase
#Author : Namitha Guruprasad
#LinkedIn : linkedin.com/in/namitha-guruprasad-216362155
#Import important libraries
import cv2
import dlib

#obtain the image feed
#path of the file - if stored in the same directory. Else, give the relative path
frame = cv2.imread('harvey.jpg')
#for detecting faces
detector = dlib.get_frontal_face_detector()
#facial landmarks - identifier
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#clahe - Contrast Limited Adaptive Histogram Equalization - to avoid over - brightness, contrast limitation and amplification of noise
#image is divided into small blocks called "tiles" (tileSize is 8x8 by default in OpenCV)
#if histogram bin is above the specified contrast limit - pixels are clipped and distributed uniformly
#here bi-polar interpolation is implemented
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)) #optional arguments
clahe_image = clahe.apply(gray)

#detect the faces in the image
detections = detector(clahe_image, 1) 
for k,d in enumerate(detections):
    #for every detected face
    #create a text file to save the landmark coordinates
    coordinates_txt = open("harv" + ".txt","w")
    shape = predictor(clahe_image, d) 
    for i in range(1,68):
        #68 landmark points on each face
        cv2.circle(frame, (shape.part(i).x, shape.part(i).y), 1, (0,0,255), thickness=2) #For each point, draw a red circle with thickness2 on the frame
        coordinates_txt.write(str(shape.part(i).x) + " " + str(shape.part(i).y) + "\n")
        #drawing the landmarks with thickness 3 - yellow in color

#display the imageframe
cv2.imshow("image", frame) 


