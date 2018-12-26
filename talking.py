# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 10:26:24 2018

@author: Aolme8
"""
import cv2 as cv2
import time as time
import imutils
import numpy as np
from matplotlib import pyplot as plt
from imutils import face_utils
import imutils
import dlib
#from keras import layers
#from keras import models
#import scipy.ndimage
#import os


video = 'test.mp4'


# Function from: http://programmingcomputervision.com/
def draw_flow(im,flow,step=8):
    """ Plot optical flow at sample points
        spaced step pixels apart. """
        
    h,w = im.shape[:2]
    y,x = np.mgrid[step/2:h:step,step/2:w:step].reshape(2,-1)
    fx,fy = flow[y.astype(int),x.astype(int)].T
        
    # create line endpoints
    lines = np.vstack([x,y,x+fx,y+fy]).T.reshape(-1,2,2)
    lines = np.int32(lines)
    
    # create image and draw
    vis = cv2.cvtColor(im,cv2.COLOR_GRAY2BGR)
    for (x1,y1),(x2,y2) in lines:
        cv2.line(vis,(x1,y1),(x2,y2),(0,255,0),1)
        cv2.circle(vis,(x1,y1),1,(0,255,0), -1)
    
    return vis

# Modified Function from: https://www.pyimagesearch.com/2017/04/10/detect-eyes-nose-lips-jaw-dlib-opencv-python/
def detect_face(image_file, shape_predictor):
    roi = []
    # initialize dlib's face detector (HOG-based) and then create
    # the facial landmark predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(shape_predictor)

    image = image_file
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # detect faces in the grayscale image
    rects = detector(gray, 1)
    
    # loop over the face detections
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the landmark (x, y)-coordinates to a NumPy array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

    # loop over the face parts individually
        for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
            if (name == 'mouth'):
                # loop over the subset of facial landmarks, drawing the
                # specific face part
                for (x, y) in shape[i:j]:
                    cv2.circle(image, (x, y), 1, (0, 0, 255), -1)

                # extract the ROI of the face region as a separate image
                (x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))
                #r = cv2.selectROI(clone)
                roi.append((x, y, w, h))

    return roi

        
def talking(video):
    false_positives = 0
    false_negatives = 0
    true_positives = 0
    true_negatives = 0
    
    count = 0

    print ("Start")
    camera = cv2.VideoCapture(video)
    time.sleep(0.25)
    
    ret, frame1 = camera.read()
        
    prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    prvs  = imutils.resize(prvs, width=700)
    frame1  = imutils.resize(frame1, width=700)
    hsv = np.zeros_like(frame1)
    hsv[...,1] = 255
    
    print("Start getting frames")
    camera.set(cv2.CAP_PROP_FPS, 10)
    while True:
        count = count + 1
        green = False
            #slow down the fps
        grabbed, frame = camera.read()
        if grabbed:
            frame = imutils.resize(frame, width=700)
            pass
        else:
            break
        
        color = frame[3,3]
        print(color)
        if (color[1] == 255):
            green = True
            
        mouth_roi = detect_face(frame, 'shape_predictor_68_face_landmarks.dat')

        # blur the frame and convert it to the HSV color space
        blurred = cv2.GaussianBlur(frame, (11,11), 0)
        kernel2 = np.array([[0, -1, 0],
                           [-1, 5, -1],
                            [0, -1, 0]])
        blurred = cv2.filter2D(blurred, -1, kernel2)
        blurred = cv2.medianBlur(blurred, 15)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        
        grayframe = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) 
        grayframe = cv2.medianBlur(grayframe, 5)       
        current_frame = grayframe
    
        flow = cv2.calcOpticalFlowFarneback(prvs, grayframe, None, 0.5, 3, 18, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        mag[mag == -np.inf] = 0.001
        hsv[...,0] = ang*180/np.pi/2
        ang = ang*180/np.pi/2
        
                
        prvs = current_frame.astype(np.uint8)

        motion_vectors = draw_flow(current_frame, flow)
        
        count_roi = 0
        count_r = 0
        talking = False
        for roi in mouth_roi:
            for x in range(roi[0], roi[0]+roi[2]):
                for y in range(roi[1], roi[1]+roi[3]):
                    if (mag[y, x] >= 1 and mag[y, x] <=2.5):
                        talking = True
                        if (green == True):
                            count_roi = count_roi + 1
                        elif (green == False):
                            count_r = count_r + 1
            if (talking == True):
                cv2.putText(motion_vectors, "talking", (roi[0],roi[1]+15), cv2.FONT_HERSHEY_DUPLEX, 0.3, (255,0,0), 1)
            cv2.rectangle(motion_vectors, (roi[0],roi[1]), (roi[0]+roi[2], roi[1]+roi[3]), (255,0,0), 1)
        
        if (talking == False and green == False):
            true_negatives = true_negatives + 1
        elif (talking == False and green == True):
            false_negatives = false_negatives + 1
            
        
        if (count_roi != 0):
            true_positives = true_positives + 1
        elif(count_r != 0):
            false_positives = false_positives + 1
            
      
        cv2.imshow('Motion vector plot', motion_vectors)
        
        print ("True Positives: ")
        print (true_positives)
        print ("True Negatives: ")
        print (true_negatives)
        print ("False Positives: ")
        print (false_positives)
        print ("False Negatives: ")
        print (false_negatives)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        


    camera.release()
    cv2.destroyAllWindows()
    print ("FINAL RESULTS")
    print ("True Positives: ")
    print (true_positives)
    print ("True Negatives: ")
    print (true_negatives)
    print ("False Positives: ")
    print (false_positives)
    print ("False Negatives: ")
    print (false_negatives)
    return (true_positives + true_negatives)/count



### Implementation Starts Here###

accuracy = talking(video)
print (accuracy)
print ("finish")

#model = models.Sequential()
#model.add(layers.Conv2D(32, (3, 3), activation='relu',
#                        input_shape=(150, 150, 3)))
#model.add(layers.MaxPooling2D((2, 2)))
#model.add(layers.Conv2D(64, (3, 3), activation='relu'))
#model.add(layers.MaxPooling2D((2, 2)))
#model.add(layers.Conv2D(128, (3, 3), activation='relu'))
#model.add(layers.MaxPooling2D((2, 2)))
#model.add(layers.Conv2D(128, (3, 3), activation='relu'))
#model.add(layers.MaxPooling2D((2, 2)))
#model.add(layers.Flatten())
#model.add(layers.Dropout(0.5))   # Dropout layer at 50%
#model.add(layers.Dense(512, activation='relu'))
#model.add(layers.Dense(1, activation='sigmoid'))
#
#
#model.summary()
#
#
#from keras import optimizers
#
#model.compile(loss='binary_crossentropy',
#              optimizer=optimizers.RMSprop(lr=1e-4),
#              metrics=['acc'])
#
#
#
#
#''' Data Processing '''
#
#
#from keras.preprocessing.image import ImageDataGenerator
#
#train_datagen = ImageDataGenerator(rescale=1./255)
#test_datagen = ImageDataGenerator(rescale=1./255)
#
#
#base_dir = '../Keras/cats_and_dogs_small'    # Need to change the dictionary if it is necessary
#train_dir = os.path.join(base_dir, 'train')
#validation_dir = os.path.join(base_dir, 'validation')
#
#train_generator = train_datagen.flow_from_directory(
#        train_dir,              # Images are here.
#        target_size=(150, 150), # All images will be resized to 150x150
#        batch_size=20,          # Process twenty images at a time.
#        class_mode='binary')    # 0/1 entries.
#
#
## validation_generator processes data from given directory.
#validation_generator = test_datagen.flow_from_directory(
#        validation_dir,
#        target_size=(150, 150),
#        batch_size=20,
#        class_mode='binary')
#
#for data_batch, labels_batch in train_generator:
#    print('data batch shape:', data_batch.shape)
#    print('labels batch shape:', labels_batch.shape)
#    break
#
#history = model.fit_generator(
#      train_generator,
#      steps_per_epoch=100, # MUST be carefully set. See above.
#      epochs=30,           # MUST be carefully set. See above.
#      validation_data=validation_generator,
#      validation_steps=50) # MUST be carefully set. See above.
#
#model.save('talking_1.h5')
#
## Plots
#''' Plot the loss and accuracy of the model '''
#
#import matplotlib.pyplot as plt
#
#acc = history.history['acc']
#val_acc = history.history['val_acc']
#loss = history.history['loss']
#val_loss = history.history['val_loss']
#
#epochs = range(len(acc))
#
#
#plt.plot(epochs, acc, 'bo', label='Training acc')
#plt.plot(epochs, val_acc, 'b', label='Validation acc')
#plt.title('Training and validation accuracy')
#plt.legend()
#
#plt.figure()
#
#plt.plot(epochs, loss, 'bo', label='Training loss')
#plt.plot(epochs, val_loss, 'b', label='Validation loss')
#plt.title('Training and validation loss')
#plt.legend()
#
#plt.show()