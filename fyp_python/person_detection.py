# -*- coding: utf-8 -*-
"""
Created on Tue May 11 02:35:00 2021

@author: muneeb
"""
from removebgg import RemoveBg
import cv2
import numpy as np
import os
import ntpath
from num2words import num2words

class myclass():
    def __init__(self,frame,impath):
        mode = 0o666
        self.path=os.path.join(impath,"person_detection_outputs/")
        self.pathbg=os.path.join(impath,"output_without_bg/")
        self.blank_path=os.path.join(impath,"blank.png")
        try:
            os.makedirs(self.path,mode)
        except:
            pass
        try:
            os.makedirs(self.pathbg,mode)
        except:
            pass
        self.apikey='W3zaorheWPAXV7hrJXuq7kBj'
        self.img=[]
        weight_path="./fyp_persondetection_files/yolov3-tiny.weights"
        cfg_path="./fyp_persondetection_files/yolov3-tiny.cfg"
        self.net = cv2.dnn.readNet(weight_path, cfg_path)
        layer_names = self.net.getLayerNames()
        self.output_layers = [layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
        self.detectperson(frame)
        print("class object")
    def createperson(self):
        print("person created ")
    def detectperson(self,frame):
        self.frame1=frame.copy()
        height, width, channels = frame.shape
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        
        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)
        self.confidences = []
        self.boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.2:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    self.boxes.append([x, y, w, h])
                    self.confidences.append(float(confidence))
        self.indexes = cv2.dnn.NMSBoxes(self.boxes, self.confidences, 0.8, 0.3)
        img = np.zeros([100,100,3],dtype=np.uint8)
        img.fill(255) # or img[:] = 255  
        
    def draw_rectangle(self):
        for i in range(len(self.boxes)):
            nimg=self.frame1.copy()
            for j in range(len(self.boxes)):
                if j==i:
                    print(self.boxes[j])
                    pass
                else:
                    if j in self.indexes:
                        x, y, w, h = self.boxes[j]
                        cv2.rectangle(nimg, (x, y), (x + w, y + h), (255,255,255), -1)
            self.img.append(self.frame1[y:y+h,x:x+w])
#            filename=num2words(i)
            filename=num2words(i)
            cv2.imwrite(self.path+filename+".png",nimg)
                
    def remove_bg(self):
        filenames=[]
        rmbg = RemoveBg(self.apikey, "error.log")
        for i in os.listdir(self.path):
            filenames.append(self.path+"/"+i)
        if filenames:
            for j in filenames:
                a=rmbg.remove_background_from_img_file(j)
                with open(self.pathbg+str(ntpath.basename(j)), 'wb') as removed_bg_file:
                    removed_bg_file.write(a.content)
    def changebg(self):
        for i in range(len(self.boxes)):
            filename=num2words(i)
            if i in self.indexes:
                x, y, w, h = self.boxes[i]
                blank=cv2.imread(self.blank_path)
                newbg=cv2.imread(self.pathbg+str(filename)+".png",-1)
                blank1=cv2.addWeighted(blank[y:y+h,x:x+w],0.4,newbg,0.1,0)
                cv2.imshow("f",blank1)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
    
                