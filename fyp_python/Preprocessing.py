# -*- coding: utf-8 -*-
"""
Created on Tue May 11 00:12:16 2021

@author: munee
"""
import cv2
import os
import numpy as np
from person_detection import myclass
import getopt, sys

inputfile=''
outputfile=''

argumentList=sys.argv[1:]
options='hm:o:'
long_options=['help','input','output']

try:
    args,values=getopt.getopt(argumentList,options,long_options)
    for carg,aval in args:
        if carg in ('-h','--hlep'):
            print("help")
        elif carg in ('-m','--input'):
            inputfile=aval
            print(inputfile)
        elif carg in ('-o','--output'):
            outputfile=aval
            print(outputfile)
except getopt.error as err:
    print(str(err))
    sys.exit(2)

#print(os.path.split(inputfile))
inp=os.path.split(inputfile)
width=512
height=600
dim=(width,height)
image_in=cv2.imread(inputfile)
image_in=cv2.resize(image_in,dim, interpolation = cv2.INTER_AREA)
#img_1=np.zeros([height,width,3],dtype=np.uint8)

#img_1.fill(255)
obj=myclass(image_in,inp[0])
obj.draw_rectangle()
obj.remove_bg()

print("done")

#cv2.imshow("input_img",image_in)
#cv2.imshow("img",img_1)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
