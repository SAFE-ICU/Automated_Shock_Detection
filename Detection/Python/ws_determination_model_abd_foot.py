from __future__ import print_function
from sklearn.externals import joblib
from skimage.feature import hog
import numpy as np
import argparse
import datetime
import imutils
import cv2
import os
from PIL import Image, ImageChops, ImageOps
import csv
#from resizeimage import resizeimage
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from numpy import genfromtxt, savetxt
from random import sample
import re
import _pickle as cPickle
all_img_path = 'C:/aditya/Thermal_images/ai_2.0/All_images_training/'
lst_all_images= os.listdir('C:/aditya/Thermal_images/ai_2.0/All_images_training/')
list_size_leg = []
TAR=[]

save_path = 'C:/aditya/Thermal_images/ai_2.0/regr_models/sk_18_py3/'

os.chdir('C:/aditya/Thermal_images/ai_2.0/All_training_data/leg/')
lst=os.listdir('./')
for name in lst:
    name1=re.sub('.png|.tif','.jpg',name)
    if name1 in lst_all_images:
        im = cv2.imread(all_img_path+name1)    
        im_leg = cv2.imread(name)
        size= (im_leg.shape[0],im_leg.shape[1])
        if len(im.shape)==2:
            im=im
        else:
            im=0.2126*im[:,:,0]+0.7152*im[:,:,1]+0.0712*im[:,:,2]
        im = im.astype(np.uint8)    
        ret ,im_thresh_binary = cv2.threshold(im,127,255,cv2.THRESH_BINARY)
        r= len(im_thresh_binary[np.nonzero(im_thresh_binary)])*1000/(im.shape[0]*im.shape[1])
        TAR.append(r)
        list_size_leg.append(size)
        
size_tb = np.array(list_size_leg)
size_x = np.ravel(size_tb[:,1])
size_y = np.ravel(size_tb[:,0])
TAR= np.array(TAR)
TAR= np.ravel(TAR) 
regr_L_tar_y= linear_model.LinearRegression()

regr_L_tar_y.fit(TAR.reshape(-1,1), size_y.reshape(-1,1)) # fit the data to the algorithm

regr_L_y_x = linear_model.LinearRegression()
regr_L_y_x.fit(size_y.reshape(-1,1),size_x.reshape(-1,1)) # fit the data to the algorithm
###############################
with open(save_path + 'regr_L_y_x', 'wb') as f:
    cPickle.dump(regr_L_y_x, f)
with open(save_path + 'regr_L_tar_y', 'wb') as f:
    cPickle.dump(regr_L_tar_y, f)

###########################
'''
lst_all_images= os.listdir(all_img_path)
list_size_leg = []
TAR=[]

os.chdir('F:/therml_images/ml_based_image_analysis/HOG_Feature_Testing/leg_foot_croped_data/')
lst=os.listdir('./')
for name in lst:
    name1=re.sub('.png|.tif','.jpg',name)
    if name1 in lst_all_images:
        im = cv2.imread('F:/therml_images/ml_based_image_analysis/all_images/'+name1)    
        im_leg = cv2.imread(name)
        size= (im_leg.shape[0],im_leg.shape[1])
        if len(im.shape)==2:
            im=im
        else:
            im=0.2126*im[:,:,0]+0.7152*im[:,:,1]+0.0712*im[:,:,2]
        im = im.astype(np.uint8)    
        ret ,im_thresh_binary = cv2.threshold(im,127,255,cv2.THRESH_BINARY)
        r= len(im_thresh_binary[np.nonzero(im_thresh_binary)])*1000/(624*1108)
        TAR.append(r)
        list_size_leg.append(size)
        
size_tb = np.array(list_size_leg)
size_x = np.ravel(size_tb[:,1])
size_y = np.ravel(size_tb[:,0])
TAR= np.array(TAR)
TAR= np.ravel(TAR) 
regr_Lf_tar_y= linear_model.LinearRegression()

regr_Lf_tar_y.fit(TAR.reshape(-1,1), size_y.reshape(-1,1)) # fit the data to the algorithm

regr_Lf_y_x = linear_model.LinearRegression()
regr_Lf_y_x.fit(size_y.reshape(-1,1),size_x.reshape(-1,1)) # fit the data to the algorithm
###########################################################################
with open('F:/therml_images/ml_based_image_analysis/HOG_Feature_Testing/regr_models/regr_Lf_y_x', 'wb') as f:
    cPickle.dump(regr_Lf_y_x, f)
with open('F:/therml_images/ml_based_image_analysis/HOG_Feature_Testing/regr_models/regr_Lf_tar_y', 'wb') as f:
    cPickle.dump(regr_Lf_tar_y, f)
#######################################################################
'''
TAR=[]
list_size_foot= []
os.chdir('C:/aditya/Thermal_images/ai_2.0/All_training_data/foot/')
lst=os.listdir('./')
for name in lst:
    name1=re.sub('.png|.tif','.jpg',name)
    if name1 in lst_all_images:
        im = cv2.imread(all_img_path+name1)    
        im_leg = cv2.imread(name)
        size= (im_leg.shape[0],im_leg.shape[1])
        if len(im.shape)==2:
            im=im
        else:
            im=0.2126*im[:,:,0]+0.7152*im[:,:,1]+0.0712*im[:,:,2]
        im = im.astype(np.uint8)    
        ret ,im_thresh_binary = cv2.threshold(im,127,255,cv2.THRESH_BINARY)
        r= len(im_thresh_binary[np.nonzero(im_thresh_binary)])*1000/(im.shape[0]*im.shape[1])
        TAR.append(r)
        list_size_foot.append(size)
        
size_tb = np.array(list_size_foot)
size_x = np.ravel(size_tb[:,1])
#size_y = size_tb[:,0].reshape(size_tb[:,1].shape[0],1)
size_y = np.ravel(size_tb[:,0])
TAR= np.array(TAR)
TAR= np.ravel(TAR)

regr_f_tar_y = linear_model.LinearRegression()
regr_f_tar_y.fit(TAR.reshape(-1,1), size_y.reshape(-1,1)) # fit the data to the algorithm

regr_f_y_x = linear_model.LinearRegression()
regr_f_y_x.fit(size_y.reshape(-1,1),size_x.reshape(-1,1)) # fit the data to the algorithm
###########################################################################
with open(save_path + 'regr_f_y_x', 'wb') as f:
    cPickle.dump(regr_f_y_x, f)
with open(save_path + 'regr_f_tar_y' , 'wb') as f:
    cPickle.dump(regr_f_tar_y, f)

############################ 
TAR=[]
list_size_abd= []
os.chdir('C:/aditya/Thermal_images/ai_2.0/All_training_data/abd/')
lst=os.listdir('./')
for name in lst:
    name1=re.sub('.png|.tif','.jpg',name)
    if name1 in lst_all_images:
        im = cv2.imread(all_img_path +name1)    
        im_leg = cv2.imread(name) 
        size= (im_leg.shape[0],im_leg.shape[1])
        if len(im.shape)==2:
            im=im
        else:
            im=0.2126*im[:,:,0]+0.7152*im[:,:,1]+0.0712*im[:,:,2]
        im = im.astype(np.uint8)    
        ret ,im_thresh_binary = cv2.threshold(im,127,255,cv2.THRESH_BINARY)
        r= len(im_thresh_binary[np.nonzero(im_thresh_binary)])*1000/(im.shape[0]*im.shape[1])
        TAR.append(r)
        list_size_abd.append(size)

size_tb = np.array(list_size_abd)
size_x = np.ravel(size_tb[:,1])
size_y = np.ravel(size_tb[:,0])
TAR= np.array(TAR)
TAR= np.ravel(TAR) 
        
regr_A_tar_y = linear_model.LinearRegression()
regr_A_tar_y.fit(TAR.reshape(-1,1), size_y.reshape(-1,1)) # fit the data to the algorithm

regr_A_y_x = linear_model.LinearRegression()
regr_A_y_x.fit(size_y.reshape(-1,1),size_x.reshape(-1,1)) # fit the data to the algorithm
###########################################################################
with open(save_path + 'regr_A_y_x', 'wb') as f:
    cPickle.dump(regr_A_y_x, f)
with open(save_path + 'regr_A_tar_y', 'wb') as f:
    cPickle.dump(regr_A_tar_y, f)



################################################################################################################
'''
list_size_diaper = []
TAR=[]

os.chdir('F:/therml_images/ml_based_image_analysis/HOG_Feature_Testing/Diaper_data/new_Diaper/')
lst=os.listdir('./')
for name in lst:
    name1=re.sub('.png|.tif','.jpg',name)
    if name1 in lst_all_images:
        im = cv2.imread('F:/therml_images/ml_based_image_analysis/all_images/'+name1)    
        im_leg = cv2.imread(name)
        size= (im_leg.shape[0],im_leg.shape[1])
        if len(im.shape)==2:
            im=im
        else:
            im=0.2126*im[:,:,0]+0.7152*im[:,:,1]+0.0712*im[:,:,2]
        im = im.astype(np.uint8)    
        ret ,im_thresh_binary = cv2.threshold(im,127,255,cv2.THRESH_BINARY)
        r= len(im_thresh_binary[np.nonzero(im_thresh_binary)])*1000/(624*1108)
        TAR.append(r)
        list_size_diaper.append(size)
        
size_tb = np.array(list_size_diaper)
size_x = np.ravel(size_tb[:,1])
size_y = np.ravel(size_tb[:,0])
TAR= np.array(TAR)
TAR= np.ravel(TAR) 
regr_D_tar_y= linear_model.LinearRegression()

regr_D_tar_y.fit(TAR.reshape(-1,1), size_y.reshape(-1,1)) # fit the data to the algorithm

regr_D_y_x = linear_model.LinearRegression()
regr_D_y_x.fit(size_y.reshape(-1,1),size_x.reshape(-1,1)) # fit the data to the algorithm
###############################
with open('F:/therml_images/ml_based_image_analysis/HOG_Feature_Testing/regr_models/regr_D_y_x', 'wb') as f:
    cPickle.dump(regr_D_y_x, f)
with open('F:/therml_images/ml_based_image_analysis/HOG_Feature_Testing/regr_models/regr_D_tar_y', 'wb') as f:
    cPickle.dump(regr_D_tar_y, f)


'''
















