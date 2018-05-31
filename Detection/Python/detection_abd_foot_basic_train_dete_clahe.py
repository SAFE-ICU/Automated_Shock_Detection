from __future__ import print_function
from sklearn.externals import joblib
from skimage.feature import hog
import numpy as np
import argparse
import datetime
import imutils
import cv2
import os
import re
from PIL import Image, ImageChops, ImageOps
import csv
#from resizeimage import resizeimage
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from numpy import genfromtxt, savetxt
import cPickle
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage.feature import hog
from skimage import data, color, exposure
from scipy import ndimage
import math

########## Loading pre-trained models ################
with open('C:/aditya/Thermal_images/ai_2.0/All_training_data/models/finals/sk18_rf_F_NF_100pad5_in127_ori12', 'rb') as f:
   rf_F = cPickle.load(f)


with open('C:/aditya/Thermal_images/ai_2.0/All_training_data/models/finals/sk18_rf_A_NA_50pad_in127_ori9', 'rb') as f:
    rf_A = cPickle.load(f)

'''
with open('F:/therml_images/ml_based_image_analysis/HOG_Feature_Testing/rf_models/rf_D_ND_model2', 'rb') as f:
    rf_D = cPickle.load(f)     
with open('F:/therml_images/ml_based_image_analysis/HOG_Feature_Testing/rf_models/rf_H_NH_model2', 'rb') as f:
    rf_H = cPickle.load(f)
'''
with open('C:/aditya/Thermal_images/ai_2.0/regr_models/sk_18/regr_A_tar_y', 'rb') as f:
    regr_A_tar_y = cPickle.load(f)     
with open('C:/aditya/Thermal_images/ai_2.0/regr_models/sk_18/regr_A_y_x', 'rb') as f: 
    regr_A_y_x = cPickle.load(f)     
'''
with open('C:/aditya/Thermal_images/HOG_Feature_Testing/regr_models/regr_L_tar_y', 'rb') as f:
    regr_L_tar_y = cPickle.load(f)     
with open('C:/aditya/Thermal_images/HOG_Feature_Testing/regr_models/regr_L_y_x', 'rb') as f:
    regr_L_y_x = cPickle.load(f)      
'''
with open('C:/aditya/Thermal_images/ai_2.0/regr_models/sk_18/regr_f_tar_y', 'rb') as f:
    regr_f_tar_y = cPickle.load(f)     
with open('C:/aditya/Thermal_images/ai_2.0/regr_models/sk_18/regr_f_y_x', 'rb') as f:
    regr_f_y_x = cPickle.load(f)      



list_profiles = []
list_profile_name = []
#detection_path = "C:/aditya/Thermal_images/ai_2.0/Detection_All_Collected_Data/Data_for_Detection/Detections/sept2016/dete_foot_abd_clahe_with_rot/"
detection_path = "C:/aditya/Thermal_images/ai_2.0/Detection_All_Collected_Data/Data_for_Detection/Detections/May_july_august2016/dete_foot_abd_clahe_not_rot2/"
if not os.path.exists(detection_path):
    os.makedirs(detection_path)
profile_path = "C:/aditya/Thermal_images/ai_2.0/Detection_All_Collected_Data/Data_for_Detection/Detections/May_july_august2016/profiles_clahe_not_rot2/"  
if not os.path.exists(profile_path):
    os.makedirs(profile_path)    

def grep(pattern,word_list):
    expr = re.compile(pattern)
    return [elem for elem in word_list if expr.match(elem)]
#main_path = 'C:/aditya/Thermal_images/ai_2.0/Analysed_data_dete/Analysed_data_clahe1/Analysed_data/'
#main_path = 'C:/aditya/Thermal_images/ai_2.0/Detection_All_Collected_Data/Data_for_Detection/Sept2016/manual_detection_ambika/'
main_path = 'C:/aditya/Thermal_images/ai_2.0/Detection_All_Collected_Data/Data_for_Detection/May_July_August2016/'
#main_path = 'C:/aditya/Thermal_images/ai_2.0/Detection_All_Collected_Data/Data_for_Detection/feb_march_april2017/'

folders = os.listdir(main_path)
for folder in folders :
    bed_path= main_path + folder + '/'

    beds = grep('^bed|control',os.listdir(bed_path))
    #save_path = 'F:/therml_images/detection_res_profiles/handed over by ambika_thermal Images/' +folder+'/'
    for bed in beds:
        img_path = bed_path + bed + '/'
        grp = grep('During|Pre|Post|Non-Shock|Shock|New folder|Resolved',os.listdir(img_path))
        #names = grep('IMG|img|vlc',os.listdir(img_path))
        #save_img_path = save_path + bed +'/'
        #name_edit.append(re.sub('detection_profile_','',names))
        if  len(grp)!= 0:
            for gr in grp:
                grp_path = img_path + gr + '/'
                imgs = grep('IMG|img|vlc|Img',os.listdir(grp_path))
                if len(imgs)==2:
                    imgs = imgs[1]
                else:
                    imgs = imgs[0]
                dete = grep("dete_abd_foot_" + imgs,os.listdir(detection_path))
                if len(dete)==0:
                    name = imgs                   
                    new_img1 = cv2.imread(grp_path+name)
                    new_img = cv2.imread(grp_path+name)
                    ####### fig ######
                    fig,ax = plt.subplots(2)
                    ax[0]
                    if len(new_img.shape)==2:
                        new_img = new_img
                    else:
                        new_img= cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
                        #new_img = 0.2126*new_img[:,:,0]+0.7152*new_img[:,:,1]+0.0712*new_img[:,:,2]
                    new_img = new_img.astype(np.uint8)
                    #clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                    #new_img_patch = clahe.apply(new_img)
                    #min_in = 150
                    #ret , new_img = cv2.threshold(new_img_patch,min_in,255,cv2.THRESH_TOZERO)
                    [R,C] = new_img.shape
                    if R > C :
                        new_img = ndimage.rotate(new_img, 90)
                    ret ,im_thresh_binary = cv2.threshold(new_img,127,255,cv2.ADAPTIVE_THRESH_MEAN_C)
                    if len(new_img.shape)==3:
                        TAR = len(im_thresh_binary[np.nonzero(im_thresh_binary)])*1000/(new_img.shape[0]*new_img.shape[1])
                    else:
                        TAR = len(im_thresh_binary[np.nonzero(im_thresh_binary)])*1000/(new_img.size)
               ###### calculating the adaptive window size ############################################################## 
                    
                    
                    y = regr_f_tar_y.predict(TAR)[0,0]
                    x = regr_f_y_x.predict(y)[0,0]
                    win_size= (int(y),int(x))
                    #win_size= (250,550)
               #################### Foot detection ############################################         
                    list_test_hog = []
                    list_box_coord = []
                #cv2.imwrite('F:/therml_images/ml_based_image_analysis/HOG_Feature_Testing/shock_no_shock/shock_img_detection/'+'grey'+name,new_img)
                #ret,new_img = cv2.threshold(new_img,127,255,cv2.THRESH_TOZERO)
                #cv2.imwrite('F:/therml_images/ml_based_image_analysis/HOG_Feature_Testing/shock_no_shock/shock_img_detection/'+'grey_thresh'+name,new_img)
                #hog_image, hog_fig = hog(new_img, orientations=8, pixels_per_cell=(64,64),cells_per_block=(1, 1), visualise=True)
                #hog_image_rescaled = exposure.rescale_intensity(hog_fig, in_range=(0, 0.02))
                #cv2.imwrite('F:/therml_images/ml_based_image_analysis/HOG_Feature_Testing/shock_no_shock/shock_img_detection/'+'hog_grey_thresh'+name,hog_fig)
                    ax[0].imshow(new_img, cmap = 'gray')#ax[0].imshow(new_img) #ax[0].imshow(new_img,cmap = 'gray')
                    #ax[0,1].imshow(new_img,cmap = 'gray')
                    for i in range(0,(new_img.shape[0]-win_size[0]),10):
                        for j in range(0,(new_img.shape[1]-win_size[1]),15):
                            new_img_patch = new_img[i:(i+win_size[0]),j:(j+win_size[1])]
                            #new_img_patch = new_img_patch.astype(np.uint8)
                            #min_in = np.amin(new_img_patch)
                            min_in = 127
                            #new_img_patch = cv2.equalizeHist(new_img_patch)
                            #clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(12,12))
                            #new_img_patch = clahe.apply(new_img_patch)
                            ret ,im_thresh= cv2.threshold(new_img_patch,min_in,255,cv2.THRESH_TOZERO)
                            im_thresh = np.lib.pad(im_thresh,5, 'constant', constant_values = 0)
                            red_patch = cv2.resize(im_thresh, (100,100))
                            hog_image= hog(red_patch, orientations=12, pixels_per_cell=(8, 8),cells_per_block=(2, 3), visualise=False)            
                            box_position_l = j
                            box_position_t = i
                            box_position_r = new_img.shape[1]-(j+win_size[1])
                            box_position_b = new_img.shape[0]-(i+win_size[0])
                            box_coord=(box_position_l,box_position_t, box_position_r, box_position_b)  
                            list_test_hog.append(hog_image)
                            list_box_coord.append(box_coord)
                            
                    box_coord= np.array(list_box_coord)
                    test_hog = np.array(list_test_hog)
                    pred_win = rf_F.predict(test_hog)
                    pred_prob = rf_F.predict_proba(test_hog)
                    high_prob_pos = box_coord[pred_prob[:,0]>0.98]
                    n_leg = box_coord[pred_prob[:,1]==max(pred_prob[:,1])]
                    cds= range(len(n_leg))
                    ###### top and btm img weights #######                
                    top_img = new_img[0:(new_img.shape[0]),0:((new_img.shape[1])/2)]
                    ret ,top_thresh= cv2.threshold(top_img,min_in,255,cv2.THRESH_TOZERO)
                    btm_img = new_img[0:(new_img.shape[0]),((new_img.shape[1])/2):((new_img.shape[1]))]
                    ret ,btm_thresh= cv2.threshold(btm_img,min_in,255,cv2.THRESH_TOZERO)
                    ####### condition to choose from #######    
                    if(len(top_img[np.nonzero(top_thresh)]) > len(top_img[np.nonzero(btm_thresh)])):
                        n_leg_extream = n_leg[(new_img.shape[1] - n_leg[:,0])==min(new_img.shape[1] - n_leg[:,0])]
                    else:    
                        n_leg_extream = n_leg[(new_img.shape[1] - n_leg[:,0])==max(new_img.shape[1] - n_leg[:,0])]
                    ##### react at extream point #####                    
                    #if n_leg_extream.shape[0]!=1:
                    n_leg_extream = n_leg_extream[0,:]
                        
                    ax[0].add_patch(patches.Rectangle((n_leg_extream[0],n_leg_extream[1]),win_size[1],win_size[0],linewidth=1,edgecolor='b',facecolor='none'))
                    lc =(n_leg_extream[0]+(win_size[1])/2, n_leg_extream[1]+(win_size[0])/2)
                    radius = int((win_size[0])*(0.1))
                    ax[0].add_patch(patches.Circle(lc,radius))
                    patch_coordi  = ((win_size[1])/2,(win_size[0])/2)
                    new_img_patch = new_img[(n_leg_extream[1]):((n_leg_extream[1])+win_size[0]),(n_leg_extream[0]):((n_leg_extream[0])+win_size[1])]
                    new_img_patch = new_img_patch.astype(np.uint8)
                    #min_in = np.amin(new_img_patch)
                    min_in = 100
                    ret ,patch_thresh= cv2.threshold(new_img_patch,min_in,255,cv2.THRESH_TOZERO)
                    radius = int((win_size[0])*(0.1))
                    a = patch_thresh
                    cx, cy = patch_coordi[0], patch_coordi[1] # The center of circle
                    y, x = np.ogrid[-radius: radius, -radius: radius]
                    index = x**2 + y**2 <= radius**2
                    a_with_circle = a[cy-radius:cy+radius, cx-radius:cx+radius][index]    
                    a_nonzero = a_with_circle[np.nonzero(a_with_circle)]
                    if len(a_nonzero)!=0:
                        med_circle_foot = np.median(a_nonzero)
                    else:
                        list_test_hog = []
                        list_box_coord = []
                        ax[0].imshow(new_img, cmap = 'gray')#ax[0].imshow(new_img) #ax[0].imshow(new_img,cmap = 'gray') 
                        for i in range(0,(new_img.shape[0]-win_size[0]),10): ## ################# again do it for foot with clahe ##################
                            for j in range(0,(new_img.shape[1]-win_size[1]),15):
                                new_img_patch = new_img[i:(i+win_size[0]),j:(j+win_size[1])]
                                #new_img_patch = new_img_patch.astype(np.uint8)
                                #min_in = np.amin(new_img_patch)
                                min_in = 150
                                #new_img_patch = cv2.equalizeHist(new_img_patch)
                                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(12,12))
                                new_img_patch = clahe.apply(new_img_patch)
                                ret ,im_thresh= cv2.threshold(new_img_patch,min_in,255,cv2.THRESH_TOZERO)
                                im_thresh = np.lib.pad(im_thresh,5, 'constant', constant_values = 0)
                                red_patch = cv2.resize(im_thresh, (100,100))
                                hog_image= hog(red_patch, orientations=12, pixels_per_cell=(8, 8),cells_per_block=(2, 3), visualise=False)            
                                box_position_l = j
                                box_position_t = i
                                box_position_r = new_img.shape[1]-(j+win_size[1])
                                box_position_b = new_img.shape[0]-(i+win_size[0])
                                box_coord=(box_position_l,box_position_t, box_position_r, box_position_b)  
                                list_test_hog.append(hog_image)
                                list_box_coord.append(box_coord)
                                
                        box_coord= np.array(list_box_coord)
                        test_hog = np.array(list_test_hog)
                        pred_win= rf_F.predict(test_hog)
                        pred_prob= rf_F.predict_proba(test_hog)
                        high_prob_pos = box_coord[pred_prob[:,0]>0.98]
                        n_leg = box_coord[pred_prob[:,1]==max(pred_prob[:,1])]
                        cds= range(len(n_leg))
                        ###### top and btm img weights #######                
                        top_img = new_img[0:(new_img.shape[0]),0:((new_img.shape[1])/2)]
                        ret ,top_thresh= cv2.threshold(top_img,min_in,255,cv2.THRESH_TOZERO)
                        btm_img = new_img[0:(new_img.shape[0]),((new_img.shape[1])/2):((new_img.shape[1]))]
                        ret ,btm_thresh= cv2.threshold(btm_img,min_in,255,cv2.THRESH_TOZERO)
                        ####### condition to choose from #######    
                        if(len(top_img[np.nonzero(top_thresh)]) > len(top_img[np.nonzero(btm_thresh)])):
                            n_leg_extream = n_leg[(new_img.shape[1] - n_leg[:,0])==min(new_img.shape[1] - n_leg[:,0])]
                        else:    
                            n_leg_extream = n_leg[(new_img.shape[1] - n_leg[:,0])==max(new_img.shape[1] - n_leg[:,0])]
                        ##### react at extream point #####                    
                        #if n_leg_extream.shape[0]!=1:
                        n_leg_extream = n_leg_extream[0,:]
                            
                        ax[0].add_patch(patches.Rectangle((n_leg_extream[0],n_leg_extream[1]),win_size[1],win_size[0],linewidth=1,edgecolor='b',facecolor='none'))
                        lc =(n_leg_extream[0]+(win_size[1])/2, n_leg_extream[1]+(win_size[0])/2)
                        radius = int((win_size[0])*(0.1))
                        ax[0].add_patch(patches.Circle(lc,radius))
                        patch_coordi  = ((win_size[1])/2,(win_size[0])/2)
                        new_img_patch = new_img[(n_leg_extream[1]):((n_leg_extream[1])+win_size[0]),(n_leg_extream[0]):((n_leg_extream[0])+win_size[1])]
                        new_img_patch = new_img_patch.astype(np.uint8)
                        #min_in = np.amin(new_img_patch)
                        min_in = 110
                        ret ,patch_thresh= cv2.threshold(new_img_patch,min_in,255,cv2.THRESH_TOZERO)
                        radius = int((win_size[0])*(0.1))
                        a = patch_thresh
                        cx, cy = patch_coordi[0], patch_coordi[1] # The center of circle
                        y, x = np.ogrid[-radius: radius, -radius: radius]
                        index = x**2 + y**2 <= radius**2
                        a_with_circle = a[cy-radius:cy+radius, cx-radius:cx+radius][index]    
                        a_nonzero = a_with_circle[np.nonzero(a_with_circle)]
                        if len(a_nonzero)!=0:
                            med_circle_foot = np.median(a_nonzero)
                        else:
                            med_circle_foot = 0

                    #non_zero_patch_foot = patch_thresh[np.nonzero(patch_thresh)]
                    #med_patch_foot = np.median(non_zero_patch_foot)
                    print(med_circle_foot)
                    #non_zero_patch_foot = patch_thresh[np.nonzero(patch_thresh)]
                    #med_patch_foot = np.median(non_zero_patch_foot)
                    '''
                    for cd in cds:
                        ax[0].add_patch(patches.Rectangle((n_leg[cd,0],n_leg[cd,1]),win_size[1],win_size[0],linewidth=1,edgecolor='b',facecolor='none'))
                        lc =(n_leg[(len(cds)-1),0]+(win_size[1])/2, n_leg[(len(cds)-1),1]+(win_size[0])/2)
                        patch_coordi  = ((win_size[1])/2,(win_size[0])/2)
                        new_img_patch = new_img[(n_leg[cds[len(cds)-1],1]):(n_leg[(len(cds)-1),1]+win_size[1]),(n_leg[(len(cds)-1),0]):(n_leg[(len(cds)-1),0]+win_size[0])]
                        new_img_patch = new_img_patch.astype(np.uint8)
                        #min_in = np.amin(new_img_patch)
                        min_in = 127
                        ret ,patch_thresh= cv2.threshold(new_img_patch,min_in,255,cv2.THRESH_TOZERO)
                        radius = int((win_size[0])*(0.2))
                        a = patch_thresh
                        cx, cy = patch_coordi[0], patch_coordi[1] # The center of circle
                        y, x = np.ogrid[-radius: radius, -radius: radius]
                        index = x**2 + y**2 <= radius**2
                        a_with_circle = a[cy-radius:cy+radius, cx-radius:cx+radius][index]    
                        a_nonzero = a_with_circle[np.nonzero(a_with_circle)]
                        if len(a_nonzero)!=0:
                            med_patch_foot = np.median(a_nonzero)
                        else:
                            med_patch_foot = 0
                        
                        contours,hierarchy = cv2.findContours(im_thresh, 1, 2)
                        cnt = contours[0]
                        M = cv2.moments(cnt)
                        if M['m00']!=0:
                            cx = int(M['m10']/M['m00'])
                            cy = int(M['m01']/M['m00'])
                        else:
                            cx = lc[0]
                            cy = lc[1]
                        area = cv2.contourArea(cnt)
                        perimeter = cv2.arcLength(cnt,True)
                        epsilon = 0.1*cv2.arcLength(cnt,True)
                        approx = cv2.approxPolyDP(cnt,epsilon,True)
                        '''
                
                    ####### sliding window ##########-2016-09-03-20h02m21s219
                    #head_indice= np.argmax(pred_prob[pred_win=='H'])
                    #abd_indice= np.argmax(pred_prob[pred_win=='A'])
                    #diaper_indice= np.argmax(pred_prob[pred_win=='D'])
                    #leg_indice= np.argmax(pred_prob[pred_win=='L'])
                        
                    #plt.show()
                ################################### abd detection ############################# 
                    list_test_hog = []
                    list_box_coord= []
                    y = regr_A_tar_y.predict(TAR)[0,0]
                    x = regr_A_y_x.predict(y)[0,0]
                    win_size_predicted = (int(y),int(x))
                    win_size= (350,430)
                    if win_size_predicted > win_size:
                        win_size = win_size
                    else:
                        win_size = win_size_predicted   
                    for i in range(0,(new_img.shape[0]-win_size[0]),10):
                        for j in range(0,(new_img.shape[1]-win_size[1]),10):
                            new_img_patch = new_img[i:(i+win_size[0]) ,j:(j+win_size[1])]
                            #new_img_patch = new_img_patch.astype(np.uint8)
                            #min_in = np.amin(new_img_patch)
                            min_in = 127
                            #equ = cv2.equalizeHist(new_img_patch)
                            #clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(12,12))
                            #new_img_patch = clahe.apply(new_img_patch)
                            ret ,im_thresh= cv2.threshold(new_img_patch,min_in,255,cv2.THRESH_TOZERO)
                            im_thresh = np.lib.pad(im_thresh,5, 'constant', constant_values = 0 )
                            red_patch = cv2.resize(im_thresh, (50, 50))
                            #contours, hierarchy = cv2.findContours(im_thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
                            #ctr = np.array(contours)
                            #cv2.drawContours(im_thresh, ctr, 3, (0,255,0), 3) 
                            hog_image = hog(red_patch, orientations= 9 , pixels_per_cell=(8, 8),cells_per_block=(2, 3), visualise=False)
                            box_position_l = j
                            box_position_t = i
                            box_position_r = new_img.shape[1]-(j+win_size[1])
                            box_position_b = new_img.shape[0]-(i+win_size[0])
                            box_coord=(box_position_l,box_position_t, box_position_r, box_position_b)  
                            list_test_hog.append(hog_image)
                            list_box_coord.append(box_coord)    
                            
                    box_coord= np.array(list_box_coord)
                    test_hog = np.array(list_test_hog)
                    pred_win = rf_A.predict(test_hog)
                    pred_prob= rf_A.predict_proba(test_hog)
                    high_prob_pos = box_coord[pred_prob[:,0]>0.98] 
                    n_abd= box_coord[pred_prob[:,1]==max(pred_prob[:,1])]
                    cds= range(len(n_abd))
                    for cd in cds:
                        ax[0].add_patch(patches.Rectangle((n_abd[cd,0],n_abd[cd,1]),win_size[1],win_size[0],linewidth=1,edgecolor='r',facecolor='none'))
                        ac =(n_abd[(len(cds)-1),0]+(win_size[1])/2, n_abd[(len(cds)-1),1]+(win_size[0])/2)
                        new_img_patch = new_img[n_abd[(len(cds)-1),1]:(n_abd[(len(cds)-1),1]+win_size[1]),n_abd[(len(cds)-1),0]:(n_abd[(len(cds)-1),0]+win_size[0])]
                        new_img_patch = new_img_patch.astype(np.uint8)
                        patch_coordi  = ((win_size[1])/2,(win_size[0])/2)
                        #min_in = np.amin(new_img_patch)
                        min_in = 110
                        ret ,patch_thresh= cv2.threshold(new_img_patch,min_in,255,cv2.THRESH_TOZERO)
                        #patch_thresh = patch_thresh[np.nonzero(patch_thresh)]
                        radius = int((win_size[0])*(0.2))
                        a = patch_thresh
                        cx, cy = patch_coordi[0], patch_coordi[1] # The center of circle
                        y, x = np.ogrid[-radius: radius, -radius: radius]
                        index = x**2 + y**2 <= radius**2
                        a_with_circle = a[cy-radius:cy+radius, cx-radius:cx+radius][index]
                        a_nonzero = a_with_circle[np.nonzero(a_with_circle)]
                        if len(a_nonzero)!=0:
                            med_circle_abd = np.median(a_nonzero)
                        else:
                            med_circle_abd = 0
                        non_zero_patch_abd = patch_thresh[np.nonzero(patch_thresh)]
                        med_patch_abd = np.median(non_zero_patch_abd)


                    #M = cv2.moments(new_img[n_leg[:,0]:(n_leg[:,0]+win_size[0]) ,n_leg[:,1]:(n_leg[:,1]+win_size[1])]) 
                    x0 , y0 = ac[0] , ac[1]
                    x1 , y1 = lc[0] , lc[1]
                    
                    #print(x0, x1,x2 ,y0 ,y1, y2 )
                    #plt.plot((ac[0], x1),(ac[1], y1),color='white')

                    #Extract the values along the line
                    num = 100
                    x, y = np.linspace(x0, x1, num), np.linspace(y0, y1, num)
                    zi = new_img[y.astype(np.int),x.astype(np.int)]

                    ax[0].plot([x0, x1], [y0, y1], 'c')
                    #ax[0,1].plot([x0, x1], [y0, y1], 'c')
                    #ax[0,1].plot([x1, x2], [y1, y2], 'c')
                    #ax[1].plot(zi)
                    zi = np.append(zi,med_patch_abd)
                    zi = np.append(zi,med_circle_foot)
                    list_profile_name.append(name) 
                    list_profiles.append(zi)
                    #np.savetxt(grp_path + 'abd_foot_two_pt_profile_' + re.sub('.jpg|.png','',name)+'.txt', zi, delimiter=',')   # X is an array                                                                               
                    np.savetxt(profile_path + 'abd_foot_two_pt_profile_' + re.sub('.jpg|.png','',name)+'.txt', zi, delimiter=',')   # X is an array                                                                               
                    #zi = np.delete( zi , -1 ,0)    
                    #num = 50
                    #x, y = np.linspace(x1, x2, num), np.linspace(y1, y2, num)
                     #zi = np.append(zi,new_img[y.astype(np.int),x.astype(np.int)])
                    ax[1].bar( range(2), zi[-2:])
                    ax[1].set_xticklabels(('','','abd','','','','','foot' ,''))
                    #ax[1,1].plot(zi)
                    #np.savetxt(grp_path + 'three_pt_profile_' + re.sub('.jpg|.png','',name)+'.txt', zi, delimiter=',')   # X is an array
                    print(grp_path)
                    #plt.savefig(grp_path + 'dete_abd_foot_' + name)
                    plt.savefig(detection_path + 'dete_abd_foot_' + name)

    profiles = np.array(list_profiles)
    profile_names= np.array(list_profile_name)
    all_prof = np.c_[profile_names, profiles]
    #np.savetxt( profile_path + 'abd_foot_two_pt_profile_all'+'.txt', all_prof, delimiter=',', fmt="%s")
    #np.savetxt( profile_path + 'abd_foot_two_pt_profile_names'+'.txt', profile_names, delimiter=',')# X is an array                                                                               
