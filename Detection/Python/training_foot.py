from sklearn.externals import joblib
from skimage.feature import hog
import numpy as np
import argparse
import datetime
import imutils
import cv2
import random
import os
import glob
import math
from PIL import Image, ImageChops, ImageOps
import csv
#from resizeimage import resizeimage
#from sklearn import svm
from numpy import genfromtxt, savetxt
from collections import OrderedDict
import matplotlib.pyplot as plt
from random import sample
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomTreesEmbedding, RandomForestClassifier,
                              GradientBoostingClassifier)
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
import _pickle as cPickle

head_im_x = []
head_im_y = []
abd_im_x = []
abd_im_y = []
dia_im_x= []
dia_im_y= []
leg_im_x= []
leg_im_y= []
os.chdir('C:/aditya/Thermal_images/ai_2.0/All_training_data/foot/')
lst=os.listdir('./')
list_hog_fd_foot = []
#size=(300,200)
block_sz = (2,3)
resiz = (100,100)
min_in = 127
ori  = 12

for i in range(0,len(lst)):
    im = cv2.imread(lst[i])
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im = im.astype(np.uint8)
    #max_in = np.amax(im)
    #min_in = np.amin(im)    
    #im = np.divide(im, max_in)
    #im = cv2.equalizeHist(im)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    im = clahe.apply(im)
    ret ,im_thresh= cv2.threshold(im,min_in,255,cv2.THRESH_TOZERO)
    #im_thresh = cv2.cvtColor(im_thresh, cv2.COLOR_BGR2GRAY)
    im_thresh = np.lib.pad(im_thresh,5, 'constant', constant_values = 0 )
    features = cv2.resize(im_thresh, resiz )
    hog_image = hog(features, orientations=ori, pixels_per_cell=(8, 8),cells_per_block= block_sz, visualise=False)
    list_hog_fd_foot.append(hog_image)
    
os.chdir('C:/aditya/Thermal_images/ai_2.0/All_training_data/abd/')
lst=os.listdir('./')
list_hog_fd_abd = []

for i in range(0,len(lst)):
    im = cv2.imread(lst[i])
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im = im.astype(np.uint8)
    #min_in = np.amin(im)
    #im = cv2.equalizeHist(im)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    im = clahe.apply(im)
    ret ,im_thresh= cv2.threshold(im,min_in,255,cv2.THRESH_TOZERO)
    #im_thresh = cv2.cvtColor(im_thresh, cv2.COLOR_BGR2GRAY)
    im_thresh = np.lib.pad(im_thresh, 5,'constant', constant_values = 0 )
    features = cv2.resize(im_thresh, resiz)
    hog_image = hog(features, orientations=ori, pixels_per_cell=(8, 8),cells_per_block= block_sz, visualise=False)
    list_hog_fd_abd.append(hog_image)
    

os.chdir('C:/aditya/Thermal_images/HOG_Feature_Testing/head_data')
lst=os.listdir('./')
list_hog_fd_head = []

for i in range(0,len(lst)):
    im = cv2.imread(lst[i])
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im = im.astype(np.uint8)
    #min_in = np.amin(im)
    #im = cv2.equalizeHist(im)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    im = clahe.apply(im)
    ret ,im_thresh= cv2.threshold(im,min_in,255,cv2.THRESH_TOZERO)
    #im_thresh = cv2.cvtColor(im_thresh, cv2.COLOR_BGR2GRAY)
    im_thresh = np.lib.pad(im_thresh, 5,'constant', constant_values = 0 )
    features = cv2.resize(im_thresh, resiz)
    hog_image = hog(features, orientations=ori, pixels_per_cell=(8, 8),cells_per_block= block_sz, visualise=False)
    list_hog_fd_head.append(hog_image)

os.chdir('C:/aditya/Thermal_images/ai_2.0/All_training_data/Diaper_data/')
lst=os.listdir('./')
list_hog_fd_diaper = []

for i in range(0,len(lst)):
    im = cv2.imread(lst[i])   
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im = im.astype(np.uint8)
    #min_in = np.amin(im)
    #im = cv2.equalizeHist(im)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    im = clahe.apply(im)
    ret ,im_thresh= cv2.threshold(im,min_in,255,cv2.THRESH_TOZERO)
    #im_thresh = cv2.cvtColor(im_thresh, cv2.COLOR_BGR2GRAY)
    im_thresh = np.lib.pad(im_thresh,5, 'constant', constant_values = 0 )
    features = cv2.resize(im_thresh, resiz)
    hog_image = hog(features, orientations=ori, pixels_per_cell=(8, 8),cells_per_block=block_sz, visualise=False)
    list_hog_fd_diaper.append(hog_image)

os.chdir('C:/aditya/Thermal_images/ai_2.0/All_training_data/not_foot/')
lst=os.listdir('./')
list_hog_fd_no_foot = []

for i in range(0,len(lst)):
    im = cv2.imread(lst[i])
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im = im.astype(np.uint8)
    #min_in = np.amin(im)    
    #equ = cv2.equalizeHist(im)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    im = clahe.apply(im)
    ret ,im_thresh= cv2.threshold(im,min_in,255,cv2.THRESH_TOZERO)
    #im_thresh = cv2.cvtColor(im_thresh, cv2.COLOR_BGR2GRAY)
    im_thresh = np.lib.pad(im_thresh,5 ,'constant', constant_values = 0 )
    features = cv2.resize(im_thresh, resiz)
    hog_image = hog(features, orientations=ori, pixels_per_cell=(8, 8),cells_per_block= block_sz,visualise=False)
    list_hog_fd_no_foot.append(hog_image)

os.chdir('C:/aditya/Thermal_images/ai_2.0/All_training_data/no_abd_rotated/')
lst=os.listdir('./')
list_hog_fd_no_abd = []

for i in range(0,len(lst)):
    im = cv2.imread(lst[i])  
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im = im.astype(np.uint8)
    #min_in = np.amin(im)
    #equ = cv2.equalizeHist(im)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    im = clahe.apply(im)
    ret ,im_thresh= cv2.threshold(im,min_in,255,cv2.THRESH_TOZERO)
    #im_thresh = cv2.cvtColor(im_thresh, cv2.COLOR_BGR2GRAY)
    im_thresh = np.lib.pad(im_thresh, 5,'constant', constant_values = 0 )
    features = cv2.resize(im_thresh, resiz)
    hog_image = hog(features, orientations=12, pixels_per_cell=(8, 8),cells_per_block= block_sz, visualise=False)
    list_hog_fd_no_abd.append(hog_image)

os.chdir('C:/aditya/Thermal_images/ai_2.0/All_training_data/leg_wto_foot/')
lst=os.listdir('./')
list_hog_fd_legwto = []
#size=(300,200)
for i in range(0,len(lst)):
    im = cv2.imread(lst[i])
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im = im.astype(np.uint8)
    #min_in = np.amin(im)
    #equ = cv2.equalizeHist(im)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    im = clahe.apply(im)
    ret ,im_thresh= cv2.threshold(im,min_in,255,cv2.THRESH_TOZERO)
    #im_thresh = cv2.cvtColor(im_thresh, cv2.COLOR_BGR2GRAY)
    im_thresh = np.lib.pad(im_thresh,5, 'constant', constant_values = 0)
    features = cv2.resize(im_thresh, resiz)
    hog_image = hog(features, orientations=ori, pixels_per_cell=(8, 8),cells_per_block= block_sz, visualise=False)
    list_hog_fd_legwto.append(hog_image)


################# training for foot ############################
sample_abd=[]
sample_head=[]
sample_diaper=[]
sample_foot= []
sample_not_foot = []

for k in sample(range(len(list_hog_fd_abd)),int(round(len(list_hog_fd_abd)*(1)))):
    sample_abd.append(list_hog_fd_abd[k])

for k in sample(range(len(list_hog_fd_diaper)),int(round(len(list_hog_fd_diaper)*(1)))):
    sample_diaper.append(list_hog_fd_diaper[k])

for k in sample(range(len(list_hog_fd_head)),int(round(len(list_hog_fd_head)*(1)))):
    sample_head.append(list_hog_fd_head[k])

#for k in sample(range(len(list_hog_fd_no_foot)),round(len(list_hog_fd_no_foot)*(0.75))):
#    sample_not_foot.append(list_hog_fd_no_foot[k])

dat_test= list_hog_fd_foot + sample_abd + sample_diaper + list_hog_fd_no_foot 
#train_res= ['F']*len(list_hog_fd_foot)+ ['NF']*(len(sample_abd)+len(sample_diaper)+ len(list_hog_fd_no_foot)) 
train_res= [1]*len(list_hog_fd_foot)+ [0]*(len(sample_abd)+len(sample_diaper)+ len(list_hog_fd_no_foot))

#y = label_binarize(train_res, classes=[0, 1])
#dat_test= list_hog_fd_foot + list_hog_fd_abd+ list_hog_fd_diaper + list_hog_fd_head + list_hog_fd_no_foot
#train_res= ['L']*len(list_hog_fd_foot)+ ['NL']*(len(list_hog_fd_abd)+len(list_hog_fd_diaper)+len(list_hog_fd_head)+ len(list_hog_fd_no_leg)) 
#temp = dat_test[0].reshape(1,18600)

#n_classes = dat_test.shape[1]
#for i in range(dat_test.shape[0]):
#    temp =  temp.append(dat_test[i].reshape(1,18600))

dat_test = np.array(dat_test)
train_res = np.array(train_res)

########### spliting data in training and testing ##########
X = dat_test
y = train_res

random.seed(43)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3,
                                                    random_state=0)


#rt = RandomTreesEmbedding(max_depth=3, n_estimators=n_estimator,
#    random_state=0)
rf = RandomForestClassifier(max_depth=3, n_estimators=160, max_features='sqrt',
random_state=0)
'''
rt_lm = LogisticRegression()
pipeline = make_pipeline(rt, rt_lm)
pipeline.fit(X_train, y_train)
y_pred_rt = pipeline.predict_proba(X_test)[:, 1]
fpr_rt_lm, tpr_rt_lm, _ = roc_curve(y_test, y_pred_rt)

# Supervised transformation based on random forests
rf = RandomForestClassifier(max_depth=3, n_estimators=n_estimator)
rf_enc = OneHotEncoder()
rf_lm = LogisticRegression()
rf.fit(X_train, y_train)
rf_enc.fit(rf.apply(X_train))
rf_lm.fit(rf_enc.transform(rf.apply(X_train_lr)), y_train_lr)

y_pred_rf_lm = rf_lm.predict_proba(rf_enc.transform(rf.apply(X_test)))[:, 1]
fpr_rf_lm, tpr_rf_lm, _ = roc_curve(y_test, y_pred_rf_lm)

grd = GradientBoostingClassifier(n_estimators=n_estimator)
grd_enc = OneHotEncoder()
grd_lm = LogisticRegression()
grd.fit(X_train, y_train)
grd_enc.fit(grd.apply(X_train)[:, :, 0])
grd_lm.fit(grd_enc.transform(grd.apply(X_train_lr)[:, :, 0]), y_train_lr)

y_pred_grd_lm = grd_lm.predict_proba(
    grd_enc.transform(grd.apply(X_test)[:, :, 0]))[:, 1]
fpr_grd_lm, tpr_grd_lm, _ = roc_curve(y_test, y_pred_grd_lm)


# The gradient boosted model by itself
y_pred_grd = grd.predict_proba(X_test)[:, 1]
fpr_grd, tpr_grd, _ = roc_curve(y_test, y_pred_grd)
'''

# The random forest model by itself
rf.fit(X_train, y_train)
print(rf.predict(X_test))
print(rf.score(X_test, y_test))
#cnf_matrix = confusion_matrix(y_test, y_pred)

#with open('C:/aditya/Thermal_images/ai_2.0/hog_detectors/hog_rf_F_NF_100pad5_in127_py3', 'wb') as f:
#    cPickle.dump(rf, f)


y_pred_rf = rf.predict_proba(X_test)[:, 1]
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_rf)
roc_auc = auc(fpr_rf , tpr_rf)
plt.figure(1)
lw = 2
plt.plot([0, 1], [0, 1], 'k--')
#plt.plot(fpr_rt_lm, tpr_rt_lm, label='RT + LR')
plt.plot(fpr_rf, tpr_rf, lw=lw, label='RF (area = %0.2f)' % roc_auc)
#plt.plot(fpr_rf_lm, tpr_rf_lm, label='RF + LR')
#plt.plot(fpr_grd, tpr_grd, label='GBT')
#plt.plot(fpr_grd_lm, tpr_grd_lm, label='GBT + LR')
plt.xlabel('False positive rate', fontsize=22)
plt.ylabel('True positive rate', fontsize=22)
plt.title('ROC curve foot classifier', fontsize=22)
plt.tick_params(axis='both', which='major', labelsize=15)
plt.xticks(fontsize=14)
plt.legend(loc='best', fontsize = 14)
plt.show()

plt.figure(2)
plt.xlim(0, 0.2)
plt.ylim(0.8, 1)
plt.plot([0, 1], [0, 1], 'k--')
#plt.plot(fpr_rt_lm, tpr_rt_lm, label='RT + LR')
plt.plot(fpr_rf, tpr_rf, label='RF')
#plt.plot(fpr_rf_lm, tpr_rf_lm, label='RF + LR')
#plt.plot(fpr_grd, tpr_grd, label='GBT')
#plt.plot(fpr_grd_lm, tpr_grd_lm, label='GBT + LR')
plt.xlabel('False positive rate', fontsize=22)
plt.ylabel('True positive rate', fontsize=22)
plt.title('ROC curve (zoomed in at top left)', fontsize=22)
plt.xticks(fontsize=14)
plt.legend(loc='best', fontsize = 14)
plt.show()


y_pred_rf = rf.predict_proba(X_test)[:, 1]
y_pred = label_binarize(y_test, classes = [0,1])
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_rf)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_rf)

plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
#plt.plot(fpr_rt_lm, tpr_rt_lm, label='RT + LR')
plt.plot(fpr_rf, tpr_rf, label='RF')
#plt.plot(fpr_rf_lm, tpr_rf_lm, label='RF + LR')
#plt.plot(fpr_grd, tpr_grd, label='GBT')
#plt.plot(fpr_grd_lm, tpr_grd_lm, label='GBT + LR')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()

################ optimizing the n-estimators and max-features #######
RANDOM_STATE = 123

# Generate a binary classification dataset.
#X, y = make_classification(n_samples=500, n_features=25,
#                           n_clusters_per_class=1, n_informative=15,
#                           random_state=RANDOM_STATE)

# NOTE: Setting the `warm_start` construction parameter to `True` disables
# support for parallelized ensembles but is necessary for tracking the OOB
# error trajectory during training.
ensemble_clfs = [
    ("RandomForestClassifier, max_features='sqrt'",
        RandomForestClassifier(warm_start=True, oob_score=True,
                               max_features="sqrt",
                               random_state=RANDOM_STATE)),
    ("RandomForestClassifier, max_features='log2'",
        RandomForestClassifier(warm_start=True, max_features='log2',
                               oob_score=True,
                               random_state=RANDOM_STATE)),
    ("RandomForestClassifier, max_features=None",
        RandomForestClassifier(warm_start=True, max_features=None,
                               oob_score=True,
                               random_state=RANDOM_STATE))
]

#Map a classifier name to a list of (<n_estimators>, <error rate>) pairs.
error_rate = OrderedDict((label, []) for label, _ in ensemble_clfs)
X=X_train
y= y_train
#Range of `n_estimators` values to explore.
min_estimators = 50
max_estimators = 300

for label, clf in ensemble_clfs:
    for i in range(min_estimators, max_estimators + 1, 10 ):
        clf.set_params(n_estimators=i)
        clf.fit(X, y)

        # Record the OOB error for each `n_estimators=i` setting.
        oob_error = 1 - clf.oob_score_
        error_rate[label].append((i, oob_error))

# Generate the "OOB error rate" vs. "n_estimators" plot.
for label, clf_err in error_rate.items():
    xs, ys = zip(*clf_err)
    plt.plot(xs, ys, label=label)

plt.xlim(min_estimators, max_estimators)
plt.xlabel("n_estimators")
plt.ylabel("OOB error rate")
plt.legend(loc="upper right")
plt.show()

