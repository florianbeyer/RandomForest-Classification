# -*- coding: utf-8 -*-
"""
Created on Thu Jan 04 15:23:18 2018

@author: Florian Beyer

Classification using Random Forest


The following script is based on the classification script of Chris Holden:
SOURCE: http://ceholden.github.io/open-geo-tutorial/python/chapter_5_classification.html

I added an independend validation in the end of the script, and I integrated a
exception handling for memory error during the prediction part.
Depending on the size of the imaage and the number of trees the limit of available RAM
can occur very quickly.
"""

# ----PACKAGES--------------------------------------------------------
from osgeo import gdal, gdal_array
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score

# Tell GDAL to throw Python exceptions, and register all drivers
gdal.UseExceptions()
gdal.AllRegister()



# ---INPUT INFORMATION---------------------------------------------------------
# ------------------------------------------------------------
# ------------------------------------------------------------
'''This is the only section, where you have to change something:'''

# define a number of trees that should be used (default = 500)
est = 500

# the remote sensing image you want to classify
img_RS = 'C:\\Define\\the\\PATH\\to\\image.tif'


# training and validation
#    as image in the same extand as your remote sensing image
#    no data pixels = 0 or negative
#    class pixels > 0 and as integer
training = 'C:\\Define\\the\\PATH\\to\\training.tif'
validation = 'C:\\Define\\the\\PATH\\to\\validation.tif'


# directory, where the classification image should be saved:
classification_image = 'N:C:\\Where\\to\\save\\the\\RF_Classification.tif'


# ------------------------------------------------------------
# ------------------------------------------------------------
# ------------------------------------------------------------




# ---PREPARE DATASET---------------------------------------------------------

img_ds = gdal.Open(img_RS, gdal.GA_ReadOnly)
roi_ds = gdal.Open(training, gdal.GA_ReadOnly)

img = np.zeros((img_ds.RasterYSize, img_ds.RasterXSize, img_ds.RasterCount),
               gdal_array.GDALTypeCodeToNumericTypeCode(img_ds.GetRasterBand(1).DataType))
for b in range(img.shape[2]):
    img[:, :, b] = img_ds.GetRasterBand(b + 1).ReadAsArray()
    
roi = roi_ds.GetRasterBand(1).ReadAsArray().astype(np.uint8)


# Display images
plt.subplot(121)
plt.imshow(img[:, :, 0], cmap=plt.cm.Greys_r)
plt.title('RS image - first band')

plt.subplot(122)
plt.imshow(roi, cmap=plt.cm.Spectral)
plt.title('Training Image')

plt.show()

# Number of training pixels:
n_samples = (roi > 0).sum()
print('We have {n} training samples'.format(n=n_samples))

# What are our classification labels?
labels = np.unique(roi[roi > 0])
print('The training data include {n} classes: {classes}'.format(n=labels.size, 
                                                                classes=labels))

# Subset the image dataset with the training image = X
# Mask the classes on the training dataset = y
# These will have n_samples rows
X = img[roi > 0, :]
y = roi[roi > 0]

print('Our X matrix is sized: {sz}'.format(sz=X.shape))
print('Our y array is sized: {sz}'.format(sz=y.shape))



# ---TRAIN RANDOM FOREST---------------------------------------------------------

rf = RandomForestClassifier(n_estimators=est, oob_score=True)
X = np.nan_to_num(X)
rf2 = rf.fit(X, y)

# ---RF MODEL DIAGNOSTICS---------------------------------------------------------

# With our Random Forest model fit, we can check out the "Out-of-Bag" (OOB) prediction score:

print('Our OOB prediction of accuracy is: {oob}%'.format(oob=rf.oob_score_ * 100))


# we can show the band importance:
bands = range(1,img_ds.RasterCount+1)

for b, imp in zip(bands, rf2.feature_importances_):
        print('Band {b} importance: {imp}'.format(b=b, imp=imp))

    
# Let's look at a crosstabulation to see the class confusion. 
# To do so, we will import the Pandas library for some help:
# Setup a dataframe -- just like R
# Exception Handling because of possible Memory Error

try:
    df = pd.DataFrame()
    df['truth'] = y
    df['predict'] = rf.predict(X)

except MemoryError:
    print 'Crosstab not available '

else:
    # Cross-tabulate predictions
    print(pd.crosstab(df['truth'], df['predict'], margins=True))



# ---PREDICTION---------------------------------------------------------
# Predicting the rest of the image

# Take our full image and reshape into long 2d array (nrow * ncol, nband) for classification
new_shape = (img.shape[0] * img.shape[1], img.shape[2])
img_as_array = img[:, :, :14].reshape(new_shape)

print 'Reshaped from {o} to {n}'.format(o=img.shape, n=img_as_array.shape)

img_as_array = np.nan_to_num(img_as_array)


# Now predict for each pixel

# actually like this:
#class_prediction = rf.predict(img_as_array)

# but in case of a big tree we have to slice the image
# img_as_array has to be sliced in order to prevent memory error

# first in two parts
# if not possible, the slice number decreased until we don't get an memory error
slices = int(round(len(img_as_array)/2))

test = True

while test == True:
    try:
        class_preds = list()
        
        temp = rf.predict(img_as_array[0:slices+1,:])
        class_preds.append(temp)
        
        for i in range(slices,len(img_as_array),slices):
            print '{} %, derzeit: {}'.format((i*100)/(len(img_as_array)), i)
            temp = rf.predict(img_as_array[i+1:i+(slices+1),:])                
            class_preds.append(temp)
        
    except MemoryError as error:
        slices = slices/2
        print 'Not enought RAM, new slices = {}'.format(slices)
        
    else:
        test = False


# concatenate all slices and re-shape it to the orgiginal extend
class_prediction = np.concatenate(class_preds,axis = 0)
class_prediction = class_prediction.reshape(img[:, :, 0].shape)

# ---SAVE RF CLASSIFICATION IMAGE---------------------------------------------------------

cols = img.shape[1]
rows = img.shape[0]

class_prediction.shape
img.shape

class_prediction.astype(np.float16)
class_prediction[500,400]

driver = gdal.GetDriverByName("GTiff")
outdata = driver.Create(classification_image, cols, rows, 1, gdal.GDT_UInt16)
outdata.SetGeoTransform(img_ds.GetGeoTransform())##sets same geotransform as input
outdata.SetProjection(img_ds.GetProjection())##sets same projection as input
outdata.GetRasterBand(1).WriteArray(class_prediction)
outdata.FlushCache() ##saves to disk!!


# ---ACCURACY ASSESSMENT---------------------------------------------------------

# validation / accuracy assessment
roi_val = gdal.Open(validation, gdal.GA_ReadOnly)
roi_v = roi_val.GetRasterBand(1).ReadAsArray().astype(np.uint8)


# vizualise
plt.subplot(221)
plt.imshow(img[:, :, 0], cmap=plt.cm.Greys_r)
plt.title('RS_Image - first band')

plt.subplot(222)
plt.imshow(class_prediction, cmap=plt.cm.Spectral)
plt.title('Classification result')


plt.subplot(223)
plt.imshow(roi, cmap=plt.cm.Spectral)
plt.title('Training Data')

plt.subplot(224)
plt.imshow(roi_v, cmap=plt.cm.Spectral)
plt.title('Validation Data')

plt.show()


# Find how many non-zero entries we have -- i.e. how many validation data samples?
n_val = (roi_v > 0).sum()
print('We have {n} validation pixels'.format(n=n_val))

# What are our validation labels?
labels_v = np.unique(roi_v[roi_v > 0])
print('The validation data include {n} classes: {classes}'.format(n=labels_v.size, 
                                                                classes=labels_v))
# Subset the classification image with the validation image = X
# Mask the classes on the validation dataset = y
# These will have n_samples rows
X_v = class_prediction[roi_v > 0]
y_v = roi_v[roi_v > 0]

print('Our X matrix is sized: {sz_v}'.format(sz_v=X_v.shape))
print('Our y array is sized: {sz_v}'.format(sz_v=y_v.shape))

# Cross-tabulate predictions
# confusion matrix
convolution_mat = pd.crosstab(y_v, X_v, margins=True)
print(convolution_mat)
# if you want to save the confusion matrix as a CSV file:
#savename = 'C:\\save\\to\\folder\\conf_matrix_' + str(est) + '.csv'
#convolution_mat.to_csv(savename, sep=';', decimal = '.')

# information about precision, recall, f1_score, and support:
# http://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html#sklearn.metrics.precision_recall_fscore_support
target_names = list()
for name in range(1,(labels.size)+1):
    target_names.append(str(name))
sum_mat = classification_report(y_v,X_v,target_names=target_names)

print sum_mat

# Overall Accuracy (OAA)
print 'OAA = {} %'.format(accuracy_score(y_v,X_v)*100)
