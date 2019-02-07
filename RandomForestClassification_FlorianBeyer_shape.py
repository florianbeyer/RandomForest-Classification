
# coding: utf-8

# # Random Forest Classification

# @author: Florian Beyer
# 
# Version: 0.2
# 
# Datum: 2019-02-07
# 
# Classification using Random Forest
# 
# Updates: 
# - Input training and validation as shape file (add attribute with integer numbers, where one number represent one class, multiple polygons per class are possible. The validation shap needs to be the same structure.)
# 
# - The classifiaction result is masked that the black border has a pixel value of 0
# 
# 
# The following script is based on the classification script of Chris Holden:
# SOURCE: http://ceholden.github.io/open-geo-tutorial/python/chapter_5_classification.html
# I added an independend validation in the end of the script, and I integrated a
# exception handling for memory error during the prediction part.
# Depending on the size of the image and the number of trees the limit of available RAM
# can occur very quickly.
# 
# Furthermore, parts of the script (converting shape files to numpy array) base on the script of Julien Rebetez:
# https://github.com/terrai/rastercube/blob/master/rastercube/datasources/shputils.py
# 

# ### Section
# 
# Required packages

# In[15]:


# packages
from osgeo import gdal, ogr, gdal_array # I/O image data
import numpy as np # math and array handling
import matplotlib.pyplot as plt # plot figures
from sklearn.ensemble import RandomForestClassifier # classifier
import pandas as pd # handling large data as table sheets
from sklearn.metrics import classification_report, accuracy_score # calculating measures for accuracy assessment

# Tell GDAL to throw Python exceptions, and register all drivers
gdal.UseExceptions()
gdal.AllRegister()


# ### Section
# 
# Input data
# 
# - This is the only section where you have to change something.

# In[16]:


# define a number of trees that should be used (default = 500)
est = 500

# the remote sensing image you want to classify
img_RS = 'N:\\...\...\\image.tif'


# training and validation
#    as image in the same extand as your remote sensing image
#    no data pixels = 0 or negative
#    class pixels > 0 and as integer
training = 'N:\\...\\...\\cal.shp'
validation = 'N:\\...\\...\\val.shp'

# what is the attributes name of your classes in the shape file (field name of the classes)?
attribute = 'class'


# directory, where the classification image should be saved:
classification_image = 'N:\\...\\...\\RFclassification.tif'


# ### Section
# Data preparation

# In[17]:


# load image data

img_ds = gdal.Open(img_RS, gdal.GA_ReadOnly)

img = np.zeros((img_ds.RasterYSize, img_ds.RasterXSize, img_ds.RasterCount),
               gdal_array.GDALTypeCodeToNumericTypeCode(img_ds.GetRasterBand(1).DataType))
for b in range(img.shape[2]):
    img[:, :, b] = img_ds.GetRasterBand(b + 1).ReadAsArray()


# In[18]:


# laod training data from shape file

#model_dataset = gdal.Open(model_raster_fname)
shape_dataset = ogr.Open(training)
shape_layer = shape_dataset.GetLayer()
mem_drv = gdal.GetDriverByName('MEM')
mem_raster = mem_drv.Create('',img_ds.RasterXSize,img_ds.RasterYSize,1,gdal.GDT_UInt16)
mem_raster.SetProjection(img_ds.GetProjection())
mem_raster.SetGeoTransform(img_ds.GetGeoTransform())
mem_band = mem_raster.GetRasterBand(1)
mem_band.Fill(0)
mem_band.SetNoDataValue(0)

att_ = 'ATTRIBUTE='+attribute
# http://gdal.org/gdal__alg_8h.html#adfe5e5d287d6c184aab03acbfa567cb1
# http://gis.stackexchange.com/questions/31568/gdal-rasterizelayer-doesnt-burn-all-polygons-to-raster
err = gdal.RasterizeLayer(mem_raster, [1], shape_layer, None, None, [1],  [att_,"ALL_TOUCHED=TRUE"])
assert err == gdal.CE_None

roi = mem_raster.ReadAsArray()



# In[19]:


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


# ### Section
# Train Random Forest

# In[6]:


rf = RandomForestClassifier(n_estimators=est, oob_score=True)
X = np.nan_to_num(X)
rf2 = rf.fit(X, y)


# ### Section
# RF Model Diagnostics

# In[7]:


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


# ### Section
# Prediction

# In[8]:


# Predicting the rest of the image

# Take our full image and reshape into long 2d array (nrow * ncol, nband) for classification
new_shape = (img.shape[0] * img.shape[1], img.shape[2])
img_as_array = img[:, :, :np.int(img.shape[2])].reshape(new_shape)

print 'Reshaped from {o} to {n}'.format(o=img.shape, n=img_as_array.shape)

img_as_array = np.nan_to_num(img_as_array)


# In[9]:


# Now predict for each pixel
# first prediction will be tried on the entire image
# if not enough RAM, the dataset will be sliced
try:
    class_prediction = rf.predict(img_as_array)
except MemoryError:
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
else:
    print 'Class prediction was successful without slicing!'


# In[10]:


# concatenate all slices and re-shape it to the orgiginal extend
try:
    class_prediction = np.concatenate(class_preds,axis = 0)
except NameError:
    print 'No slicing was necessary!'
    
class_prediction = class_prediction.reshape(img[:, :, 0].shape)
print 'Reshaped back to {}'.format(class_prediction.shape)


# ### Section
# 
# Mask classification image (black border = 0)
# 

# In[11]:


# generate mask image from red band
mask = np.copy(img[:,:,0])
mask[mask > 0.0] = 1.0 # all actual pixels have a value of 1.0

# plot mask

plt.imshow(mask)


# In[12]:


# mask classification an plot

class_prediction.astype(np.float16)
class_prediction_ = class_prediction*mask

plt.subplot(121)
plt.imshow(class_prediction, cmap=plt.cm.Spectral)
plt.title('classification unmasked')

plt.subplot(122)
plt.imshow(class_prediction_, cmap=plt.cm.Spectral)
plt.title('classification masked')

plt.show()


# ### Section
# Save Classification Image

# In[13]:


cols = img.shape[1]
rows = img.shape[0]

class_prediction_.astype(np.float16)

driver = gdal.GetDriverByName("gtiff")
outdata = driver.Create(classification_image, cols, rows, 1, gdal.GDT_UInt16)
outdata.SetGeoTransform(img_ds.GetGeoTransform())##sets same geotransform as input
outdata.SetProjection(img_ds.GetProjection())##sets same projection as input
outdata.GetRasterBand(1).WriteArray(class_prediction_)
outdata.FlushCache() ##saves to disk!!
print 'Image saved to: {}'.format(classification_image)


# ### Section
# Accuracy Assessment

# In[20]:


# validation / accuracy assessment

# laod training data from shape file
shape_dataset_v = ogr.Open(validation)
shape_layer_v = shape_dataset_v.GetLayer()
mem_drv_v = gdal.GetDriverByName('MEM')
mem_raster_v = mem_drv_v.Create('',img_ds.RasterXSize,img_ds.RasterYSize,1,gdal.GDT_UInt16)
mem_raster_v.SetProjection(img_ds.GetProjection())
mem_raster_v.SetGeoTransform(img_ds.GetGeoTransform())
mem_band_v = mem_raster_v.GetRasterBand(1)
mem_band_v.Fill(0)
mem_band_v.SetNoDataValue(0)

# http://gdal.org/gdal__alg_8h.html#adfe5e5d287d6c184aab03acbfa567cb1
# http://gis.stackexchange.com/questions/31568/gdal-rasterizelayer-doesnt-burn-all-polygons-to-raster
err_v = gdal.RasterizeLayer(mem_raster_v, [1], shape_layer_v, None, None, [1],  [att_,"ALL_TOUCHED=TRUE"])
assert err_v == gdal.CE_None

roi_v = mem_raster_v.ReadAsArray()



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

