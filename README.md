# RandomForest-Classification

## Getting Started

This script is for classification of remote sensing multi-band images using shape files as input for training and validation.

I am using Anaconda (Python 3.8) and the following packages:
- GDAL package from OSGEO.
- OGR
- scikit learn
(- pandas/numpy/matplotlib/seaborn/...)

### You only have to prepare yor data and you have to define how many tree you want to use:
1. Choose the number of trees for the Random Forest (default = 500)
2. The multi-band Remote Sensing image as tif format.
3. training and validation data as shape files (Polygones)
IMPORTANT!!! -> classes as integer numbers (do not use class names)

Furthermore, the classification result will be masked. Thus a black border has a value of 0 in the classification.
 
 
### You find a section "INPUT INFORMATION".
This is the only section where you have to change something in the script (directories and file names).


## EXAMPLE:
This example uses a 14 bands remote sensing dataset and 8 classes as training and validation.
Finaly, you get a tif file as your classification image and a report.txt as well as many outputs in your python console!
During the process you will also see several plots...

### report.txt
```
Random Forest Classification
Processing: 2020-05-07 14:14:53.766020
-------------------------------------------------
PATHS:
Image: R:\OwnCloud\WetScapes\2020_04_23_HüMo\huemo2018_14bands_tif.tif
Training shape: R:\OwnCloud\WetScapes\2020_04_23_HüMo\cal.shp
Vaildation shape: R:\OwnCloud\WetScapes\2020_04_23_HüMo\val.shp
      choosen attribute: class
Classification image: R:\OwnCloud\WetScapes\2020_04_23_HüMo\results\HueMo2018_14bands_class_.tif
Report text file: R:\OwnCloud\WetScapes\2020_04_23_HüMo\results\results_txt_.txt
-------------------------------------------------
Image extent: 4721 x 5224 (row x col)
Number of Bands: 14
---------------------------------------
TRAINING
Number of Trees: 500
11781 training samples
training data include 8 classes: [1 2 3 4 5 6 7 8]
--------------------------------
TRAINING and RF Model Diagnostics:
OOB prediction of accuracy is: 99.53314659197012%
Band 1 importance: 0.07422191283709235
Band 2 importance: 0.03138862335047076
Band 3 importance: 0.01232741814805193
Band 4 importance: 0.06724784717595128
Band 5 importance: 0.11994202487099442
Band 6 importance: 0.050658933643359196
Band 7 importance: 0.06543997268021191
Band 8 importance: 0.27292836274508814
Band 9 importance: 0.12041183266036815
Band 10 importance: 0.030058880237602194
Band 11 importance: 0.036830909145992574
Band 12 importance: 0.01929961375159746
Band 13 importance: 0.04317281009998762
Band 14 importance: 0.056070858653231984
predict     1     2     3    4    5     6    7   8    All
truth                                                    
1        3558     0     0    0    0     0    0   0   3558
2           0  1105     0    0    0     0    0   0   1105
3           0     0  1941    0    0     0    0   0   1941
4           0     0     0  207    0     0    0   0    207
5           0     0     0    0  298     0    0   0    298
6           0     0     0    0    0  4231    0   0   4231
7           0     0     0    0    0     0  346   0    346
8           0     0     0    0    0     0    0  95     95
All      3558  1105  1941  207  298  4231  346  95  11781
------------------------------------
VALIDATION
8058 validation pixels
validation data include 8 classes: [1 2 3 4 5 6 7 8]
col_0     1    2     3    4    5     6    7   8   All
row_0                                                
1      3278    0    16    0    0     0    0   0  3294
2         0  413    29    0    0     0    0   0   442
3         0   77  1228    0    0     0    0   0  1305
4         0    0     0  105    0     0    0   0   105
5         0    0     1    0  118     0    0   0   119
6         0    0     0    0    0  2449    0   0  2449
7         0    0     0   10    0     0  246   0   256
8         0   14     0    3    0     0    0  71    88
All    3278  504  1274  118  118  2449  246  71  8058
              precision    recall  f1-score   support

           1       1.00      1.00      1.00      3294
           2       0.82      0.93      0.87       442
           3       0.96      0.94      0.95      1305
           4       0.89      1.00      0.94       105
           5       1.00      0.99      1.00       119
           6       1.00      1.00      1.00      2449
           7       1.00      0.96      0.98       256
           8       1.00      0.81      0.89        88

    accuracy                           0.98      8058
   macro avg       0.96      0.95      0.95      8058
weighted avg       0.98      0.98      0.98      8058

OAA = 98.138495904691 %
```
