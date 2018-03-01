# RandomForest-Classification

## Getting Started

### You only have to prepare yor data:
1. The multi-band Remote Sensing image as tif format.
2. an image which defines your training dataset in the same extend as the remote sensing image
3. an image which defines your validation dataset in the same extend as the remote sensing image

### You find a section "INPUT INFORMATION".
This is the only section where you have to change something in the script (directories and file names).


## EXAMPLE:
This example uses a 14 bands remote sensing dataset and 11 classes as training and validation.
In the end, a Tiff-File will be saved as your classification.
During the process you will see 2 plots and a lot of information...

### ...about the training data:
```
We have 87895 training samples
The training data include 11 classes: [ 1  2  3  4  5  6  7  8  9 10 11]
Our X matrix is sized: (87895L, 14L)
Our y array is sized: (87895L,)
```

### ...about the Random Forest Model Fit:
```
Our OOB prediction of accuracy is: 99.8634734627%
```

```
Band 1 importance: 0.0261680458145
Band 2 importance: 0.0153287127996
Band 3 importance: 0.0205070224479
Band 4 importance: 0.037036238847
Band 5 importance: 0.0310529483133
Band 6 importance: 0.0436272835319
Band 7 importance: 0.055942106673
Band 8 importance: 0.398011332962
Band 9 importance: 0.119100238254
Band 10 importance: 0.0119935702383
Band 11 importance: 0.0857394486887
Band 12 importance: 0.0216590258319
Band 13 importance: 0.0998573985857
Band 14 importance: 0.033976627012
```

```
	predict     1    2     3      4     5     6     7     8      9    10   11    All
	truth                                                                           
	1        5280    0     0      0     0     0     0     0      0     0    0   5280
	2           0  170     0      0     0     0     0     0      0     0    0    170
	3           0    0  2962      0     0     0     0     0      0     0    0   2962
	4           0    0     0  16650     0     0     0     0      0     0    0  16650
	5           0    0     0      0  6194     0     0     0      0     0    0   6194
	6           0    0     0      0     0  1691     0     0      0     0    0   1691
	7           0    0     0      0     0     0  4509     0      0     0    0   4509
	8           0    0     0      0     0     0     0  2342      0     0    0   2342
	9           0    0     0      0     0     0     0     0  41504     0    0  41504
	10          0    0     0      0     0     0     0     0      0  6218    0   6218
	11          0    0     0      0     0     0     0     0      0     0  375    375
	All      5280  170  2962  16650  6194  1691  4509  2342  41504  6218  375  87895
```

### ...about the independent validation:

```
We have 68482 validation pixels
The validation data include 11 classes: [ 1  2  3  4  5  6  7  8  9 10 11]
Our X matrix is sized: (68482L,)
Our y array is sized: (68482L,)
```

```
	col_0     1    2     3      4     5     6     7     8      9    10   11    All
	row_0                                                                         
	1      2597    0     0      0     0     0     0     0    701     0    0   3298
	2         0  125     0      0     0     0     0     0      0     0   10    135
	3         0    0  1457    743     0     0     0    16      0    85    0   2301
	4         0    0   163   9640    12     0     0   473      0    72    0  10360
	5         2    0     0    410  5505     0     0     0    858    47    0   6822
	6         0    0     0    128     0   934     0     7      0   483    0   1552
	7         0    0     0      0     0     0  2662     0      0     0    0   2662
	8         0    0    85    389     8     0     0   538      0   299    0   1319
	9       137    0     0      3  1620     0     0     0  31385     0    0  33145
	10        0    0    47    520    63   190    40    19      1  5656    0   6536
	11        0    0     0      0     0     0     0     0      0     0  352    352
	All    2736  125  1752  11833  7208  1124  2702  1053  32945  6642  362  68482
  '''
  
 ```
				precision    recall  f1-score   support
	
			1       0.95      0.79      0.86      3298
			2       1.00      0.93      0.96       135
			3       0.83      0.63      0.72      2301
			4       0.81      0.93      0.87     10360
			5       0.76      0.81      0.78      6822
			6       0.83      0.60      0.70      1552
			7       0.99      1.00      0.99      2662
			8       0.51      0.41      0.45      1319
			9       0.95      0.95      0.95     33145
			10       0.85      0.87      0.86      6536
			11       0.97      1.00      0.99       352
	
	avg / total       0.89      0.89      0.89     68482
```

```
OAA = 88.8569259075 %
```
