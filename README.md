# Jewellery Classification

### Introduction

Jewellery Classification is a Convolutional neural network (CNN ) model that designed for 5 different jewellries such as Bracelet, Earrings,  Necklace, Rings, Wristwatch and able to predict sample test data.

### Dataset

sn  | Name |Training | Test 
--- | --- | --- | --- 
1  | Bracelet | 355  | 50
2  | Earrings | 676  | 50
3 | Necklace | 251  | 50
4 | Rings | 183  | 50
5 | Wristwatch | 121  | 50

### Architecture
1. Convolutiona 2D layer with a specific requiring parameters such as `filter`:32, `kernel_size`: (3,3) image size
2. MaxPooling to reduce number of features with pool size (2,2).
3. Flatten layer to flatten matrix to vector so that it can be used in a dense layer
4. Application of a dropout of 0.2% to avoid over fitting with an activation function of `relu`.
5. Dense layer/Hidden Layer or a fully connected layer with a neurons of 128
6. Added a dropout of 0.2 to keep the model from over fitting
7. Hidden layer of 128 neurons with an activation function of `relu`.
8. Added a dropout of 0.2 to keep the model from over fitting.
9. finally added an output layer with a unit of 5 neurons(number of classes of datasets) and a softmax activation function.

### Image Augmentation
Due to the size of our dataset and class imbalance, we will not get the right accuracy. there is a need to increase the size of our dataset to get optimal result.

Image Augmentations techniques are methods of artificially increasing the variations of images in our data-set by using horizontal/vertical flips, rotations, variations in brightness of images, horizontal/vertical shifts etc.

Keras ImageDataGenerator class is used to perform this operation.

 
### Hyperparameter
* `batch_size` = 25
* `epoch` = 25
* `steps_per_epoch` = 15
* `optimizer` = adam
* `loss` = categorical_crossentropy
*  `metrics` = ['accuracy']


### Result

img size  | Training (loss) |Training(acc) | Test (loss) | Test(acc) 
---| --- | --- | --- | ---
32px vs 32px  | 0.2955 | 0.9013 | 0.6947| 0.8160
64px vs 64px  | 0.2733 | 0.9040 | 0.4626 | 0.8560 
128px vs 128px | 0.2389 | 0.9333 | 0.5702 | 0.8507

### Install Dependencies
1. Clone the repository on your system
2. Install the necsessary packages such as
	-	Python2 or Python3
	-	Tensorflow
	- 	Keras
	- 	Numpy

### Run Program
``` python classifier.py```



