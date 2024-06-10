# Convolutional Neural Network (CNN) Project

This project implements a Convolutional Neural Network (CNN) using TensorFlow and Keras to classify images of cats and dogs. The project involves data preprocessing, building the CNN, training it on a dataset, and making predictions on new images.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Data Preprocessing](#data-preprocessing)
   - [Preprocessing the Training Set](#preprocessing-the-training-set)
   - [Preprocessing the Test Set](#preprocessing-the-test-set)
3. [Building the CNN](#building-the-cnn)
   - [Initializing the CNN](#initializing-the-cnn)
   - [Adding Convolutional Layers](#adding-convolutional-layers)
   - [Flattening](#flattening)
   - [Full Connection](#full-connection)
   - [Output Layer](#output-layer)
4. [Training the CNN](#training-the-cnn)
   - [Compiling the CNN](#compiling-the-cnn)
   - [Training and Evaluation](#training-and-evaluation)
5. [Making a Single Prediction](#making-a-single-prediction)
6. [Conclusion](#conclusion)

## Project Overview

This project demonstrates the implementation of a Convolutional Neural Network (CNN) to classify images of cats and dogs. The dataset is split into a training set and a test set. The CNN is built, trained, and evaluated on these datasets. Finally, the trained model is used to make predictions on new images.

## Data Preprocessing

### Preprocessing the Training Set

The training set is preprocessed using data augmentation techniques to improve the model's generalization. The `ImageDataGenerator` class is used to apply random transformations to the images.

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

training_set = train_datagen.flow_from_directory(
    'dataset/training_set',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary'
)

test_datagen = ImageDataGenerator(rescale=1./255)

test_set = test_datagen.flow_from_directory(
    'dataset/test_set',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary'
)
```
### Building the CNN
Initializing the CNN by creating a sequential model.
```
cnn = tf.keras.models.Sequential()
```

### Adding Convolutional Layers
Two convolutional layers with ReLU activation and max pooling layers are added.
```
cnn.add(tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=[64, 64, 3]))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

cnn.add(tf.keras.layers.Conv2D(32, 3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
```

### Flattening
The feature maps are flattened into a 1D vector.
```
cnn.add(tf.keras.layers.Flatten())
```

### Full Connection
A fully connected (dense) layer with 128 units and ReLU activation is added.
```
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))
```
### Output Layer
The output layer with a single neuron and sigmoid activation is added for binary classification.
```
cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
```

### Training the CNN
Compiling the CNN
The model is compiled with the Adam optimizer and binary crossentropy loss function.
```
cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```

### Training and Evaluation
Compiling the CNN
The model is compiled with the Adam optimizer and binary crossentropy loss function.
```
cnn.fit(x=training_set, validation_data=test_set, epochs=25)
```

### Making a Single Prediction
Making a Single Prediction
A single image is loaded, preprocessed, and passed through the trained model to make a prediction.
```
import numpy as np
from keras.preprocessing import image

test_image = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size=(64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)

result = cnn.predict(test_image)
class_indices = training_set.class_indices

if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'

print(f'The predicted class is: {prediction}')
```


