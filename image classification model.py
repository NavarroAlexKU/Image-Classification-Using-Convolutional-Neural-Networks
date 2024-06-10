# %% [markdown]
# # Convolutional Neural Network

# %% [markdown]
# ### Importing the libraries

# %%
# Import python packages:
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

print(tf.__version__)

# %% [markdown]
# ## Part 1 - Data Preprocessing

# %% [markdown]
# ### Preprocessing the Training set

# %%
# Create an instance of the ImageDataGenerator with data augmentation parameters
train_datagen = ImageDataGenerator(
    # Rescale pixel values to be between 0 and 1
    rescale=1./255,
    # Apply random shear transformations
    shear_range=0.2,
    # Apply random zoom transformations
    zoom_range=0.2,
    # Randomly flip images horizontally
    horizontal_flip=True
)

# Use the flow_from_directory method to load images from the specified directory and apply the transformations
training_set = train_datagen.flow_from_directory(
    # Specify the directory containing the training images
    r'dataset\training_set',
    # Resize all images to 64x64 pixels
    target_size=(64, 64),
    # Set the batch size to 32
    batch_size=32,
    # Use binary labels (for binary classification)
    class_mode='binary'
)

# %% [markdown]
# ### Preprocessing the Test set

# %%
# Create an instance of the ImageDataGenerator for the test set with only rescaling
test_datagen = ImageDataGenerator(
    # Rescale pixel values to be between 0 and 1
    rescale=1./255
)

# Use the flow_from_directory method to load test images from the specified directory and apply the rescaling
test_set = test_datagen.flow_from_directory(
    # Specify the directory containing the test images
    r'dataset\test_set',
    # Resize all images to 64x64 pixels
    target_size=(64, 64),
    # Set the batch size to 32
    batch_size=32,
    # Use binary labels (for binary classification)
    class_mode='binary'
)

# %% [markdown]
# ## Part 2 - Building the CNN

# %% [markdown]
# ### Initialising the CNN

# %%
# Initialize a Sequential model
cnn = tf.keras.models.Sequential()

# %% [markdown]
# ### Step 1 - Convolution

# %%
# Add a convolutional layer to the Sequential model
cnn.add(
    tf.keras.layers.Conv2D(
        # Set the number of feature detectors (filters) to 32
        filters=32,
        # Set the size of the kernel (filter) to 3x3
        kernel_size=3,
        # Set the activation function to ReLU (Rectified Linear Unit)
        activation='relu',
        # Set the input shape to 64x64 pixels with 3 color channels (RGB)
        input_shape=[64, 64, 3]
    )
)

# %% [markdown]
# ### Step 2 - Pooling

# %%
# Add a max pooling layer to the Sequential model
cnn.add(
    tf.keras.layers.MaxPool2D(
        # Set the size of the pooling window to 2x2
        pool_size=2,
        # Set the stride of the pooling operation to 2
        strides=2
    )
)

# %% [markdown]
# ### Adding a second convolutional layer

# %%
# Add a convolutional layer to the Sequential model
cnn.add(
    tf.keras.layers.Conv2D(
        # Set the number of feature detectors (filters) to 32
        filters=32,
        # Set the size of the kernel (filter) to 3x3
        kernel_size=3,
        # Set the activation function to ReLU (Rectified Linear Unit)
        activation='relu'
    )
)

# Add a max pooling layer to the Sequential model
cnn.add(
    tf.keras.layers.MaxPool2D(
        # Set the size of the pooling window to 2x2
        pool_size=2,
        # Set the stride of the pooling operation to 2
        strides=2
    )
)

# %% [markdown]
# ### Step 3 - Flattening

# %%
# Add a Flatten layer to the CNN model.
# The Flatten layer converts the 2D matrix of features into a 1D vector.
# This is often used as the first step in the transition from convolutional layers to fully connected layers.
cnn.add(tf.keras.layers.Flatten())

# %% [markdown]
# ### Step 4 - Full Connection

# %%
# Add a Dense (fully connected) layer to the CNN model.
# The Dense layer has 128 units (neurons) and uses the ReLU activation function.
# This layer helps to learn complex patterns in the data by connecting each neuron to all neurons in the previous layer.
cnn.add(
    tf.keras.layers.Dense(
        units=128,          # Number of neurons in this layer
        activation='relu'   # Activation function to introduce non-linearity
    )
)

# %% [markdown]
# ### Step 5 - Output Layer

# %%
# Add the output Dense (fully connected) layer to the CNN model.
# This Dense layer has 1 unit (neuron) and uses the sigmoid activation function.
# The sigmoid activation function is typically used for binary classification tasks,
# as it outputs a value between 0 and 1, representing the probability of the positive class.
cnn.add(
    tf.keras.layers.Dense(
        units=1,            # Number of neurons in this layer (1 for binary classification)
        activation='sigmoid' # Activation function to produce a probability output
    )
)

# %% [markdown]
# ## Part 3 - Training the CNN

# %% [markdown]
# ### Compiling the CNN

# %%
# Compile the CNN model.
# The compilation process configures the model for training.

cnn.compile(
    optimizer='adam',            # The optimizer used for updating the model weights. 'adam' is an adaptive learning rate optimization algorithm.
    loss='binary_crossentropy',  # The loss function used for binary classification tasks. It measures the difference between the predicted and true labels.
    metrics=['accuracy']         # Metrics to evaluate the model's performance. Here, 'accuracy' is used to measure how often predictions match the true labels.
)

# %% [markdown]
# ### Training the CNN on the Training set and evaluating it on the Test set

# %%
# Train the CNN model.
# The fit method trains the model for a fixed number of epochs (iterations over the entire dataset).

cnn.fit(
    x=training_set,        # The training data.
    validation_data=test_set, # The validation data to evaluate the model's performance during training.
    epochs=25              # The number of epochs to train the model.
)


# %% [markdown]
# ## Part 4 - Making a single prediction

# %%
# Import necessary libraries
import numpy as np
from keras.preprocessing import image

# Load and preprocess a single test image
test_image = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg',  # Path to the test image
                            target_size=(64, 64))  # Resize the image to match the input size of the model

# Convert the image to an array
test_image = image.img_to_array(test_image)

# Expand the dimensions of the image to match the expected input shape for the model
# Model expects input shape: (batch_size, height, width, channels)
test_image = np.expand_dims(test_image, axis=0)

# Predict the class of the image using the trained model
result = cnn.predict(test_image)

# Get the class indices from the training set
# This will provide the mapping of class labels (e.g., 'cat': 0, 'dog': 1)
class_indices = training_set.class_indices

# Interpret the prediction result
# If the result is 1, the prediction is 'dog'; otherwise, it's 'cat'
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'

# Print the prediction
print(f'The predicted class is: {prediction}')


