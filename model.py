import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Input, ZeroPadding2D, BatchNormalization, Activation, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.utils import shuffle
from keras_tuner import RandomSearch
import os
import cv2
import imutils
import numpy as np
import matplotlib.pyplot as plt
import time
from os import listdir

# Function to crop brain contour from an image


def crop_brain_contour(image, plot=False):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)
    cnts = cv2.findContours(
        thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)
    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])
    new_image = image[extTop[1]:extBot[1], extLeft[0]:extRight[0]]
    if plot:
        plt.imshow(new_image)
    return new_image


# Function to load and preprocess the dataset
def load_data(dir_list, image_size):
    X = []
    y = []
    image_width, image_height = image_size
    for directory in dir_list:
        for filename in listdir(directory):
            image = cv2.imread(os.path.join(directory, filename))
            image = crop_brain_contour(image, plot=False)
            image = cv2.resize(image, dsize=(
                image_width, image_height), interpolation=cv2.INTER_CUBIC)
            image = image / 255.0
            X.append(image)
            y.append([1] if directory[-3:] == 'yes' else [0])
    X = np.array(X)
    y = np.array(y)
    X, y = shuffle(X, y)
    return X, y


# Define paths
augmented_path = 'augmented data/'
augmented_yes = os.path.join(augmented_path, 'yes')
augmented_no = os.path.join(augmented_path, 'no')
IMG_WIDTH, IMG_HEIGHT = 240, 240
X, y = load_data([augmented_yes, augmented_no], (IMG_WIDTH, IMG_HEIGHT))


def plot_sample_images(X, y, n=50):
    """
    Plots n sample images for both values of y (labels).
    Arguments:
        X: A numpy array with shape = (#_examples, image_width, image_height, #_channels)
        y: A numpy array with shape = (#_examples, 1)
    """

    for label in [0, 1]:
        # grab the first n images with the corresponding y values equal to label
        images = X[np.argwhere(y == label)]
        n_images = images[:n]

        columns_n = 10
        rows_n = int(n / columns_n)

        plt.figure(figsize=(20, 10))

        i = 1  # current plot
        for image in n_images:
            plt.subplot(rows_n, columns_n, i)
            plt.imshow(image[0])

            # remove ticks
            plt.tick_params(axis='both', which='both',
                            top=False, bottom=False, left=False, right=False,
                            labelbottom=False, labeltop=False, labelleft=False, labelright=False)

            i += 1

        def label_to_str(label): return "Yes" if label == 1 else "No"
        plt.suptitle(f"Brain Tumor: {label_to_str(label)}")
        plt.show()


# plot_sample_images(X, y)


# Split the data into training, validation, and test sets
def split_data(X, y, test_size=0.3):
    X_train, X_test_val, y_train, y_test_val = train_test_split(
        X, y, test_size=test_size, random_state=42)
    X_test, X_val, y_test, y_val = train_test_split(
        X_test_val, y_test_val, test_size=0.5, random_state=42)
    return X_train, y_train, X_val, y_val, X_test, y_test


X_train, y_train, X_val, y_val, X_test, y_test = split_data(
    X, y, test_size=0.3)

print("number of training examples = " + str(X_train.shape[0]))
print("number of development examples = " + str(X_val.shape[0]))
print("number of test examples = " + str(X_test.shape[0]))
print("X_train shape: " + str(X_train.shape))
print("Y_train shape: " + str(y_train.shape))
print("X_val (dev) shape: " + str(X_val.shape))
print("Y_val (dev) shape: " + str(y_val.shape))
print("X_test shape: " + str(X_test.shape))
print("Y_test shape: " + str(y_test.shape))


# Nicely formatted time string
def hms_string(sec_elapsed):
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60
    return f"{h}:{m}:{round(s,1)}"


def compute_f1_score(y_true, prob):
    # convert the vector of probabilities to a target vector
    y_pred = np.where(prob > 0.5, 1, 0)

    score = f1_score(y_true, y_pred)

    return score


# Build the model with hyperparameter tuning
def build_model(hp):
    X_input = Input((240, 240, 3))
    X = ZeroPadding2D((2, 2))(X_input)
    X = Conv2D(hp.Int('conv_filters', min_value=32, max_value=128,
               step=16), (7, 7), strides=(1, 1), name='conv0')(X)
    X = BatchNormalization(axis=3, name='bn0')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((4, 4), name='max_pool0')(X)
    X = MaxPooling2D((4, 4), name='max_pool1')(X)
    X = Flatten()(X)
    X = Dense(1, activation='sigmoid', name='fc')(X)
    model = Model(inputs=X_input, outputs=X, name='BrainDetectionModel')
    model.compile(optimizer=tf.keras.optimizers.Adam(hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
                  loss='binary_crossentropy', metrics=['accuracy'])
    return model


tuner = RandomSearch(build_model, objective='val_accuracy', max_trials=10,
                     executions_per_trial=1, directory='model_tuning', project_name='brain_tumor_detection')
tuner.search(x=X_train, y=y_train, epochs=10, validation_data=(X_val, y_val))

# Retrieve the best model
best_model = tuner.get_best_models(num_models=1)[0]

# After hyperparameter tuning, use the best model for further training and evaluation
best_model.summary()

# Setup logging and checkpointing
log_file_name = f'brain_tumor_detection_cnn_{int(time.time())}'
tensorboard = TensorBoard(log_dir=f'logs/{log_file_name}')
filepath = "models/cnn-parameters-improvement-{epoch:02d}-{val_accuracy:.2f}.model"
checkpoint = ModelCheckpoint(
    filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')


# Train the best model
start_time = time.time()
best_model.fit(x=X_train, y=y_train, batch_size=32, epochs=20, validation_data=(
    X_val, y_val), callbacks=[tensorboard, checkpoint])
end_time = time.time()
execution_time = end_time - start_time
print(f"Elapsed time: {hms_string(execution_time)}")
