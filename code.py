# Import necessary libraries
import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dropout, Flatten, BatchNormalization, Dense, MaxPool2D, Conv2D, Input, Activation, Add
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam, Adagrad, Adadelta, Adamax, RMSprop

import os

# Set directory path
fldr = "/content/UTKFace"
files = os.listdir(fldr)

# Initialize lists to store images, ages, and genders
ages = []
genders = []
images = []

# Loop through files to read images and extract age and gender information
for fle in files:
    age = int(fle.split('_')[0])
    gender = int(fle.split('_')[1])
    total = fldr + '/' + fle
    image = cv2.imread(total)
    image1 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image2 = cv2.resize(image1, (48, 48))
    images.append(image2)
    ages.append(age)
    genders.append(gender)

# Display a sample image with its age and gender
plt.imshow(images[87])
print(ages[87])
print(genders[87])

# Convert lists to numpy arrays
images_f = np.array(images)
ages_f = np.array(ages)
genders_f = np.array(genders)

# Save numpy arrays
np.save(fldr + 'images.npy', images_f)
np.save(fldr + 'ages.npy', ages_f)
np.save(fldr + 'genders.npy', genders_f)

# Plot gender distribution
values, counts = np.unique(genders_f, return_counts=True)
fig = plt.figure()
ax = fig.add_axes([0, 0, 1, 1])
genders1 = ['Male', 'Female']
values = [12391, 11317]
ax.bar(genders1, values)
plt.show()

# Plot age distribution
values, counts = np.unique(ages_f, return_counts=True)
val = values.tolist()
cn = counts.tolist()
plt.plot(cn)
plt.xlabel('Ages')
plt.ylabel('Distribution')
plt.show()

# Create labels array combining ages and genders
lables = []
i = 0
while i < len(ages):
    lable = [ages[i], genders[i]]
    lables.append(lable)
    i = i + 1

# Normalize images
images_f_2 = images_f / 255.0

# Convert labels to numpy array and save
lables_f = np.array(lables)
np.save(fldr + "lables.npy", lables_f)

from sklearn.model_selection import train_test_split

# Split data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(images_f_2, lables_f, test_size=0.25)

# Split labels into separate age and gender labels
Y_train_2 = [Y_train[:, 1], Y_train[:, 0]]
Y_test_2 = [Y_test[:, 1], Y_test[:, 0]]

# Define convolutional block
def convolution(input_tensor, filters):
    x = Conv2D(filters=filters, kernel_size=(3, 3), padding='same', strides=(1, 1), kernel_regularizer=l2(0.001))(input_tensor)
    x = Dropout(0.1)(x)
    x = Activation('relu')(x)
    return x

# Define the model architecture
def c_model(input_shape):
    inputs = Input(input_shape)
    conv_1 = convolution(inputs, 32)
    maxp_1 = MaxPool2D(pool_size=(2, 2))(conv_1)
    conv_2 = convolution(maxp_1, 64)
    maxp_2 = MaxPool2D(pool_size=(2, 2))(conv_2)
    conv_3 = convolution(maxp_2, 128)
    maxp_3 = MaxPool2D(pool_size=(2, 2))(conv_3)
    conv_4 = convolution(maxp_3, 256)
    maxp_4 = MaxPool2D(pool_size=(2, 2))(conv_4)
    flatten = Flatten()(maxp_4)
    dense_1 = Dense(64, activation='relu')(flatten)
    dense_2 = Dense(64, activation='relu')(flatten)
    drop_1 = Dropout(0.2)(dense_1)
    drop_2 = Dropout(0.2)(dense_2)
    output_1 = Dense(1, activation='sigmoid', name='sex_out')(drop_1)
    output_2 = Dense(1, activation='relu', name='age_out')(drop_2)
    model = Model(inputs=[inputs], outputs=[output_1, output_2])
    model.compile(loss=["binary_crossentropy", "mae"], optimizer="Adam", metrics=["accuracy"])
    return model

# Instantiate and summarize the model
Models = c_model((48, 48, 3))
Models.summary()

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Define callbacks for model training
fle_ss = 'Age_Sex_Detection.h5.keras'
checkpoint = ModelCheckpoint(
    fle_ss, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', save_freq='epoch')
Early_stop = tf.keras.callbacks.EarlyStopping(patience=75, monitor='val_loss', restore_best_weights="True")
callback_list = [checkpoint, Early_stop]

# Train the model
History = Models.fit(X_train, Y_train_2, batch_size=64, validation_data=(X_test, Y_test_2), epochs=250, callbacks=callback_list)

# Evaluate the model
Models.evaluate(X_test, Y_test_2)

# Make predictions
pred = Models.predict(X_test)

# Plot training history
plt.plot(History.history['loss'])
plt.plot(History.history['val_loss'])
plt.title("Model loss")
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.subplots_adjust(top=1.0, bottom=0.0, right=0.95, left=0, hspace=0.25, wspace=0.35)

plt.plot(History.history['sex_out_accuracy'])
plt.plot(History.history['val_sex_out_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.subplots_adjust(top=1.0, bottom=0.0, right=0.95, left=0, hspace=0.25, wspace=0.35)

# Plot actual vs predicted ages
fig, ax = plt.subplots()
ax.scatter(Y_test_2[1], pred[1])
ax.plot([Y_test_2[1].min(), Y_test_2[1].max()], [Y_test_2[1].min(), Y_test_2[1].max()], 'k--', lw=4)
ax.set_xlabel('Actual age')
ax.set_ylabel('Predicted age')
plt.show()

# Generate classification report and confusion matrix for gender prediction
from sklearn.metrics import confusion_matrix, classification_report

pred_l = [int(np.round(pred[0][i])) for i in range(len(pred[0]))]
report = classification_report(Y_test_2[0], pred_l)
print(report)

results = confusion_matrix(Y_test_2[0], pred_l)
import seaborn as sns
sns.heatmap(results, annot=True)

# Function to test model on a single image
def test_image(ind, images_f, images_f_2, Model):
    plt.imshow(images_f[ind])
    images_test = images_f_2[ind]
    pred_l = Model.predict(np.array([images_test]))
    sex_f = ['Male', 'Female']
    age = int(np.round(pred_l[1][0]))
    sex = int(np.round(pred_l[0][0]))
    print("Predicted Age is " + str(age))
    print("Predicted Gender is " + sex_f[sex])

test_image(4, images_f, images_f_2, Models)
test_image(2, images_f, images_f_2, Models)
