import scipy.io as sio
import tensorflow as tf
from keras import backend as K
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Reshape, ZeroPadding2D, Convolution2D, MaxPooling2D, Dense, Dropout, Softmax, Flatten, Activation, BatchNormalization
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from PIL import Image
import numpy as np
from tqdm import tqdm
import os
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report

# Read .mat file for Model Description
mat = sio.loadmat('data/vgg_face.mat', struct_as_record=False)
net = mat['net'][0][0]
mat_model = net.layers
mat_model_layers = mat_model[0]
num_mat_layers = mat_model_layers.shape[0]

# VGG Architecture Implementation
def vgg_tf():
    model = Sequential()
    for i in range(num_mat_layers):
        mat_model_layer = mat_model_layers[i][0][0].name[0]
        if mat_model_layer.find("conv") == 0 or mat_model_layer.find("fc") == 0:
            weights = mat_model_layers[i][0,0].weights
            weights_shape = weights[0][0].shape
            filter_x = weights_shape[0]; filter_y = weights_shape[1]
            number_of_filters = weights_shape[3]

            if mat_model_layer.find("conv") == 0:
                if i == 0:
                    model.add(ZeroPadding2D((1,1),input_shape=(224,224, 3)))
                else:
                    model.add(ZeroPadding2D((1,1)))
            model.add(Convolution2D(number_of_filters, (filter_x, filter_y), name= mat_model_layer))
        else:
            if mat_model_layer.find("relu") == 0:
                model.add(Activation('relu', name=mat_model_layer))
            elif mat_model_layer.find("dropout") == 0:
                model.add(Dropout(0.5, name=mat_model_layer))
            elif mat_model_layer.find("pool") == 0:
                model.add(MaxPooling2D((2,2), strides=(2,2), name=mat_model_layer))
            elif mat_model_layer.find("softmax") == 0:
                model.add(Activation('softmax', name=mat_model_layer))
    return model

# Save model to .h5 file for use later.
model_ = vgg_tf()
model_.save('data/vgg_face.h5')
model = Model(inputs=model_.layers[0].input,outputs=model_.layers[-2].output)

# Load dataset
def pad(img):
    ht, wd, cc= img.shape
    ww = 224
    hh = 224
    color = (0,0,0)
    result = np.full((hh,ww,cc), color, dtype=np.uint8)
    xx = (ww - wd) // 2
    yy = (hh - ht) // 2
    result[yy:yy+ht, xx:xx+wd] = img
    return result

def load_img_file(path, model):
    img = Image.open(path)
    x = img_to_array(img) 
    x = pad(x)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    x = model(x)
    return x

def load_data(path, model):
    x = []
    y = []
    labels = {}
    folders = os.listdir(path)
    for i in tqdm(range(len(folders))):
        labels[i] = folders[i]
        label_path = path + '/' + folders[i]
        images = os.listdir(label_path)
        for image in images:
            img = load_img_file(label_path + '/' + image, model)
            x.append(np.squeeze(K.eval(img)).tolist())
            y.append(i)
    return np.array(x), np.array(y), labels

# training set preparation
X_train = y_train = labels = None
try:
    X_train = np.load('data/combined/X_train.npy', allow_pickle=True)
    y_train = np.load('data/combined/y_train.npy', allow_pickle=True)
    labels = np.load('data/combined/labels.npy', allow_pickle=True)[()]
except:
    X_train, y_train, labels = load_data('data/combined/aligned', model)
    np.save('data/combined/X_train', X_train)
    np.save('data/combined/y_train', y_train)
    np.save('data/combined/labels', labels)

# testing set preparation
X_test = y_test = None
try:
    X_test = np.load('data/combined/X_test.npy', allow_pickle=True)
    y_test = np.load('data/combined/y_test.npy', allow_pickle=True)
    labels = np.load('data/combined/labels.npy', allow_pickle=True)[()]
except:
    X_test, y_test, labels = load_data('data/combined/valid', model)
    np.save('data/combined/X_test', X_test)
    np.save('data/combined/y_test', y_test)

# Construct Classifier
def classier_model(input_dim, num_classes):
    classifier = Sequential()
    classifier.add(Dense(num_classes, input_dim=input_dim))
    classifier.add(Activation('softmax'))
    classifier.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer='nadam', metrics=['accuracy'])
    return classifier

# Gender classfication
y_train_gender = np.copy(y_train)
y_test_gender = np.copy(y_test)
y_train_gender[y_train_gender % 2 == 0] = 0
y_train_gender[y_train_gender % 2 != 0] = 1
y_test_gender[y_test_gender % 2 == 0] = 0
y_test_gender[y_test_gender % 2 != 0] = 1

classifier = classier_model(2622, 2)
history = classifier.fit(X_train, y_train_gender, batch_size=16, epochs=100, validation_data=(X_test, y_test_gender))
classifier.save('data/gender_classifier.h5')

# Plot model accuracy graph with epoch
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

# Model Evaluation: classification metrics such as accuracy, precision, recall, f1 score
print(classification_report(y_test_gender, history.predict(X_test)))
