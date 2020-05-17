import os


import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense, Flatten
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import *
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import DataHandler
os.chdir('C:\\Users\\hassanelhajj\\desktop\\docs2\\training\\client_training_2')
train_path = 'train'
valid_path = 'valid'
test_path = 'test'


def plots(ims, figsize=(12,6), rows=1, interp=False, titles=None):
    if type(ims[0]) is np.ndarray:
        ims = np.array(ims).astype(np.uint8)
        if (ims.shape[-1] != 3):
            ims = ims.transpose((0,2,3,1))
    f = plt.figure(figsize=figsize)
    cols = len(ims)//rows if len(ims) % 2 == 0 else len(ims)//rows + 1
    for i in range(len(ims)):
        sp = f.add_subplot(rows, cols, i+1)
        sp.axis('Off')
        if titles is not None:
            sp.set_title(titles[i], fontsize=16)
    plt.imshow(ims[i], interpolation=None if interp else 'none')

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def array_to_class(arr):
    l =[]
    for i in arr:
        print(i)
        max = np.max(i)
        l.append(np.where(i==max)[0][0])
    return l

def classes_to_labels():
    #gets the list of labels from a generator and turns it into a numpy array
    l = list(validation_generator.class_indices.keys())
    return np.array(l)

def ResolveFolderNameToKey(generator,name):
  #returns the keys for the generator indices
  return generator.class_indices[name]

def GetKey(generator,value):
  #finds the key from a value in a dictionnary generator indices
  key = {k:v for k, v in generator.class_indices.items() if v == value}
  return [*key.keys()][0]


def RevertDictionnary(generator):
  return {v: k for k, v in generator.class_indices.items()}
def GetTrueLabelsArray(liste,revertedDic):
  #gets a list of the true classes labels / folder names
  l=[]
  for i in liste:
    l.append(revertedDic[i])
  return l
def PlotTrainingAndValidationAccuracy(history):
  acc = history.history['accuracy']
  val_acc = history.history['val_accuracy']
  epochs = range(len(acc))

  plt.plot(epochs, acc, 'b', label='Training acc')
  plt.plot(epochs, val_acc, 'r', label='Validation acc')
  plt.title('Training and validation accuracy')
  plt.legend()
  plt.show()


def PlotTrainingAndValidationLoss(history):
  acc = history.history['accuracy']
  loss = history.history['loss']
  val_loss = history.history['val_loss']
  epochs = range(len(acc))
  plt.figure()
  plt.plot(epochs, loss, 'b', label='Training loss')
  plt.plot(epochs, val_loss, 'r', label='Validation loss')
  plt.title('Training and validation loss')
  plt.legend()
  plt.show()


def DataLoader(nb_epochs,batch_size,desired_batch_size):
  os.chdir('C:\\Users\\hassanelhajj\\desktop\\docs2\\training\\client_training_2')
  train_path = 'train'
  valid_path = 'valid'
  test_path = 'test'

  img_height = 224
  img_width =224
  train_datagen = ImageDataGenerator(
      validation_split=0.3,rescale=1./255) # set validation split


  train_generator = train_datagen.flow_from_directory(
      train_path,
      target_size=(224, 224),
      batch_size=batch_size,
      class_mode='categorical',
      subset='training') # set as training data


  validation_generator = train_datagen.flow_from_directory(
      train_path, # same directory as training data
      target_size=(224, 224),
      batch_size=batch_size,
      class_mode='categorical',
      subset='validation') # set as validation data

  test_batches = ImageDataGenerator(rescale=1./255).flow_from_directory(test_path, target_size=(224,224),
    class_mode='categorical', batch_size=desired_batch_size)

  return (train_generator,validation_generator,test_batches)

def CreateModel(numberOfDenseLayers):
  # number_of_dense_layers  = len(train_generator.class_indices.keys())
  vgg16_model = keras.applications.vgg16.VGG16(input_shape=(224,224,3))
  type(vgg16_model)
  #we convert vgg to a sequential model and remove the last layer that we're not gonna use
  model = Sequential()
  for layer in vgg16_model.layers[:-1]:
      model.add(layer)
  for layer in model.layers:
      layer.trainable = False
  model.add(Dense(numberOfDenseLayers, activation='softmax'))
  model.compile(optimizer=Adam(lr=.001), loss='categorical_crossentropy', metrics=['accuracy'])
  return model

def TrainModel(model,dataLoader,nb_epochs):
  overfitcallback = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=15,            verbose=0, mode='min')
  reduce_lr =keras.callbacks.ReduceLROnPlateau(
      monitor='val_loss', factor=0.1, patience=10, verbose=0, mode='min',
      min_delta=0.0001, cooldown=0, min_lr=0
  )
  history =model.fit_generator(
      dataLoader[0],
      steps_per_epoch =    1,
      validation_data = dataLoader[1],
      validation_steps =   1,
      epochs = nb_epochs,
      callbacks=[overfitcallback,reduce_lr])
  return history

def PredictOnModel(test_batches,model):
  test_imgs, test_labels = next(test_batches)
  #test_labels = test_labels[:,0]
  filenames = test_batches.filenames
  nb_samples = len(filenames)
  predictions = model.predict_generator(test_batches,steps =1,verbose=0)
  test_classes = array_to_class(test_labels)
  train_predictions = array_to_class(predictions)
  cm = confusion_matrix(test_classes,train_predictions)
  cm_plot_labels = classes_to_labels()
  plot_confusion_matrix(cm, cm_plot_labels, title='Confusion Matrix')
  PlotTrainingAndValidationAccuracy(history)
  PlotTrainingAndValidationLoss(history)

def SaveModelWeights(model,modelName):
  os.chdir('C:\\Users\\hassanelhajj\\desktop\\docs2\\FypApi\\foodclusteringapp')
  model.save_weights(modelName,True)
  del model
def LoadModel(modelName,denselayers):
  #this function recreates the VGG16 sequential model
  vgg16_model = keras.applications.vgg16.VGG16(input_shape=(224,224,3))
  type(vgg16_model)
  #we convert vgg to a sequential model and remove the last layer that we're not gonna use
  model = Sequential()
  for layer in vgg16_model.layers[:-1]:
      model.add(layer)
  for layer in model.layers:
      layer.trainable = False
  model.add(Dense(denselayers, activation='softmax'))
  model.compile(optimizer=Adam(lr=.001), loss='categorical_crossentropy', metrics=['accuracy'])
  model.load_weights(modelName)
  return model


Email = "hassanitohajj@gmail.com"
Password = "gimzjzkyjbgneuzo"
Recipient = "hassanlhage@hotmail.com"
def TrainServer(number):
    import time
    DataHandler.send_email(Email,Password,Recipient,"Training model ", "Dear user your model has started trianing")
    time.sleep(number)
    print("Training done")
    DataHandler.send_email(Email,Password,Recipient,"Training model ", "Dear user your model has finished trianing")

