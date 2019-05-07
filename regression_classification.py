""" build a CNN capable of detecting positive selection on human genome - predicting a continious value though instead of the discrete label of binary_classification.py """

# imports

import os
import gzip
import pickle

import numpy as np
import scipy.stats

import skimage.transform
from keras import models, layers, activations, optimizers, regularizers
from keras.utils import plot_model
from keras.models import load_model
from keras import backend as K

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import pymc3 # this will be removed
import pydot # optional

import pathlib

# run the ImaGene.py file

execfile('/home/nathanrobins/UG_proj/ImaGene/ImaGene.py')

# create a new folder to store model

folder='/home/nathanrobins/UG_proj/regression_results'
print(folder)
#pathlib.Path(folder).mkdir(parents=True, exist_ok=True) 

# this is not overly important...

m = 'RowsCols'

# set up a while loop in order to train & test the neural network

x = 1
while x <= 2:
    # read the simulations & store into objs
    myfile =""" build a CNN capable of detecting positive selection on human genome - predicting a continious value though instead of the discrete label of binary_classification.py """

# imports

import os
import gzip
import pickle

import numpy as np
import scipy.stats

import skimage.transform
from keras import models, layers, activations, optimizers, regularizers
from keras.utils import plot_model
from keras.models import load_model
from keras import backend as K

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import pymc3 # this will be removed
import pydot # optional

import pathlib

# run the ImaGene.py file

execfile('/home/nathanrobins/UG_proj/ImaGene/ImaGene.py')

# create a new folder to store model

folder='/home/nathanrobins/UG_proj/regression_results'
print(folder)
#pathlib.Path(folder).mkdir(parents=True, exist_ok=True) 

# this is not overly important...

m = 'RowsCols'

# set up a while loop in order to train & test the neural network

x = 1
while x <= 2:
    # read the simulations & store into objs
    myfile = ImaFile(simulations_folder='/home/nathanrobins/UG_proj/reg_simdata', nr_samples=128, model_name='Marth-3epoch-CEU')
    mygene = myfile.read_simulations(parameter_name='selection_coeff_hetero', max_nrepl=10)

    mygene.majorminor()
    mygene.filter_freq(0.01)
    if (m =='Rows') | (m == 'RowsCols'):
        mygene.sort('rows_freq')
    if (m =='Cols') | (m == 'RowsCols'):
        mygene.sort('cols_freq')
    mygene.resize((128, 128))
    mygene.convert()

    # first iteration
    if x == 1:


        # build the neural network
        model = models.Sequential([
                    layers.Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), activation='relu', kernel_regularizer=regularizers.l1_l2(l1=0.005, l2=0.005), padding='valid', input_shape=mygene.data.shape[1:4]),
                    layers.MaxPooling2D(pool_size=(2,2)),
                    layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), activation='relu', kernel_regularizer=regularizers.l1_l2(l1=0.005, l2=0.005), padding='valid'),
                    layers.MaxPooling2D(pool_size=(2,2)),
                    layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), activation='relu', kernel_regularizer=regularizers.l1_l2(l1=0.005, l2=0.005), padding='valid'),
                    layers.MaxPooling2D(pool_size=(2,2)),
                    layers.Flatten(),
                    layers.Dense(units=1, activation='relu')])
        model.compile(optimizer='rmsprop',
                    loss='mse',
                    metrics=['mae'])
        plot_model(model, folder + '/model.png')

        mynet = ImaNet(name='[C32+P]+[C64+P]x2')

    # training
    if x < 2:
        score = model.fit(mygene.data, mygene.targets, batch_size=32, epochs=1, verbose=1, validation_split=0.10)
        print(score)
        mynet.update_scores(score)
    else:
        # testing
        mynet.test = model.evaluate(mygene.data, mygene.targets, batch_size=None, verbose=1)
#        mynet.predict(mygene, model)
#        model.predict(mygene, model)

    x += 1
""" build a CNN capable of detecting positive selection on human genome - predicting a continious value though instead of the discrete label of binary_classification.py """

# imports

import os
import gzip
import pickle

import numpy as np
import scipy.stats

import skimage.transform
from keras import models, layers, activations, optimizers, regularizers
from keras.utils import plot_model
from keras.models import load_model
from keras import backend as K

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import pymc3 # this will be removed
import pydot # optional

import pathlib

# run the ImaGene.py file

execfile('/home/nathanrobins/UG_proj/ImaGene/ImaGene.py')

# create a new folder to store model

folder='/home/nathanrobins/UG_proj/regression_results'
print(folder)
#pathlib.Path(folder).mkdir(parents=True, exist_ok=True) 

# this is not overly important...

m = 'RowsCols'

# set up a while loop in order to train & test the neural network

x = 1
while x <= 2:
    # read the simulations & store into objs
    myfile = ImaFile(simulations_folder='/home/nathanrobins/UG_proj/reg_simdata', nr_samples=128, model_name='Marth-3epoch-CEU')
    mygene = myfile.read_simulations(parameter_name='selection_coeff_hetero', max_nrepl=10)

    mygene.majorminor()
    mygene.filter_freq(0.01)
    if (m =='Rows') | (m == 'RowsCols'):
        mygene.sort('rows_freq')
    if (m =='Cols') | (m == 'RowsCols'):
        mygene.sort('cols_freq')
    mygene.resize((128, 128))
    mygene.convert()

    # first iteration
    if x == 1:


        # build the neural network
        model = models.Sequential([
                    layers.Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), activation='relu', kernel_regularizer=regularizers.l1_l2(l1=0.005, l2=0.005), padding='valid', input_shape=mygene.data.shape[1:4]),
                    layers.MaxPooling2D(pool_size=(2,2)),
                    layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), activation='relu', kernel_regularizer=regularizers.l1_l2(l1=0.005, l2=0.005), padding='valid'),
                    layers.MaxPooling2D(pool_size=(2,2)),
                    layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), activation='relu', kernel_regularizer=regularizers.l1_l2(l1=0.005, l2=0.005), padding='valid'),
                    layers.MaxPooling2D(pool_size=(2,2)),
                    layers.Flatten(),
                    layers.Dense(units=1, activation='relu')])
        model.compile(optimizer='rmsprop',
                    loss='mse',
                    metrics=['mae'])
        plot_model(model, folder + '/model.png')

        mynet = ImaNet(name='[C32+P]+[C64+P]x2')

    # training
    if x < 2:
        score = model.fit(mygene.data, mygene.targets, batch_size=32, epochs=1, verbose=1, validation_split=0.10)
        print(score)
        mynet.update_scores(score)
    else:
        # testing
        mynet.test = model.evaluate(mygene.data, mygene.targets, batch_size=None, verbose=1)
#        mynet.predict(mygene, model)
#        model.predict(mygene, model)

    x += 1

# plot?
mynet.plot_train()


# save final (trained) model
#model.save(folder + '/model.h5')

# save testing data
#mygene.save(folder + '/mygene')

# save final network
#mynet.save(folder + '/mynet')

print(mynet.test)
# plot?
mynet.plot_train()


# save final (trained) model
#model.save(folder + '/model.h5')

# save testing data
#mygene.save(folder + '/mygene')

# save final network
#mynet.save(folder + '/mynet')

print(mynet.test) ImaFile(simulations_folder='/home/nathanrobins/UG_proj/reg_simdata', nr_samples=128, model_name='Marth-3epoch-CEU')
    mygene = myfile.read_simulations(parameter_name='selection_coeff_hetero', max_nrepl=10)

    mygene.majorminor()
    mygene.filter_freq(0.01)
    if (m =='Rows') | (m == 'RowsCols'):
        mygene.sort('rows_freq')
    if (m =='Cols') | (m == 'RowsCols'):
        mygene.sort('cols_freq')
    mygene.resize((128, 128))
    mygene.convert()

    # first iteration
    if x == 1:


        # build the neural network
        model = models.Sequential([
                    layers.Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), activation='relu', kernel_regularizer=regularizers.l1_l2(l1=0.005, l2=0.005), padding='valid', input_shape=mygene.data.shape[1:4]),
                    layers.MaxPooling2D(pool_size=(2,2)),
                    layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), activation='relu', kernel_regularizer=regularizers.l1_l2(l1=0.005, l2=0.005), padding='valid'),
                    layers.MaxPooling2D(pool_size=(2,2)),
                    layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), activation='relu', kernel_regularizer=regularizers.l1_l2(l1=0.005, l2=0.005), padding='valid'),
                    layers.MaxPooling2D(pool_size=(2,2)),
                    layers.Flatten(),
                    layers.Dense(units=1, activation='relu')])
        model.compile(optimizer='rmsprop',
                    loss='mse',
                    metrics=['mae'])
        plot_model(model, folder + '/model.png')

        mynet = ImaNet(name='[C32+P]+[C64+P]x2')

    # training
    if x < 2:
        score = model.fit(mygene.data, mygene.targets, batch_size=32, epochs=1, verbose=1, validation_split=0.10)
        print(score)
        mynet.update_scores(score)
    else:
        # testing
        mynet.test = model.evaluate(mygene.data, mygene.targets, batch_size=None, verbose=1)
#        mynet.predict(mygene, model)
#        model.predict(mygene, model)

    x += 1

# plot?
mynet.plot_train()


# save final (trained) model
#model.save(folder + '/model.h5')

# save testing data
#mygene.save(folder + '/mygene')

# save final network
#mynet.save(folder + '/mynet')

print(mynet.test)
