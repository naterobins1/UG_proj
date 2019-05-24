
# reproduce regression analysis (no dense layer, (3,3) filter)

import os
import gzip
import _pickle as pickle

import numpy as np
import scipy.stats

import skimage.transform
from keras import models, layers, activations, optimizers, regularizers
from keras.utils import plot_model
from keras.models import load_model

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
#import pymc3 # this will be removed
import pydot # optional

import itertools

exec(open('/home/nathanrobins/UG_proj/ImaGene/ImaGene.py').read())

i = 1
while i <= 10:

    #change these files
    myfile = ImaFile(simulations_folder='/home/nathanrobins/UG_proj/code/simulation.Continuous.'+str(i), nr_samples=128, model_name='Marth-3epoch-CEU')
    mygene = myfile.read_simulations(parameter_name='selection_coeff_hetero', max_nrepl=100)

    #mygene.majorminor()
    mygene.filter_freq(0.01)
    mygene.sort('rows_freq')
    mygene.sort('cols_freq')
    mygene.resize((128, 128))
    mygene.convert()

    # first iteration
    if i == 1:

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
#        plot_model(model, folder + '/model.png')

        mynet = ImaNet(name='[C32+P]+[C64+P]x2')

    # training
    if i < 10:
        score = model.fit(mygene.data, mygene.targets, batch_size=32, epochs=1, verbose=1, validation_split=0.10)
        print(score)
        mynet.update_scores(score)
    else:
        # testing
        mynet.test = model.evaluate(mygene.data, mygene.targets, batch_size=None, verbose=1)
        mynet.predict(mygene, model)
        print(mynet.test)
        #mynet.plot_train()
        #plot a confusion matrix
        #mynet.values.shape
        mynet.plot_cm(mygene.classes, file= '/home/nathanrobins/UG_proj/plots/regression/cm')
        #mynet.plot_cm(mygene.classes)
        #plot a scatter
        mynet.plot_scatter(file= '/home/nathanrobins/UG_proj/plots/regression/scatter')
        #mynet.plot_scatter()
    i += 1