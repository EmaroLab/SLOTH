#!/usr/bin/python

import numpy as np
import tensorflow as tf
from keras.models import load_model
import matplotlib.pyplot as plt


class sloth:
    def __init__(self, model_path, window_size, class_size, feature_size, rho, tau, c):
        self.model_path = model_path
        self.window_size = window_size
        self.class_size = class_size
        self.feature_size = feature_size
        self.probabilities_size = window_size

        self.graph = tf.get_default_graph()
        self.model = load_model(self.model_path)
        self.model.summary()
        self.window = np.empty((1,self.window_size,self.feature_size))
        self.window[:] = np.nan

        self.probabilities = np.empty((1,self.probabilities_size,self.class_size))
        self.probabilities[:] = np.nan

        self.rho = rho
        self.tau = tau
        self.c = c

        self.time = 0
        self.peaks = np.zeros((1,self.class_size))

        self.gestures = []
        self.prob_mean = []

    def classify(self):
        if not np.any(np.isnan(self.window)):
            with self.graph.as_default():
                self.probabilities = np.roll(self.probabilities,self.probabilities_size-1,1)
                self.probabilities[0,-1,:] = self.model.predict(self.window, batch_size=self.window_size, verbose=2)
        else:
            print "The sliding window is not completely full"

    def detect(self):
        delta_prob = (self.probabilities[0,-1,:] - self.probabilities[0,-1-1,:]) 
        possible_peaks = np.where(delta_prob > self.rho)
        possible_peaks = possible_peaks[0]

        for ids in possible_peaks:
            if self.peaks[0, ids] == 0:
                self.peaks[0, ids] = self.time
            else:
                time_diff = self.time - self.peaks[0, ids]
                if time_diff >= self.c[ids]:
                    self.peaks[0, ids] = self.time
        active_peaks = np.where(self.peaks[0,:]> 0)
        active_peaks = active_peaks[0]

        for ids in active_peaks:
            time_diff = self.time - self.peaks[0, ids] + 1
            if time_diff >= self.c[ids]:
                start = int(self.probabilities_size-time_diff)
                prob_mean = np.mean(self.probabilities[0,start:,ids])
                if prob_mean > self.tau[ids]:
                    self.peaks[0, ids] = 0
                    self.gestures.append(ids+1)
                    self.prob_mean.append(prob_mean)

    def window_update(self, x, y, z):
        self.window = np.roll(self.window,self.window_size-1,1)
        self.window[:,-1,0] = x
        self.window[:,-1,1] = y
        self.window[:,-1,2] = z
        self.time += 1

    def get_gestures(self):
        temp = self.gestures
        self.gestures = []
        temp2 = self.prob_mean
        self.prob_mean = []
        return temp, temp2
