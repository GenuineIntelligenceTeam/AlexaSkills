%matplotlib notebook
import matplotlib.pyplot as plt
from camera import take_picture
import face_recognition
import numpy as np
import cv2
import mxnet as mx
import pandas as pd
import random
import os
from PIL import Image, ImageDraw
import pickle
from mynn.layers.dense import dense
from mynn.layers.conv import conv
from mynn.initializers.he_normal import he_normal
from mynn.activations.relu import relu
from mynn.activations.softmax import softmax
from mynn.losses.cross_entropy import softmax_cross_entropy 
from mynn.optimizers.sgd import SGD
from mygrad.nnet.layers import max_pool
import mygrad as mg
from mynn.initializers.glorot_uniform import glorot_uniform

class Model:
    def __init__(self):
        self.conv1 = conv(1, 10, 5, padding=0, weight_initializer=glorot_uniform)
        self.conv2 = conv(5, 20, 5, padding=0, weight_initializer=glorot_uniform)
        self.dense1 = dense(290, 20, weight_initializer=glorot_uniform)
        self.dense2 = dense(20, 2, weight_initializer=glorot_uniform)
        
    def __call__(self, x):
        ''' Forward data through the network.

        This allows us to conveniently initialize a model `m` and then send data through it
        to be classified by calling `m(x)`.
        
        Parameters
        ----------
        x : Union[numpy.ndarray, mygrad.Tensor], shape=(N, D)
            The data to forward through the network.
            
        Returns
        -------
        mygrad.Tensor, shape=(N, 1)
            The model outputs.
        '''
        # STUDENT CODE
        # if num_filters = 10; (N, C, 32, 32) --> (N, 10, 28, 28)
        # if num_filters = 10; (N, C, 32, 32) --> (N, 10, 28, 28)
        x = self.conv1(x)
        # (N, 10, 28, 28) --> (N, 10, 14, 14)
        x = max_pool(x, (2,2), 2)
        # if num_filters = 20; (N, 10, 14, 14) --> (N, 20, 10, 10)
        x = self.conv2(x)
        # (N, 20, 10, 10) --> (N, 20, 5, 5)
        x = max_pool(x, (2,2), 2)
        # (N, 20, 5, 5) -reshape-> (N, 500) x (500, 20) -> (N, 20)
        x = relu(self.dense1(x.reshape(x.shape[0], -1)))
        # (N, 20) -> (N, 10)
        return self.dense2(x)
        
    @property
    def parameters(self):
        ''' A convenience function for getting all the parameters of our model. '''
                # STUDENT CODE
        params = []
        for layer in (self.conv1, self.conv2, self.dense1, self.dense2):
            params += list(layer.parameters)
        return params
    
angry_model = pickle.load(open('angry_model.sav', 'rb'))
disg_model = pickle.load(open('disg_model.sav', 'rb'))
fear_model = pickle.load(open('fear_model.sav', 'rb'))
happy_model = pickle.load(open('happy_model.sav', 'rb'))
Neutral_model = pickle.load(open('Neutral_model.sav', 'rb'))
sad_model = pickle.load(open('sad_model.sav', 'rb'))
surprise_model = pickle.load(open('surprise_model.sav', 'rb'))

from collections import defaultdict
def emotion_test(pic_test):
    allModel = [angry_model,fear_model,happy_model,sad_model,surprise_model,Neutral_model]
    allModelName = ['angry','fearful','happy','sad','surprised','neutral']
    myDic = defaultdict(list)
    ans = np.zeros((6))
    for index,i in enumerate(allModel):
        try:
            encode_test = face_recognition.face_encodings(pic_test)[0].reshape(1,1,128)
            myDic[allModelName[index]] = mg.abs(i(encode_test)[0,0])
        except:
            print("oof")
    ansTrue = []
    myDic['fearful'] += 0.2
    if myDic['angry'] < 1:
        ansTrue += ["angry"]
    if myDic['sad'] < 0.9:
        ansTrue += ["sad"]
        del myDic['happy']
    if myDic['fearful'] < 0.7:
        ansTrue += ["fearful"]
        try:
            del myDic['happy']
        except:
            print("oof")
    if myDic['surprised'] < 0.5:
        ansTrue += ['surprised']
    if min(myDic, key=myDic.get) == 'neutral' and myDic['neutral'] > 0.5:
        del myDic['neutral']
    ansTrue += [min(myDic, key=myDic.get)]
    ansTrue = list(set(ansTrue))
    return ' and '.join(ansTrue)

#Use this to take picture and store: pic_test = take_picture()
#Then do this to get the emotion in text: emotion_test(pic_test)