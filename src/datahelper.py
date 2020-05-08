import numpy as np
import os
import sys
import keras
from keras.models import Sequential, Model
from keras.layers.core import Dense, Activation
from keras.layers import LSTM, Input, Flatten, Concatenate, Embedding, Convolution1D,Dropout, Conv2D, Conv1D, Bidirectional
from keras.layers.wrappers import TimeDistributed

from keras.optimizers import SGD, Adam, RMSprop
from keras.layers.normalization import BatchNormalization
from sklearn.preprocessing import label_binarize
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing import sequence
from keras.utils import to_categorical

import librosa
import argparse

import tensorflow as tf
from keras import backend as K
from keras import regularizers, constraints, initializers, activations
from keras.layers.recurrent import Recurrent
from keras.engine import InputSpec
from keras.callbacks import EarlyStopping,TensorBoard, ModelCheckpoint
from keras_self_attention import SeqSelfAttention
from keras_multi_head import MultiHeadAttention

import cv2
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
import argparse
import imutils
import dlib



detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')
fa = FaceAligner(predictor, desiredFaceWidth=64)
POINTS_NUM_LANDMARK = 68

def face_align(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 2)
    print(np.shape(rects))
    (x, y, w, h) = rect_to_bb(rects[0])
    faceOrig = imutils.resize(img[y:y + h, x:x + w], width=64)
    faceAligned = fa.align(img, gray, rects[0])
    return faceOrig, faceAligned


def entropy_weight(data, alpha):
    data = np.square(data)
#     print(data)
    max_v = np.max(data)
#     print('max',max_v)
    A = []
    for d_t in data:
        x = d_t/max_v
        A.append(x)
    A = np.asarray(A)
#     print(A)

    rows, cols = A.shape
    k = 1.0 / math.log(rows)
    lnf = [[None] * cols for i in range(rows)]
    for i in range(0, rows):
        for j in range(0, cols):
            if A[i][j] == 0:
                lnfij = 0.0
            else:
                p = A[i][j] / A.sum(axis=0)[j]
                lnfij = math.log(p) * p * (-k) 
            lnf[i][j] = lnfij
    
    E = [[None] * 1 for i in range(rows)]
    for i in range(rows):
        e = lnf[i]
#         e = np.exp(e)
        x = e / np.sum(e)
        E[i] = x
#     print(E)
    return alpha*np.asarray(E)

def wav2vec(wav_file, label):
    y, sr = librosa.load(wav_file, sr=16000)
    print('audio reading finished')
    wf = librosa.feature.melspectrogram(y, sr, n_fft=8000, hop_length=400, n_mels=384)
    if type(wf) is tuple:
        wf = wf[0]
    wf = sequence.pad_sequences(wf, maxlen=256, dtype='float32', padding='post', value=0)
#     wf = librosa.power_to_db(wf, ref=np.max)
    librosa.display.specshow(wf, x_axis='time', y_axis='mel',sr=16000, hop_length=400)
#     plt.set_cmap('hot')
#     plt.colorbar()
#     plt.title(label)
#     plt.savefig(label+'.jpg')
    return wf

def wav2vec_withEntnoise(wav_file, label, alpha):
    y, sr = librosa.load(wav_file, sr=16000)
    print('audio reading finished')
    wf = librosa.feature.melspectrogram(y, sr, n_fft=8000, hop_length=400, n_mels=384)
    if type(wf) is tuple:
        wf = wf[0]
    wf = sequence.pad_sequences(wf, maxlen=256, dtype='float32', padding='post', value=0)
#     wf = librosa.power_to_db(wf, ref=np.max)
    wf = wf*entropy_weight(wf, alpha)
    librosa.display.specshow(wf, x_axis='time', y_axis='mel',sr=16000, hop_length=400)
#     plt.set_cmap('hot')
#     plt.colorbar()
#     plt.title(label)
#     plt.savefig(label+'.jpg')
    return wf

def load_mocab(root_path, Session, avi_name, max_len = 500):
    avi_dict = '_'.join(avi_name.split('_')[:-1])
    file = root_path+Session+mocab_head_dict+avi_dict+'/'+avi_name+'.txt'
    print(file)
    pad_mocab = np.zeros((6))
    mocab_data = []
    try:
        f = open(file, 'r').readlines()
#         print(f)
        for d in f[2:]:
            d = d.split()
            mocab_data.append([float(digit) for digit in d[2:]])
        len_mocab = len(mocab_data)
#         print('** MOCAB: len', len_mocab)
        if len_mocab>max_len:
            mocab_data_tmp = mocab_data
            mocab_data = []
            
#             if (len_mocab/max_len) - int(len_mocab/max_len) > 0.5:
#                 step = int(len_mocab/max_len)+1
#             else:
            step = int(len_mocab/max_len)
                
            for i in range(0, len_mocab, step):
                mocab_data.append(mocab_data_tmp[i])
        elif len(mocab_data)<max_len:
            print('** MOCAB: pad mocab', (max_len-len(mocab_data)))
            while len(mocab_data)<max_len:
                mocab_data.append(pad_mocab)
    except Exception as e:
        print('========== MOCAB ==========')
        print(e)
        print('========== MOCAB ==========')
        mocab_data = np.zeros((max_len, 6))
        
#     print('** MOCAB: len', np.shape(mocab_data))
    while len(mocab_data)<max_len:
                mocab_data.append(pad_mocab)
    mocab_data_final = np.asarray(mocab_data[:max_len])
#     print('** MOCAB: final', np.shape(mocab_data_final))
    return np.asarray(mocab_data_final)

