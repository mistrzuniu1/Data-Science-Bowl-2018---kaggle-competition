import numpy as np
import pathlib
import imageio
from skimage.transform import resize
from skimage.io import imread,imshow
import os
import matplotlib.pyplot as plt
from model import get_unet
from skimage.morphology import label
import pandas as pd

def savePreparedData(X_test,X_train,Y_train,sizes_test):
    np.save('X_test', X_test)
    np.save('X_train', X_train)
    np.save('Y_train', Y_train)
    np.save('sizes_test', sizes_test)

def loadPreparedData():
    X_test=np.load('X_test.npy')
    X_train=np.load('X_train.npy')
    Y_train=np.load('Y_train.npy')
    sizes_test=np.load('sizes_test.npy')
    return X_test,X_train,Y_train,sizes_test


def rle_encoding(x):
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths

def prob_to_rles(x, cutoff=0.5):
    lab_img = label(x > cutoff)
    for i in range(1, lab_img.max() + 1):
        yield rle_encoding(lab_img == i)

def getImagesPath(name):
    paths = pathlib.Path('venv/'+name).glob('*/images/*.png')
    paths_list = sorted([x for x in paths])
    return paths_list

def getMaskPath(im_path):
    paths = pathlib.Path(str(im_path)+'/masks').glob('*.png')
    paths_list = sorted([x for x in paths])
    return paths_list

def resizeTest(path_list_test,preds_test,sizes_test):
    preds_test_upsampled = []
    for i in range(len(path_list_test)):
        preds_test_upsampled.append(resize(np.squeeze(preds_test[i]),
                                           (sizes_test[i][0], sizes_test[i][1]),
                                           mode='constant', preserve_range=True))
    return preds_test_upsampled
def saveOutputTocsv(new_test_ids,rles):
    sub = pd.DataFrame()
    sub['ImageId'] = new_test_ids
    sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))
    sub.to_csv('out.csv', index=False)

def getRLE(path_list_test,preds_test_upsampled):
    new_test_ids = []
    rles = []
    for n, id_ in enumerate(path_list_test):
        rle = list(prob_to_rles(preds_test_upsampled[n]))
        rles.extend(rle)
        new_test_ids.extend([os.path.basename(os.path.normpath(str(id_)))] * len(rle))
    return new_test_ids,rles

def plotResult(preds_test_t,X_test):
    for i in range(len(preds_test_t)):
        plt.subplot(1,2,1)
        imshow(np.squeeze(preds_test_t[i]))
        plt.subplot(1,2,2)
        imshow(X_test[i])
        plt.show()

def PreprocessData(path_list,label=True):
    X = np.zeros((len(path_list), 128, 128, 3),dtype=np.uint8)
    Y = np.zeros((len(path_list), 128, 128, 1),dtype=np.bool)
    sizes_test = []
    for i,path in enumerate(path_list):
        img=imread(str(path))[:,:,:3]
        if(label==False):
            sizes_test.append([img.shape[0], img.shape[1]])
        img=resize(img,(128,128),mode='constant',preserve_range=True)
        X[i]=img
        if(label):
            masks_path=os.path.dirname(os.path.dirname(str(path)))
            masks_list=getMaskPath(masks_path)
            mask = np.zeros((128, 128, 1), dtype=np.bool)
            for mask_path in masks_list:
                mask_img=imread(mask_path)
                mask_img=resize(mask_img, (128,128),mode='constant',preserve_range=True)
                mask_img=np.expand_dims(mask_img,axis=-1)
                mask=np.maximum(mask,mask_img)
            Y[i]=mask
    if(label):
        return X, Y
    else:
        return X,sizes_test
