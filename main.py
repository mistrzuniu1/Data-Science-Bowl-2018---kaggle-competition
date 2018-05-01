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
import data as d
import model as m
from sklearn.model_selection import train_test_split

path_list_train=d.getImagesPath('train')
path_list_test=d.getImagesPath('test')
X_train,Y_train=d.PreprocessData(path_list_train)
X_test,sizes_test=d.PreprocessData(path_list_test,False)
d.savePreparedData(X_test,X_train,Y_train,sizes_test)


xtr, xval, ytr, yval = train_test_split(X_train, Y_train, test_size=0.1, random_state=7)
#X_gen,Y_gen=d.genImagesAndMasks(X_train,Y_train)
#X_train=np.concatenate(X_train,X_gen)
#Y_train=np.concatenate(Y_train,Y_gen)
train_generator, val_generator = d.generator(xtr, xval, ytr, yval, 16)
model = get_unet(256,256,3)
model.fit_generator(train_generator, steps_per_epoch=len(xtr)/6, epochs=250,
                        validation_data=val_generator, validation_steps=len(xval)/16)
preds_test = model.predict(X_test, verbose=1)


preds_test_t = (preds_test > 0.5).astype(np.uint8)
preds_test_upsampled =d.resizeTest(path_list_test,preds_test,sizes_test)

new_test_ids = []
rles = []
for n, path in enumerate(path_list_test):
    rle = list(d.prob_to_rles(preds_test_upsampled[n]))
    rles.extend(rle)
    new_test_ids.extend([os.path.splitext(os.path.basename(os.path.normpath(str(path))))[0]] * len(rle))


sub = pd.DataFrame()
sub['ImageId'] = new_test_ids
sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))
sub.to_csv('x.csv', index=False)

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
