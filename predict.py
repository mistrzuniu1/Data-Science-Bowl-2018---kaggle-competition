from skimage.measure import regionprops
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
from skimage.morphology import label
import numpy as np
import os
import data as d
import model as m
import cv2
import pandas as pd
import postprocessing

path_list_test=d.getImagesPath('test')
X_test,sizes_test=d.PreprocessData(path_list_test,False)
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[m.mean_iou])
preds_test = loaded_model.predict(X_test, verbose=1)

preds_test_t = (preds_test > 0.5).astype(np.uint8)

#d.plotResult(preds_test_t,X_test)

test_connected_components=[postprocessing.process(img)  for img in preds_test_t]
test_connected_components_split=[postprocessing.split_and_relabel(img)  for img in test_connected_components]

preds_test_upsampled =d.resizeTest(path_list_test,test_connected_components,sizes_test)

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