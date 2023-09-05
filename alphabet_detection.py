import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from PIL import Image
import PIL.ImageOps
import os, ssl, time

X = np.load('image.npz')['arr_0']
y = pd.read_csv("labels.csv")["labels"]
print(pd.Series(y).value_counts())
classes = ['A', 'B', 'C', 'D', 'E','F', 'G', 'H', 'I', 'J', "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
nclasses = len(classes)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=9, train_size=3500, test_size=500)
X_train_scaled = X_train/255.0
X_test_scaled = X_test/255.0

clf = LogisticRegression(solver='saga', multi_class='multinomial').fit(X_train_scaled, y_train)

y_pred = clf.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print("The accuracy is :- ",accuracy)

def get_prediction(image):
    img=Image.open(image)
    img_bw = img.convert('L')
    img_bw_resized = img_bw.resize((28,28), Image.ANTIALIAS)

    img_bw_resized_inverted = PIL.ImageOps.invert(img_bw_resized)
    pixel_filter = 20
    min_pixel = np.percentile(img_bw_resized_inverted, pixel_filter)
    img_bw_resized_inverted_scaled = np.clip(img_bw_resized_inverted-min_pixel, 0, 255)
    max_pixel = np.max(img_bw_resized_inverted)
    img_bw_resized_inverted_scaled = np.asarray(img_bw_resized_inverted_scaled)/max_pixel
    test_sample = np.array(img_bw_resized_inverted_scaled).reshape(1,784)
    test_pred = clf.predict(test_sample)
    print( test_pred)

