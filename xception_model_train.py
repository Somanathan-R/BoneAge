# %%
import numpy as np 
import pandas as pd 
import matplotlib.pyplot  as plt
from matplotlib.pyplot import imshow,imread
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import load_img, img_to_array
import tensorflow as tf
import os
from sklearn.preprocessing import scale,minmax_scale
from keras.utils.vis_utils import plot_model

# %%
train_df = pd.read_csv('dataset/RSNA_Annotations/RSNA_Annotations/BONEAGE/boneage_train.csv')
train_df ['ID'] = train_df['ID'].map(lambda x: f'{x}.png')
train_df['Male']= train_df['Male'].map(lambda x: 1 if (x == True) else 0)
mean = train_df['Boneage'].mean()
stdd = train_df['Boneage'].std()
train_df['Zscore']= train_df['Boneage'].map(lambda x:(x-mean)/stdd)

# %%
test_df = pd.read_csv('dataset/RSNA_Annotations/RSNA_Annotations/BONEAGE/gender_test.csv')
test_df['ID'] = test_df['ID'].map(lambda x: f'{x}.png')

# %%
val_df = pd.read_csv('dataset/RSNA_Annotations/RSNA_Annotations/BONEAGE/boneage_val.csv')
val_df ['ID'] = val_df['ID'].map(lambda x: f'{x}.png')
val_df['Male']= val_df['Male'].map(lambda x: 1 if (x == True) else 0)
mean = val_df['Boneage'].mean()
stdd = val_df['Boneage'].std()
val_df['Zscore']= val_df['Boneage'].map(lambda x:(x-mean)/stdd)

# %%
train_dir='dataset/RSNA_train/images'
test_dir= 'dataset/RSNA_test/images'
val_dir='dataset/RSNA_val/images/' 
# %%
core_dg = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=1,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    zoom_range=0.1,
    fill_mode="nearest",
    cval=0.0,
    horizontal_flip=True,
    vertical_flip=True,
    validation_split=0.2,
    dtype=None,
    preprocessing_function = tf.keras.applications.xception.preprocess_input
)

# %%
train_dg = core_dg.flow_from_dataframe(
    train_df,
    directory=train_dir,
    x_col="ID",
    y_col="Boneage",
    weight_col=None,
    target_size=(128, 128),
    color_mode="rgb",
    classes=None,
    class_mode="raw",
    batch_size=32,
    shuffle=True,
    seed=None,
    save_to_dir=None,
    save_prefix="",
    save_format="png",
    subset=None,
    interpolation="nearest",
    validate_filenames=True,
)

# %%
val_dg = core_dg.flow_from_dataframe(
    val_df,
    directory=val_dir,
    x_col="ID",
    y_col="Boneage",
    weight_col=None,
    target_size=(128, 128),
    color_mode="rgb",
    classes=None,
    class_mode="raw",
    batch_size=32,
    shuffle=True,
    seed=None,
    save_to_dir=None,
    save_prefix="",
    save_format="png",
    subset=None,
    interpolation="nearest",
    validate_filenames=True,
)

# %%
test_dg_core = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function = tf.keras.applications.xception.preprocess_input)
test_dg = test_dg_core.flow_from_dataframe(test_df,
                                          x_col='ID',
                                          directory = test_dir,
                                          class_mode = None)

# %%
from keras import Sequential
from keras.layers import Dense,MaxPooling2D,Input,Conv2D,Dropout,BatchNormalization,GlobalMaxPooling2D,Flatten
from keras.applications.xception import Xception
from keras.metrics import MeanAbsoluteError
from keras.models import Model

# %%
input_shape=(128,128,3)
dropout_rate = 0.35

model_base = Xception(include_top = False,input_shape=input_shape)
model_base.trainable = True
model = Sequential()
model.add(model_base)
model.add(GlobalMaxPooling2D())
model.add(Dense(10,activation='relu'))
model.add(Dense(1,activation = 'linear'))
model.compile(loss='mse', optimizer= 'adam', metrics=[MeanAbsoluteError()])
model.summary()

# %%
model.fit_generator(train_dg, epochs = 10)

# %%
model.save(filepath='checkpoint_xce.h5')