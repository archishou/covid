from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import pandas as pd
import numpy as np
from scripts import utils
import os
import librosa
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
data_set = "/Users/Archish/Documents/CodeProjects/Python/ipf-new/raw_data"

num_rows = 40
num_columns = 174
num_channels = 1

def build_model(params):
    # next we can build the model exactly like we would normally do it
    create_model_folder(params=params)
    x_train, x_val, y_train, y_val = pre_process(params['test_size'])

    x_train = x_train.reshape(x_train.shape[0], num_rows, num_columns, num_channels)
    x_val = x_val.reshape(x_val.shape[0], num_rows, num_columns, num_channels)

    model = Sequential()
    model.add(Conv2D(filters=params['filter_0'], kernel_size=params['kernel_size_0'], input_shape=(num_rows, num_columns, num_channels), activation=params['activation_0']))
    model.add(MaxPooling2D(pool_size=params['pool_size_0']))
    model.add(Dropout(params['dropout_0']))

    model.add(Conv2D(filters=params['filter_1'], kernel_size=params['kernel_size_1'], activation=params['activation_1']))
    model.add(MaxPooling2D(pool_size=params['pool_size_1']))
    model.add(Dropout(params['dropout_1']))

    model.add(Conv2D(filters=params['filter_2'], kernel_size=params['kernel_size_2'], activation=params['activation_2']))
    model.add(MaxPooling2D(pool_size=params['pool_size_2']))
    model.add(Dropout(params['dropout_2']))

    model.add(Conv2D(filters=params['filter_3'], kernel_size=params['kernel_size_3'], activation=params['activation_3']))
    model.add(MaxPooling2D(pool_size=params['pool_size_3']))
    model.add(Dropout(params['dropout_3']))

    model.add(GlobalAveragePooling2D())

    model.add(Dense(3, activation=params['activation_4']))

    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=params['optimizer'])

    model.summary()

    history = model.fit(x_train, y_train,
                        validation_data=[x_val, y_val],
                        batch_size=params['batch_size'],
                        epochs=params['epochs'],
                        verbose=0)

    # finally we have to make sure that history object and model are returned
    return history, model

def pre_process(test_size):
    features = []
    for file in os.listdir(data_set):
        if file.endswith(".wav"):
            class_label = utils.class_name(file)
            data_file = os.path.join(data_set, file)
            audio, sample_rate = utils.load_audio(data_file)
            raw_data = utils.extract_features(audio, sample_rate)
            features = utils.append_features(features, class_label, raw_data)

    featuresdf = pd.DataFrame(features, columns=['feature', 'class_label'])
    x = np.array(featuresdf.feature.tolist())
    y = np.array(featuresdf.class_label.tolist())
    le = LabelEncoder()
    yy = to_categorical(le.fit_transform(y))
    x_train, x_test, y_train, y_test = train_test_split(x, yy, test_size=test_size, random_state=42)
    return x_train, x_test, y_train, y_test

def create_model_folder(params):
    # define the name of the directory to be created
    path = "/Users/Archish/Documents/CodeProjects/Python/ipf-new/models"

    try:
        os.mkdir(path)
    except OSError:
        print("Creation of the directory %s failed" % path)
    else:
        print("Successfully created the directory %s " % path)