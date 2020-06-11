import itertools
import model_constructor
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
import pandas as pd
import numpy as np
import utils
import os
data_set = '/home/archi/env/project/raw_data/audio_and_txt_files/'

param_matrix = {
    'filter_0':[32,
    'filter_1':[64],
    'filter_2':[18],
    'filter_3':[256],
    'kernel_size_0':[2],
    'kernel_size_1':[2],
    'kernel_size_2':[2],
    'kernel_size_3':[2],
    'activation_0':['sigmoid'],
    'activation_1':['relu'],
    'activation_2':['relu'],
    'activation_3':['relu'],
    'activation_4':['softmax'],
    'pool_size_0':[2],
    'pool_size_1':[2],
    'pool_size_2':[2],
    'pool_size_3':[2],
    'dropout_0':[0.2],
    'dropout_1':[0.2],
    'dropout_2':[0.2],
    'dropout_3':[0.2],
    'optimizer':['adam'],
    'batch_size':[256],
    'epochs':[100],
    'test_size':[0.2]
}

def load_data():
    features = []
    for file in os.listdir(data_set):
        if file.endswith(".wav") and ("Al" in file or 'Ar' in file or 'Pr' in file or 'Pl' in file) and 'COPD' in file and 'AKGC417L' in file:
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
    return x, yy


if __name__ == '__main__':
    x, yy = load_data()
    tunable_param_keys = list(param_matrix.keys())
    vals = list(param_matrix.values())
    parm_combos = list(itertools.product(*vals))
    num_models = len(parm_combos)
    print(num_models)
    for index in range(num_models):
        dict = {}
        build_vals = list(parm_combos[index])
        for parm_num, param_name in enumerate(tunable_param_keys, start=0):
            dict[param_name] = build_vals[parm_num]
        model_constructor.create(params=dict, model_num=index, x=x, yy=yy)
        print(dict)
