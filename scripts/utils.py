import librosa
import os
import numpy as np
max_pad_len = 4000
def load_audio(file_name):
    try:
        audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
        return audio, sample_rate
    except Exception as e:
        print("Error encountered while parsing file: ", file_name)

def extract_features(audio, sample_rate):
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    print(mfccs.shape[1])
    pad_width = max_pad_len - mfccs.shape[1]
    mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')

    return mfccs

def class_name(file):
    file_str = str(file)
    if file_str.find("COPD") > 0:
        return "COPD"
    else:
        return "NONE"

def append_features(features, label, *augmented_data):
    for d in augmented_data:
        features.append([d, label])
    return features


def create_dir(path):
    try:
        os.mkdir(path)
    except OSError:
        print("Creation of the directory %s failed" % path)
    else:
        print("Successfully created the directory %s " % path)
