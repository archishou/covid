def load_audio(file_name):
    try:
        audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
        return audio, sample_rate
    except Exception as e:
        print("Error encountered while parsing file: ", file_name)

def extract_features(audio, sample_rate):
    max_pad_len = 174
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    pad_width = max_pad_len - mfccs.shape[1]
    mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')

    return mfccs

def class_name(file):
    if file.startswith("ipf"):
        return "ipf"
    if file.startswith("healthy"):
        return "healthy"
    if file.startswith("copd"):
        return "copd"

def append_features(features, label, *augmented_data):
    for d in augmented_data:
        features.append([d, label])
    return features
