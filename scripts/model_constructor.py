from sklearn.model_selection import train_test_split
import csv
from keras.callbacks import ModelCheckpoint
import utils
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
data_set = '/Volumes/ArchishmaanHD1/data/raw_data/Respiratory_Sound_Database/audio_and_txt_files/'
num_rows = 40
num_columns = utils.max_pad_len
num_channels = 1

def build_model(x, yy, params, model_path):
    # next we can build the model exactly like we would normally do it
    x_train, x_val, y_train, y_val = train_test_split(x, yy, test_size=params['test_size'], random_state=42)

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
    saved_models = model_path + 'saved_models/'
    utils.create_dir(path=saved_models)
    filepath = saved_models + "weights-{epoch:02d}-{val_accuracy:.2f}.hdf5"
    print(filepath)
    checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, mode='max')

    history = model.fit(x_train, y_train,
                        validation_data=[x_val, y_val],
                        batch_size=params['batch_size'],
                        epochs=params['epochs'],
                        verbose=0, callbacks=[checkpoint])

    hist_df = pd.DataFrame(history.history)

    with open(model_path + 'history.json', mode='w+') as f:
        hist_df.to_json(f)

    save_figures(model_path=model_path, hist=history.history)

def create(params, model_num, x, yy):
    # define the name of the directory to be created
    model_id = "model_" + str(model_num) + '/'
    model_path = "/Users/Archish/Documents/CodeProjects/Python/ipf-new/models/" + model_id
    utils.create_dir(path=model_path)

    with open(model_path + 'params.csv', 'w') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in params.items():
            writer.writerow([key, value])

    build_model(params=params, x=x, yy=yy, model_path=model_path)

def save_figures(model_path, hist):
    figure_path = model_path + 'figures/'
    utils.create_dir(path=figure_path)
    plt.clf()
    plt.plot(hist['accuracy'])
    plt.plot(hist['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(figure_path + 'accuracy.png')
    # "Loss"
    plt.clf()
    plt.plot(hist['loss'])
    plt.plot(hist['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(figure_path + 'loss.png')