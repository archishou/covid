
data_set = "/Users/Archish/Documents/CodeProjects/Python/IPF/datafiles/all_data"

param_matrix = {
    'filter_0':[16, 32],
    'filter_1':[32],
    'filter_2':[64],
    'filter_3':[128],
    'kernel_size_0':[2],
    'kernel_size_1':[2],
    'kernel_size_2':[2],
    'kernel_size_3':[2],
    'activation_0':['relu'],
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
    'epochs':[72],
    'test_size':[0.2]
}

num_models = 1

for index in range(num_models):
    tunable_params = list(param_matrix.keys())
    for params in tunable_params:
        num_models = num_models * len(param_matrix[params])

    print(num_models)
