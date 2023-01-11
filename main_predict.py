import os
import functions
import numpy as np
import h5py
from config import Settings
from generator_test import generator

data_path = 'C:\\Users\\biomed\\Desktop\\MyProjects\\seizure_detection_code\\val_data'  # path to data
config_path = 'Configs' # path to configuration files
model_weights_path = 'Model_weights'

model_name = 'ChronoNet_test'   # name of the model


# Configuration for the data generator, model and training routine:
config = Settings(name=model_name, dataset='SZ2')
config.frame = 2  # segment window size in seconds

#######################################################################################################################
### Get model's predictions on test set ###
#######################################################################################################################

if not os.path.exists('Predictions'):
    os.mkdir('Predictions')
if not os.path.exists(os.path.join('Predictions', config.name)):
    os.mkdir(os.path.join('Predictions', config.name))
if not os.path.exists(os.path.join('Predictions', config.name, 'validation_set')):
    os.mkdir(os.path.join('Predictions', config.name, 'validation_set'))


patients = [x for x in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, x))]

for pat in patients:

    recordings = [x for x in os.listdir(os.path.join(data_path, pat)) if '.edf' in x]

    for rec in recordings:
        print(rec)
        rec_path = os.path.join(data_path, pat, rec)

        raw, segments = functions.get_data(rec_path, config)

        gen_data = generator(raw, segments, batch_size=len(segments[0]))

        y_pred, y_true = functions.predict_net(gen_data, model_weights_path, config)

        with h5py.File(os.path.join('Predictions', config.name, 'validation_set', rec[0:-4] + '.h5'), 'w') as f:
            f.create_dataset('y_pred', data=y_pred)
            f.create_dataset('y_true', data=y_true)
            


