import os
import h5py
import functions

pred_path = 'Predictions/ChronoNet_test/validation_set' # path to the prediction file group

pred_files_list = [os.path.join(pred_path, x) for x in os.listdir(pred_path) if '.h5' in x]

#######################################################################################################################
### Get metrics and score ###
#######################################################################################################################

score, sens_ovlp, FA_epoch = functions.get_metrics_scoring(pred_files_list, th=0.5)
