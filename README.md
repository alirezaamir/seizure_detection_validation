# BIOMED Seizure Detection Challenge - Validation

This repository contains the pipeline to evaluate the models submitted in task 2 of the Seizure Detection Challenge, in the context of ICASSP 2023. For further information, please visit our [website](https://biomedepi.github.io/seizure_detection_challenge/).

Please visit our other [repository](https://github.com/biomedepi/seizure_detection_code) containing a more complete version for the development of Task 2 of the challenge, as well as requirements and instructions.

## How to run

1) run the 'main_predict.py' script to obtain the prediction files of the model on the validation dataset.
2) run the 'main_evaluate.py' script to obtain the score and metrics of the model.

#### Notes:
- The way the code is written, the model weights are saved in the 'Model_weights' folder, as an h5py file (used tensorflow's API for saving weights to disk & loading them back)
- There is a slight difference in the routine for loading the validation and test dataset's annotations. The routine for this is included in 'functions.wrangle_tsv_sz2'.

The evaluation metrics are explained in the challenge website. The implementation of the metrics is based on a prediction vector containing consecutive probabilities of a 2-second window to be a seizure. The functions functions.perf_measure_epoch and routines.perf_measure_ovlp calculate the metrics according to the EPOCH and any-overlap methods (for more information, check https://biomedepi.github.io/seizure_detection_challenge/regulations/). The final scoring is calculated with the function routines.get_metrics_scoring.
