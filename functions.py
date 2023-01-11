import os
import numpy as np
import mne.io
import pandas as pd
from scipy import signal
import ChronoNet
from tensorflow.keras import backend as K
import h5py


def load_recording(rec_path):
    """ Function to load a recording as an mne.io.raw instance.
    
    Args:
        rec_path: Path to the recording file
    
    Return:
        raw: mne.io.raw instance of the recording (loaded into memory)
    """

    sd_uni = ['bteleft sd', 'bteright sd']  # unilateral channel names
    sd_cross = ['crosstop sd']  # cross head channel name

    raw = mne.io.read_raw_edf(rec_path, preload=False, verbose=False)

    sd_uni_index = [i for i in range(len(raw.ch_names)) if raw.ch_names[i].lower() in sd_uni]

    sd_cross_index = [i for i in range(len(raw.ch_names)) if raw.ch_names[i].lower() in sd_cross]
    
    raw.pick_channels([raw.ch_names[sd_uni_index[0]] , raw.ch_names[sd_cross_index[0]]])

    raw.load_data(verbose=False)

    return raw


def get_data(rec_path, config):
    """ Function to load the recording as an mne.io.raw instance and build the segments for the data generator.
    
    Args:
        rec_path: Path to the recording file
        config: configuration object containing all parameters
    
    Return:
        raw: mne.io.raw instance of the recording (loaded into memory)
        segments: list of lists containing a list with the start time in seconds of the segments, a list with
                  the end time in seconds of the segments and a list with the labels (0 or 1-seizure) of each
                  segment
    """

    raw = load_recording(rec_path)

    ann_path = rec_path[0:-4] + '_a1.tsv'

    seiz_events = wrangle_tsv_sz2(ann_path)

    start_seg = np.arange(0, np.floor(raw.tmax)-config.frame+1, config.frame).tolist()
    end_seg = np.arange(config.frame, np.floor(raw.tmax)+1, config.frame).tolist()

    labels = eventList2Mask(seiz_events, len(start_seg), 1/config.frame)

    segments = [start_seg, end_seg, labels.tolist()]

    return raw, segments


def wrangle_tsv_sz2(tsv_path):
    """ Function to process a given .tsv annotation file (from the SeizeIT2 dataset) into a list of events.
    
    Args:
        tsv_path: Path to .tsv file
    
    Return:
        events_times: list of lists with length 2, 1st item is the start of a seizure, 2nd is the stop, both
                    in seconds

        (check the dataset description for more information)
    """

    df = pd.read_csv(tsv_path, sep='\t', header=None, names=[0, 1, 2, 3])
    nb_events = df.shape[0] - 5
    events_times = []
    for i in range(nb_events):
        if df[0][5 + i] != 'None' and df[1][5 + i] != 'None':
            if df[2][5 + i] == 'seizure':
                start_sec = int(df[0][5 + i])
                stop_sec = int(df[1][5 + i])
                events_times.append([start_sec, stop_sec])

    return events_times


def pre_process_ch(ch_data, fs):
    ''' Pre-process EEG data by applying a 0.5 Hz highpass filter, a 60  Hz lowpass filter and a 50 Hz notch filter,
    all 4th order Butterworth filters. The data is resampled to 200 Hz.

    Args:
        ch_data: a list or numpy array containing the data of an EEG channel
        fs: the sampling frequency of the data
    
    Returns:
        ch_data: a numpy array containing the processed EEG data
        fs_resamp: the sampling frequency of the processed EEG data
    '''

    fs_resamp = 200
    
    b, a = signal.butter(4, 0.5/(fs/2), 'high')
    ch_data = signal.filtfilt(b, a, ch_data)

    b, a = signal.butter(4, 60/(fs/2), 'low')
    ch_data = signal.filtfilt(b, a, ch_data)

    b, a = signal.butter(4, [49.5/(fs/2), 50.5/(fs/2)], 'bandstop')
    ch_data = signal.filtfilt(b, a, ch_data)

    ch_data = signal.resample(ch_data, int(fs_resamp*len(ch_data)/fs))

    return ch_data, fs_resamp


def pre_process_raw(raw, normalize):
    ''' Load data from a mne raw instance and apply the bi-channel montage (ipsilateral channel - depending on the dominant
    hemisphere where seizures occur in a patient - and cross-lateral) used on the data to train the benchmark model. The
    channel data is pre-processed after the montage is applied.
    
    Args:
        raw: raw object of the mne package containing an EEG recording (loaded offline)
        normalize: bool, indicating whether the channel data should be normalized or not (with the z-score method)

    Returns:
        ch_focal: a list of arrays containing the data of the focal/ipsilateral channel
        ch_cross: a list of arrays containing the data of the cross-lateral channel
        fs_resamp: the sampling frequency of the data
    '''

    ch_focal = raw.get_data(0)[0]
    ch_cross = raw.get_data(1)[0]    

    ch_focal, fs_resamp = pre_process_ch(ch_focal, raw.info['sfreq'])
    ch_cross, _ = pre_process_ch(ch_cross, raw.info['sfreq'])
    
    if normalize:
        ch_focal = (ch_focal - np.mean(ch_focal))/np.std(ch_focal)
        ch_cross = (ch_cross - np.mean(ch_cross))/np.std(ch_cross)

    return ch_focal, ch_cross, fs_resamp


def predict_net(generator, model_weights_path, config):
    ''' Routine to obtain predictions from the trained model with the desired configurations.

    Args:
        generator: a keras data generator containing the data to predict
        model_weights_path: path to the folder containing the models' weights
        config: configuration object containing all parameters

    Returns:
        y_pred: array with the probability of seizure occurences (0 to 1) of each consecutive
                window of the recording.
        y_true: analogous to y_pred, the array contains the label of each segment (0 or 1)
    '''

    K.set_image_data_format('channels_last')

    model = ChronoNet.net()
    model.load_weights(os.path.join(model_weights_path, config.name + '.h5'))

    y_aux = []
    for j in range(len(generator)):
        _, y = generator[j]
        y_aux.append(y)
    true_labels = np.vstack(y_aux)

    prediction = model.predict(generator)

    y_pred = np.empty(len(prediction), dtype='float32')
    for j in range(len(y_pred)):
        y_pred[j] = prediction[j][1]

    y_true = np.empty(len(true_labels), dtype='uint8')
    for j in range(len(y_true)):
        y_true[j] = true_labels[j][1]

    return y_pred, y_true


# ## EVENT & MASK MANIPULATION ###

def eventList2Mask(events, totalLen, fs):
    """Convert list of events to mask.
    
    Returns a logical array of length totalLen.
    All event epochs are set to True
    
    Args:
        events: list of events times in seconds. Each row contains two
                columns: [start time, end time]
        totalLen: length of array to return in samples
        fs: sampling frequency of the data in Hertz
    Return:
        mask: logical array set to True during event epochs and False the rest
              if the time.
    """
    mask = np.zeros((totalLen,))
    for event in events:
        for i in range(min(int(event[0]*fs), totalLen), min(int(event[1]*fs), totalLen)):
            mask[i] = 1
    return mask


def mask2eventList(mask, fs):
    """Convert mask to list of events.
        
    Args:
        mask: logical array set to True during event epochs and False the rest
          if the time.
        fs: sampling frequency of the data in Hertz
    Return:
        events: list of events times in seconds. Each row contains two
                columns: [start time, end time]
    """
    events = list()
    tmp = []
    start_i = np.where(np.diff(np.array(mask, dtype=int)) == 1)[0]
    end_i = np.where(np.diff(np.array(mask, dtype=int)) == -1)[0]
    
    if len(start_i) == 0 and mask[0]:
        events.append([0, (len(mask)-1)/fs])
    else:
        # Edge effect
        if mask[0]:
            events.append([0, (end_i[0]+1)/fs])
            end_i = np.delete(end_i, 0)
        # Edge effect
        if mask[-1]:
            if len(start_i):
                tmp = [[(start_i[-1]+1)/fs, (len(mask))/fs]]
                start_i = np.delete(start_i, len(start_i)-1)
        for i in range(len(start_i)):
            events.append([(start_i[i]+1)/fs, (end_i[i]+1)/fs])
        events += tmp
    return events


def merge_events(events_list, distance):
    """ Merge events.
    
    Args:
        events: list of events times in seconds. Each row contains two
                columns: [start time, end time]
        distance: maximum distance (in seconds) between events to be merged
    Return:
        events: list of events (after merging) times in seconds.
    """
    i = 1
    tot_len = len(events_list)
    while i < tot_len:
        if events_list[i][0] - events_list[i-1][1] <= distance:
            events_list[i-1][1] = events_list[i][1]
            events_list.pop(i)
            tot_len -= 1
        else:
            i += 1
    return


def get_events(events, margin):
    ''' Converts the unprocessed events to the post-processed events based on physiological constrains:
    - seizure alarm events distanced by 0.2*margin (in seconds) are merged together
    - only events with a duration longer than margin*0.8 are kept
    (for more info, check: K. Vandecasteele et al., “Visual seizure annotation and automated seizure detection using
    behind-the-ear elec- troencephalographic channels,” Epilepsia, vol. 61, no. 4, pp. 766–775, 2020.)

    Args:
        events: list of events times in seconds. Each row contains two
                columns: [start time, end time]
        margin: float, the desired margin in seconds

    Returns:
        ev_list: list of events times in seconds after merging and discarding short events.
    '''
    merge_events(events, margin*0.2)
    ev_list = []
    for i in range(len(events)):
        if events[i][1] - events[i][0] >= margin:
            ev_list.append(events[i])

    return ev_list



def post_processing(y_pred, fs, th, margin):
    ''' Post process the predictions given by the model based on physiological constraints: a seizure is
    not shorter than 10 seconds and events separated by 2 seconds are merged together.

    Args:
        y_pred: array with the seizure classification probabilties (of each segment)
        fs: sampling frequency of the y_pred array (1/window length - in this challenge fs = 1/2)
        th: threshold value for seizure probability (float between 0 and 1)
        margin: float, the desired margin in seconds (check get_events)
    
    Returns:
        pred: array with the processed classified labels by the model
    '''
    pred = (y_pred > th)
    events = mask2eventList(pred, fs)
    events = get_events(events, margin)
    pred = eventList2Mask(events, len(y_pred), fs)

    return pred


def getOverlap(a, b):
    ''' If > 0, the two intervals overlap.
    a = [start_a, end_a]; b = [start_b, end_b]
    '''
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def perf_measure_epoch(y_true, y_pred):
    ''' Calculate the performance metrics based on the EPOCH method.
    
    Args:
        y_true: array with the ground-truth labels of the segments
        y_pred: array with the predicted labels of the segments

    Returns:
        TP: true positives
        FP: false positives
        TN: true negatives
        FN: false negatives
    '''

    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_pred)): 
        if y_true[i] == y_pred[i] == 1:
           TP += 1
        if y_pred[i] == 1 and y_true[i] != y_pred[i]:
           FP += 1
        if y_true[i] == y_pred[i] == 0:
           TN += 1
        if y_pred[i] == 0 and y_true[i] != y_pred[i]:
           FN += 1

    return TP, FP, TN, FN


def perf_measure_ovlp(y_true, y_pred, fs):
    ''' Calculate the performance metrics based on the any-overlap method.
    
    Args:
        y_true: array with the ground-truth labels of the segments
        y_pred: array with the predicted labels of the segments
        fs: sampling frequency of the predicted and ground-truth label arrays
            (in this challenge, fs = 1/2)

    Returns:
        TP: true positives
        FP: false positives
        FN: false negatives
    '''
    true_events = mask2eventList(y_true, fs)
    pred_events = mask2eventList(y_pred, fs)

    TP = 0
    FP = 0
    FN = 0

    for pr in pred_events:
        found = False
        for tr in true_events:
            if getOverlap(pr, tr) > 0:
                TP += 1
                found = True
        if not found:
            FP += 1
    for tr in true_events:
        found = False
        for pr in pred_events:
            if getOverlap(tr, pr) > 0:
                found = True
        if not found:
            FN += 1

    return TP, FP, FN


def get_metrics_scoring(pred_files_list, th):
    ''' Get the score for the challenge.

    Args:
        pred_files_list: list with prediction files path containing the objects 'y_pred' and 'y_true',
                         related to the prediction and true labels vectors of a recording returned by
                         the 'predict_net' function
    
    Returns:
        score: the score of the challenge
        sens_ovlp: sensitivity calculated with the any-overlap method
        FA_epoch: false alarm rate (false alarms per hour) calculated with the EPOCH method
    '''

    total_N = 0
    total_TP_epoch = 0
    total_FP_epoch = 0
    total_FN_epoch = 0
    total_TP_ovlp = 0
    total_FP_ovlp = 0
    total_FN_ovlp = 0
    total_seiz = 0

    for pred_file in pred_files_list:
        with h5py.File(pred_file, 'r') as f:
            y_pred = np.array(f['y_pred'])
            y_true = np.array(f['y_true'])

        total_N += len(y_pred)*2
        total_seiz += np.sum(y_true)

        # Post process predictions (merge predicted events separated by 2 seconds and discard events smaller than 10 seconds)
        y_pred = post_processing(y_pred, fs=1/2, th=th, margin=10)

        TP_epoch, FP_epoch, TN_epoch, FN_epoch = perf_measure_epoch(y_true, y_pred)
        total_TP_epoch += TP_epoch
        total_FP_epoch += FP_epoch
        total_FN_epoch += FN_epoch

        TP_ovlp, FP_ovlp, FN_ovlp = perf_measure_ovlp(y_true, y_pred, fs=1/2)
        total_TP_ovlp += TP_ovlp
        total_FP_ovlp += FP_ovlp
        total_FN_ovlp += FN_ovlp


    if total_seiz == 0:
        sens_ovlp = float("nan")
    else:
        sens_ovlp = total_TP_ovlp/(total_TP_ovlp + total_FN_ovlp)
        
    FA_epoch = total_FP_epoch*3600/total_N

    score = sens_ovlp*100 - 0.4*FA_epoch

    print('Sensitivity (ovlp): ' + str(sens_ovlp*100) + ' %')
    print('False alarm per hour (epoch): ' + str(FA_epoch))
    print('Final score: ' + str(score))

    return score, sens_ovlp, FA_epoch