import numpy as np
from tensorflow import keras
import functions
from tqdm import tqdm


class generator(keras.utils.Sequence):
    ''' Class where the keras data generator is built.

    Args:
        files_list: list of raw instances (of the mne package) containing EEG recordings
        segments: list of keys (each key is a list [1x4] containing the recording index in the files_list,
                  the start and stop of the segment in seconds and the label of the segment)
        montages: a list of montages (list of lists with pairs of strings where each pair is an electrode name
                  used for each channel in the montage) corresponding to each recording in the files_list
        normalize: boolean, if True the channel data is normalized with the z-score method
        batch_size: batch size of the generator
        shuffle: boolean, if True, the segments are randomly mixed in every batch
    
    '''

    def __init__(self, raw, segments, normalize=True, batch_size=32, shuffle=False):
        
        'Initialization'
        self.segments = segments
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.data_segs = np.empty(shape=[len(self.segments[0]), 400, 2])
        self.labels = np.empty(shape=[len(self.segments[0]), 2])
        
        
        ch_focal, ch_cross, fs_out = functions.pre_process_raw(raw, True)
        del raw

        for s in range(len(self.segments[0])):
            start_seg = int(self.segments[0][s]*fs_out)
            end_seg = int(self.segments[1][s]*fs_out)

            self.data_segs[s, :, 0] = ch_focal[start_seg:end_seg]
            self.data_segs[s, :, 1] = ch_cross[start_seg:end_seg]

            if self.segments[2][s] == 1:
                self.labels[s, :] = [0, 1]
            elif self.segments[2][s] == 0:
                self.labels[s, :] = [1, 0]
        
        self.key_array = np.arange(len(self.labels))

        self.on_epoch_end()


    def __len__(self):
        return len(self.key_array) // self.batch_size

    def __getitem__(self, index):
        keys = np.arange(start=index * self.batch_size, stop=(index + 1) * self.batch_size)
        x, y = self.__data_generation__(keys)
        return x, y

    def on_epoch_end(self):
        if self.shuffle:
            self.key_array = np.random.permutation(self.key_array)

    def __data_generation__(self, keys):
        return self.data_segs[self.key_array[keys], :, :], self.labels[self.key_array[keys]]
