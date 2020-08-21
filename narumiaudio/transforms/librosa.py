import math

import librosa
import numpy as np
import torch


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class ToTensor(object):
    def __call__(self, x):
        return torch.from_numpy(x)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class Transpose(object):
    def __call__(self, x):
        return x.T

    def __repr__(self):
        return self.__class__.__name__ + '()'


class Spectrogram(object):
    def __init__(self,
                 n_fft=1024,
                 hop_length=256,
                 win_length=1024,
                 window='hann',
                 center=True,
                 pad_mode='reflect',
                 power=1.0):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length or n_fft
        self.window = window
        self.center = center
        self.pad_mode = pad_mode
        self.power = power

    def __call__(self, y):
        s = librosa.stft(y,
                         n_fft=self.n_fft,
                         hop_length=self.hop_length,
                         win_length=self.win_length,
                         window=self.window,
                         center=self.center,
                         pad_mode=self.pad_mode)
        s = np.abs(s)**self.power
        return s

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'n_fft={}'.format(self.n_fft)
        format_string += ', hop_length={}'.format(self.hop_length)
        format_string += ', win_length={}'.format(self.win_length)
        format_string += ', window={}'.format(self.window)
        format_string += ', center={}'.format(self.center)
        format_string += ', pad_mode={}'.format(self.pad_mode)
        format_string += ', power={}'.format(self.power)
        format_string += ')'
        return format_string


class MelScale(object):
    def __init__(self,
                 sample_rate=22050,
                 n_fft=1024,
                 n_mels=80,
                 fmin=0,
                 fmax=8000):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax

        self.mel_basis = librosa.filters.mel(sr=self.sample_rate,
                                             n_fft=self.n_fft,
                                             n_mels=self.n_mels,
                                             fmin=self.fmin,
                                             fmax=self.fmax)

    def __call__(self, s):
        return np.dot(self.mel_basis, s)

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'sample_rate={}'.format(self.sample_rate)
        format_string += ', n_fft={}'.format(self.n_fft)
        format_string += ', fmin={}'.format(self.fmin)
        format_string += ', fmax={}'.format(self.fmax)
        format_string += ')'
        return format_string


class AmplitudeToDB(object):
    def __init__(self,
                 multiplier=20,
                 amin=1e-5,
                 ref_value=1.0,
                 ref_level_db=None):
        self.multiplier = multiplier
        self.amin = amin
        if ref_level_db is not None:
            ref_value = 10**(ref_level_db / self.multiplier)
        self.ref_value = ref_value
        self.db_multiplier = np.log10(np.maximum(self.amin, self.ref_value))

    def __call__(self, s):
        s_db = self.multiplier * np.log10(np.maximum(s, self.amin))
        s_db -= self.multiplier * self.db_multiplier
        return s_db

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'multiplier={}'.format(self.multiplier)
        format_string += ', amin={}'.format(self.amin)
        format_string += ', ref_value={}'.format(self.ref_value)
        format_string += ', db_multiplier={}'.format(self.db_multiplier)
        format_string += ')'
        return format_string


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, x):
        return (x - self.mean) / self.std

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'mean={}'.format(self.mean)
        format_string += ', std={}'.format(self.std)
        format_string += ')'
        return format_string
