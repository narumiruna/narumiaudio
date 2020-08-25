from torch.utils.data import Dataset
from pathlib import Path

import librosa
import torch

import torch.nn.functional as F

from typing import Callable


def load_audio(path, sr=None, normalize=True):
    y, _ = librosa.load(path, sr=sr)
    if normalize:
        y = librosa.util.normalize(y)
    return y


class SpeakerFolder(Dataset):
    def __init__(self,
                 root: str,
                 ext: str = '.wav',
                 sample_rate: int = 22050,
                 transform: Callable = None):
        self.root = Path(root)
        self.ext = ext
        self.sample_rate = sample_rate
        self.transform = transform
        self.speakers = sorted(
            [d.stem for d in self.root.iterdir() if d.is_dir()])
        self.speaker_to_id = {s: i for i, s in enumerate(self.speakers)}

        self.samples = []
        for speaker in self.speakers:
            speaker_dir = self.root / speaker
            paths = list(speaker_dir.rglob('*{}'.format(self.ext)))

            for path in paths:
                self.samples.append((path, speaker))

    def __getitem__(self, index):
        path, speaker = self.samples[index]

        data = load_audio(path, sr=self.sample_rate)
        if self.transform is not None:
            data = self.transform(data)

        speaker_id = self.speaker_to_id[speaker]

        one_hot = F.one_hot(torch.tensor(speaker_id),
                            num_classes=len(self.speakers))

        return data, one_hot

    def __len__(self):
        return len(self.samples)
