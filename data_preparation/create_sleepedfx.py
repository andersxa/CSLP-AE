import argparse
import logging
from functools import partial
from pathlib import Path
from typing import List, Optional

import librosa
import mne
import numpy as np
import torch
from scipy import signal
from sklearn.model_selection import GroupKFold
from tqdm import tqdm

logger = logging.getLogger(__name__)


class STFTTransform:
    """
    STFTTransform: Short-time Fourier Transform

    Attributes:
        fs (int): sampling frequency.
        segment_size (int): window length in samples.
        step_size (int): step size between successive windows in samples.
        nfft (int): number of points for FFT.
        transform_fn (Callable): partial function containing the STFT operation.
    """

    def __init__(self, fs: int, segment_size: int, step_size: int, nfft: int, *args, **kwargs) -> None:
        """
        Args:
            fs (int): sampling frequency.
            segment_size (int): window length in samples.
            step_size (int): step size between successive windows in samples.
            nfft (int): number of points for FFT.
        """
        self.fs = fs
        self.segment_size = segment_size
        self.step_size = step_size
        self.nfft = nfft
        self.transform_fn = partial(
            signal.stft,
            fs=self.fs,
            nperseg=self.segment_size,
            noverlap=self.segment_size - self.step_size,
            nfft=self.nfft,
        )

    def calculate_output_dims(self, window_size: int) -> List[int]:
        """calculates output dimensions after STFT operation.
        The STFT windows and optionally overlaps data, which causes reductions in the time vectors.

        Args:
            window_size (int): Size of windows in samples.

        Returns:
            List[int]: List containing the frequency and time dimensions.
        """
        T = librosa.samples_to_frames(window_size, hop_length=self.step_size) + 1
        F = self.nfft // 2 + 1
        return [F, T]

    def __call__(self, X: np.ndarray, annotations: Optional[np.ndarray] = None) -> np.ndarray:
        _, _, Zxx = self.transform_fn(X)
        Zxx = np.abs(Zxx) ** 2
        # Zxx = np.abs(self.transform_fn(X)) ** 2
        Zxx = librosa.power_to_db(Zxx)

        return Zxx[:, 1:, :-1]
        # return Zxx[:, :, 1:]  # Remove DC component

def process_sleepedfx_data(args):
    N_SUBJECTS = args.subjects
    SAMPLING_RATE = args.fs
    DATA_DIR = Path(args.data)
    OUTPUT_DIR = Path(args.output)
    EEG_PICK = args.pick
    STEP_SIZE = args.step_size
    NFFT = args.nfft

    filelist = mne.datasets.sleep_physionet.age.fetch_data(
        subjects=list(range(N_SUBJECTS)), path=DATA_DIR, on_missing="warn"
    )
    annotation_desc_2_event_id = {
        "Sleep stage W": 1,
        "Sleep stage 1": 2,
        "Sleep stage 2": 3,
        "Sleep stage 3": 4,
        "Sleep stage 4": 4,
        "Sleep stage R": 5,
    }
    # create a new event_id that unifies stages 3 and 4
    event_id = {
        "Sleep stage W": 1,
        "Sleep stage 1": 2,
        "Sleep stage 2": 3,
        "Sleep stage 3/4": 4,
        "Sleep stage R": 5,
    }
    Zxx_array = None
    subjects = []
    nights = []
    tasks = []
    for idx, file_pair in enumerate(tqdm(filelist)):
        edf_path = Path(file_pair[0])
        hyp_path = Path(file_pair[1])

        raw = mne.io.read_raw_edf(edf_path, preload=True, stim_channel="Event marker", infer_types=True)
        annot = mne.read_annotations(hyp_path)
        raw.set_annotations(annot, emit_warning=False)

        # keep last 30-min wake events before sleep and first 30-min wake events after
        # sleep and redefine annotations on raw data
        first_wake = [i for i, x in enumerate(annot.description) if x == "Sleep stage W"][0]
        last_wake = [i for i, x in enumerate(annot.description) if x == "Sleep stage W"][-1]
        if first_wake is not None and last_wake is not None:
            annot.crop(annot[first_wake + 1]["onset"] - 30 * 60, annot[last_wake]["onset"] + 30 * 60)
        else:
            ...
            # continue
        raw.set_annotations(annot, emit_warning=False)
        events, _ = mne.events_from_annotations(raw, event_id=annotation_desc_2_event_id, chunk_duration=30.0)

        # Resample data and events
        logger.info(f"Resampling data from {raw.info['sfreq']} to {SAMPLING_RATE} Hz")
        raw.resample(SAMPLING_RATE, events=events, npad="auto")

        tmax = 30.0 - 1.0 / SAMPLING_RATE
        epochs = mne.Epochs(
            raw=raw, events=events, event_id=event_id, tmin=0.0, tmax=tmax, baseline=None, preload=True, on_missing="warn"
        )

        # Normalize data
        mu = epochs.pick(EEG_PICK).get_data().mean()
        std = epochs.pick(EEG_PICK).get_data().std()
        X = (epochs.pick(EEG_PICK).get_data() - mu) / std
        Zxx = STFTTransform(fs=SAMPLING_RATE, segment_size=2 * SAMPLING_RATE, step_size=STEP_SIZE, nfft=NFFT)(X.squeeze())[
            :, : SAMPLING_RATE // 2, :
        ]
        Zxx = (Zxx - Zxx.mean()) / Zxx.std()
        if Zxx_array is None:
            Zxx_array = Zxx
        else:
            Zxx_array = np.concatenate((Zxx_array, Zxx), axis=0)

        # Save metadata
        N, F, T = Zxx.shape
        subjects.extend([int(edf_path.stem[3:5])] * N)
        nights.extend([int(edf_path.stem[5:6])] * N)
        tasks.extend(epochs.events[:, -1].tolist())

    Zxx_array = torch.from_numpy(Zxx_array).float()

    # Do splits and calculate split mean and std
    splits = list(GroupKFold(n_splits=5).split(X=Zxx_array, y=tasks, groups=subjects))
    fold_info = {i: {"train_idx": None, "test_idx": None, "mean": None, "std": None} for i, _ in enumerate(splits)}
    for i, (train_index, test_index) in enumerate(splits):
        fold_info[i]["train_idx"] = train_index
        fold_info[i]["test_idx"] = test_index
        fold_info[i]["mean"] = torch.mean(Zxx_array[train_index])
        fold_info[i]["std"] = torch.std(Zxx_array[train_index])
    torch.save(
        dict(
            data=Zxx_array,
            subjects=torch.tensor(subjects),
            tasks=torch.tensor(tasks),
            runs=torch.tensor(nights),
            labels=torch.tensor(tasks),
            fold_info=fold_info,
        ),
        OUTPUT_DIR / "sleepedfx_data.pt",
    )

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="SleepEDFx data processing")
    parser.add_argument('-d', '--data', type=str, help='Data directory path')
    parser.add_argument('-o', '--output', type=str, help='Output file path')
    parser.add_argument('-n', '--subjects', type=int, default=83, help='Number of subjects to process')
    parser.add_argument('--fs', type=int, default=128, help='Sampling frequency')
    parser.add_argument('--pick', type=str, default='Fpz-Cz', choices=['Fpz-Cz', 'Pz-Oz'], help='Channel to pick')
    parser.add_argument('--step-size', type=int, default=16, help='Step size for STFT')
    parser.add_argument('--nfft', type=int, default=256, help='Number of points for FFT')
    args = parser.parse_args()

    process_sleepedfx_data(args)
