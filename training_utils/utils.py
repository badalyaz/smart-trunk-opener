import os
import random
import shutil
from glob import glob
from pathlib import Path
from typing import Callable, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import scipy
import tensorflow as tf
import torch
import torch.nn.functional as fun
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle
from tensorflow import keras
from tqdm import tqdm

def clean_data(input_path, correcting_size=12):
    structure = np.ones(correcting_size)
    is_directory = os.path.isdir(input_path)
    
    if is_directory:
        for folder in glob(os.path.join(input_path, "*")):
            ref_kicks_path = os.path.join(folder, "ref_kicks.npy")
            ref_kicks = np.load(ref_kicks_path)
            
            label_kick_filled_dilated = scipy.ndimage.binary_dilation(ref_kicks, structure=structure)
            label_kick_filled_erosed = scipy.ndimage.binary_erosion(label_kick_filled_dilated, structure=structure).astype(int)
            
            np.save(ref_kicks_path, label_kick_filled_erosed)
    else:
        label_kick_filled_dilated = scipy.ndimage.binary_dilation(input_path, structure=structure)
        label_kick_filled_erosed = scipy.ndimage.binary_erosion(label_kick_filled_dilated, structure=structure).astype(int)

        return label_kick_filled_erosed


def decode_data(input_data):
    N = input_data.shape[0]
    tmp = input_data.reshape(N, -1)
    
    target_bins = tmp[:, ::2]
    target_amps = tmp[:, 1::2]
    
    decoded_bins = target_bins & 0x00FF
    decoded_amps = ((target_bins & 0xFF00) >> 8) + ((target_amps & 0x00FF) << 4)

    amps = decoded_amps.reshape(N, 2, 2)
    bins = decoded_bins.reshape(N, 2, 2)

    return amps, bins


def plot_predictions(predictions, labels, size=1000):
    pred = np.array(tf.math.argmax(predictions, axis=1))
    
    plt.plot(np.arange(len(labels))[:size], labels[:size], label="True labels")
    plt.plot(np.arange(len(pred))[:size], pred[:size], label="Predictions")
    plt.yticks([0, 1], ["No kick", "Kick"])
    plt.legend(loc="upper left")
    plt.show()


def calculate_accuracy(predictions, labels):
    pred = np.array(tf.math.argmax(predictions, axis=1))
    acc = round(100 * (pred == labels).mean(), 2)

    return acc


def _gaussian(n: int) -> np.ndarray:  
    n += 1
    x = np.arange(-n // 2 + 1, n // 2)
    x = 2 * x / n
    y = np.exp(-1 / (1 - x**2))

    return y / y.sum()


def perform_fft(raw_data: np.ndarray, window_fn: Optional[Callable[[int], np.ndarray]] = None) -> np.ndarray:
    data = np.vectorize(complex)(raw_data[..., 0, :], raw_data[..., 1, :])
    data -= data.mean(axis=-1, keepdims=True)

    if window_fn is not None:
        window = window_fn(raw_data.shape[-1])
        window /= window.sum()
        data = data * window

    fft_data: np.ndarray = np.fft.fft(data, axis=-1)

    return fft_data


def extract_targets_from_raw(fft_data, num_targets=2):
    amp = np.abs(fft_data)

    bins = np.argpartition(amp, -num_targets, axis=-1)[..., -num_targets:][..., ::-1]
    values = np.take_along_axis(amp, indices=bins, axis=-1)

    values = values.astype(np.uint16).reshape(-1, 4)
    bins = bins.astype(np.uint16).reshape(-1, 4)

    return values, bins


def raw_data_to_target(raw_data, num_targets=2):
    fft_data = perform_fft(raw_data, window_fn=_gaussian)
    target_data = extract_targets_from_raw(fft_data=fft_data, num_targets=num_targets)
    result = np.concatenate(target_data, axis=1)

    return result
