import librosa
import numpy as np
import random
import os


def load_wavs(wav_dir, sr):

    wavs = list()
    for file in os.listdir(wav_dir):
        file_path = os.path.join(wav_dir, file)
        wav, _ = librosa.load(file_path, sr=sr, mono=True)
        print(wav_dir)
        wavs.append(wav)

    return wavs


def wavs_to_mfccs(wavs, sr, n_fft=1024, hop_length=None, n_mels=5, n_mfcc=39):

    mfccs = list()
    for wav in wavs:
        #  n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
        mfcc = librosa.feature.mfcc(
            y=wav, sr=sr, hop_length=hop_length,  n_mfcc=n_mfcc)
        print(mfcc.shape)
        mfccs.append(mfcc)

    return mfccs


def wav_to_mfcc(wav_dir, file, sr, frame_length=2048, hop_length=None, n_mfcc=39, n=10):
    file_path = os.path.join(wav_dir, file)
    wav, _ = librosa.load(file_path, sr=sr, mono=True)
    mfcc = librosa.feature.mfcc(
        y=wav, sr=sr, hop_length=hop_length,  n_mfcc=n_mfcc, n_fft=frame_length)
    return mfcc


def wav_to_numpy(wav_dir, file, out_dir, sr, hop_length=None, n_mfcc=39, n=10):
    file_path = os.path.join(wav_dir, file)
    wav, _ = librosa.load(file_path, sr=sr, mono=True)
    mfcc = librosa.feature.mfcc(
        y=wav, sr=sr, hop_length=hop_length,  n_mfcc=n_mfcc)
    # print('{} {}'.format(file, mfcc.shape))
    size = mfcc.shape[1]
    for start in range(size):
        # window = np.zeros([n_mfcc, n])
        if(size < start + n):
            num_pad_right = n - (size - start)
        else:
            num_pad_right = 0
        tmp = mfcc[:, start: start + n]
        window = np.pad(tmp,
                        ((0, 0), (0, num_pad_right)), 'constant', constant_values=0)
        # print(window.shape)
        np.save('{}{}_{}'.format(out_dir, file.replace('.wav', ''), start), window)
    return size


def mfcc_to_numpy(mfcc, file, out_dir, sr, n=10, start_indexs=None):
    size = mfcc.shape[1]
    if start_indexs is None:
        start_indexs = range(size)
    for start in range(size):
        # window = np.zeros([n_mfcc, n])
        if(size < start + n):
            num_pad_right = n - (size - start)
        else:
            num_pad_right = 0
        tmp = mfcc[:, start: start + n]
        window = np.pad(tmp,
                        ((0, 0), (0, num_pad_right)), 'constant', constant_values=0)
        # print(window.shape)
        np.save('{}{}_{}'.format(out_dir, file.replace('.wav', ''), start), window)
    return size


def wavs_to_numpy(wav_dir, out_dir, sr, hop_length=None, n_mfcc=39, n=10):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    total = 0
    for file in os.listdir(wav_dir):
        if file.find('.wav') == -1:
            continue
        size = wav_to_numpy(wav_dir, file, out_dir, sr,
                            hop_length=None, n_mfcc=39, n=10)
        total += size
    print('{} {}'.format(wav_dir, total))
    return total


def dataset_to_numpy(dataset_dir, out_dir, sr, hop_length=None, n_mfcc=39, n=10):
    total = 0
    real_total = 0
    clear_dir = dataset_dir + 'clean/'
    multi_dir = dataset_dir + 'multi/'
    clear_out_dir = out_dir + 'trainA/'
    multi_out_dir = out_dir + 'trainB/'
    if not os.path.exists(clear_out_dir):
        os.makedirs(clear_out_dir)
    if not os.path.exists(multi_out_dir):
        os.makedirs(multi_out_dir)
    list_dir = os.listdir(clear_dir)
    i = 0
    for file in list_dir:
        i += 1
        print('{}/{}'.format(i, len(list_dir)))
        if file.find('.wav') == -1:
            continue
        clear_mfcc = wav_to_mfcc(
            clear_dir, file, sr, frame_length=200, hop_length=80, n_mfcc=39, n=n)
        multi_mfcc = wav_to_mfcc(
            multi_dir, file, sr, frame_length=200, hop_length=80, n_mfcc=39, n=n)
        mfcc_size = multi_mfcc.shape[1]
        start_indexs = random.sample(range(mfcc_size), 10)
        mfcc_to_numpy(clear_mfcc, file, clear_out_dir, sr,
                      n=n, start_indexs=start_indexs)
        mfcc_to_numpy(multi_mfcc, file, multi_out_dir, sr,
                      n=n, start_indexs=start_indexs)
        total += len(start_indexs)
        real_total += mfcc_size
    print('{} total = {} origin = {}'.format(clear_dir, total, real_total))
    return total


if __name__ == '__main__':
    sample_rate = 8000
    dataset_to_numpy('data/train/',
                     'datasets/aurora2_n39_4/', sample_rate, n=39)
    # wavs_to_numpy('data/train/clean/','datasets/aurora2_n10_hop_len100/clean/', sample_rate,n = 10,hop_length=100)
    # wavs_to_numpy('data/train/multi/','datasets/aurora2_n10_hop_len100/multi/', sample_rate,n = 10,hop_length=100)
    print('success')
