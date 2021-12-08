import librosa.feature as ft
import librosa
from librosa.feature.spectral import mfcc
import numpy as np
import pandas as pd
import uuid
import os


class AudioFeatureExtractor:

    def __init__(self):
        self.root_path = './data/raw'
        self.spectrogram_path = './data/processed/spectrograms'
        self.chroma_stft_path = './data/processed/chroma_stft'
        self.chroma_cens_path = './data/processed/chroma_cens'
        self.spectral_flatness_path = './data/processed/sp_flat'
        self.spectral_contrast_path = './data/processed/sp_contrast'
        self.big_db_save_path = './data/unsplit.csv'
        self.sr = 22050
        self.full_song_data = pd.DataFrame()
        self.makedirs()

    def makedirs(self):
        if not os.path.exists(self.spectrogram_path):
            os.makedirs(self.spectrogram_path)

        if not os.path.exists(self.chroma_stft_path):
            os.makedirs(self.chroma_stft_path)

        if not os.path.exists(self.chroma_cens_path):
            os.makedirs(self.chroma_cens_path)

        if not os.path.exists(self.spectral_flatness_path):
            os.makedirs(self.spectral_flatness_path)

        if not os.path.exists(self.spectral_contrast_path):
            os.makedirs(self.spectral_contrast_path)

    def build_30s_dataset(self):
        dataset = []
        for filename in os.listdir(self.root_path):
            if '.wav' in filename:
                file_uuid = uuid.uuid1()
                y, sr = librosa.load(
                    self.root_path+f'/{filename}', self.sr)
                chroma_stft = ft.chroma_stft(y, sr)
                rms = ft.rms(y, sr)
                chroma_cens = ft.chroma_cens(y, sr)
                mfcc = ft.mfcc(y, sr, n_mfcc=20)
                spectral_centroid = ft.spectral_centroid(y, sr)
                spectral_bandwidth = ft.spectral_bandwidth(y, sr)
                spectral_contrast = ft.spectral_contrast(y, sr)
                spectral_flatness = ft.spectral_flatness(y)
                spectral_rolloff = ft.spectral_rolloff(y, sr)
                zcr = np.mean(ft.zero_crossing_rate(y))
                tempo = librosa.beat.tempo(y)

                rmsmean = np.mean(rms)
                mfccsmean = np.mean(mfcc, axis=1)
                spectral_centroid_mean = np.mean(spectral_centroid)
                spectral_bandwidth_mean = np.mean(spectral_bandwidth)
                spectral_rolloff_mean = np.mean(spectral_rolloff)

                rmsstd = np.std(rms)
                mfccsstd = np.std(mfcc, axis=1)
                spectral_centroid_std = np.std(spectral_centroid)
                spectral_bandwidth_std = np.std(spectral_bandwidth)
                spectral_rolloff_std = np.std(spectral_rolloff)
                len_s = librosa.get_duration(y)

                features = [filename.split('.')[0], file_uuid, len_s, zcr, tempo[0], rmsmean, rmsstd, spectral_centroid_mean, spectral_centroid_std,
                            spectral_bandwidth_mean, spectral_bandwidth_std, spectral_rolloff_mean, spectral_rolloff_std, np.mean(chroma_stft), np.std(chroma_stft), np.mean(chroma_cens), np.std(chroma_cens), np.mean(spectral_contrast), np.std(spectral_contrast), np.mean(spectral_flatness), np.std(spectral_flatness)]

                for k in range(len(mfccsstd)):
                    features.append(mfccsmean[k])
                    features.append(mfccsstd[k])
                dataset.append(features)
        columns = ['filename', 'uuid', 'duration    ', 'zero_crossing_rate', 'tempo', 'rms_mean', 'rms_std', 'spectral_centroid_mean',
                   'spectral_centroid_std', 'spectral_bandwidth_mean', 'spectral_bandwidth_std', 'spectral_rolloff_mean', 'spectral_rolloff_std', 'chroma_stft_mean', 'chroma_stft_std', 'chroma_cens_mean', 'chroma_cens_std', 'spectral_contrast_mean', 'spectral_contrast_std', 'spectral_flatness_mean', 'spectral_flatness_std']
        for k in range(len(mfccsmean)):
            columns.append(f'mfcc_{k}_mean')
            columns.append(f'mfcc_{k}_std')
        dataset = pd.DataFrame(dataset, columns=columns)
        dataset.to_csv(self.big_db_save_path, index=False)

        return dataset
