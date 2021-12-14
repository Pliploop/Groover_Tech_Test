from librosa.effects import harmonic, percussive
import librosa.feature as ft
import librosa
from librosa.feature.spectral import mfcc
import numpy as np
from numpy.lib.arraysetops import isin
import pandas as pd
import uuid
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import pydub
from pydub import AudioSegment
from faker import Faker


def compute_power_db(y, sr, win_len_sec=0.2, power_ref=10**(-12)):
    
    win_len = round(win_len_sec * sr)
    win = np.ones(win_len) / win_len
    power_db = 10 * np.log10(np.convolve(y**2, win, mode='same') / power_ref)
    return power_db



class AudioFeatureExtractor:

    def __init__(self):
        ## Initialize paths to save processed data to
        self.root_path = './data/raw'
        self.spectrogram_path = './data/processed/spectrograms'
        self.chroma_stft_path = './data/processed/chroma_stft'
        self.chroma_cens_path = './data/processed/chroma_cens'
        self.spectral_flatness_path = './data/processed/sp_flat'
        self.spectral_contrast_path = './data/processed/sp_contrast'
        self.big_db_save_path = './data/unsplit.csv'
        self.small_db_save_path = './data/split.csv'
        
        self.sr = 22050 ## Standard sample rate
        self.n_mfccs = 20
        self.full_song_data = pd.DataFrame()
        self.makedirs() # Make directories if non-existent
        self.f1 = Faker()
        Faker.seed(314156) # Seed to get same uuids
        

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
        """builds song-level feature dataset"

        Returns:
            pd.DataFrame: dataset of song-level features
        """
        dataset = []
        for filename in tqdm(os.listdir(self.root_path)):
            if '.wav' in filename:
                file_uuid = self.f1.uuid4()
                features = self.extract_features(filename,file_uuid,True) # Extract features to list
                dataset.append(features)
        columns = ['filename', 'uuid', 'duration', 'zero_crossing_rate', 'tempo','dynamic_range', 'rms_mean', 'rms_std','harmonic_rms','percussive_rms', 'spectral_centroid_mean',
                   'spectral_centroid_std', 'HPR','spectral_bandwidth_mean', 'spectral_bandwidth_std', 'spectral_rolloff_mean', 'spectral_rolloff_std', 'chroma_stft_mean', 'chroma_stft_std', 'chroma_cens_mean', 'chroma_cens_std', 'spectral_contrast_mean', 'spectral_contrast_std', 'spectral_flatness_mean', 'spectral_flatness_std']
        for k in range(self.n_mfccs):
            columns.append(f'mfcc_{k}_mean')
            columns.append(f'mfcc_{k}_std')
        dataset = pd.DataFrame(dataset, columns=columns) # Create dataset
        dataset.to_csv(self.big_db_save_path, index=False) # Save dataset
        return dataset

    def build_sub_datasets(self,path):
        """Builds instance-level dataset with extracted audio features and without processed data

        Args:
            path (str): path to save instance level dataset to

        Returns:
            pd.DataFrame: dataframe containing all data for instance-level features
        """

        if path is not None:
            try:
                dataset = pd.read_csv(path)
            except:
                dataset = self.build_30s_dataset()
        else:
            dataset = pd.read_csv(self.big_db_save_path)

        big_dataset = []
        for sample_loc in tqdm(range(len(dataset))):
            sample = dataset.iloc[sample_loc]
            filename = f'{self.root_path}/{sample["filename"]}.wav'
            uuid = sample['uuid']
            sound = AudioSegment.from_file(filename)
            file_tempo = sample.tempo
            bar_duration_ms = int(60/float(file_tempo)*4*1000) #We assume the music to be in 4/4, though a supervised model could be trained to identify the time signature
            chunks  = pydub.utils.make_chunks(sound,bar_duration_ms) # Create beat-level chunks
            
            for i,chunk in enumerate(chunks):
                chunks = chunk.get_array_of_samples()
                y = np.nan_to_num(np.array(chunks).astype(np.float32))
                new_uuid = f'{str(uuid)}_{i}' #create new identifier based on song name
                features = self.extract_features(y,new_uuid) # extract features for each chunk
                big_dataset.append(features)
        columns = ['filename', 'uuid', 'duration', 'zero_crossing_rate', 'tempo','dynamic_range', 'rms_mean', 'rms_std','harmonic_rms','percussive_rms', 'spectral_centroid_mean',
                'spectral_centroid_std', 'HPR','spectral_bandwidth_mean', 'spectral_bandwidth_std', 'spectral_rolloff_mean', 'spectral_rolloff_std', 'chroma_stft_mean', 'chroma_stft_std', 'chroma_cens_mean', 'chroma_cens_std', 'spectral_contrast_mean', 'spectral_contrast_std', 'spectral_flatness_mean', 'spectral_flatness_std']
        for k in range(self.n_mfccs):
            columns.append(f'mfcc_{k}_mean')
            columns.append(f'mfcc_{k}_std')
        big_dataset = pd.DataFrame(big_dataset,columns=columns)
        big_dataset.to_csv(self.small_db_save_path)
        return big_dataset
        

            
            

                


    def extract_features(self,fn,uuid,save_2D = False):
        """extracts audio features from audio clip given as array or filepath

        Args:
            fn (array of filepath): switch to find the audio file
            uuid (str): uuid of the audio file in the dataset
            save_2D (bool, optional): instruction to save spectrograms/chromagrams. Defaults to False.

        Returns:
            array: array of features for the given clip
        """

        if isinstance(fn,str):
            y, sr = librosa.load(
            self.root_path+f'/{fn}', self.sr)
        else:
            y,sr = fn,self.sr
            fn = uuid.split('_')[0]
        
        ### extracting all relevant features
        chroma_stft = ft.chroma_stft(y, sr)
        rms = ft.rms(y, sr)
        chroma_cens = ft.chroma_cens(y, sr)
        mfcc = ft.mfcc(y, sr, n_mfcc=self.n_mfccs)
        spectral_centroid = ft.spectral_centroid(y, sr)
        spectral_bandwidth = ft.spectral_bandwidth(y, sr)
        spectral_contrast = ft.spectral_contrast(y, sr)
        spectral_flatness = ft.spectral_flatness(y)
        spectral_rolloff = ft.spectral_rolloff(y, sr)
        zcr = np.mean(ft.zero_crossing_rate(y))
        tempo = librosa.beat.tempo(y)
        melspec = librosa.power_to_db(S = ft.melspectrogram(y,sr,n_mels=128),ref=np.max)

        ## Only for song-level clips
        if save_2D:
            self.save_2D(chroma_stft,chroma_cens,spectral_contrast,spectral_flatness,melspec,uuid)

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

        power_db = compute_power_db(y,sr)
        dynamic_range = np.max(power_db) - np.min(power_db)

        harmonic_elements = librosa.effects.harmonic(y)
        percussive_elements = librosa.effects.percussive(y)
        ## Custom harmonic-to-percussive ratio feature
        try:
            HPR = len(np.where(harmonic_elements>0.00001)[0])/len(np.where(percussive_elements>0.00001)[0])
        except:
            HPR = 1
        

        harmonic_rms = np.mean(ft.rms(harmonic_elements,sr))
        percussive_rms = np.mean(ft.rms(percussive_elements,sr))

        features = [fn.split('.')[0], uuid, len_s, zcr, tempo[0], dynamic_range, rmsmean, rmsstd,harmonic_rms,percussive_rms, spectral_centroid_mean, spectral_centroid_std, HPR,
                    spectral_bandwidth_mean, spectral_bandwidth_std, spectral_rolloff_mean, spectral_rolloff_std, np.mean(chroma_stft), np.std(chroma_stft), np.mean(chroma_cens), np.std(chroma_cens), np.mean(spectral_contrast), np.std(spectral_contrast), np.mean(spectral_flatness), np.std(spectral_flatness)]
        
        for k in range(len(mfccsstd)):
            features.append(mfccsmean[k])
            features.append(mfccsstd[k])
            
        return features

    def save_2D(self,chroma_stft,chroma_cens,spectral_contrast,spectral_flatness,mel_spec,uuid):
        plt.imsave(f'{self.spectrogram_path}/{uuid}.png',mel_spec)
        plt.imsave(f'{self.chroma_stft_path}/{uuid}.png',chroma_stft)
        plt.imsave(f'{self.chroma_cens_path}/{uuid}.png',chroma_cens)
        plt.imsave(f'{self.spectral_contrast_path}/{uuid}.png',spectral_contrast)
        plt.imsave(f'{self.spectral_flatness_path}/{uuid}.png',spectral_flatness)
