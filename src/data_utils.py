import os
import numpy as np
import matplotlib.pyplot as plt

import librosa
import librosa.display

from tqdm import tqdm

def mp3_to_spectrogram(mp3_path, img_path):
    # Load the MP3 file and extract audio data
    y, sr = librosa.load(mp3_path)

    # Convert the audio data into a spectrogram
    spectrogram = librosa.feature.melspectrogram(y = y, sr = sr)

    # Plot the spectrogram
    plt.figure(figsize = (10, 4))
    librosa.display.specshow(librosa.power_to_db(spectrogram, ref = np.max))
    plt.axis('off')

    # Save the spectrogram as an image
    plt.savefig(img_path, transparent = True)
    plt.close()



if __name__ == '__main__':
    mp3_dir = '../../data/train'
    img_dir = '../../data/imgs'
    for mp3_name in tqdm(os.listdir(mp3_dir)):
        mp3_path = os.path.join(mp3_dir, mp3_name)

        img_name = mp3_name.replace('.mp3', '.png')
        img_path = os.path.join(img_dir, img_name)
        mp3_to_spectrogram(mp3_path, img_path)
        
