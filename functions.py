import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

def extract_mfcc(filename, n_mfcc=13):
    """
    Ekstrakcja współczynników MFCC z pliku audio.
    
    Args:
        filename (str): Ścieżka do pliku WAV.
        n_mfcc (int): Liczba współczynników MFCC do obliczenia.
    
    Returns:
        np.ndarray: Tablica współczynników MFCC.
    """
    # Załaduj plik audio
    y, sr = librosa.load(filename, sr=None)
    # Oblicz MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return mfcc

def plot_mfcc(mfcc, sr):
    """
    Rysuje wykres MFCC.
    
    Args:
        mfcc (np.ndarray): Współczynniki MFCC.
        sr (int): Częstotliwość próbkowania.
    """
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfcc, x_axis='time', sr=sr)
    plt.colorbar(format='%+2.0f dB')
    plt.title('MFCC')
    plt.tight_layout()
    plt.show()


def plot_fft(filename):
    """
    Rysuje wykres FFT z pliku audio.
    
    Args:
        filename (str): Ścieżka do pliku WAV.
    """
    # Załaduj plik audio
    y, sr = librosa.load(filename, sr=None)
    # Oblicz FFT
    fft = np.fft.fft(y)
    magnitude = np.abs(fft)
    frequency = np.linspace(0, sr, len(magnitude))
    
    # Wykres FFT
    plt.figure(figsize=(10, 4))
    plt.plot(frequency[:len(frequency)//2], magnitude[:len(magnitude)//2])
    plt.title("Widmo częstotliwości (FFT)")
    plt.xlabel("Częstotliwość (Hz)")
    plt.ylabel("Amplituda")
    plt.grid()
    plt.show()