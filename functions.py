import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine

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

def calculate_similarity(mfcc1, mfcc2):
    """
    Oblicza podobieństwo między dwoma zestawami cech MFCC.
    Używa odległości kosinusowej.
    
    Args:
        mfcc1 (np.ndarray): Współczynniki MFCC próbki referencyjnej.
        mfcc2 (np.ndarray): Współczynniki MFCC próbki testowej.
    
    Returns:
        float: Odległość kosinusowa (mniejsza wartość = większe podobieństwo).
    """
    # Oblicz średnie wartości MFCC
    mean_mfcc1 = np.mean(mfcc1, axis=1)
    mean_mfcc2 = np.mean(mfcc2, axis=1)
    # Oblicz odległość kosinusową
    distance = cosine(mean_mfcc1, mean_mfcc2)
    return distance

def verify_user(reference_file, test_file, threshold=0.3):
    """
    Weryfikuje użytkownika na podstawie podobieństwa głosów.
    
    Args:
        reference_file (str): Ścieżka do próbki referencyjnej (plik WAV).
        test_file (str): Ścieżka do próbki testowej (plik WAV).
        threshold (float): Próg akceptacji. Im mniejszy, tym bardziej restrykcyjny.
    
    Returns:
        bool: True, jeśli użytkownik jest zweryfikowany, False w przeciwnym razie.
    """
    # Oblicz MFCC dla obu próbek
    mfcc_ref = extract_mfcc(reference_file)
    mfcc_test = extract_mfcc(test_file)
    # Oblicz podobieństwo
    similarity = calculate_similarity(mfcc_ref, mfcc_test)
    print(f"Odległość kosinusowa: {similarity:.4f}")
    # Porównaj z progiem
    return similarity < threshold