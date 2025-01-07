import sounddevice as sd
from scipy.io.wavfile import write

def record_voice(duration=5, samplerate=16000, filename="recording.wav"):
    """
    Nagrywa głos użytkownika przez określony czas i zapisuje plik WAV.
    
    Args:
        duration (int): Czas nagrania w sekundach.
        samplerate (int): Częstotliwość próbkowania.
        filename (str): Nazwa pliku wynikowego.
    """
    print("Nagrywanie... Powiedz coś!")
    recording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='int16')
    sd.wait()  # Czeka na zakończenie nagrywania
    write(filename, samplerate, recording)  # Zapisuje plik WAV
    print(f"Nagranie zakończone i zapisane jako {filename}")