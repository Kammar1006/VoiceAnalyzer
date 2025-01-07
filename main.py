import os
from record import record_voice
from functions import extract_mfcc, plot_mfcc, plot_fft

def menu():
    """
    Proste menu w konsoli do obsługi programu.
    """
    while True:
        print("\n=== MENU ===")
        print("1. Nagraj głos")
        print("2. Ekstrakcja i wizualizacja MFCC")
        print("3. FFT (Widmo częstotliwości)")
        print("4. Wyjście")
        choice = input("Wybierz opcję: ")

        if choice == "1":
            # Opcja 1: Nagrywanie głosu
            filename = input("Podaj nazwę pliku do zapisania (np. 'sample.wav'): ")
            record_voice(duration=5, filename=filename)
        elif choice == "2":
            # Opcja 2: Ekstrakcja MFCC
            filename = input("Podaj nazwę pliku audio (np. 'sample.wav'): ")
            if os.path.exists(filename):
                mfcc_features = extract_mfcc(filename)
                plot_mfcc(mfcc_features, 16000)
            else:
                print(f"Błąd: Plik '{filename}' nie istnieje.")
        elif choice == "3":
            # Opcja 3: Analiza FFT
            filename = input("Podaj nazwę pliku audio (np. 'sample.wav'): ")
            if os.path.exists(filename):
                plot_fft(filename)
            else:
                print(f"Błąd: Plik '{filename}' nie istnieje.")
        elif choice == "4":
            # Wyjście
            print("Do zobaczenia!")
            break
        else:
            print("Nieprawidłowy wybór, spróbuj ponownie.")

if __name__ == "__main__":
    menu()