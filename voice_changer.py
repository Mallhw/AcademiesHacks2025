import numpy as np
import sounddevice as sd
import librosa

SAMPLE_RATE = 44100
BLOCKSIZE = 2048

def audio_callback(indata, outdata, frames, time, status):
    if status:
        print("Audio status:", status)

    audio_input = indata[:, 0] if indata.ndim > 1 else indata

    try:
        audio_output = librosa.effects.pitch_shift(audio_input, sr=SAMPLE_RATE, n_steps=PITCH_SHIFT)
    except Exception as e:
        print("Error during pitch shifting:", e)
        audio_output = audio_input

    if len(audio_output) < len(audio_input):
        audio_output = np.pad(audio_output, (0, len(audio_input) - len(audio_output)), mode='constant')
    elif len(audio_output) > len(audio_input):
        audio_output = audio_output[:len(audio_input)]

    outdata[:] = audio_output.reshape(-1, 1)

def main(pitch_shift=-3):
    global PITCH_SHIFT
    PITCH_SHIFT = pitch_shift
    print("Voice changer running. Press Ctrl+C to stop.")
    
    with sd.Stream(samplerate=SAMPLE_RATE,
                   blocksize=BLOCKSIZE,
                   channels=1,
                   callback=audio_callback):
        try:
            while True:
                sd.sleep(1000)
        except KeyboardInterrupt:
            print("Voice changer stopped.")

if __name__ == "__main__":
    main()
