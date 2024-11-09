import numpy as np
import pyaudio
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
  
Volume = 0.3
MusicLength = 2 # Length in seconds
MusicRate = 44100 # Sampling rate per second
Music = np.array([0] * MusicRate * MusicLength).astype(np.float32)


def envelope (input, attack, decay, release): 
    input[0:attack[0]] *= np.arange(0, attack[1], attack[1]/attack[0])
    input[attack[0]:attack[0]+decay] *= np.arange(attack[1], 1, (1-attack[1])/decay)
    input[-1-release:-1] *= np.arange(1, 0, -1/release)
    return input

"""
A "step" here is a unit of time; 1 sec / MusicRate.
freq - The frequency of the note (in Hz).
amp - Amplitude of the note. 
start - Numer of steps into the music which note starts.
length - Number of steps note lasts.
attack - [Number of steps note initially increases, gain (must be <=1)]
decay - Number of steps note takes to decrease to gain 1 (after the initial attack)
release - Number of steps note takes to decrease to 0 (at the end of its length)
"""
def AddTone (freq, amp, start, length, attack, decay, release):
    scale = freq * (2* np.pi) / MusicRate
    wave = amp * np.sin(np.arange(length) * scale)  # A pure tone
    tone = envelope(wave, attack, decay, release) # The wave with ADSR envelope
    end = min(start+len(tone), len(Music)) # Fit the tone to within music boundaries
    Music[start:end] += wave[:end-start]

AddTone(440, 0.2, 0, 44100, [1000, 3], 1000, 3000)
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paFloat32, channels=1, rate=MusicRate, output=1)
stream.write(Music.astype(np.float32).tobytes())
stream.close()
p.terminate()

fig, axs = plt.subplots(2, 1)
axs[0].plot(range(len(Music)), Music)
axs[0].set_xlabel('Time (in discrete samples)')
axs[0].set_ylabel('Amplitude')
axs[0].set_title('Time Domain')

fft_values = fft(Music)
frequencies = fftfreq(len(Music), 1/MusicRate)
axs[1].plot(frequencies, np.abs(fft_values))
axs[1].set_xlabel('Frequency (Hz)')
axs[1].set_ylabel('Amplitude')
axs[1].set_title('Frequency Spectrum')

plt.subplots_adjust(hspace=0.6)
plt.show()