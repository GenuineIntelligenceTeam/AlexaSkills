'''
    'loud.py' uses PortAudio to record audio and depending on the audio data,
    various mp3 files will play based on loudness in order to trigger the
    Alexa Skill "Study Mode".
    The recording time is at the user's discretion.
'''

import numpy as np
import playsound
from microphone import record_audio

listen_time = 30
frames, sample_rate = record_audio(listen_time)
audio_data = np.hstack([np.frombuffer(i, np.int16) for i in frames])

count1, count2 = 0, 0
for signal in audio_data:
    if np.abs(signal) > 10000:
        count1 += 1
    elif np.abs(signal) > 20000:
        count2 += 1
if count1 > 100:
    playsound.playsound('distracting_file.mp3', True)
elif count2 > 20:
    playsound.playsound('loud_file.mp3', True)
