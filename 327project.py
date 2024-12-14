import librosa
import numpy as np
import librosa.display
import matplotlib.pyplot as plt
from collections import Counter


#identifying each note's major and minor chords
def get_chord_templates():
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    major_intervals = [0, 4, 7]
    minor_intervals = [0, 3, 7]

    chords = {}
    for i, n in enumerate(notes):
        major_template = np.zeros(12)
        for intv in major_intervals:
            major_template[(i + intv) % 12] = 1
        chords[f"{n}"] = major_template

        minor_template = np.zeros(12)
        for intv in minor_intervals:
            minor_template[(i + intv) % 12] = 1
        chords[f"{n}m"] = minor_template
    return chords

def chord_detection(chroma, smoothing_window):
    
    chords = get_chord_templates()
    chord_names = list(chords.keys())
    chord_templates = np.array([chords[ch] for ch in chord_names])
    chord_templates = chord_templates / np.linalg.norm(chord_templates, axis=1, keepdims=True)

    detected_chords = []
    for frame in chroma.T:
        similarities = chord_templates.dot(frame)
        best_chord_index = np.argmax(similarities)
        detected_chords.append(chord_names[best_chord_index])


    # Smooth chord sequence to remove rapid fluctuations
    if len(detected_chords) > 0:
        smoothed_chords = []
        for i in range(len(detected_chords)):
            start = max(0, i - smoothing_window)
            end = min(len(detected_chords), i + smoothing_window + 1)
            window_chords = detected_chords[start:end]
            # Choose the most common chord in the window
            most_common_chord = Counter(window_chords).most_common(1)[0][0]
            smoothed_chords.append(most_common_chord)
        detected_chords = smoothed_chords

    return detected_chords




#Loading music file
music_file="yoklugundaV2.wav"
measurements, sample_rate = librosa.load(music_file, sr=44100, mono=True)
#Sampling rate contains number of samples that taken per second
#Measurements contains amplitude of waveform per sample

#For calculating bpm more accurate
a, b = librosa.load(music_file, sr=None)

# Calculate BPM
bpm, _ = librosa.beat.beat_track(y=a, sr=b)
if isinstance(bpm, np.ndarray):
   bpm = bpm.item()
   print("Estimated tempo (BPM):", bpm)
   
#☺ interval to convert measurements to chromagram 
hop_length = 512

#Calculate chroma with cqt(Constant Q Transform)
chroma = librosa.feature.chroma_cqt(y=measurements, sr=sample_rate, hop_length=hop_length)

# Smooth the chroma features slightly for both axis
chroma = chroma / (np.linalg.norm(chroma, axis=0, keepdims=True) + 1e-9)

chroma = np.apply_along_axis(
    lambda x: np.convolve(x, np.ones(5) / 5, mode='same'),
    axis=1,
    arr=chroma
)

#Calculating chroma frame per second
frames_per_second = sample_rate / hop_length

# Calculates the number of frames per beat based on bpm and doubles it for the window size.
frames_per_beat = (int)(frames_per_second * (60.0 / bpm))
# Detect chords
detected_chords = chord_detection(chroma, frames_per_beat)

# Convert frames to time
times = librosa.frames_to_time(np.arange(chroma.shape[1]), sr=sample_rate)
# Convert detected chords to chord segments
chord_changes = []
if len(detected_chords) > 0:
    current_chord = detected_chords[0]
    start_time = times[0]
    for i in range(1, len(detected_chords)):
        if detected_chords[i] != current_chord:
            end_time = times[i]
            chord_changes.append((current_chord, start_time, end_time))
            current_chord = detected_chords[i]
            start_time = times[i]
    # Add the last segment
    end_time = times[-1]
    chord_changes.append((current_chord, start_time, end_time))


# Get unique chords
unique_chords = list(dict.fromkeys(detected_chords))
chord_to_y = {ch: i for i, ch in enumerate(unique_chords)}

plt.figure(figsize=(10, 6))

# Plot Chromagram on top
ax1 = plt.subplot(2, 1, 1)
librosa.display.specshow(chroma, x_axis='time', y_axis='chroma', sr=sample_rate, cmap='coolwarm')
plt.colorbar()
plt.title('Chroma Representation')
plt.yticks(np.arange(12), ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'])

# Plot chord segments as horizontal bars
ax2 = plt.subplot(2, 1, 2, sharex=ax1)
cmap = plt.get_cmap('tab20')
for chord, start, end in chord_changes:
    y_pos = chord_to_y[chord]
    duration = end - start
    color = cmap(y_pos % cmap.N)
    ax2.barh(y_pos, duration, left=start, height=0.6, color=color, edgecolor='black')

ax2.set_yticks(range(len(unique_chords)))
ax2.set_yticklabels(unique_chords)
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Chord')
ax2.set_title('Detected Chords as Horizontal Bars')
ax2.invert_yaxis()  # Optional: puts first chord at the top

plt.tight_layout()
plt.show()
