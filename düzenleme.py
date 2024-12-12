import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from music21 import chord, stream, note
import noisereduce as nr

def identify_major_minor_chord(note_list):
    """
    Verilen nota listesine en uygun majör veya minör akor adını belirler.
    - note_list: Liste halinde notalar (['C', 'E', 'G'])
    """
    try:
        c = chord.Chord(note_list)
        if c.quality == 'major':
            return f"{c.root().name} Major"
        elif c.quality == 'minor':
            return f"{c.root().name} Minor"
        else:
            return "Unknown"
    except:
        return "Unknown"

def get_key_music21(y, sr):
    """
    Şarkının tonalitesini belirler.
    """
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    y_harmonic, _ = librosa.effects.hpss(y)
    chroma = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)
    energy_threshold = 0.3  # Daha hassas bir eşik değeri belirlendi
    notes_present = [notes[i] for i, energy in enumerate(chroma_mean) if energy > energy_threshold]
    if len(notes_present) > 5:
        sorted_notes = sorted(zip(notes_present, chroma_mean), key=lambda x: x[1], reverse=True)
        notes_present = [n for n, e in sorted_notes[:5]]
    if not notes_present:
        return "Unknown", "Unknown"
    s = stream.Stream()
    for n in notes_present:
        s.append(note.Note(n))
    try:
        detected_key = s.analyze('key', keyWeightAlgorithm='correlation')  # Tonalite algılamasında daha hassas bir yöntem kullanıldı
        return detected_key.tonic.name, detected_key.mode
    except:
        return "Unknown", "Unknown"

def reduce_noise(y, sr):
    noise_clip = y[:int(sr * 0.5)]
    y_denoised = nr.reduce_noise(y=y, sr=sr, y_noise=noise_clip, prop_decrease=1.0)
    return y_denoised

# Audio dosyasını yükle
file_path = r"C:\Users\bulen\OneDrive\Masaüstü\samples\sample3\sample3_C#m_125bpm.wav"
y, sr = librosa.load(file_path, sr=None)

# Gürültü azalt ve normalleştir
# Gürültü azaltma işlemi geçici olarak kaldırıldı
y = y
y = librosa.util.normalize(y)

# Tempo ve tonalite
tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
if isinstance(tempo, np.ndarray):
    tempo = tempo.item()
print(f"BPM: {int(tempo)}")

tonic, mode = get_key_music21(y, sr)
print(f"Detected Key: {tonic} {mode}")

# Harmonik bileşen ve chroma hesaplama
y_harmonic, y_percussive = librosa.effects.hpss(y)
chroma = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr, hop_length=512)

# Chroma indekslerini nota isimlerine eşleştir
notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# Her notanın genel yoğunluğunu hesapla
note_intensities = np.sum(chroma, axis=1)

# En yoğun 3 notayı bul
top_3_notes_idx = np.argsort(note_intensities)[-3:][::-1]
top_3_notes = [notes[i] for i in top_3_notes_idx]
top_1_note = top_3_notes[0]
print(f"Most intense 3 notes: {top_3_notes}")
print(f"Most intense note: {top_1_note}")

# Her saniye için en yoğun 3 notayı belirle
duration = librosa.get_duration(y=y, sr=sr)
time_bins = np.arange(0, duration, 1.0)
frames_per_second = librosa.time_to_frames(time_bins, sr=sr, hop_length=512)
top_notes_per_second = []

for i in range(len(frames_per_second) - 1):
    start_frame = frames_per_second[i]
    end_frame = frames_per_second[i + 1]
    if end_frame > chroma.shape[1]:
        end_frame = chroma.shape[1]
    chroma_slice = chroma[:, start_frame:end_frame]
    mean_chroma = np.mean(chroma_slice, axis=1)
    top_notes_idx = np.argsort(mean_chroma)[-3:][::-1]
    top_notes = [notes[idx] for idx in top_notes_idx]
    top_notes_per_second.append(top_notes)

# Akor isimlerini belirle
chords_per_second = []
for top_notes in top_notes_per_second:
    chord_name = identify_major_minor_chord(top_notes)
    chords_per_second.append(chord_name)

# Chord Chromagram oluştur
chord_chromagram = np.zeros_like(chroma)
for i in range(len(frames_per_second) - 1):
    start_frame = frames_per_second[i]
    end_frame = frames_per_second[i + 1]
    if end_frame > chroma.shape[1]:
        end_frame = chroma.shape[1]
    top_notes_idx = [notes.index(n) for n in top_notes_per_second[i] if n in notes]
    for idx in top_notes_idx:
        chord_chromagram[idx, start_frame:end_frame] = chroma[idx, start_frame:end_frame]

# Görselleştirme
plt.figure(figsize=(14, 8))

# Subplot 1: SoundCloud tarzı yansımalı Waveform (daha detaylı ve RGB renklerle)
S = np.abs(librosa.stft(y, n_fft=8192, hop_length=128))**2  # Daha detaylı bir çözünürlük için n_fft ve hop_length iyileştirildi  # Daha detaylı bir çözünürlük için n_fft ve hop_length ayarlandı
frequencies = librosa.fft_frequencies(sr=sr, n_fft=8192)

low_freq_range = (frequencies >= 20) & (frequencies < 250)
mid_freq_range = (frequencies >= 250) & (frequencies < 4000)
high_freq_range = (frequencies >= 4000)

low_energy = np.sum(S[low_freq_range, :], axis=0)
mid_energy = np.sum(S[mid_freq_range, :], axis=0)
high_energy = np.sum(S[high_freq_range, :], axis=0)

# Normalize
low_energy = low_energy / np.max(low_energy)
mid_energy = mid_energy / np.max(mid_energy)
high_energy = high_energy / np.max(high_energy)

# Zaman ekseni
frames = np.arange(len(low_energy))
time = librosa.frames_to_time(frames, sr=sr, hop_length=256)

plt.subplot(3, 1, 1)
plt.plot(time, low_energy, color='red', alpha=0.8, label='Low Frequencies(bass,kick)')
plt.plot(time, mid_energy, color='blue', alpha=0.8, label='Mid Frequencies(melody,vocals)')
plt.plot(time, high_energy, color='green', alpha=0.8, label='High Frequencies(cymbals,atmosphere)')
plt.plot(time, -low_energy, color='red', alpha=0.4)
plt.plot(time, -mid_energy, color='blue', alpha=0.4)
plt.plot(time, -high_energy, color='green', alpha=0.4)
plt.xlabel('Time (s)')
plt.ylabel('Normalized Amplitude')
plt.title(f'SoundCloud-Style RGB Waveform (BPM: {int(tempo)})')
plt.legend()

# Subplot 2: Chroma Spectrogram
plt.subplot(3, 1, 2)
librosa.display.specshow(chroma, x_axis='time', y_axis='chroma')
plt.colorbar(format='%+2.0f dB')
plt.title('Chroma Spectrogram')

# Subplot 3: Chord Chromagram
plt.subplot(3, 1, 3)
librosa.display.specshow(chord_chromagram, x_axis='time', y_axis='chroma', )
plt.colorbar(format='%+2.0f dB')
plt.title('Chord Chromagram (Top 3 Notes per Second)')

plt.tight_layout()
plt.show()

# Sonuçları yazdır
print("\nMost intense 3 notes overall:")
print(top_3_notes)

print("\nMost intense note overall:")
print(top_1_note)

print("\nTop 3 most played notes per second and chords formed:")
for i, (top_notes, chord_name) in enumerate(zip(top_notes_per_second, chords_per_second)):
    time_sec = i
    print(f"Second {time_sec}-{time_sec + 1}: Top notes: {top_notes}, Chord: {chord_name}")
plt.yticks(np.arange(len(notes)), notes)
plt.yticks(np.arange(len(notes)), notes)
