import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from music21 import chord, stream, note
import noisereduce as nr


def identify_major_minor_chord(note_list):
    try:
        # music21 ile akor nesnesi oluştur
        c = chord.Chord(note_list)

        # Akor kalitesini kontrol et
        if c.quality == 'major':
            return f"{c.root().name} Major"
        elif c.quality == 'minor':
            return f"{c.root().name} Minor"
        else:
            return "Unknown"
    except:
        return "Unknown"


def get_key_music21(y, sr):
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F',
             'F#', 'G', 'G#', 'A', 'A#', 'B']

    # Harmonik bileşeni izole et
    y_harmonic, y_percussive = librosa.effects.hpss(y)

    # Chroma feature hesapla
    chroma = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)

    # Enerji eşiği belirleyin ve sadece bu eşiğin üzerindeki notaları seçin
    energy_threshold = 0.3  # Eşik değerini artırdık
    notes_present = [notes[i] for i, energy in enumerate(chroma_mean) if energy > energy_threshold]

    # Çok fazla nota seçilmişse, sadece en yoğun birkaçını alarak süreci sadeleştirin
    max_notes = 5  # İhtiyaca göre artırabilir veya azalt.
    if len(notes_present) > max_notes:
        # En yoğun notaları seçin
        sorted_notes = sorted(zip(notes_present, chroma_mean), key=lambda x: x[1], reverse=True)
        notes_present = [n for n, e in sorted_notes[:max_notes]]

    if not notes_present:
        return "Unknown", "Unknown"

    # music21 stream oluşturun
    s = stream.Stream()
    for n in notes_present:
        s.append(note.Note(n))

    try:
        detected_key = s.analyze('key')
        return detected_key.tonic.name, detected_key.mode
    except:
        return "Unknown", "Unknown"


def reduce_noise(y, sr):
    """
    Gürültüyü azaltmak için noisereduce kütüphanesini kullanır.
    """
    # İlk birkaç saniyeyi gürültü profili olarak kullanın
    noise_clip = y[:int(sr * 0.5)]
    y_denoised = nr.reduce_noise(y=y, sr=sr, y_noise=noise_clip, prop_decrease=1.0)
    return y_denoised


# Audio dosyasını yükle
file_path = 'yoklugundaV2.wav'  # Kendi dosya yolunuzu kullanın
y, sr = librosa.load(file_path, sr=None)

# Gürültüyü azalt
y = reduce_noise(y, sr)

# Normalize et
y = librosa.util.normalize(y)

# Tempo ve beat analizini yap
tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)

# Tempo değerinin skaler olduğunu doğrula
if isinstance(tempo, np.ndarray):
    tempo = tempo.item()

print(f"BPM: {int(tempo)}")

# Tonalite analizi
tonic, mode = get_key_music21(y, sr)
print(f"Detected Key: {tonic} {mode}")

# Harmonik bileşeni izole et ve chroma hesapla
y_harmonic, y_percussive = librosa.effects.hpss(y)
chroma = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr, hop_length=512)  # win_length kaldırıldı

# Chroma indekslerini nota isimlerine eşleştir
notes = ['C', 'C#', 'D', 'D#', 'E', 'F',
         'F#', 'G', 'G#', 'A', 'A#', 'B']

# Her notanın genel yoğunluğunu hesapla
note_intensities = np.sum(chroma, axis=1)

# En yoğun 3 notayı bul
top_3_notes_idx = np.argsort(note_intensities)[-3:][::-1]
top_3_notes = [notes[i] for i in top_3_notes_idx]
print(f"Most intense 3 notes: {top_3_notes}")

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
    top_notes_idx = np.argsort(mean_chroma)[-3:][::-1]  # 3 nota seç
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

# Subplot 1: Waveform
plt.subplot(3, 1, 1)
librosa.display.waveshow(y, sr=sr, alpha=0.8, color='grey')
plt.title(f'Waveform of "{file_path}"', fontsize=16)
plt.xlabel('Time (s)', fontsize=14)
plt.ylabel('Amplitude', fontsize=14)


# Subplot 2: Chord Chromagram
plt.subplot(3, 1, 2)
librosa.display.specshow(chord_chromagram, x_axis='time', y_axis='chroma', cmap='coolwarm', sr=sr, hop_length=512)
cbar = plt.colorbar()
cbar.set_ticks([])
cbar.set_label('Density', fontsize=16)
plt.title('Chord Chromagram (Top 3 Notes per Second)', fontsize=16)



plt.tight_layout()
plt.show()

print("\nMost intense 3 notes overall:")
print(top_3_notes)

print("\nTop 3 most played notes per second and chords formed:")
for i, (top_notes, chord_name) in enumerate(zip(top_notes_per_second, chords_per_second)):
    time_sec = i
    print(f"Second {time_sec}-{time_sec + 1}: Top notes: {top_notes}, Chord: {chord_name}")
