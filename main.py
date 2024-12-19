import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QVBoxLayout,
    QFileDialog, QMessageBox, QLabel
)
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt
import librosa
import numpy as np
import librosa.display
import matplotlib.pyplot as plt
from collections import Counter
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

# Define chord templates
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

# Chord detection function
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


# Main Application Class
class App(QWidget):
    def __init__(self):
        super().__init__()
        self.title = 'Chromagram Analysis and Chord Detection'
        self.left = 100
        self.top = 100
        self.width = 900
        self.height = 700
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        layout = QVBoxLayout()

        # Load Music File Button
        self.button = QPushButton('Load Music File', self)
        self.button.clicked.connect(self.openFileNameDialog)
        layout.addWidget(self.button)

        # Music Name Label
        self.music_label = QLabel("Music Name: --", self)
        music_font = QFont()
        music_font.setBold(True)
        music_font.setPointSize(12)  # Adjust font size as needed
        self.music_label.setFont(music_font)
        self.music_label.setAlignment(Qt.AlignCenter)  # Center the text
        layout.addWidget(self.music_label)

        # BPM Label
        self.bpm_label = QLabel("BPM: --", self)
        bpm_font = QFont()
        bpm_font.setBold(True)
        bpm_font.setPointSize(12)  # Adjust font size as needed
        self.bpm_label.setFont(bpm_font)
        self.bpm_label.setAlignment(Qt.AlignCenter)  # Center the text
        layout.addWidget(self.bpm_label)

        # Matplotlib Canvas for Plotting
        self.canvas = FigureCanvas(plt.Figure(figsize=(10, 8)))
        layout.addWidget(self.canvas)

        self.setLayout(layout)
        self.show()

    def openFileNameDialog(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(
            self,
            "Select Music File",
            "",
            "WAV Files (*.wav);;All Files (*)",
            options=options
        )
        if fileName:
            self.process_audio(fileName)

    def process_audio(self, file_path):
        try:
            import os
            music_name = os.path.basename(file_path)
            self.music_label.setText(f"Music Name: <b>{music_name}</b>")

            # Load audio file
            measurements, sample_rate = librosa.load(file_path, sr=44100, mono=True)

            # Load again for BPM detection with original sample rate
            a, b = librosa.load(file_path, sr=None)
            bpm, _ = librosa.beat.beat_track(y=a, sr=b)
            bpm = bpm.item() if isinstance(bpm, np.ndarray) else bpm
            self.bpm_label.setText(f"BPM: <b>{int(bpm)}</b>")

            # Chromagram Calculation
            hop_length = 512
            chroma = librosa.feature.chroma_cqt(y=measurements, sr=sample_rate, hop_length=hop_length)
            chroma = chroma / (np.linalg.norm(chroma, axis=0, keepdims=True) + 1e-9)
            chroma = np.apply_along_axis(
                lambda x: np.convolve(x, np.ones(5) / 5, mode='same'),
                axis=1,
                arr=chroma
            )
            frames_per_second = sample_rate / hop_length
            frames_per_beat = int(frames_per_second * (60.0 / bpm))
            detected_chords = chord_detection(chroma, frames_per_beat)

            # Time Conversion for Chords
            times = librosa.frames_to_time(np.arange(chroma.shape[1]), sr=sample_rate)
            chord_changes = []
            if detected_chords:
                current_chord = detected_chords[0]
                start_time = times[0]
                for i in range(1, len(detected_chords)):
                    if detected_chords[i] != current_chord:
                        end_time = times[i]
                        chord_changes.append((current_chord, start_time, end_time))
                        current_chord = detected_chords[i]
                        start_time = times[i]
                end_time = times[-1]
                chord_changes.append((current_chord, start_time, end_time))

            unique_chords = list(dict.fromkeys(detected_chords))
            chord_to_y = {ch: i for i, ch in enumerate(unique_chords)}

            # Clear the previous plot
            self.canvas.figure.clf()

            # Plot waveform
            ax1 = self.canvas.figure.add_subplot(3, 1, 1)
            S = np.abs(librosa.stft(measurements, n_fft=8192, hop_length=128)) ** 2
            frequencies = librosa.fft_frequencies(sr=sample_rate, n_fft=8192)

            low_freq_range = (frequencies >= 20) & (frequencies < 250)
            mid_freq_range = (frequencies >= 250) & (frequencies < 4000)
            high_freq_range = (frequencies >= 4000)

            low_energy = np.sum(S[low_freq_range, :], axis=0)
            mid_energy = np.sum(S[mid_freq_range, :], axis=0)
            high_energy = np.sum(S[high_freq_range, :], axis=0)

            # Normalize energies
            low_energy /= np.max(low_energy) if np.max(low_energy) > 0 else 1
            mid_energy /= np.max(mid_energy) if np.max(mid_energy) > 0 else 1
            high_energy /= np.max(high_energy) if np.max(high_energy) > 0 else 1

            time = librosa.frames_to_time(np.arange(len(low_energy)), sr=sample_rate, hop_length=128)

            ax1.plot(time, low_energy, color='red', alpha=0.8, label='Low Frequencies (Bass, Kick)')
            ax1.plot(time, mid_energy, color='blue', alpha=0.8, label='Mid Frequencies (Melody, Vocal)')
            ax1.plot(time, high_energy, color='green', alpha=0.8, label='High Frequencies (Cymbals, Atmosphere)')
            ax1.plot(time, -low_energy, color='red', alpha=0.4)
            ax1.plot(time, -mid_energy, color='blue', alpha=0.4)
            ax1.plot(time, -high_energy, color='green', alpha=0.4)
            ax1.set_xlabel('Time (s)')
            ax1.set_ylabel('Normalized Amplitude')
            ax1.set_title('RGB Waveform')
            ax1.legend()

            # Plot Chromagram
            ax2 = self.canvas.figure.add_subplot(3, 1, 2, sharex=ax1)
            librosa.display.specshow(
                chroma,
                x_axis='time',
                y_axis='chroma',
                sr=sample_rate,
                cmap='coolwarm',
                ax=ax2
            )
            ax2.set_title('Chroma Representation')
            ax2.set_yticks(np.arange(12))
            ax2.set_yticklabels(['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'])

            # Plot Chord Segments
            ax3 = self.canvas.figure.add_subplot(3, 1, 3, sharex=ax1)
            cmap = plt.get_cmap('tab20')
            for chord, start, end in chord_changes:
                y_pos = chord_to_y[chord]
                duration = end - start
                color = cmap(y_pos % cmap.N)
                ax3.barh(y_pos, duration, left=start, height=0.6, color=color, edgecolor='black')

            ax3.set_yticks(range(len(unique_chords)))
            ax3.set_yticklabels(unique_chords)
            ax3.set_xlabel('Time (s)')
            ax3.set_ylabel('Chord')
            ax3.set_title('Detected Chords as Horizontal Bars')
            ax3.invert_yaxis()

            # Adjust layout
            self.canvas.figure.tight_layout()
            self.canvas.draw()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred: {e}")


# Run
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())
