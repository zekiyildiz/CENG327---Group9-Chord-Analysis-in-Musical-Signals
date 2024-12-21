import sys
import os
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QVBoxLayout,
    QFileDialog, QMessageBox, QLabel, QProgressBar
)
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
import librosa
import numpy as np
import librosa.display
import matplotlib.pyplot as plt
import csv
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


class AudioProcessor(QThread):
    finished = pyqtSignal(tuple)
    error = pyqtSignal(str)

    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path

    def run(self):
        try:
            # Audio processing code
            measurements, sample_rate = librosa.load(self.file_path, sr=44100, mono=True)
            a, b = librosa.load(self.file_path, sr=None)
            bpm, _ = librosa.beat.beat_track(y=a, sr=b)

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
            
            # Emit all the data needed for updating the plots
            self.finished.emit((measurements, sample_rate, chroma, chord_changes, 
                              unique_chords, chord_to_y, bpm))
        except Exception as e:
            self.error.emit(str(e))

class App(QWidget):
    def __init__(self):
        super().__init__()
        self.title = 'Chromagram Analysis and Chord Detection'
        self.left = 100
        self.top = 100
        self.width = 1200
        self.height = 800
        self.chord_changes = []
        self.bpm = None
        self.progress_timer = None
        self.audio_processor = None
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        # Set minimum size to prevent window from being too small
        self.setMinimumSize(1000, 700)
        
        # Load Music File Button
        self.button = QPushButton('Load Music File', self)
        self.button.clicked.connect(self.openFileNameDialog)
        layout.addWidget(self.button)

        # Music Name Label
        self.music_label = QLabel("Music Name: --", self)
        music_font = QFont()
        music_font.setBold(True)
        music_font.setPointSize(12)
        self.music_label.setFont(music_font)
        self.music_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.music_label)

        # BPM Label
        self.bpm_label = QLabel("BPM: --", self)
        bpm_font = QFont()
        bpm_font.setBold(True)
        bpm_font.setPointSize(12)
        self.bpm_label.setFont(bpm_font)
        self.bpm_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.bpm_label)

        # Progress Bar
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat('Processing: %p%')
        self.progress_bar.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.progress_bar)
        self.progress_bar.hide()

        # Matplotlib Canvas
        self.canvas = FigureCanvas(plt.Figure(figsize=(10, 8)))
        layout.addWidget(self.canvas)
        self.canvas.hide()

        # Export Button
        self.export_button = QPushButton('Export Chords', self)
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self.export_chords)
        layout.addWidget(self.export_button)

        self.setLayout(layout)
        self.show()

    def start_progress_bar(self):
        self.progress_bar.setValue(0)
        self.progress_bar.show()
        self.progress_timer = QTimer()
        self.progress_timer.timeout.connect(self.update_progress)
        self.progress_timer.start(50)  # Update every 50ms

    def update_progress(self):
        current_value = self.progress_bar.value()
        if current_value < 95:  # Stop at 95% to show we're still processing
            self.progress_bar.setValue(current_value + 1)

    def stop_progress_bar(self):
        if self.progress_timer:
            self.progress_timer.stop()
        self.progress_bar.setValue(100)  # Show 100% when done
        QTimer.singleShot(200, self.progress_bar.hide)  # Hide after a brief delay

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
            # Update UI for processing state
            music_name = os.path.basename(fileName)
            self.music_label.setText(f"Music Name: <b>{music_name}</b>")
            self.button.setEnabled(False)
            self.export_button.setEnabled(False)
            self.canvas.hide()
            
            # Start progress bar
            self.start_progress_bar()
            
            # Start processing in separate thread
            self.audio_processor = AudioProcessor(fileName)
            self.audio_processor.finished.connect(self.on_processing_finished)
            self.audio_processor.error.connect(self.on_processing_error)
            self.audio_processor.start()

    def on_processing_finished(self, result):
        measurements, sample_rate, chroma, chord_changes, unique_chords, chord_to_y, bpm = result
        
        # Update instance variables
        self.chord_changes = chord_changes
        self.bpm = bpm
        self.bpm_label.setText(f"BPM: <b>{int(self.bpm)}</b>")
        
        # Update plots
        self.update_plots(measurements, sample_rate, chroma, chord_changes, unique_chords, chord_to_y)
        
        # Reset UI state
        self.stop_progress_bar()
        self.button.setEnabled(True)
        self.export_button.setEnabled(True)
        self.canvas.show()

    def on_processing_error(self, error_message):
        self.stop_progress_bar()
        self.button.setEnabled(True)
        QMessageBox.critical(self, "Error", f"An error occurred: {error_message}")

            

    def update_plots(self, measurements, sample_rate, chroma, chord_changes, unique_chords, chord_to_y):
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
            
            
    def process_audio(self, file_path):
        try:
            # Your existing audio processing code here
            measurements, sample_rate = librosa.load(file_path, sr=44100, mono=True)
            a, b = librosa.load(file_path, sr=None)
            self.bpm, _ = librosa.beat.beat_track(y=a, sr=b)
            self.bpm_label.setText(f"BPM: <b>{int(self.bpm)}</b>")

            hop_length = 512
            chroma = librosa.feature.chroma_cqt(y=measurements, sr=sample_rate, hop_length=hop_length)
            chroma = chroma / (np.linalg.norm(chroma, axis=0, keepdims=True) + 1e-9)
            chroma = np.apply_along_axis(
                lambda x: np.convolve(x, np.ones(5) / 5, mode='same'),
                axis=1,
                arr=chroma
            )
            frames_per_second = sample_rate / hop_length
            frames_per_beat = int(frames_per_second * (60.0 / self.bpm))
            detected_chords = chord_detection(chroma, frames_per_beat)

            times = librosa.frames_to_time(np.arange(chroma.shape[1]), sr=sample_rate)
            self.chord_changes = []
            if detected_chords:
                current_chord = detected_chords[0]
                start_time = times[0]
                for i in range(1, len(detected_chords)):
                    if detected_chords[i] != current_chord:
                        end_time = times[i]
                        self.chord_changes.append((current_chord, start_time, end_time))
                        current_chord = detected_chords[i]
                        start_time = times[i]
                end_time = times[-1]
                self.chord_changes.append((current_chord, start_time, end_time))

            self.update_plots(
                measurements, sample_rate, chroma, self.chord_changes,
                list(dict.fromkeys(detected_chords)),
                {ch: i for i, ch in enumerate(list(dict.fromkeys(detected_chords)))}
            )

            # Processing complete
            self.stop_progress_bar()
            self.button.setEnabled(True)
            self.export_button.setEnabled(True)
            self.canvas.show()

        except Exception as e:
            self.stop_progress_bar()
            self.button.setEnabled(True)
            QMessageBox.critical(self, "Error", f"An error occurred: {e}")
    
    
    def export_chords(self):
        if not self.chord_changes:
            QMessageBox.warning(self, "No Data", "No chord data to export. Please load and process a music file first.")
            return

        options = QFileDialog.Options()
        save_file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Chord Data",
            "chords",
            "CSV Files (*.csv);;Text Files (*.txt);;All Files (*)",
            options=options
        )

        if save_file_path:
            try:
                ext = os.path.splitext(save_file_path)[1]
                with open(save_file_path, mode='w', newline='') as file:
                    if ext == ".csv":
                        writer = csv.writer(file)
                        writer.writerow([f"BPM: {int(self.bpm)}"])
                        writer.writerow(["Start Time", "End Time", "Chord"])
                        for chord, start, end in self.chord_changes:
                            writer.writerow([start, end, chord])
                    else:
                        file.write(f"BPM: {int(self.bpm)}\n")
                        for chord, start, end in self.chord_changes:
                            file.write(f"{start} - {end}: {chord}\n")

                QMessageBox.information(self, "Export Successful", f"Chord data saved to {save_file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"An error occurred while exporting: {e}")


# Run
if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setApplicationName('Chord Recognition Tool')
    ex = App()
    sys.exit(app.exec_())