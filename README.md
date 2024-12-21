# CENG327 - Chord Analysis in Musical Signals #
This is a chord recognition project, coded in Python for the CENG327 Scientific Computing course. The project utilizes signal processing algorithms and
many Python libraries.
1. Librosa was used for loading and processing sound files and also for signal proccessing operations.
2. Numpy was used for vectorization.
3. Matplotlib was used for visualization of data.
4. PyQt5 was used for making the GUI.

The program can predict simple songs fairly easily and has about %70 accuracy rate for complex songs and songs with lots of noise. It takes a
sound file as input (primarily works with .wav files, but .mp3 files also work) and displays the waveform of frequencies, the energy densities of
notes in a chromagram and lastly, detected chords in horizontal bars according to the corresponding time intervals.

![chordrecognition_result](https://github.com/user-attachments/assets/2fcfb4a7-2ff3-4ebb-8482-70a0d040ae60)

The program also has an export chords option where it can export the detected chords in corresponding time intervals (from start time to end time) in either .txt or .csv
format, in an annotated manner.

![chordrecognition_csv](https://github.com/user-attachments/assets/5f598419-8148-4b2e-a95e-bf0e0d99d479)



