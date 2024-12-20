# Audio Trim

Python script that slices audio with silence detection


This is a highly modified version of openvpi/audio-slicer made for my own personal use

---

## Modifications done:

- Made quicker workflow for end user by making it to where you just place ```trim.py```  in a folder of audio files and execute by ``` cd \yourpath\audio\wav trim.py ```.
- Improved script by allowing the removal of silence from all .wav files in a folder instead of one at a time.
-  Output Directory: Created a "trimmed" folder within the current working directory to store output files.
- CLI Integration: Added argparse for command-line argument parsing.
- Silence Detection: Modified the Slicer class to only trim silence without adding a "_trimmed" suffix.
- Audio Length Enforcement: Introduced a function to pad or truncate audio to a fixed length (default: 8 seconds).
- No Filename Changes: Ensured output filenames remain unchanged (no "_trimmed" suffix).

## What I use this script for:
I am a music producer and sample all 88 keys of the analog synths I own, but my audio interface has about 3ms latency. Using this script enables me to trim all of the latency so that I can create kontakt libraries and sampler VSTs out of my analog synth samples.

## Original source code:
(https://github.com/openvpi/audio-slicer)

## Algorithm:
Silence detection:
This script uses RMS (root mean score) to measure the quiteness of the audio and detect silent parts. RMS values of each frame (frame length set as hop size) are calculated and all frames with an RMS below the threshold will be regarded as silent frames.

Audio slicing: 
Once silent audio is detected, it is removed from the beginning and end of each file.

## Requirements:

```bash
pip install numpy soundfile librosa
```

or

```shell
pip install -r requirements.txt
```
## Usage

```bash
cd \yourpath\audio\wav trim.py
```
