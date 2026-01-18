import os, glob
import numpy as np
import soundfile as sf

# Noise floor in dB
NOISE_FLOOR = -60

# Trim silence from start?
TRIM_START = True

# Trim silence from end?
TRIM_END = True

# Padding to apply around detected edge of silence
PADDING = 32

# Target length for all audio files in seconds
TARGET_LENGTH_SECONDS = 8

epsilon = 10**(NOISE_FLOOR/20)

def process(filename, target_length_samples):
    # Load the wav file
    data, samplerate = sf.read(filename, always_2d=True)
    # Find indices of samples above noise floor
    indices = (np.abs(data) >= epsilon).any(axis=1).nonzero()[0]
    # Trim silence (if it found any non-silent samples)
    if len(indices) > 1:
        # Find start and end index to trim (+/- padding)
        start = max(0, indices[0] - PADDING) if TRIM_START else 0
        end = min(len(data), indices[-1] + PADDING) if TRIM_END else len(data)
        trim = data[start: end]
    else:
        trim = data
    
    # Ensure trimmed audio is the target length
    if len(trim) < target_length_samples:
        # Pad with zeros if shorter
        padding_needed = target_length_samples - len(trim)
        padding = np.zeros((padding_needed, trim.shape[1]))
        trimmed_padded = np.concatenate((trim, padding), axis=0)
    else:
        # Crop if longer
        trimmed_padded = trim[:target_length_samples]
    
    # Write trimmed and resized wave to subfolder called 'trimmed' with the same file name
    sf.write(os.path.join('trimmed', os.path.basename(filename)), trimmed_padded, samplerate)


if __name__ == "__main__":
    # Prepare output directory
    os.makedirs('trimmed', exist_ok=True)
    
    # Find and process all wav files
    wav_files = glob.glob("*.wav")
    
    # Set target length in samples
    target_length_samples = None
    for filename in wav_files:
        # Check the sample rate of the first file
        data, samplerate = sf.read(filename, always_2d=True)
        target_length_samples = int(TARGET_LENGTH_SECONDS * samplerate)
        break
    
    for i, filename in enumerate(wav_files):
        print(f"{i}/{len(wav_files)}: {filename}")
        try:
            process(filename, target_length_samples)
        except Exception as ex:
            print("Trimming failed!")
            print(ex)
