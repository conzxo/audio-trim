import os
import numpy as np
import librosa
import soundfile as sf

# Function to calculate RMS (Root Mean Square)
def get_rms(y, *, frame_length=2048, hop_length=512, pad_mode="constant"):
    padding = (int(frame_length // 2), int(frame_length // 2))
    y = np.pad(y, padding, mode=pad_mode)

    axis = -1
    out_strides = y.strides + tuple([y.strides[axis]])
    x_shape_trimmed = list(y.shape)
    x_shape_trimmed[axis] -= frame_length - 1
    out_shape = tuple(x_shape_trimmed) + tuple([frame_length])
    xw = np.lib.stride_tricks.as_strided(y, shape=out_shape, strides=out_strides)

    if axis < 0:
        target_axis = axis - 1
    else:
        target_axis = axis + 1

    xw = np.moveaxis(xw, -1, target_axis)
    slices = [slice(None)] * xw.ndim
    slices[axis] = slice(0, None, hop_length)
    x = xw[tuple(slices)]

    power = np.mean(np.abs(x) ** 2, axis=-2, keepdims=True)
    return np.sqrt(power)

# Slicer class to trim leading and trailing silence
class Slicer:
    def __init__(self, sr: int, threshold: float = -40., hop_size: int = 20):
        self.threshold = 10 ** (threshold / 20.)
        self.hop_size = round(sr * hop_size / 1000)

    def _apply_slice(self, waveform, begin, end):
        if len(waveform.shape) > 1:
            return waveform[:, begin * self.hop_size: end * self.hop_size]
        else:
            return waveform[begin * self.hop_size: end * self.hop_size]

    def slice(self, waveform):
        if len(waveform.shape) > 1:
            samples = waveform.mean(axis=0)
        else:
            samples = waveform

        # Calculate RMS to detect silent frames
        rms_list = get_rms(y=samples, frame_length=2048, hop_length=self.hop_size).squeeze(0)

        # Detect leading and trailing silence
        non_silent_frames = np.where(rms_list >= self.threshold)[0]
        if len(non_silent_frames) == 0:
            return waveform, True  # Return original if the entire file is silent

        start_frame = non_silent_frames[0]  # First non-silent frame
        end_frame = non_silent_frames[-1] + 1  # Last non-silent frame

        trimmed_audio = self._apply_slice(waveform, start_frame, end_frame)

        return trimmed_audio, False

# Function to pad or truncate audio to the desired length
def pad_or_truncate(audio, sr, target_length_sec=8):
    target_length = target_length_sec * sr
    if len(audio) > target_length:
        # Truncate if longer than target length
        return audio[:target_length]
    else:
        # Pad with silence if shorter than target length
        padding = target_length - len(audio)
        if len(audio.shape) > 1:
            return np.pad(audio, ((0, 0), (0, padding)), mode='constant')
        else:
            return np.pad(audio, (0, padding), mode='constant')

# Main function to process all WAV files in the current directory
def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--db_thresh', type=float, default=-40, help='The dB threshold for silence detection')
    parser.add_argument('--hop_size', type=int, default=10, help='Frame length in milliseconds')
    parser.add_argument('--duration', type=int, default=8, help='Target length of each audio clip in seconds')

    args = parser.parse_args()

    # Get the current working directory
    input_folder = os.getcwd()

    # Create the 'trimmed' folder inside the current directory
    output_folder = os.path.join(input_folder, 'trimmed')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Loop through all WAV files in the current directory
    for filename in os.listdir(input_folder):
        if filename.endswith('.wav'):
            file_path = os.path.join(input_folder, filename)
            print(f'Processing {file_path}...')
            audio, sr = librosa.load(file_path, sr=None, mono=False)
            slicer = Slicer(
                sr=sr,
                threshold=args.db_thresh,
                hop_size=args.hop_size,
            )
            trimmed_audio, _ = slicer.slice(audio)

            # Ensure the audio is exactly the specified duration (default: 8 seconds)
            final_audio = pad_or_truncate(trimmed_audio, sr, target_length_sec=args.duration)

            # Save the processed audio without adding "_trimmed" suffix to the filename
            output_filename = f'{filename}'
            output_path = os.path.join(output_folder, output_filename)

            print(f"Saving '{output_filename}' with a forced duration of {args.duration} seconds.")

            sf.write(output_path, final_audio.T if len(final_audio.shape) > 1 else final_audio, sr)

if __name__ == '__main__':
    main()
