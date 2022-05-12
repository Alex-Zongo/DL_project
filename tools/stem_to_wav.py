"""
Output .wav files containing 5 channels
- `0` - The mixture,
- `1` - The drums,
- `2` - The bass,
- `3` - The rest of the accompaniment,
- `4` - The vocals.
"""
import os
import subprocess
import tempfile

import librosa
import numpy as np
import soundfile as sf


def converter(origin_dataset_path, new_dataset_path, sr):
    if os.path.exists(new_dataset_path):
        raise FileExistsError(f'{new_dataset_path} already exists.')
    else:
        os.mkdir(new_dataset_path)

    os.mkdir(os.path.join(new_dataset_path, 'train'))
    os.mkdir(os.path.join(new_dataset_path, 'test'))

    with tempfile.TemporaryDirectory() as tmpdir:
        for subdir in ('train', 'test'):
            origin_path = os.path.join(origin_dataset_path, subdir)
            files = [f for f in os.listdir(origin_path) if os.path.splitext(f)[1]=='.mp4']
            for file in files:
                path = os.path.join(origin_path, file)
                name = os.path.splitext(file)[0]
                wav_data = []
                for ch in range(5):
                    temp_f = f'{name}.{ch}.wav'
                    out_path = os.path.join(tmpdir, temp_f)
                    subprocess.run(['ffmpeg', '-i', path, '-map', f'0:{ch}', out_path])
                    sound, _ = librosa.load(out_path, sr=sr, mono=True)
                    wav_data.append(sound)
                wav_data = np.stack(wav_data, axis=1)
                out_path = os.path.join(
                    new_dataset_path, subdir, f'{name}.wav'
                )
                sf.write(out_path, wav_data, sr)

