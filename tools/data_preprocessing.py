import os
import glob
import subprocess
import tempfile

import numpy as np
import musdb
from tools.utils import write_wav, load
import librosa


# get the dataset
def get_data_hq(database_path):
    '''
    Retrieve audio file paths for MUSDB HQ dataset
    :param database_path: MUSDB HQ root directory
    :return: dictionary with train and test keys, each containing list of samples, each sample containing all audio paths
    '''
    subsets = list()

    for subset in ["train", "test"]:
        print("Loading " + subset + " set...")
        tracks = glob.glob(os.path.join(database_path, subset, "*"))
        samples = list()
        print(tracks)

        # Go through tracks
        for track_folder in sorted(tracks):
            # Skip track if mixture is already written, assuming this track is done already
            example = dict()
            for stem in ["mix", "bass", "drums", "other", "vocals"]:
                filename = stem if stem != "mix" else "mixture"
                audio_path = os.path.join(track_folder, filename + ".wav")
                example[stem] = audio_path

            # Add other instruments to form accompaniment
            acc_path = os.path.join(track_folder, "accompaniment.wav")

            if not os.path.exists(acc_path):
                print("Writing accompaniment to " + track_folder)
                stem_audio = []
                for stem in ["bass", "drums", "other"]:
                    audio, sr = load(example[stem], sr=None, mono=False)
                    stem_audio.append(audio)
                acc_audio = np.clip(sum(stem_audio), -1.0, 1.0)
                write_wav(acc_path, acc_audio, sr)

            example["accompaniment"] = acc_path

            samples.append(example)

        subsets.append(samples)

    return subsets

def get_data(data_path):
    """
    return the audio file paths for the dataset
    :param data_path:
    :return: dictionary with train and test keys
    """

    subsets = list()
    stems = {0: "mix", 1: "drums", 2: "bass", 3: "other", 4: "vocals"}
    with tempfile.TemporaryDirectory() as tmpdir:
        for subdir in ["train", "test"]:
            tracks_dir = os.path.join(data_path, subdir)
            tracks = [t for t in os.listdir(tracks_dir) if os.path.splitext(t)[1] == '.mp4']
            samples = list()
            for track in sorted(tracks):
                track_path = os.path.join(tracks_dir, track)
                track_name = os.path.splitext(track)[0]

                mix_track = f'{track_name}.mix.wav'
                acc_track = f'{track_name}.accompaniment.wav'
                if os.path.exists(mix_track):
                    print("WARNING: Skipping track " + mix_track + " since it exists already")

                    # Add paths and then skip
                    paths = {"mix": mix_track, "accompaniment": acc_track}
                    paths.update({key: os.path.join(data_path, f'{track_name}.{key}.wav') for key in ["bass", "drums", "other", "vocals"]})
                    samples.append(paths)

                    continue

                paths = dict()
                stem_audio = dict()
                for ch in range(5):
                    temp_f = f'{track_name}.{stems[ch]}.wav'
                    out_path = os.path.join(tmpdir, temp_f)
                    subprocess.run(['ffmpeg', '-i', track_path, '-map', f'0:{ch}', out_path])
                    audio, sr = load(out_path, sr=None, mono=False)
                    # if stems[ch] != "vocals":
                    #     acc_audio.append(audio)
                    out_path = os.path.join(data_path, subdir, temp_f)
                    write_wav(out_path, audio, sr)
                    stem_audio[stems[ch]] = audio
                    paths[stems[ch]] = out_path

                acc_audio = np.clip(sum([stem_audio[key] for key in list(stem_audio.keys()) if key != "vocals"]), -1.0, 1.0)
                acc_file = f'{track_name}.accompaniment.wav'
                acc_path = os.path.join(data_path, subdir, acc_file)
                write_wav(acc_path, acc_audio, sr)
                paths["accompaniment"] = acc_path

                # mixture
                mix_audio = stem_audio["mix"]

                # check if acc + vocal = mix
                diff_signal = np.abs(mix_audio - acc_audio - stem_audio["vocals"])
                print("Maximum absolute deviation from source additivity constraint: " + str(
                    np.max(diff_signal)))
                print("Mean absolute deviation from source additivity constraint: " + str(np.mean(diff_signal)))

                samples.append(paths)
                print(len(samples))
            subsets.append(samples)

    print("Dataset Preparation COMPLETE")
    return subsets


def get_data_folds(root_path, cat="hq"):
    if cat == "hq":
        dataset = get_data_hq(root_path)
    else:
        dataset = get_data(root_path)
    train_val_list = dataset[0]
    print(len(train_val_list))
    test_list = dataset[1]

    np.random.seed(1337)
    train_list = np.random.choice(train_val_list, 75, replace=False)
    val_list = [elem for elem in train_val_list if elem not in train_list]

    return {"train": train_list, "val": val_list, "test": test_list}


# if __name__ == '__main__':
#     path = "dataset"
#     print(get_data_folds(path))
#     # dataset = musdb.DB(root=path, download=True, is_wav=True)
#     # train_tracks = glob.glob(os.path.join(path, "train", "*"))
#
#     # train_data = dataset.load_mus_tracks(subsets="train")
#     # print(len(train_data))
#


