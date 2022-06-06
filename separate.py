from model.my_model import Alexunet
from tools.utils import write_wav
from model.utils import load_model
from test import predict_song
import os
import torch
import argparse


def main(args):
    # parameters
    checkpoint_dir = "checkpoints/alexunet_gpu_depth1_level6_res_fixed_transformer_oneModel_hq/checkpoint_31175"
    sr = 44100
    instruments = ["bass", "drums", "other", "vocals"]
    features = 32  # number of feature channels per layer
    channels = 2
    kernel_size = 5
    strides = 4
    output_size = 3.0  # output duration
    feature_growth = "double"  # double/add
    levels = 6  # number of DS/US blocks
    res = "fixed"  # resampling strategy ("fixed" or "learned")
    separate = 0  # train separate model for each source (1) or only one (0)
    sample_freq = 200  # Write an audio summary into Tensorboard logs every X training iterations
    num_features = [features * i for i in range(1, levels + 1)] if feature_growth == "add" else \
        [features * 2 ** i for i in range(0, levels)]
    target_outputs = int(output_size * sr)
    num_workers = 4
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = Alexunet(channels, num_features, channels, instruments, kernel_size, target_output_size=target_outputs,
                     strides=strides, res=res, separate=separate)

    if device == "cuda":
        model = model.to(device)

    print("Loading model from checkpoint " + str(checkpoint_dir))
    state = load_model(model, None, checkpoint_dir, device)
    print('Step', state['step'])

    preds = predict_song(channels, sr, args.input, model)

    output_folder = os.path.dirname(args.input) if args.output is None else args.output
    for inst in preds.keys():
        write_wav(os.path.join(output_folder, os.path.basename(args.input) + "_" + inst + ".wav"),
                  preds[inst], sr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help="Path to the input mixture to be separated")
    parser.add_argument('--output', type=str, default=None, help="Output path (same as the input if not set)")

    args = parser.parse_args()

    main(args)
