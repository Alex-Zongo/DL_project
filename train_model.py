import os
import pickle
import time
from functools import partial
import numpy as np
import torch.cuda
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import utils
from model.utils import compute_loss, save_model, load_model
from model.my_model import Alexunet
from test import validate, evaluate
from tools.data_loader import SeparationDataset
from tools.data_preprocessing import get_data_folds
from tools.stem_to_wav import converter
from tools.utils import crop_targets, random_amplify
from utils import worker_init_fn

if __name__ == '__main__':
    torch.cuda.empty_cache()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset_dir = "dataset"
    checkpoint_dir = "checkpoints/alexunet_gpu_depth1_level6_res_learned_lstm_with_conv"

    # waveunet_gpu ==> depth=1 and levels=2 load from ckp2025
    # hdf => waveunet_gpu
    # HYPER-PARAMETERS
    sr = 44100
    instruments = ["bass", "drums", "other", "vocals"]
    features = 32  # number of feature channels per layer
    channels = 2
    kernel_size = 5
    batch_size = 4
    cycles = 2  # Number of LR cycles per epoch
    depth = 1  # number of convolution per block
    strides = 4
    output_size = 2.0  # output duration
    feature_growth = "double"  # double/add
    levels = 6  # number of DS/US blocks
    # conv_type = "gn"  # (normal, BN-normalised, GN-normalised): normal/bn/gn"
    res = "learned"  # resampling strategy ("fixed" or "learned")
    separate = 1  # train separate model for each source (1) or only one (0)
    sample_freq = 100  # Write an audio summary into Tensorboard logs every X training iterations
    num_features = [features * i for i in range(1, levels + 1)] if feature_growth == "add" else \
        [features * 2 ** i for i in range(0, levels)]
    target_outputs = int(output_size * sr)
    num_workers = 4

    # MODEL
    model = Alexunet(channels, num_features, channels, instruments, kernel_size, target_output_size=target_outputs,
                     depth=depth, strides=strides, res=res, separate=separate)
    model = model.to(device)

    log_dir = './runs/alexunet_gpu_depth1_level6_res_learned_lstm_with_conv'
    writer = SummaryWriter(log_dir)

    # DATASET
    hdf_dir = "alex_hdf_depth1_level6_"
    dataset = get_data_folds(root_path=dataset_dir)
    crop_func = partial(crop_targets, shapes=model.shapes)
    augment_func = partial(random_amplify, shapes=model.shapes, min=0.7, max=1.0)
    training_data = SeparationDataset(dataset, "train", instruments, sr, channels, model.shapes, random_hops=True,
                                      hdf_dir=hdf_dir, audio_transform=augment_func)
    val_data = SeparationDataset(dataset, "val", instruments, sr, channels, model.shapes, random_hops=False,
                                 hdf_dir=hdf_dir, audio_transform=crop_func)
    test_data = SeparationDataset(dataset, "test", instruments, sr, channels, model.shapes, random_hops=False,
                                  hdf_dir=hdf_dir, audio_transform=crop_func)

    data_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                             worker_init_fn=worker_init_fn)

    example, _ = next(iter(data_loader))
    print('model', model)
    print('parameter count:', str(sum(p.numel() for p in model.parameters())))
    # writer.add_graph(model, [example, instruments], True)
    print(example, example.shape)

    # #### Training #####
    # loss function
    loss = "L2"  # MSE loss
    criterion = nn.MSELoss()

    # Optimizer
    lr = 1e-3
    min_lr = 5e-5
    optimizer = Adam(model.parameters(), lr=lr)

    # # training state dict
    state = {
        "step": 0,
        "epoch": 0,
        "worse_epoch": 0,
        "best_loss": np.Inf
    }
    #
    # Load model from checkpoint and continue training
    is_load_model = "load"
    if is_load_model is not None:
        print("Continuing training full model from checkpoint " + str(load_model))
        state = load_model(model, optimizer, "checkpoints/alexunet_gpu_depth1_level6_res_learned_lstm_with_conv/checkpoint_3375", device)

    patience = 20
    while state["worse_epoch"] < patience:
        print("Training one epoch from iteration " + str(state["step"]))
        avg_time = 0.0
        model.train()
        with tqdm(total=len(training_data) // batch_size) as pbar:
            np.random.seed()
            for sample_id, (x, targets) in enumerate(data_loader):
                x = x.to(device)
                for k in list(targets.keys()):
                    targets[k] = targets[k].to(device)
                t = time.time()

                # set lr
                utils.set_cyclic_lr(optimizer, sample_id, len(training_data) // batch_size, cycles, min_lr, lr)
                writer.add_scalar("lr", utils.get_lr(optimizer), state["step"])

                # compute loss for each instrument
                optimizer.zero_grad()
                outputs, avg_loss = compute_loss(model, x, targets, criterion, compute_grad=True)

                optimizer.step()

                state["step"] += 1

                t = time.time() - t
                avg_time += (1. / float(sample_id + 1)) * (t - avg_time)

                writer.add_scalar("train_loss", avg_loss, state["step"])

                if sample_id % sample_freq == 0:
                    input_centre = torch.mean(
                        x[0, :, model.shapes["output_start_frame"]:model.shapes["output_end_frame"]],
                        0)  # Stereo not supported for logs yet
                    writer.add_audio("input", input_centre, state["step"], sample_rate=sr)

                    for inst in outputs.keys():
                        writer.add_audio(inst + "_pred", torch.mean(outputs[inst][0], 0), state["step"],
                                         sample_rate=sr)
                        writer.add_audio(inst + "_target", torch.mean(targets[inst][0], 0), state["step"],
                                         sample_rate=sr)

                pbar.update(1)
        print("Average training time: ", avg_time)
        # validate
        val_loss = validate(batch_size, num_workers, device, model, criterion, val_data)
        print("VALIDATION FINISHED: LOSS: " + str(val_loss))
        writer.add_scalar("val_loss", val_loss, state["step"])

        # EARLY STOPPING CHECK
        checkpoint_path = os.path.join(checkpoint_dir, "checkpoint_" + str(state["step"]))
        if val_loss >= state["best_loss"]:
            state["worse_epoch"] += 1
        else:
            print("MODEL IMPROVED ON VALIDATION SET!")
            state["worse_epoch"] = 0
            state["best_loss"] = val_loss
            state["best_checkpoint"] = checkpoint_path

        state["epoch"] += 1
        # CHECKPOINT
        print("Saving model...")
        save_model(model, optimizer, state, checkpoint_path)

    # TODO
    #### TESTING ####
    # Test loss
    print("TESTING")

    # Load best model based on validation loss
    state = load_model(model, None, state["best_checkpoint"], device)
    test_loss = validate(batch_size, num_workers, device, model, criterion, test_data)
    print("TEST FINISHED: LOSS: " + str(test_loss))
    writer.add_scalar("test_loss", test_loss, state["step"])

    # Mir_eval metrics
    test_metrics = evaluate(channels, sr, dataset["test"], model, instruments)

    # Dump all metrics results into pickle file for later analysis if needed
    with open(os.path.join(checkpoint_dir, "results.pkl"), "wb") as f:
        pickle.dump(test_metrics, f)

    # Write most important metrics into Tensorboard log
    avg_SDRs = {inst: np.mean([np.nanmean(song[inst]["SDR"]) for song in test_metrics]) for inst in
                instruments}
    avg_SIRs = {inst: np.mean([np.nanmean(song[inst]["SIR"]) for song in test_metrics]) for inst in
                instruments}
    for inst in instruments:
        writer.add_scalar("test_SDR_" + inst, avg_SDRs[inst], state["step"])
        writer.add_scalar("test_SIR_" + inst, avg_SIRs[inst], state["step"])
    overall_SDR = np.mean([v for v in avg_SDRs.values()])
    writer.add_scalar("test_SDR", overall_SDR)
    print("SDR: " + str(overall_SDR))

    writer.close()
