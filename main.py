import torch
import musdb

if __name__ == '__main__':
    # Download dataset
    mus = musdb.DB(root='./dataset_wav', download=True, is_wav=True)
