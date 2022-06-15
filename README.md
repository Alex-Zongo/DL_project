# DL_project
Deep Learning Class final project
Title: Music Source Separation
In this project we implemented and modified the existing Wave-U-Net architecture and trained it on MusDB18-HQ dataset.
The overall SDR exceeded the performance of Wave-U-Net under similar settings. 
This improvement is imputed to the improved version of transformer encoder implemented in DPTNet.

The model architecture can be found in the file "my_model.py".
To train and test the model, simply download the dataset and store it in "dataset_hq" folder, then run the file "train_model_hq.py".

NB: You can change the hyper-parameters or others customizable settings in the training file as you wish.
During training, the evaluation and performance of your model can be observed on tensorboard.

After training, to separate your own track, look inside the file "separate.py" and make sure the path to your checkpoint is correct
as well as the hyper-parameters of your model. Then run it accordingly.


References:

https://github.com/f90/Wave-U-Net-Pytorch

https://github.com/ujscjj/DPTNet

papers consulted:

[1] Jingjing Chen, Qirong Mao, and Dong Liu. Dual-path trans- former network: Direct context-aware modeling for end-to- end monaural speech separation, 2020. 

[2] Alexandre De ́fossez, Nicolas Usunier, Le ́on Bottou, and Francis R. Bach. Music source separation in the waveform domain. CoRR, abs/1911.13254, 2019.

[3] Alexandre De ́fossez. Hybrid spectrogram and waveform source separation, 2021. 1, 4, 5

[4] Zafar Rafii, Antoine Liutkus, Fabian-Robert Sto ̈ter, Stylianos Ioannis Mimilakis, and Rachel Bittner. MUSDB18-HQ - an uncompressed version of musdb18, Dec. 2019. 

[5] Zafar Rafii, Antoine Liutkus, Fabian-Robert Sto ̈ter, Stylianos Ioannis Mimilakis, and Rachel M. Bittner. Musdb18 - a corpus for music separation. 2017. 

[6] Xuchen Song, Qiuqiang Kong, Xingjian Du, and Yuxuan Wang. Catnet: music source separation system with mix- audio augmentation, 2021. 

[7] Daniel Stoller, Sebastian Ewert, and Simon Dixon. Wave- u-net: A multi-scale neural network for end-to-end audio source separation. 2018. 

[8] Fabian-Robert Sto ̈ter, Antoine Liutkus, and Nobutaka Ito. The 2018 signal separation evaluation campaign. ArXiv, abs/1804.06267, 2018.

[9] Naoya Takahashi, Nabarun Goswami, and Yuki Mitsufuji. Mmdenselstm: An efficient combination of convolutional and recurrent neural networks for audio source separation, 2018.

[10] E. Vincent, R. Gribonval, and C. Fevotte. Performance measurement in blind audio source separation. IEEE Transactions on Audio, Speech, and Language Processing, 14(4):1462–1469, 2006.
