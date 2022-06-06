import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.crop import centre_crop
from model.transformer import SingleTransformer

# changed version of convLayer
class ConvLayer(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, conv_type, transpose=False):
        super(ConvLayer, self).__init__()
        self.transpose = transpose
        self.stride = stride
        self.kernel_size = kernel_size
        self.conv_type = conv_type  # encoder/decoder

        # How many channels should be normalised as one group if GroupNorm is activated
        # WARNING: Number of channels has to be divisible by this number!
        NORM_CHANNELS = 8
        assert (n_outputs % NORM_CHANNELS == 0)
        self.norm = nn.GroupNorm(n_outputs // NORM_CHANNELS, n_outputs)

        # if self.transpose:
        #     self.filter = nn.ConvTranspose1d(n_inputs, n_outputs, self.kernel_size, stride, padding=kernel_size-1)
        # else:
        #     self.filter = nn.Conv1d(n_inputs, n_outputs, self.kernel_size, stride)

        if conv_type == "encoder" or conv_type == "bottleneck":
            self.transpose = False
            self.filter = nn.Conv1d(n_inputs, n_outputs, self.kernel_size, stride)
            self.conv2 = nn.Conv1d(n_outputs, 2 * n_outputs, kernel_size=1, stride=1)
            self.activation = nn.GLU(dim=1)
        elif conv_type == "decoder":
            self.transpose = True
            self.filter = nn.ConvTranspose1d(n_inputs, n_outputs, self.kernel_size, stride, padding=kernel_size - 1)

    def forward(self, x):
        # Apply the convolution
        # if self.conv_type == "gn" or self.conv_type == "bn":
        #     out = F.relu(self.norm((self.filter(x))))
        # Add your own variations here with elif conditioned on "conv_type" parameter!
        if self.conv_type == "encoder" or self.conv_type == "bottleneck":
            out = self.activation(self.conv2(F.relu(self.norm(self.filter(x)))))
        elif self.conv_type == "decoder":
            out = F.relu(self.norm(self.filter(x)))
        return out

    def get_input_size(self, output_size):
        # Strided conv/decimation
        if not self.transpose:
            curr_size = (output_size - 1) * self.stride + 1  # o = (i-1)//s + 1 => i = (o - 1)*s + 1
        else:
            curr_size = output_size

        # Conv
        curr_size = curr_size + self.kernel_size - 1  # o = i + p - k + 1

        # Transposed
        if self.transpose:
            assert ((curr_size - 1) % self.stride == 0)  # We need to have a value at the beginning and end
            curr_size = ((curr_size - 1) // self.stride) + 1
        assert (curr_size > 0)
        return curr_size

    def get_output_size(self, input_size):
        # Transposed
        if self.transpose:
            assert (input_size > 1)
            curr_size = (input_size - 1) * self.stride + 1  # o = (i-1)//s + 1 => i = (o - 1)*s + 1
        else:
            curr_size = input_size

        # Conv
        curr_size = curr_size - self.kernel_size + 1  # o = i + p - k + 1
        assert (curr_size > 0)

        # Strided conv/decimation
        if not self.transpose:
            assert ((curr_size - 1) % self.stride == 0)  # We need to have a value at the beginning and end
            curr_size = ((curr_size - 1) // self.stride) + 1

        return curr_size


# TODO idea ==> change resample to synthesis
class Resample1d(nn.Module):
    def __init__(self, channels, kernel_size, stride, transpose=False, padding="reflect", trainable=True):
        '''
        Creates a resampling layer for time series data (using 1D convolution) - (N, C, W) input format
        :param channels: Number of features C at each time-step
        :param kernel_size: Width of sinc-based lowpass-filter (>= 15 recommended for good filtering performance)
        :param stride: Resampling factor (integer)
        :param transpose: False for down-, true for upsampling
        :param padding: Either "reflect" to pad or "valid" to not pad
        :param trainable: Optionally activate this to train the lowpass-filter, starting from the sinc initialisation
        '''
        super(Resample1d, self).__init__()

        self.padding = padding
        self.kernel_size = kernel_size
        self.stride = stride
        self.transpose = transpose
        self.channels = channels

        cutoff = 0.5 / stride

        assert (kernel_size > 2)
        assert ((kernel_size - 1) % 2 == 0)
        assert (padding == "reflect" or padding == "valid")

        filter = build_sinc_filter(kernel_size, cutoff)

        self.filter = torch.nn.Parameter(
            torch.from_numpy(np.repeat(np.reshape(filter, [1, 1, kernel_size]), channels, axis=0)),
            requires_grad=trainable)

    def forward(self, x):
        # Pad here if not using transposed conv
        input_size = x.shape[2]
        if self.padding != "valid":
            num_pad = (self.kernel_size - 1) // 2
            out = F.pad(x, (num_pad, num_pad), mode=self.padding)
        else:
            out = x

        # Low-pass filter (+ 0 insertion if transposed)
        if self.transpose:
            expected_steps = ((input_size - 1) * self.stride + 1)
            if self.padding == "valid":
                expected_steps = expected_steps - self.kernel_size + 1

            out = F.conv_transpose1d(out, self.filter, stride=self.stride, padding=0, groups=self.channels)
            diff_steps = out.shape[2] - expected_steps
            if diff_steps > 0:
                assert (diff_steps % 2 == 0)
                out = out[:, :, diff_steps // 2:-diff_steps // 2]
        else:
            assert (input_size % self.stride == 1)
            out = F.conv1d(out, self.filter, stride=self.stride, padding=0, groups=self.channels)

        return out

    def get_output_size(self, input_size):
        '''
        Returns the output dimensionality (number of timesteps) for a given input size
        :param input_size: Number of input time steps (Scalar, each feature is one-dimensional)
        :return: Output size (scalar)
        '''
        assert (input_size > 1)
        if self.transpose:
            if self.padding == "valid":
                return ((input_size - 1) * self.stride + 1) - self.kernel_size + 1
            else:
                return (input_size - 1) * self.stride + 1
        else:
            assert (input_size % self.stride == 1)  # Want to take first and last sample
            if self.padding == "valid":
                return input_size - self.kernel_size + 1
            else:
                return input_size

    def get_input_size(self, output_size):
        '''
        Returns the input dimensionality (number of time steps) for a given output size
        :param input_size: Number of input time steps (Scalar, each feature is one-dimensional)
        :return: Output size (scalar)
        '''

        # Strided conv/decimation
        if not self.transpose:
            curr_size = (output_size - 1) * self.stride + 1  # o = (i-1)//s + 1 => i = (o - 1)*s + 1
        else:
            curr_size = output_size

        # Conv
        if self.padding == "valid":
            curr_size = curr_size + self.kernel_size - 1  # o = i + p - k + 1

        # Transposed
        if self.transpose:
            assert ((curr_size - 1) % self.stride == 0)  # We need to have a value at the beginning and end
            curr_size = ((curr_size - 1) // self.stride) + 1
        assert (curr_size > 0)
        return curr_size


def build_sinc_filter(kernel_size, cutoff):
    # FOLLOWING https://www.analog.com/media/en/technical-documentation/dsp-book/dsp_book_Ch16.pdf
    # Sinc lowpass filter
    # Build sinc kernel
    assert (kernel_size % 2 == 1)
    M = kernel_size - 1
    filter = np.zeros(kernel_size, dtype=np.float32)
    for i in range(kernel_size):
        if i == M // 2:
            filter[i] = 2 * np.pi * cutoff
        else:
            filter[i] = (np.sin(2 * np.pi * cutoff * (i - M // 2)) / (i - M // 2)) * \
                        (0.42 - 0.5 * np.cos((2 * np.pi * i) / M) + 0.08 * np.cos(4 * np.pi * M))

    filter = filter / np.sum(filter)
    return filter


class Encoder(nn.Module):
    def __init__(self, n_inputs, n_shortcut, n_outputs, kernel_size, stride, res):
        super(Encoder, self).__init__()
        assert (stride > 1)

        self.kernel_size = kernel_size
        self.stride = stride
        self.conv_type = "encoder"

        # CONV 1
        self.pre_shortcut_convs = nn.ModuleList(
            [ConvLayer(n_inputs, n_shortcut, kernel_size, 1, conv_type=self.conv_type)])

        self.post_shortcut_convs = nn.ModuleList(
            [ConvLayer(n_shortcut, n_outputs, kernel_size, 1, conv_type=self.conv_type)])

        # CONV 2 with decimation
        if res == "fixed":
            self.downconv = Resample1d(n_outputs, 21, stride)  # Resampling with fixed-size sync low-pass filter
        else:
            self.downconv = ConvLayer(n_outputs, n_outputs, kernel_size, stride, conv_type=self.conv_type)

    def forward(self, x):
        # PREPARING SHORTCUT FEATURES
        shortcut = x
        for conv in self.pre_shortcut_convs:
            shortcut = conv(shortcut)

        # PREPARING FOR DOWNSAMPLING
        out = shortcut
        for conv in self.post_shortcut_convs:
            out = conv(out)

        # DOWN-SAMPLING
        out = self.downconv(out)

        return out, shortcut

    def get_input_size(self, output_size):
        curr_size = self.downconv.get_input_size(output_size)

        for conv in reversed(self.post_shortcut_convs):
            curr_size = conv.get_input_size(curr_size)

        for conv in reversed(self.pre_shortcut_convs):
            curr_size = conv.get_input_size(curr_size)
        return curr_size


class Decoder(nn.Module):
    def __init__(self, n_inputs, n_shortcut, n_outputs, kernel_size, stride, res):
        super(Decoder, self).__init__()
        assert (stride >= 1)
        self.conv_type = "decoder"
        # CONV 1 for UPSAMPLING
        if res == "fixed":
            self.upconv = Resample1d(n_inputs, 21, stride, transpose=True)
        else:
            self.upconv = ConvLayer(n_inputs, n_inputs, kernel_size, stride, conv_type=self.conv_type, transpose=True)

        self.pre_shortcut_convs = nn.ModuleList(
            [ConvLayer(n_inputs, n_outputs, kernel_size, 1, conv_type=self.conv_type)])

        # CONVS to combine high- with low-level information (from shortcut)
        self.post_shortcut_convs = nn.ModuleList(
            [ConvLayer(n_outputs + n_shortcut, n_outputs, kernel_size, 1, conv_type=self.conv_type)])

    def forward(self, x, shortcut):
        # UPSAMPLE HIGH-LEVEL FEATURES
        upsampled = self.upconv(x)

        for conv in self.pre_shortcut_convs:
            upsampled = conv(upsampled)

        # Prepare shortcut connection
        combined = centre_crop(shortcut, upsampled)

        # Combine high- and low-level features
        for conv in self.post_shortcut_convs:
            combined = conv(torch.cat([combined, centre_crop(upsampled, combined)], dim=1))
        return combined

    def get_output_size(self, input_size):
        curr_size = self.upconv.get_output_size(input_size)

        # Upsampling convs
        for conv in self.pre_shortcut_convs:
            curr_size = conv.get_output_size(curr_size)

        # Combine convolutions
        for conv in self.post_shortcut_convs:
            curr_size = conv.get_output_size(curr_size)

        return curr_size


class Bottleneck(nn.Module):
    def __init__(self, in_chan, hidden_dim, n_layers):
        super(Bottleneck, self).__init__()
        self.in_chan = in_chan
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.lstm = nn.LSTM(in_chan, hidden_size=hidden_dim, bidirectional=True, num_layers=n_layers,
                            batch_first=True)
        self.conv = nn.Conv1d(in_channels=2*hidden_dim, out_channels=hidden_dim, kernel_size=1, stride=1)

    def forward(self, x):
        # X: N, C, T
        x_w = x.permute(0, 2, 1)  # B, T, C
        out = self.conv(self.lstm(x_w)[0].permute(0, 2, 1))  # out: B, hidden, T
        return F.relu(out)  # return [B, hidden_dim, T]

    def get_input_size(self, out_size):
        input_size = out_size
        return input_size

    def get_output_size(self, in_size):
        return in_size


# waveunet(channels, num_features, channels, instruments, kernel_size, target_output_size=target_outputs,
#                      conv_type=conv_type, depth=depth, strides=strides, res=res, separate=separate)
class Alexunet(nn.Module):
    def __init__(self, num_inputs, num_channels, num_outputs, instruments, kernel_size, target_output_size,
                 res, separate=False, strides=4):
        super(Alexunet, self).__init__()

        self.num_levels = len(num_channels)
        self.strides = strides
        self.kernel_size = kernel_size
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        # self.depth = depth
        self.instruments = instruments
        self.separate = separate

        # Only odd filter kernels allowed
        assert (kernel_size % 2 == 1)

        self.alexunets = nn.ModuleDict()

        model_list = instruments if separate else ["ALL"]
        # Create a model for each source if we separate sources separately, otherwise only one (model_list=["ALL"])
        for instrument in model_list:
            module = nn.Module()

            module.encoder_blocks = nn.ModuleList()
            module.decoder_blocks = nn.ModuleList()

            for i in range(self.num_levels - 1):
                in_ch = num_inputs if i == 0 else num_channels[i]

                module.encoder_blocks.append(
                    Encoder(in_ch, num_channels[i], num_channels[i + 1], kernel_size, strides, res))

            for i in range(0, self.num_levels - 1):
                module.decoder_blocks.append(
                    Decoder(num_channels[-1 - i], num_channels[-2 - i], num_channels[-2 - i], kernel_size, strides, res))

            # TODO ==> modify the bottleneck

            # module.bottlenecks = nn.ModuleList(
            #     [ConvLayer(num_channels[-1], num_channels[-1], kernel_size, 1, conv_type="bottleneck") for _ in range(depth)])

            # module.bottlenecks = nn.ModuleList([Bottleneck(num_channels[-1], num_channels[-1], n_layers=3)])

            module.bottlenecks = nn.ModuleList(
                [SingleTransformer(input_size=num_channels[-1], hidden_size=num_channels[-1], dropout=0.2)]
            )


            # Output conv
            outputs = num_outputs if separate else num_outputs * len(instruments)
            module.output_conv = nn.Conv1d(num_channels[0], outputs, 1)

            self.alexunets[instrument] = module

        self.set_output_size(target_output_size)

    def set_output_size(self, target_output_size):
        self.target_output_size = target_output_size

        self.input_size, self.output_size = self.check_padding(target_output_size)
        print("Using valid convolutions with " + str(self.input_size) + " inputs and " + str(
            self.output_size) + " outputs")

        assert ((self.input_size - self.output_size) % 2 == 0)
        self.shapes = {"output_start_frame": (self.input_size - self.output_size) // 2,
                       "output_end_frame": (self.input_size - self.output_size) // 2 + self.output_size,
                       "output_frames": self.output_size,
                       "input_frames": self.input_size}

    def check_padding(self, target_output_size):
        # Ensure number of outputs covers a whole number of cycles so each output in the cycle is weighted equally
        # during training
        bottleneck = 1

        while True:
            out = self.check_padding_for_bottleneck(bottleneck, target_output_size)
            if out is not False:
                return out
            bottleneck += 1

    def check_padding_for_bottleneck(self, bottleneck, target_output_size):
        module = self.alexunets[[k for k in self.alexunets.keys()][0]]
        try:
            curr_size = bottleneck
            for idx, block in enumerate(module.decoder_blocks):
                curr_size = block.get_output_size(curr_size)
            output_size = curr_size

            # Bottleneck-Conv
            curr_size = bottleneck
            for block in reversed(module.bottlenecks):
                curr_size = block.get_input_size(curr_size)
            for idx, block in enumerate(reversed(module.encoder_blocks)):
                curr_size = block.get_input_size(curr_size)

            assert (output_size >= target_output_size)
            return curr_size, output_size
        except AssertionError as e:
            return False

    def forward_module(self, x, module):
        '''
        A forward pass through a single Wave-U-Net (multiple Wave-U-Nets might be used, one for each source)
        :param x: Input mix
        :param module: Network module to be used for prediction
        :return: Source estimates
        '''
        shortcuts = []
        out = x

        # DOWNSAMPLING BLOCKS
        for block in module.encoder_blocks:
            out, short = block(out)
            shortcuts.append(short)

        # BOTTLENECK CONVOLUTION
        for conv in module.bottlenecks:
            out = conv(out)

        # UPSAMPLING BLOCKS
        for idx, block in enumerate(module.decoder_blocks):
            out = block(out, shortcuts[-1 - idx])

        # OUTPUT CONV
        out = module.output_conv(out)
        if not self.training:  # At test time clip predictions to valid amplitude range
            out = out.clamp(min=-1.0, max=1.0)
        return out

    def forward(self, x, inst=None):
        curr_input_size = x.shape[-1]
        assert (
                curr_input_size == self.input_size)  # User promises to feed the proper input himself, to get the pre-calculated (NOT the originally desired) output size

        if self.separate:
            return {inst: self.forward_module(x, self.alexunets[inst])}
        else:
            assert (len(self.alexunets) == 1)
            out = self.forward_module(x, self.alexunets["ALL"])

            out_dict = {}
            for idx, inst in enumerate(self.instruments):
                out_dict[inst] = out[:, idx * self.num_outputs:(idx + 1) * self.num_outputs]
            return out_dict
