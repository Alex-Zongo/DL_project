import torch
import torch.nn as nn
import torch.nn.functional as F
from model.transformer import SingleTransformer
from model.crop import centre_crop


class ConvLayer(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, conv_type, transpose=False, with_relu=True):
        super(ConvLayer, self).__init__()
        self.transpose = transpose
        self.stride = stride
        self.kernel_size = kernel_size
        self.conv_type = conv_type
        self.with_relu = with_relu

        # How many channels should be normalised as one group if GroupNorm is activated
        # WARNING: Number of channels has to be divisible by this number!
        NORM_CHANNELS = 8

        if self.transpose:
            self.filter = nn.ConvTranspose1d(n_inputs, n_outputs, self.kernel_size, stride, padding=kernel_size - 1)
        else:
            self.filter = nn.Conv1d(n_inputs, n_outputs, self.kernel_size, stride)

        # if not self.with_relu:
        #     self.filter = nn.Conv1d(n_inputs, n_outputs, self.kernel_size, stride)

        if conv_type == "gn":
            assert (n_outputs % NORM_CHANNELS == 0)
            self.norm = nn.GroupNorm(n_outputs // NORM_CHANNELS, n_outputs)
        elif conv_type == "bn":
            self.norm = nn.BatchNorm1d(n_outputs, momentum=0.01)
        # Add you own types of variations here!

    def forward(self, x):
        # Apply the convolution
        if self.with_relu:
            if self.conv_type == "gn" or self.conv_type == "bn":
                out = F.relu(self.norm((self.filter(x))))
            else:  # Add your own variations here with elifs conditioned on "conv_type" parameter!
                assert (self.conv_type == "normal")
                out = F.leaky_relu(self.filter(x))
        else:
            out = self.filter(x)
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


class Encoder(nn.Module):
    def __init__(self,  in_ch, out_ch):
        super(Encoder, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.k = 8
        self.stride = 4
        self.lstm = nn.LSTM(input_size=out_ch, hidden_size=out_ch, bidirectional=True, num_layers=1, batch_first=True)
        # TODO => replace lstm with transformers
        self.pre_shortcut = nn.ModuleList([
            ConvLayer(in_ch, out_ch, self.k, self.stride, conv_type="gn", transpose=False),
            ConvLayer(out_ch, 2 * out_ch, 1, 1, conv_type="gn", transpose=False),
            nn.GLU(dim=1)
        ]
        )
        self.post_shortcut = nn.Sequential(
            nn.Conv1d(2*out_ch, out_ch, 1, 1),
            nn.ReLU()
        )

    def forward(self, x):
        #  shortcut
        shortcut = x
        for layer in self.pre_shortcut:
            shortcut = layer(shortcut)  # shortcut size = [B, out_ch, T'] same as output

        # out no down-sampling
        output = shortcut  # size: [B, out_ch, T]
        output = self.lstm(output.transpose(1, 2))[0]  # size: [B, L, 2*out_ch]
        output = self.post_shortcut(output.transpose(1, 2))

        return output, shortcut

    def get_input_size(self, output_size):
        curr_size = output_size
        curr_size = (curr_size-1)*self.stride + self.k
        assert(curr_size > 0)
        return curr_size

    def get_output_size(self, input_size):
        curr_size = input_size
        curr_size = (curr_size-self.k)//2 + 1
        return curr_size


class Decoder(nn.Module):
    def __init__(self, in_ch, out_ch, with_relu=False):
        super(Decoder, self).__init__()
        self.in_ch, self.out_ch = in_ch, out_ch
        self.k1, self.k2 = 3, 8
        self.stride1, self.stride2 = 1, 4

        self.pre_shortcut = ConvLayer(in_ch, in_ch, 3, 1, conv_type="gn", transpose=False)
        # TODO => add transformer
        self.post_shortcut = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(2*in_ch, 2*in_ch, 1, 1),
                nn.GLU(dim=1)
            ),
            ConvLayer(in_ch, out_ch, 8, 4, conv_type="gn", transpose=True, with_relu=with_relu)
        ])

    def forward(self, x, shortcut):
        out = self.pre_shortcut(x)

        cropped = centre_crop(shortcut, out)

        combined = torch.cat([out, cropped], dim=1)
        out = combined
        for layer in self.post_shortcut:
            out = layer(out)
        return out

    def get_input_size(self, output_size):
        curr_size = output_size
        curr_size = self.post_shortcut[-1].get_input_size(curr_size)
        curr_size = (curr_size-1)*self.stride1 + self.k1
        return curr_size

    def get_output_size(self, input_size):
        curr_size = self.pre_shortcut.get_output_size(input_size)
        curr_size = self.post_shortcut[-1].get_output_size(curr_size)
        return curr_size


# bottleneck=lstm or transformer encoder (SingleTransformer)
class Bottlenecks(nn.Module):
    def __init__(self, in_ch, hidden_size, num_layers=2):
        super(Bottlenecks, self).__init__()
        self.in_ch = in_ch
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=in_ch, hidden_size=hidden_size, bidirectional=True, num_layers=self.num_layers,
                            batch_first=True)
        self.conv = nn.Conv1d(in_channels=2 * hidden_size, out_channels=hidden_size, kernel_size=1, stride=1)

    def forward(self, x):
        out = x.transpose(1, 2)
        out = self.lstm(out)[0].transpose(1, 2)
        out = F.relu(self.conv(out))
        return out

    def get_output_size(self, input_size):
        curr_size = input_size
        return curr_size

    def get_input_size(self, output_size):
        return output_size

# channels = [2, 48] + [48*2**i for i in range(1, 6)]
# 2, 48, 96, 192, 384, 768, 1536


# n_in = n_out = 2
# num_levels = 6
class AlexDemucs(nn.Module):
    def __init__(self, n_in, channel_features, n_out, target_output_size, num_levels, separate, instruments):
        super(AlexDemucs, self).__init__()
        self.n_in, self.n_out = n_in, n_out
        assert (n_in == n_out)
        self.instruments = instruments
        self.channel_features = channel_features
        self.target_output = target_output_size
        self.num_levels = num_levels
        self.separate = separate

        self.separator = nn.ModuleDict()
        model_list = instruments if separate else ["ALL"]
        for instrument in model_list:
            module = nn.Module()

            module.encoders = nn.ModuleList()
            module.decoders = nn.ModuleList()
            for level in range(num_levels):
                module.encoders.append(
                    Encoder(in_ch=channel_features[level], out_ch=channel_features[level+1])
                )

            for level in range(num_levels-1):
                module.decoders.append(
                    Decoder(in_ch=channel_features[-1-level], out_ch=channel_features[-2-level])
                )

            outputs = self.n_out if separate else self.n_out*len(instruments)
            module.decoders.append(
                Decoder(in_ch=channel_features[1], out_ch=outputs, with_relu=False)
            )

            # module.bottlenecks = nn.ModuleList(
            #     [Bottlenecks(in_ch=channel_features[-1], hidden_size=channel_features[-1], num_layers=3)]
            # )
            module.bottlenecks = nn.ModuleList(
                [SingleTransformer(input_size=channel_features[-1], hidden_size=channel_features[-1], dropout=0.5)]
            )
            self.separator[instrument] = module

        self.set_output_size(target_output_size)

    def forward_module(self, x, module):
        '''
        A forward pass through a single Wave-U-Net (multiple Wave-U-Nets might be used, one for each source)
        :param x: Input mix
        :param module: Network module to be used for prediction
        :return: Source estimates
        '''
        shortcuts = []
        out = x

        # Encoding BLOCKS
        for block in module.encoders:
            out, short = block(out)
            shortcuts.append(short)

        # BOTTLENECKS
        for layer in module.bottlenecks:
            out = layer(out)

        # Decoder BLOCKS and output
        for idx, block in enumerate(module.decoders):
            out = block(out, shortcuts[-1 - idx])

        # # OUTPUT CONV
        # out = module.output_conv(out)
        if not self.training:  # At test time clip predictions to valid amplitude range
            out = out.clamp(min=-1.0, max=1.0)
        return out

    def forward(self, x, inst=None):
        curr_input_size = x.shape[-1]
        assert(curr_input_size == self.input_size) # User promises to feed the proper input himself, to get the pre-calculated (NOT the originally desired) output size

        if self.separate:
            return {inst: self.forward_module(x, self.separator[inst])}
        else:
            assert(len(self.separator) == 1)
            out = self.forward_module(x, self.separator["ALL"])

            out_dict = {}
            for idx, inst in enumerate(self.instruments):
                out_dict[inst] = out[:, idx * self.n_out:(idx + 1) * self.n_out]
            return out_dict

    def set_output_size(self, target_output_size):
        self.target_output_size = target_output_size

        self.input_size, self.output_size = self.check_padding(target_output_size)
        print("Using valid convolutions and operations with " + str(self.input_size) + " inputs and " + str(self.output_size) + " outputs")

        assert((self.input_size - self.output_size) % 2 == 0)
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
        module = self.separator[[k for k in self.separator.keys()][0]]
        try:
            curr_size = bottleneck
            for idx, block in enumerate(module.decoders):
                curr_size = block.get_output_size(curr_size)
            output_size = curr_size

            # Bottleneck-Conv
            curr_size = bottleneck
            for block in reversed(module.bottlenecks):
                curr_size = block.get_input_size(curr_size)
            for idx, block in enumerate(reversed(module.encoders)):
                curr_size = block.get_input_size(curr_size)

            assert(output_size >= target_output_size)
            return curr_size, output_size
        except AssertionError as e:
            return False

