import torch.nn as nn
from channel import Channel
import torch


def a_inv_times_b(a, b):
    """
    Perform in an efficient way the A^{-1}B.
    """
    
    try:
        c = torch.linalg.solve(a, b)

    except RuntimeError as e:
        if 'The input tensor A must have at least 2 dimensions' in str(e):
            if len(a.shape) == 2 and a.shape[0] == 1 and a.shape[1] == 1:
                c = (1 / a) * b

            else:
                raise e
            
        else:
            raise e

    return c


class _LinearAlignment(nn.Module):
    """
    Aligner class that uses a linear layer.

    Used both for least squares and Adam optimization forms.
    """

    def __init__(self, size=None, align_matrix=None):
        super(_LinearAlignment, self).__init__()

        if align_matrix is not None:
            self.align_matrix = nn.Parameter(align_matrix, requires_grad=False)

        else:
            self.align_matrix = nn.Parameter(torch.empty(size, size))
            nn.init.xavier_uniform_(self.align_matrix)

    def forward(self, x):
        # get shape of input
        shape = x.shape

        # flatten input
        x = x.flatten(start_dim=1)

        # apply alignment
        x = x @ self.align_matrix

        # return to original shape
        return x.reshape(shape)


class _MLPAlignment(nn.Module):
    """
    Aligner class that uses a Multi-Layer Perceptron (MLP).
    """

    def __init__(self, input_dim, hidden_dims, output_dim=None, nonlinearity=nn.PReLU):
        super(_MLPAlignment, self).__init__()

        if output_dim is None:
            output_dim = input_dim

        layers = []
        dims = [input_dim] + hidden_dims + [output_dim]

        for i in range(len(dims) - 1):
            linear = nn.Linear(dims[i], dims[i + 1])
            nn.init.xavier_uniform_(linear.weight)
            nn.init.zeros_(linear.bias)
            layers.append(linear)

            if i < len(dims) - 2:
                layers.append(nonlinearity())

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        # get shape of input
        shape = x.shape

        # flatten input
        x = x.flatten(start_dim=1)

        # apply alignment
        x = self.mlp(x)

        # return to original shape
        return x.reshape(shape)
    

class _ConvolutionalAlignment(nn.Module):
    """
    Aligner class that uses one convolutional layer.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(_ConvolutionalAlignment, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2)

        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='leaky_relu')

    def forward(self, x):
        return self.conv(x)


class _TwoConvAlignment(nn.Module):
    """
    Aligner class that uses two convolutional layers.
    """

    def __init__(self, in_channels, hidden_channels, out_channels, kernel_size=3):
        super(_TwoConvAlignment, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size, padding=kernel_size//2)
        self.non_linearity = nn.PReLU()
        self.conv2 = nn.Conv2d(hidden_channels, out_channels, kernel_size, padding=kernel_size//2)

        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.conv2.weight, mode='fan_out', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.conv1(x)
        x = self.non_linearity(x)
        x = self.conv2(x)
        return x
    

class _ZeroShotAlignment(nn.Module):
    """
    Aligner class that uses a zeroshot approach.
    """

    def __init__(self, F_tilde, G_tilde, G, L, mean):
        super(_ZeroShotAlignment, self).__init__()

        self.F_tilde = nn.Parameter(F_tilde, requires_grad=False)
        self.G_tilde = nn.Parameter(G_tilde, requires_grad=False)
        self.G = nn.Parameter(G, requires_grad=False)
        self.L = nn.Parameter(L, requires_grad=False)
        self.mean = nn.Parameter(mean, requires_grad=False)

    def compression(self, input):
        x_hat = input.T

        # go to similarity scores
        x_hat = self.F_tilde @ x_hat

        # prewhitening
        x_hat = a_inv_times_b(self.L, x_hat - self.mean)

        # multiply by F is ignored because it is always 1
        return x_hat
    
    def decompression(self, input):
        # y_hat = input * self.G
        y_hat = input

        # dewhitening
        y_hat = self.L @ y_hat + self.mean

        # go back to image
        y_hat = self.G_tilde @ y_hat

        return y_hat.T


class AlignedDeepJSCC(nn.Module):
    """
    DeepJSCC class that supports aligners.
    """

    def __init__(self, encoder, decoder, aligner, snr, channel_type):
        super(AlignedDeepJSCC, self).__init__()

        # get encoder from model1
        self.encoder = encoder
        self.encoder.requires_grad = False
        
        # setup channel
        self.snr = snr
        if self.snr is not None:
            self.channel = Channel(channel_type, snr)
        else:
            self.channel = None

        # get aligner
        self.aligner = aligner
        if aligner is not None:
            self.aligner.requires_grad = False

        # get decoder from model2
        self.decoder = decoder
        self.decoder.requires_grad = False

        # get forward function
        if type(self.aligner) == _ZeroShotAlignment:
            self.forward = self._forward_zeroshot
        
        else:
            self.forward = self._forward_default

    def _forward_default(self, x):
        """
        Forward function for most aligners.
        """

        z = self.encoder(x)

        if self.channel is not None:
            z = self.channel(z)

        if self.aligner is not None:
            z = self.aligner(z)
        
        x_hat = self.decoder(z)

        return x_hat

    def _forward_zeroshot(self, x):
        """
        Forward function for zeroshot aligner.
        """

        z = self.encoder(x)

        # get shape of input
        shape = z.shape

        # flatten input
        if z.dim() == 3:
            z = z.unsqueeze(0)

        z = z.flatten(start_dim=1)

        # zeroshot compression
        z = self.aligner.compression(z)

        if self.channel is not None:
            z = self.channel(z)

        # zeroshot decompression
        z = self.aligner.decompression(z)

        z = z.reshape(shape)
        
        x_hat = self.decoder(z)

        return x_hat

    def change_channel(self, channel_type='AWGN', snr=None):
        if snr is None:
            self.channel = None
        else:
            self.channel = Channel(channel_type, snr)

    def get_channel(self):
        if hasattr(self, 'channel') and self.channel is not None:
            return self.channel.get_channel()
        
        return None

    def loss(self, prd, gt):
        criterion = nn.MSELoss(reduction='mean')
        loss = criterion(prd, gt)
        
        return loss