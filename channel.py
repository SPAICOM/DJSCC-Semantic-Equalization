import torch
import torch.nn as nn


class Channel(nn.Module):
    def __init__(self, channel_type='AWGN', snr=20):
        if channel_type not in ['AWGN', 'Rayleigh']:
            raise Exception('Unknown type of channel')
        
        super(Channel, self).__init__()
        self.channel_type = channel_type

        # if constant snr this is single integer/float,
        # otherwise it is a tuple of the snr range
        # distribuited is set as uniform
        self.snr = snr

    def forward(self, z_hat):
        if z_hat.dim() == 3 or z_hat.dim() == 1:
            z_hat = z_hat.unsqueeze(0)
        
        k = z_hat[0].numel()

        if z_hat.dim() == 4:
            sig_pwr = torch.sum(torch.abs(z_hat).square(), dim=(1, 2, 3), keepdim=True) / k

        elif z_hat.dim() == 2:
            sig_pwr = torch.sum(torch.abs(z_hat).square(), dim=1, keepdim=True) / k

        # constant snr
        if isinstance(self.snr, (int, float)):
            noi_pwr = sig_pwr / (10 ** (self.snr / 10))
            noise = torch.randn_like(z_hat) * torch.sqrt(noi_pwr/2)

        # variable snr
        else:
            noi_snr = torch.empty_like(z_hat).uniform_(self.snr[0], self.snr[1])
            noi_pwr = sig_pwr / (10 ** (noi_snr / 10))
            noise = torch.randn_like(z_hat) * torch.sqrt(noi_pwr/2)

        if self.channel_type == 'Rayleigh':
            # hc = torch.randn_like(z_hat)  wrong implement before
            # hc = torch.randn(1, device = z_hat.device) 
            hc = torch.randn(2, device = z_hat.device) 
        
            # clone for in-place operation  
            z_hat = z_hat.clone()
            z_hat[:,:z_hat.size(1)//2] = hc[0] * z_hat[:,:z_hat.size(1)//2]
            z_hat[:,z_hat.size(1)//2:] = hc[1] * z_hat[:,z_hat.size(1)//2:]
            
            # z_hat = hc * z_hat

        return z_hat + noise

    def get_channel(self):
        return self.channel_type, self.snr


if __name__ == '__main__':
    # test
    channel = Channel(channel_type='AWGN', snr=10)
    z_hat = torch.randn(64, 10, 5, 5)
    z_hat = channel(z_hat)
    print(z_hat)

    channel = Channel(channel_type='Rayleigh', snr=10)
    z_hat = torch.randn(10, 5, 5)
    z_hat = channel(z_hat)
    print(z_hat)
