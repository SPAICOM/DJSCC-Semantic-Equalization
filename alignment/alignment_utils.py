import torch
from model import DeepJSCC
import torch.nn.functional as F

def load_deep_jscc(path, snr, c, channel_type):

    state_dict = torch.load(path, map_location=torch.device('cpu'))
    from collections import OrderedDict
    new_state_dict = OrderedDict()

    for k, v in state_dict.items():
        name = k.replace('module.','') # remove `module.`
        new_state_dict[name] = v

    model = DeepJSCC(c=c, channel_type=channel_type, snr=snr)

    model.load_state_dict(new_state_dict)
    model.change_channel(channel_type, snr)

    return model


def get_batch_psnr(images, gts, max_val=255):
    # assumes shape: (B, C, H, W)
    batch_mse = F.mse_loss(images, gts, reduction='none')

    batch_mse = batch_mse.view(batch_mse.shape[0], -1).mean(dim=1) # mean over each image
    psnr = 10 * torch.log10(max_val**2 / batch_mse)
    return psnr  # shape: (B,)