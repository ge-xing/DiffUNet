from .nnunet3d_denoise import get_nnunet3d_denoise
from .nnunet3d import get_nnunet3d
import torch.nn as nn 

class DiffUNet(nn.Module):
    def __init__(self, in_channels, out_channels, 
                 ddim_steps=3, rand_steps=1, bta=True, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.edge_model = get_nnunet3d(in_chans=in_channels, out_chans=out_channels)
        self.denoise_model = get_nnunet3d_denoise(in_chans=in_channels, out_chans=out_channels, 
                                          ddim_step=ddim_steps,
                                          rand_step=rand_steps,
                                          bta=bta)

    # def forward(self, image, gt=None, ddim=False):
    #     pred_edge, embeddings = self.edge_model(image)

    #     if ddim:
    #         pred = self.denoise_model(image, gt=gt, 
    #                                     embeddings=embeddings, 
    #                                     ddim=True)
    #         return pred + pred_edge
    #     else :
    #         pred, uncertainty = self.denoise_model(image, gt=gt, 
    #                                     embeddings=embeddings, 
    #                                     ddim=False)
    #         return pred, pred_edge, uncertainty

    def forward(self, image, gt=None, ddim=False):
        pred_edge, embeddings = self.edge_model(image)

        pred = self.denoise_model(image, gt=gt, 
                                        embeddings=embeddings, 
                                        ddim=True)
        return pred + pred_edge
