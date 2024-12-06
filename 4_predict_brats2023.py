import numpy as np
from light_training.dataloading.dataset import get_train_test_loader_from_test_list
import torch 
import torch.nn as nn 
from monai.networks.nets.basic_unet import BasicUNet
from monai.networks.nets.swin_unetr import SwinUNETR
from monai.inferers import SlidingWindowInferer
from light_training.evaluation.metric import dice
from light_training.trainer import Trainer
from monai.utils import set_determinism
from light_training.utils.files_helper import save_new_model_and_delete_last
from models.uent3d import UNet3D
from monai.networks.nets.segresnet import SegResNet
from models.transbts.TransBTS_downsample8x_skipconnection import TransBTS
from einops import rearrange
from models.modelgenesis.unet3d import UNet3DModelGen
from models.transvw.models.ynet3d import UNet3DTransVW
from monai.networks.nets.basic_unet import BasicUNet
from monai.networks.nets.attentionunet import AttentionUnet
from light_training.loss.compound_losses import DC_and_CE_loss
from light_training.loss.dice import MemoryEfficientSoftDiceLoss
from light_training.evaluation.metric import dice
set_determinism(123)
from light_training.loss.compound_losses import DC_and_CE_loss
import os
from medpy import metric
from light_training.prediction import Predictor


data_dir = "./data/fullres/train"
env = "pytorch"
max_epoch = 1000
batch_size = 2
val_every = 2
num_gpus = 1
device = "cuda:6"
patch_size = [128, 128, 128]

class BraTSTrainer(Trainer):
    def __init__(self, env_type, max_epochs, batch_size, device="cpu", val_every=1, num_gpus=1, logdir="./logs/", master_ip='localhost', master_port=17750, training_script="train.py"):
        super().__init__(env_type, max_epochs, batch_size, device, val_every, num_gpus, logdir, master_ip, master_port, training_script)
        
        self.patch_size = patch_size
        self.augmentation = False
        from models.nnunet3d import get_nnunet3d
        # self.model = get_nnunet3d(4, 3)

        # self.model = SwinUNETR(patch_size, 1, 3, feature_size=48)
        # self.model = UNet2D()
        # self.model = BasicUNet(3, 1, 17)
        # self.load_state_dict(os.path.join(logdir, "model", "final_model_0.8864.pt"))
        # _, model = TransBTS(dataset='brats', _conv_repr=True, _pe_type="learned")
        # self.model = model

        # self.model = SegResNet(3, 32, 1, 17)


    def convert_labels(self, labels):
        ## TC, WT and ET
        result = [(labels == 1) | (labels == 3), (labels == 1) | (labels == 3) | (labels == 2), labels == 3]
        
        return torch.cat(result, dim=1).float()
    
    def get_input(self, batch):
        image = batch["data"]
        label = batch["seg"]
        properties = batch["properties"]
        label = self.convert_labels(label)
        # label = label[:, 0].long()
        
        return image, label, properties 

    def define_model_nnunet(self):
        from models.nnunet3d import get_nnunet3d
        # self.print_time = True
        model = get_nnunet3d(4, 4)
        # model_path = "/home/xingzhaohu/jiuding_code/brats23/logs_gpu4/nnunet/model/final_model_0.8994.pt"
        
        model_path = "/data/xingzhaohu/brats23/logs/nnunet/model/final_model_0.8592.pt"
        # model_path = "/home/xingzhaohu/jiuding_code/brats23/logs_gpu4/nnunet/model/tmp_model_ep799_0.9173.pt"
        # model_path = "/home/xingzhaohu/jiuding_code/brats23/logs_gpu4/nnunet/model/tmp_model_ep299_0.9235.pt"
        # model_path = "/home/xingzhaohu/jiuding_code/brats23/logs_gpu4/nnunet_gpu1/model/final_model_0.9024.pt"
        new_sd = self.filte_state_dict(torch.load(model_path, map_location="cpu"))
        model.load_state_dict(new_sd)
        model.eval()
        window_infer = SlidingWindowInferer(roi_size=patch_size,
                                        sw_batch_size=2,
                                        overlap=0.3,
                                        progress=True,
                                        mode="gaussian")

        predictor = Predictor(window_infer=window_infer,
                              mirror_axes=[0,1,2])
        
        # save_path = "./prediction_results_v2/nnunet_ep899"
        # save_path = "./prediction_results_v2/nnunet_ep299"
        save_path = "./prediction_results/nnunet_gpu1_final_all"
        
        os.makedirs(save_path, exist_ok=True)

        return model, predictor, save_path
    
    # def define_model_segmambav2(self):
    #     from models_segmamba.segmamba_v10_mamba_v2 import define_segmambav2

    #     # model = define_segmambav2("segmambav2-l", 4, 4)
    #     model = define_segmambav2("segmambav2-b", 4, 4)
    #     # model_path = "/data/xingzhaohu/brats23/logs/segmambav2-l_mamba_v2/model/final_model_0.8330.pt"
    #     model_path = "/data/xingzhaohu/brats23/logs/segmambav2-b_mamba_v2/model/final_model_0.8584.pt"
    #     new_sd = self.filte_state_dict(torch.load(model_path, map_location="cpu"))
    #     model.load_state_dict(new_sd)
    #     model.eval()
    #     window_infer = SlidingWindowInferer(roi_size=patch_size,
    #                                     sw_batch_size=2,
    #                                     overlap=0.3,
    #                                     progress=True,
    #                                     mode="gaussian")

    #     predictor = Predictor(window_infer=window_infer,
    #                           mirror_axes=[0,])

    #     # save_path = "./prediction_results/segmambav2-base"
    #     save_path = "./prediction_results/segmambav2-large"
    #     os.makedirs(save_path, exist_ok=True)

    #     return model, predictor, save_path

    def define_model_segmambav2(self):
        # from models_segmamba.segmamba_v10_mamba_v2 import define_segmambav2
        # from models_segmamba.segmambav2_3D_pool_conv_no_gsc import define_segmambav2
        # from models_segmamba.segmambav2_3D_pool_conv_gsc import define_segmambav2
        # from models_segmamba.segmambav2_final_v3_dwconv import define_segmambav2
        # from models_segmamba.segmambav2_final_v4_dwconv import define_segmambav2
        # from models_segmamba.segmambav2_final_v5 import define_segmambav2
        # from models_segmamba.segmambav2_final_v11 import get_umamba_bot_3d
        from models_segmamba.segmambav2_final_v12 import get_umamba_bot_3d
        model = get_umamba_bot_3d(4, 4)
        # model = define_segmambav2("segmambav2-l", 4, 4)
        # model = define_segmambav2("segmambav2-b", 4, 4)
        # model_path = "/data/xingzhaohu/brats23/logs/segmambav2-l_mamba_v2/model/final_model_0.8330.pt"
        # model_path = "/data/xingzhaohu/brats23/logs/segmambav2_final_v2/model/final_model_0.8560.pt"
        # model_path = "/data/xingzhaohu/brats23/logs/segmambav2_final_v11/model/final_model_0.8766.pt"
        model_path = "/data/xingzhaohu/brats23/logs/segmambav2_final_v12/model/final_model_0.8670.pt"
        new_sd = self.filte_state_dict(torch.load(model_path, map_location="cpu"))
        model.load_state_dict(new_sd)
        model.eval()
        window_infer = SlidingWindowInferer(roi_size=patch_size,
                                        sw_batch_size=2,
                                        overlap=0.5,
                                        progress=True,
                                        mode="gaussian")

        predictor = Predictor(window_infer=window_infer,
                              mirror_axes=[0,1,2])

        # save_path = "./prediction_results/segmambav2-base"
        # save_path = "./prediction_results/segmambav2-final_dwconv_aug_v5"
        save_path = "./prediction_results/segmambav2-final_v12"
        os.makedirs(save_path, exist_ok=True)

        return model, predictor, save_path
    
    def define_model_segmambav1(self):
        from models_segmamba.segmamba import SegMamba
        model = SegMamba(4, 4)

        # model_path = "/data/xingzhaohu/brats23/logs/segmambav2-l_mamba_v2/model/final_model_0.8330.pt"
        model_path = "/data/xingzhaohu/brats23/logs/segmambav1_ep100/model/tmp_model_ep99_0.8103.pt"
        new_sd = self.filte_state_dict(torch.load(model_path, map_location="cpu"))
        model.load_state_dict(new_sd)
        model.eval()
        window_infer = SlidingWindowInferer(roi_size=patch_size,
                                        sw_batch_size=2,
                                        overlap=0.0,
                                        progress=True,
                                        mode="gaussian")

        predictor = Predictor(window_infer=window_infer,
                              mirror_axes=None)

        # save_path = "./prediction_results/segmambav2-base"
        # save_path = "./prediction_results/segmambav2-large"
        save_path = "./prediction_results/segmambav1-test"
        # save_path = "./prediction_results/segmambav2-large"
        os.makedirs(save_path, exist_ok=True)

        return model, predictor, save_path

    def define_model_segresnet(self):
        from monai.networks.nets.segresnet import SegResNet
        model = SegResNet(3, 16, 4, 3)
        model_path = "/home/xingzhaohu/jiuding_code/brats23/logs/segresnet_lr1ef2ep1000_bceloss_gpu1/model/final_model_0.9113.pt"
        new_sd = self.filte_state_dict(torch.load(model_path, map_location="cpu"))
        model.load_state_dict(new_sd)
        model.eval()
        window_infer = SlidingWindowInferer(roi_size=patch_size,
                                        sw_batch_size=2,
                                        overlap=0.0,
                                        progress=True,
                                        mode="gaussian")

        predictor = Predictor(window_infer=window_infer,
                              mirror_axes=None)

        save_path = "./prediction_results/segresnet"
        os.makedirs(save_path, exist_ok=True)

        return model, predictor, save_path


    def define_model_swinunetr(self):
        model = SwinUNETR(in_channels=4,
                          out_channels=4,
                          img_size=patch_size,
                          num_heads=[3,6,12,24],
                          drop_rate=0.0)
        # model_path = "/home/xingzhaohu/Liver_2017/logs/swinunetr/model/final_model_0.7214.pt"
        # new_sd = self.filte_state_dict(torch.load(model_path, map_location="cpu"))
        # model.load_state_dict(new_sd)
        model.eval()
        window_infer = SlidingWindowInferer(roi_size=patch_size,
                                        sw_batch_size=2,
                                        overlap=0.0,
                                        progress=True,
                                        mode="gaussian")

        predictor = Predictor(window_infer=window_infer,
                              mirror_axes=None)

        save_path = "./prediction_results/test"
        os.makedirs(save_path, exist_ok=True)

        return model, predictor, save_path
    

    def define_model_unetr(self):
        from monai.networks.nets.unetr import UNETR
        model = UNETR(4, 4, [128, 128, 128])

        model_path = "/data/xingzhaohu/brats23/logs/unetr/model/final_model_0.8367.pt"
        new_sd = self.filte_state_dict(torch.load(model_path, map_location="cpu"))
        model.load_state_dict(new_sd)
        model.eval()
        window_infer = SlidingWindowInferer(roi_size=patch_size,
                                        sw_batch_size=2,
                                        overlap=0.5,
                                        progress=True,
                                        mode="gaussian")

        predictor = Predictor(window_infer=window_infer,
                              mirror_axes=[0,])

        save_path = "./prediction_results/unetr"
        os.makedirs(save_path, exist_ok=True)

        return model, predictor, save_path
    

    def define_model_uxnet(self):
        from models.UXNet_3D.network_backbone import UXNET
        model = UXNET(
                    in_chans=4,
                    out_chans=4,
                    depths=[2, 2, 2, 2],
                    feat_size=[48, 96, 192, 384],
                    drop_path_rate=0,
                    layer_scale_init_value=1e-6,
                    spatial_dims=3)
        # model_path = "/home/xingzhaohu/jiuding_code/brats23/logs/UXNet_ep1000_gpu1_addaug/model/final_model_0.8970.pt"
        # new_sd = self.filte_state_dict(torch.load(model_path, map_location="cpu"))
        # model.load_state_dict(new_sd)
        model.eval()
        window_infer = SlidingWindowInferer(roi_size=patch_size,
                                        sw_batch_size=2,
                                        overlap=0.0,
                                        progress=True,
                                        mode="gaussian")

        predictor = Predictor(window_infer=window_infer,
                              mirror_axes=None)

        save_path = "./prediction_results/uxnet_over_test"
        os.makedirs(save_path, exist_ok=True)

        return model, predictor, save_path
    
    def define_model_umamba(self):
        from models.umamba import get_umamba_bot_3d

        model = get_umamba_bot_3d(4, 4)

        model_path = "/data/xingzhaohu/brats23/logs/umamba/model/final_model_0.8542.pt"
        new_sd = self.filte_state_dict(torch.load(model_path, map_location="cpu"))
        model.load_state_dict(new_sd)
        model.eval()
        window_infer = SlidingWindowInferer(roi_size=patch_size,
                                        sw_batch_size=2,
                                        overlap=0.5,
                                        progress=True,
                                        mode="gaussian")

        predictor = Predictor(window_infer=window_infer,
                              mirror_axes=[0,1,2])

        save_path = "./prediction_results/umamba-aug"
        os.makedirs(save_path, exist_ok=True)

        return model, predictor, save_path
    
    def define_model_diffseg3d(self):
        from models_minor.unet3d import UNetModel

        model = UNetModel(dims=3, in_channels=4,
                        model_channels=32, out_channels=4, 
                        num_res_blocks=1,
                        channel_mult=[1, 2, 3, 4],
                        num_heads=8).to(device)

        model_path = "/data/xingzhaohu/brats23/logs/diffseg_3d/model/final_model_0.8477.pt"
        new_sd = self.filte_state_dict(torch.load(model_path, map_location="cpu"))
        model.load_state_dict(new_sd)
        model.eval()
        window_infer = SlidingWindowInferer(roi_size=patch_size,
                                        sw_batch_size=2,
                                        overlap=0.0,
                                        progress=True,
                                        mode="gaussian")

        predictor = Predictor(window_infer=window_infer,
                              mirror_axes=None)

        save_path = "./prediction_results/diffseg_3d_none"
        os.makedirs(save_path, exist_ok=True)

        return model, predictor, save_path
    

    def define_model_diffunet(self):
        from models_diffunet.nnunet_denoise_ddp_infer.get_unet3d_denoise_uncer_edge import DiffUNet
        model = DiffUNet(4, 4).to(device)
        # model_path = "/data/xingzhaohu/brats23/logs/diffseg_3d/model/final_model_0.8477.pt"
        # new_sd = self.filte_state_dict(torch.load(model_path, map_location="cpu"))
        # model.load_state_dict(new_sd)
        model.eval()
        window_infer = SlidingWindowInferer(roi_size=patch_size,
                                        sw_batch_size=2,
                                        overlap=0.0,
                                        progress=True,
                                        mode="gaussian")

        predictor = Predictor(window_infer=window_infer,
                              mirror_axes=None)

        save_path = "./prediction_results/test"
        os.makedirs(save_path, exist_ok=True)

        return model, predictor, save_path

    def convert_labels_dim0(self, labels):
        ## TC, WT and ET
        result = [(labels == 1) | (labels == 3), (labels == 1) | (labels == 3) | (labels == 2), labels == 3]
        
        return torch.cat(result, dim=0).float()
    
    def validation_step(self, batch):
        image, label, properties = self.get_input(batch)
        # model, predictor, save_path = self.define_model_diffunet_v2()
        # model, predictor, save_path = self.define_model_diffunet_v3()
        ddim = False
        # model, predictor, save_path = self.define_model_nnunet()
        # model, predictor, save_path = self.define_model_segmambav2()
        model, predictor, save_path = self.define_model_diffseg3d()
        
        # model, predictor, save_path = self.define_model_umamba()
        # model, predictor, save_path = self.define_model_segmambav1()
       
        # model, predictor, save_path = self.define_model_diffunet()
        # model, predictor, save_path = self.define_model_unetr()
        # model, predictor, save_path = self.define_model_segresnet()
        # model, predictor, save_path = self.define_model_diffunet_skipconn()
        # model, predictor, save_path = self.define_model_diffunet_new_vae()
        # model, predictor, save_path = self.define_model_diffunet_new_vae_edge_sdm()
        # model, predictor, save_path = self.define_model_segresnet_vae()
        # model, predictor, save_path = self.define_model_modelsgen()
        # model, predictor, save_path = self.define_model_swinunetr_dynamic()
        # model, predictor, save_path = self.define_model_swinunetr()
        # model, predictor, save_path = self.define_model_uxnet()
        # model, predictor, save_path = self.define_model_transvw()
        # model, predictor, save_path = self.define_model_universal_model()
        
        # model, predictor, save_path = self.define_model_unetformer()
        # model, predictor, save_path = self.define_deepunet_v2()

        # model_output = predictor.maybe_mirror_and_predict(image, model, device=device, pred_type="ddim_sample")
        if ddim:
            model_output = predictor.maybe_mirror_and_predict(image, model, device=device, ddim=True)
        else :
            model_output = predictor.maybe_mirror_and_predict(image, model, device=device)

        ## add the visualization for boundary
        # 
        # import matplotlib.pyplot as plt

        # plt_output = model_output.detach().cpu().numpy()[0, :, 80].sum(axis=0)
        # plt.imshow(plt_output, cmap="jet")
        # plt.colorbar()
        # plt.savefig("./test11.png")
        # exit(0)

        model_output = predictor.predict_raw_probability(model_output, 
                                                         properties=properties)
        

        # model_output = model_output > 0
        model_output = model_output.argmax(dim=0)[None]
        model_output = self.convert_labels_dim0(model_output)

        label = label[0]
        c = 3
        dices = []
        for i in range(0, c):
            output_i = model_output[i].cpu().numpy()
            label_i = label[i].cpu().numpy()
            d = dice(output_i, label_i)
            dices.append(d)


        print(dices)

        model_output = predictor.predict_noncrop_probability(model_output, properties)
        predictor.save_to_nii(model_output, 
                              raw_spacing=[1,1,1],
                              case_name = properties['name'][0],
                              save_dir=save_path)
        
        return 0

    def convert_labels_dim0(self, labels):
        ## TC, WT and ET
        result = [(labels == 1) | (labels == 3), (labels == 1) | (labels == 3) | (labels == 2), labels == 3]
        
        return torch.cat(result, dim=0).float()
    

    def filte_state_dict(self, sd):
        if "module" in sd :
            sd = sd["module"]
        new_sd = {}
        for k, v in sd.items():
            k = str(k)
            new_k = k[7:] if k.startswith("module") else k 
            new_sd[new_k] = v 
        del sd 
        return new_sd
    
if __name__ == "__main__":

    trainer = BraTSTrainer(env_type=env,
                            max_epochs=max_epoch,
                            batch_size=batch_size,
                            device=device,
                            logdir="",
                            val_every=val_every,
                            num_gpus=num_gpus,
                            master_port=17751,
                            training_script=__file__)

    from data.test_list import test_list
    train_ds, test_ds = get_train_test_loader_from_test_list(data_dir=data_dir, test_list=test_list)

    trainer.validation_single_gpu(test_ds)

    # print(f"result is {v_mean}")


