import numpy as np
from light_training.dataloading.dataset import get_train_test_loader_from_test_list
import torch 
import torch.nn as nn 
from monai.inferers import SlidingWindowInferer
from light_training.evaluation.metric import dice
from light_training.trainer import Trainer
from monai.utils import set_determinism
set_determinism(123)
import os
from light_training.prediction import Predictor

data_dir = "./data/fullres/train"
env = "pytorch"
max_epoch = 1000
batch_size = 2
val_every = 2
num_gpus = 1
device = "cuda:0"
patch_size = [128, 128, 128]


class BraTSTrainer(Trainer):
    def __init__(self, env_type, max_epochs, batch_size, device="cpu", val_every=1, num_gpus=1, logdir="./logs/", master_ip='localhost', master_port=17750, training_script="train.py"):
        super().__init__(env_type, max_epochs, batch_size, device, val_every, num_gpus, logdir, master_ip, master_port, training_script)
        
        self.patch_size = patch_size

    # for image, label in data_loader:
    def get_input(self, batch):
        image = batch["data"]
        label = batch["seg"]
        properties = batch["properties"]
        # print(f"properties is {properties}")
        # label = self.convert_labels(label)
        label = label[:, 0].long()

        return image, label, properties 

    def define_model_diffunet(self):
        from diffunet.diffunet_model import DiffUNet
        model = DiffUNet(1, 2)
        model_path = "/data/xingzhaohu/aiib23/logs/diffunet_final/model/final_model_0.9460.pt"
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

        save_path = "./prediction_results/diffunet"
        os.makedirs(save_path, exist_ok=True)

        return model, predictor, save_path
   
    
    def validation_step(self, batch):
        image, label, properties = self.get_input(batch)
        
        # model, predictor, save_path = self.define_model_segmambav2()
        # model, predictor, save_path = self.define_model_nnunet2d()
        # model, predictor, save_path = self.define_model_diffunet()
        model, predictor, save_path = self.define_model_diffseg3d()
        # model, predictor, save_path = self.define_model_umamba()

        model_output = predictor.maybe_mirror_and_predict(image, model, device=device)

        model_output = predictor.predict_raw_probability(model_output, 
                                                         properties=properties)
        
        # print(f"model_output is {model_output.shape}")
        # model_output = model_output > 0
        model_output = model_output.argmax(dim=0)

        print(f"noncrop start ...")
        model_output = predictor.predict_noncrop_probability(model_output, properties)
        print(f"noncrop end ...")

        print(f"save start ...")

        predictor.save_to_nii(model_output, 
                              raw_spacing=[1,1,1],
                              case_name = properties['name'][0],
                              save_dir=save_path,
                              postprocess=False)
        
        print(f"save end ...")

        return 0

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

    from test_list_aiib23 import test_list
    train_ds, test_ds = get_train_test_loader_from_test_list(data_dir=data_dir, test_list=test_list)

    trainer.validation_single_gpu(test_ds)

    # print(f"result is {v_mean}")


