import numpy as np
import torch 
import torch.nn as nn 
from monai.inferers import SlidingWindowInferer
from light_training.evaluation.metric import dice
from light_training.trainer import Trainer
from light_training.utils.files_helper import save_new_model_and_delete_last

import os

def func(m, epochs):
    return np.exp(-10*(1- m / epochs)**2)

import torch

data_dir = "./data/fullres/train"

fold = 0

logdir = "./logs/diffunet"

env = "pytorch"
model_save_path = os.path.join(logdir, "model")
max_epoch = 1000
batch_size = 2
val_every = 2
num_gpus = 1
device = "cuda:0"
patch_size = [128, 128, 128]
augmentation = True 

class BraTSTrainer(Trainer):
    def __init__(self, env_type, max_epochs, batch_size, device="cpu", val_every=1, num_gpus=1, logdir="./logs/", master_ip='localhost', master_port=17750, training_script="train.py"):
        super().__init__(env_type, max_epochs, batch_size, device, val_every, num_gpus, logdir, master_ip, master_port, training_script)
        self.window_infer = SlidingWindowInferer(roi_size=patch_size,
                                        sw_batch_size=2,
                                        overlap=0.5)
        self.patch_size = patch_size
        self.augmentation = augmentation
        self.train_process = 12

        from diffunet.diffunet_model import DiffUNet
        self.model = DiffUNet(1, 2)

        self.best_mean_dice = 0.0
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-2, weight_decay=3e-5,
                                    momentum=0.99, nesterov=True)
        self.scheduler_type = "poly"
        
        self.loss_func = nn.CrossEntropyLoss()

    def training_step(self, batch):
        import time 
        image, label = self.get_input(batch)

        pred, pred_edge, uncertainty = self.model(image, label)
        uncertainty = torch.clamp(uncertainty, 0.0, 1.0)

        loss = self.loss_func(pred, label)
        loss_edge = self.loss_func(pred_edge, label)

        scale = func(self.epoch, max_epoch)

        loss = loss.mean() + loss_edge.mean() + (loss * uncertainty).mean() * scale

        self.log("training_loss", loss.mean(), step=self.global_step)
        self.log("training_loss_edge", loss_edge.mean(), step=self.global_step)

        return loss 

    def get_input(self, batch):
        image = batch["data"]
        label = batch["seg"]

        label = label[:, 0].long()

        return image, label 

    def cal_metric(self, gt, pred, voxel_spacing=[1.0, 1.0, 1.0]):
        if pred.sum() > 0 and gt.sum() > 0:
            d = dice(pred, gt)
            # hd95 = metric.binary.hd95(pred, gt)
            return np.array([d, 50])
        
        elif gt.sum() == 0 and pred.sum() == 0:
            return np.array([1.0, 50])
        
        else:
            return np.array([0.0, 50])
    
    def validation_step(self, batch):
        image, label = self.get_input(batch)
       
        # output, _ = self.model(image)
        output = self.model(image, ddim=True)

        output = output.argmax(dim=1)

        output = output.cpu().numpy()
        target = label.cpu().numpy()

        cal_dice, _ = self.cal_metric(output, target)
        
        return cal_dice
    
    def validation_end(self, val_outputs):
        dices = val_outputs

        d = dices.mean()

        print(f"dices is {d}")
        
        self.log("d", d, step=self.epoch)


        if d > self.best_mean_dice:
            self.best_mean_dice = d
            save_new_model_and_delete_last(self.model, 
                                            os.path.join(model_save_path, 
                                            f"best_model_{d:.4f}.pt"), 
                                            delete_symbol="best_model")

        save_new_model_and_delete_last(self.model, 
                                        os.path.join(model_save_path, 
                                        f"final_model_{d:.4f}.pt"), 
                                        delete_symbol="final_model")


        if (self.epoch + 1) % 100 == 0:
            torch.save(self.model.state_dict(), os.path.join(model_save_path, f"tmp_model_ep{self.epoch}_{d:.4f}.pt"))

if __name__ == "__main__":

    trainer = BraTSTrainer(env_type=env,
                            max_epochs=max_epoch,
                            batch_size=batch_size,
                            device=device,
                            logdir=logdir,
                            val_every=val_every,
                            num_gpus=num_gpus,
                            master_port=17752,
                            training_script=__file__)
    
    from light_training.dataloading.dataset import get_train_test_loader_from_test_list
    from test_list_aiib23 import test_list
    
    train_ds, test_ds = get_train_test_loader_from_test_list(data_dir=data_dir, test_list=test_list)

    trainer.train(train_dataset=train_ds, val_dataset=test_ds)
