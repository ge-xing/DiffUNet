from light_training.dataloading.dataset import get_train_test_loader_from_test_list
from monai.utils import set_determinism

import torch 
import os
import numpy as np
import SimpleITK as sitk
from medpy import metric
import argparse
from tqdm import tqdm 
from scoring_metrics.task1 import evaluation_branch_metrics
set_determinism(123)

parser = argparse.ArgumentParser()

parser.add_argument("--pred_name", required=True, type=str)

results_root = "prediction_results"
# results_root = "prediction_results_v2"
args = parser.parse_args()

pred_name = args.pred_name

def cal_metric(gt, pred, voxel_spacing):
    iou, DLR, DBR, precision, leakages, total_length, detected_num = evaluation_branch_metrics("1", gt, pred)
    
    return np.array([iou, DLR, DBR, precision, leakages, total_length, detected_num])

if __name__ == "__main__":
    data_dir = "./data/fullres/train"
    raw_data_dir = "./data/raw_data/AIIB23_Train_T1/"
    from test_list_aiib23 import test_list
    train_ds, test_ds = get_train_test_loader_from_test_list(data_dir, test_list)
    print(len(test_ds))
    
    all_results = np.zeros((24,7))

    ind = 0
    for batch in tqdm(test_ds, total=len(test_ds)):
        properties = batch["properties"]
        case_name = properties["name"]
        gt_itk = os.path.join(raw_data_dir, "gt", f"{case_name}.nii.gz")
        voxel_spacing = [1, 1, 1]
        gt_itk = sitk.ReadImage(gt_itk)
        gt_array = sitk.GetArrayFromImage(gt_itk).astype(np.int32)
        
        pred_itk = sitk.ReadImage(f"./{results_root}/{pred_name}/{case_name}.nii.gz")
        pred_array = sitk.GetArrayFromImage(pred_itk)
        
        m = cal_metric(gt_array, pred_array, [1, 1, 1])

        print(f"m is {m}")

        all_results[ind] = m
    
        ind += 1

    os.makedirs(f"./{results_root}/result_metrics/", exist_ok=True)
    np.save(f"./{results_root}/result_metrics/{pred_name}.npy", all_results) 
    
    result = np.load(f"./{results_root}/result_metrics/{pred_name}.npy")
    print(result.shape)
    print(result.mean(axis=0))
    print(result.std(axis=0))



