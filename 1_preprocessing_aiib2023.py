
from light_training.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor 

base_dir = "./data/raw_data/AIIB23_Train_T1"
image_dir = "img"
gt_dir = "gt"

def process_train():
    # fullres spacing is [0.5        0.70410156 0.70410156]
    # median_shape is [602.5 516.5 516.5]
    preprocessor = DefaultPreprocessor(base_dir=base_dir, 
                                    image_dir=image_dir,
                                    label_dir=gt_dir,
                                   )

    out_spacing = [0.5, 0.70410156, 0.70410156]
    output_dir = "./data/fullres/train/"
    
    with open("./data_analysis_result.txt", "r") as f:
        content = f.read().strip("\n")
        print(content)
    content = eval(content)
    foreground_intensity_properties_per_channel = content["intensity_statistics_per_channel"]

    preprocessor.run(output_spacing=out_spacing, 
                     output_dir=output_dir, 
                     all_labels=[1],
                     foreground_intensity_properties_per_channel=foreground_intensity_properties_per_channel
    )

def plan():
    
    preprocessor = DefaultPreprocessor(base_dir=base_dir, 
                                    image_dir=image_dir,
                                    label_dir=gt_dir,
                                   )
    preprocessor.run_plan()


if __name__ == "__main__":
# 
    plan()
    process_train()
