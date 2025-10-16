import torch

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from metrics import Metrics
from helper_functions import HelperFunctions

import argparse

# Defines the ArgumentParser object
parser = argparse.ArgumentParser()

# Hyperparameters
parser.add_argument("--file_start_number", type=int, default=1)
parser.add_argument("--file_end_number", type=int, default=10)
parser.add_argument("--width", type=int, default=1024)
parser.add_argument("--height", type=int, default=1024)
parser.add_argument("--multi_mask", type=bool, default=True)
parser.add_argument("--sam2_checkpoint", type=str, default='C:/model_weights/sam-2/sam2_hiera_base_plus.pt')
parser.add_argument("--model_cfg", type=str, default='C:/Users/r02sw23/PycharmProjects/pythonProject1/.venv/PANet-master-borebreen-sam2/sam2_configs/sam2_hiera_b+.yaml')
# ------------------------------------------------------------------------

def main():
    # Creates the ArgumentParser object in the main function
    args = parser.parse_args()

    # Use bfloat16 for the entire runtime
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

    if torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Sets the device and the random seeds
    if torch.cuda.is_available():
        # Sets the device to CUDA GPU
        device = 'cuda'
        # Set random seed for GPU
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)

    else:
        # Sets the device to CUDA GPU
        device = 'cpu'
        # Set random seed for CPU
        torch.cuda.manual_seed(42)

    # Defiens the HelperFunctions object
    helper = HelperFunctions()

    for i in range(args.file_start_number, args.file_end_number + 1):
        image_path = f"C:/Users/r02sw23/Documents/borebreen-drone-image-data/masks/borebreen_crop_drone_{i}.png"
        voc_mask_path = f'C:/Users/r02sw23/PycharmProjects/pythonProject1/.venv/A13_Supervised_LR_FNN_LSTM_borebreen/LR/test_results/borebreen_crop_drone_{i}.png'

        # Load model
        predictor = load_sam2_model(args.sam2_checkpoint, args.model_cfg)

        # Option 1: Use all object pixels as prompt
        masks, scores, logits = segment_with_pascal_voc_mask(
            predictor, image_path, voc_mask_path
        )

        # # Option 2: Use specific class ID as prompt (e.g., class 1)
        # masks, scores, logits = segment_with_pascal_voc_mask(
        #     predictor, image_path, voc_mask_path, class_id=1
        # )

        ######################################## Muti-Mask 1 ##########################################
        if args.multi_mask:
            path = f'./results/multi_mask/base_plus/image_{i}/'  # Creates the file path for the output results
            # Calls the make_dir Python helper function
            os.makedirs(path, exist_ok=True)

            # Stores the output masks from the SAM 2 model inside a Python list
            out_masks = []

            # Iterates through the SAM 2 model predictions to output images to the screen as a figure plot
            for i, (mask, score) in enumerate(zip(masks, scores)):
                plt.figure(figsize=(10, 10))
                plt.imshow(image)
                helper.show_mask(mask, plt.gca())
                helper.show_points(input_point, input_label, plt.gca())
                plt.title(f"Mask {i + 1}, Score: {score:.3f}", fontsize=18)
                plt.axis('on')
                plt.savefig(path + 'output_image_' + str(i + 1) + '.png')
                out_masks.append(mask)
                plt.show()

            # Prints out the number of output masks saved to the out_masks Python list
            print("\nNumber of Output Masks: ", len(out_masks))

            # Displays the first output mask from the SAM 2 model and saves to a plot figure and an image
            plt.figure(figsize=(10.24, 10.24))
            plt.imshow(out_masks[0])
            plt.axis('off')
            plt.savefig(path + 'mask_1.png')
            segmented = out_masks[0].astype(int)
            cv2.imwrite(path + output_mask_1, segmented)
            plt.show()

            # Setting the number of rows and columns for the subplot figures
            rows = 1
            columns = 3

            # Creates the subplot figure
            fig = plt.figure(figsize=(10, 7))
            fig.add_subplot(rows, columns, 1)
            plt.imshow(image)
            plt.axis('off')

            # Adds a subplot at the 2nd position
            fig.add_subplot(rows, columns, 2)

            # Segmentation mask subplot
            plt.imshow(out_masks[0])
            plt.axis('off')

            # Adds a subplot at the 3rd position
            fig.add_subplot(rows, columns, 3)

            # Segmentation mask over image subplot
            image0 = cv2.imread(path + 'output_image_1.png')
            image_temp = cv2.cvtColor(image0, cv2.COLOR_BGR2RGB)
            image0 = cv2.resize(image_temp, (1000, 1000), interpolation=cv2.INTER_LINEAR)
            plt.imshow(image0)
            plt.axis('off')

            # Saves the Matplotlib figure to the outputs file directory
            plt.savefig(path + 'output_image_all_1.png')

            # Displays the subplot figure to the screen
            plt.show()

            # Prints got to the end of the prediction iteration to the screen
            print('\nMask 1')
            print("SAM 2 Model Predictions Complete")

            # Defines the Metrics and HelperFunctions Python objects
            metric = Metrics()

            # Stores the input image, segmentation masks and ground truth masks as NumPy arrays
            image = cv2.imread(dir_images)
            mask = cv2.imread(path + output_mask_1, 0)
            gt = cv2.imread(gt_name, 0)

            # Stores the mask and ground truth as a Pytorch tensor from the to_tensor helper function
            tensor_mask, tensor_gt = helper.to_tensor(mask, gt)

            # Reshape the PyTroch tensors
            tensor_mask = tensor_mask.reshape(1, 1, args.width, args.height)
            tensor_gt = tensor_gt.reshape(1, 1, args.width, args.height)

            # Find maximum value and its index and replaces with the new value
            mask_max_value, mask_max_index = tensor_mask.max(), tensor_mask.argmax()
            new_value = 1

            # Replace the maximum value in the tensor
            tensor_mask[tensor_mask == mask_max_value] = new_value

            # Find the maximum pixel value and its index
            gt_max_value, gt_max_index = tensor_gt.max(), tensor_gt.argmax()
            new_value = 1

            # Replace the maximum pixel value in the tensor
            tensor_gt[tensor_gt == gt_max_value] = new_value

            # Segemntation metric calculations
            iou = metric.iou(tensor_mask, tensor_gt.long())
            ds = metric.dsc(tensor_mask, tensor_gt.long())

            # Prints out the segmentation metric results to the screen
            print('Dice Score Coefficient (DCE): ', ds.item())
            print('Intersection Over Union (IoU): ', iou.item())

            # Display the model segmentation mask on the screen and save it as an image
            figure_size = 10.24
            plt.figure(figsize=(figure_size, figure_size))
            plt.imshow(mask)
            plt.title('Segmented Image'), plt.xticks([]), plt.yticks([])
            plt.axis('off')
            plt.savefig(path + 'mask_only_1.png')
            plt.show()

            # Display the ground truth mask on the screen and save it as an image
            figure_size = 10.24
            plt.figure(figsize=(figure_size, figure_size))
            plt.imshow(gt)
            plt.title('Ground Truth'), plt.xticks([]), plt.yticks([])
            plt.savefig(path + 'gt_only_1.png')
            plt.show()

            ######################################## Muti-Mask 2 ##########################################
            # Display the second mask
            print("\nMask 2")

            # Displays the first output mask from the SAM model and saves to a plot figure and an image
            plt.figure(figsize=(10.24, 10.24))
            plt.imshow(out_masks[1])
            plt.axis('off')
            plt.savefig(path + 'mask_2.png')
            segmented = out_masks[1].astype(int)
            cv2.imwrite(path + output_mask_2, segmented)
            plt.show()

            # Creates the sublot figure
            fig = plt.figure(figsize=(10, 7))
            fig.add_subplot(rows, columns, 1)
            plt.imshow(image)
            plt.axis('off')

            # Adds a subplot at the 2nd position
            fig.add_subplot(rows, columns, 2)

            # Segmentation mask subplot
            plt.imshow(out_masks[1])
            plt.axis('off')

            # Adds a subplot at the 3rd position
            fig.add_subplot(rows, columns, 3)

            # Segmentation mask over image subplot
            image0 = cv2.imread(path + 'output_image_2.png')
            image_temp = cv2.cvtColor(image0, cv2.COLOR_BGR2RGB)
            image0 = cv2.resize(image_temp, (1000, 1000), interpolation=cv2.INTER_LINEAR)
            plt.imshow(image0)
            plt.axis('off')

            # Saves the Matplotlib figure to the outputs file directory
            plt.savefig(path + 'output_image_all_2.png')

            # Displays the subplot figure to the screen
            plt.show()

            # Prints got to the end of the prediction iteration to the screen
            print("SAM 2 Model Predictions Complete")

            # Stores the input image, segmentation masks and ground truth masks as NumPy arrays
            image = cv2.imread(dir_images)
            mask = cv2.imread(path + output_mask_2, 0)
            gt = cv2.imread(gt_name, 0)

            # Stores the mask and ground truth as a Pytorch tensor from teh to_tensor helper function
            tensor_mask, tensor_gt = helper.to_tensor(mask, gt)

            # Reshape the PyTroch tensors
            tensor_mask = tensor_mask.reshape(1, 1, args.width, args.height)
            tensor_gt = tensor_gt.reshape(1, 1, args.width, args.height)

            # Find maximum value and its index and replaces with the new value
            mask_max_value, mask_max_index = tensor_mask.max(), tensor_mask.argmax()
            new_value = 1

            # Replace the maximum value in the tensor
            tensor_mask[tensor_mask == mask_max_value] = new_value

            # Find the maximum pixel value and its index
            gt_max_value, gt_max_index = tensor_gt.max(), tensor_gt.argmax()
            new_value = 1

            # Replace the maximum pixel value in the tensor
            tensor_gt[tensor_gt == gt_max_value] = new_value

            # Segemntation metric calculations
            iou = metric.iou(tensor_mask, tensor_gt.long())
            ds = metric.dsc(tensor_mask, tensor_gt.long())

            # Prints out the segmentation metric results to the screen
            print('Dice Score Coefficient (DCE): ', ds.item())
            print('Intersection Over Union (IoU): ', iou.item())

            # Display the ground truth mask to the screen and saves as an image
            figure_size = 10.24
            plt.figure(figsize=(figure_size, figure_size))
            plt.imshow(mask)
            plt.title('Segmented Image'), plt.xticks([]), plt.yticks([])
            plt.axis('off')
            plt.savefig(path + 'mask_only_2.png')
            plt.show()

            # Displays the segmentation mask on the screen and saves it as an image
            figure_size = 10.24
            plt.figure(figsize=(figure_size, figure_size))
            plt.imshow(gt)
            plt.title('Ground Truth'), plt.xticks([]), plt.yticks([])
            plt.savefig(path + 'gt_only_2.png')
            plt.show()

            ######################################## Muti-Mask 3 ##########################################
            # Display the third mask
            print("\nMask 3")

            # Displays the first output mask from the SAM model and saves to a plot figure and an image
            plt.figure(figsize=(10.24, 10.24))
            plt.imshow(out_masks[2])
            plt.axis('off')
            plt.savefig(path + 'mask_3.png')
            segmented = out_masks[2].astype(int)
            cv2.imwrite(path + output_mask_3, segmented)
            plt.show()

            # Creates the sublot figure
            fig = plt.figure(figsize=(10, 7))
            fig.add_subplot(rows, columns, 1)
            plt.imshow(image)
            plt.axis('off')

            # Adds a subplot at the 2nd position
            fig.add_subplot(rows, columns, 2)

            # Segmentation mask subplot
            plt.imshow(out_masks[2])
            plt.axis('off')

            # Adds a subplot at the 3rd position
            fig.add_subplot(rows, columns, 3)

            # Segmentation mask over image subplot
            image0 = cv2.imread(path + 'output_image_3.png')
            image_temp = cv2.cvtColor(image0, cv2.COLOR_BGR2RGB)
            image0 = cv2.resize(image_temp, (1000, 1000), interpolation=cv2.INTER_LINEAR)
            plt.imshow(image0)
            plt.axis('off')

            # Saves the Matplotlib figure to the outputs file directory
            plt.savefig(path + 'output_image_all_3.png')

            # Displays the subplot figure to the screen
            plt.show()

            # Prints got to the end of the prediction iteration to the screen
            print("SAM 2 Model Predictions Complete")

            # Defines the Metrics and HelperFunctions Python objects
            metric = Metrics()

            # Stores the input image, segmentation masks and ground truth masks as NumPy arrays
            image = cv2.imread(dir_images)
            mask = cv2.imread(path + output_mask_3, 0)
            gt = cv2.imread(gt_name, 0)

            # Stores the mask and ground truth as a Pytorch tensor from teh to_tensor helper function
            tensor_mask, tensor_gt = helper.to_tensor(mask, gt)

            # Reshape the PyTroch tensors
            tensor_mask = tensor_mask.reshape(1, 1, args.width, args.height)
            tensor_gt = tensor_gt.reshape(1, 1, args.width, args.height)

            # Find maximum value and its index and replaces with the new value
            mask_max_value, mask_max_index = tensor_mask.max(), tensor_mask.argmax()
            new_value = 1

            # Replace the maximum value in the tensor
            tensor_mask[tensor_mask == mask_max_value] = new_value

            # Find the maximum pixel value and its index
            gt_max_value, gt_max_index = tensor_gt.max(), tensor_gt.argmax()
            new_value = 1

            # Replace the maximum pixel value in the tensor
            tensor_gt[tensor_gt == gt_max_value] = new_value

            # Segemntation metric calculations
            iou = metric.iou(tensor_mask, tensor_gt.long())
            ds = metric.dsc(tensor_mask, tensor_gt.long())

            # Prints out the segmentation metric results to the screen
            print('Dice Score Coefficient (DCE): ', ds.item())
            print('Intersection Over Union (IoU): ', iou.item())

            # Display the ground truth mask to the screen and saves as an image
            figure_size = 10.24
            plt.figure(figsize=(figure_size, figure_size))
            plt.imshow(mask)
            plt.title('Segmented Image'), plt.xticks([]), plt.yticks([])
            plt.axis('off')
            plt.savefig(path + 'mask_only_3.png')
            plt.show()

            # Displays the segmentation mask on the screen and saves it as an image
            figure_size = 10.24
            plt.figure(figsize=(figure_size, figure_size))
            plt.imshow(gt)
            plt.title('Ground Truth'), plt.xticks([]), plt.yticks([])
            plt.savefig(path + 'gt_only_3.png')
            plt.show()

        else:
            output_path = f'./results/single_mask/{args.model_cfg}/image_{i}/'  # Output filepath directory for the segmentation mask results
            # Calls the make_dir Python helper function
            os.makedirs(output_path, exist_ok=True)

            # Stores the output masks from the SAM 2 model inside a Python list
            out_masks = []

            # Iterates through the SAM 2 model predictions to output images to the screen as a figure plot
            for i, (mask, score) in enumerate(zip(masks, scores)):
                plt.figure(figsize=(10, 10))
                plt.imshow(image)
                helper.show_mask(mask, plt.gca())
                helper.show_points(input_point, input_label, plt.gca())
                plt.title(f"Mask {i + 1}, Score: {score:.3f}", fontsize=18)
                plt.axis('on')
                plt.savefig(output_path + 'output_image_' + str(i + 1) + '.png')
                out_masks.append(mask)
                plt.show()

            # Prints out the number of output masks saved to the out_masks Python list
            print("\nNumber of Output Masks: ", len(out_masks))

            # Displays the first output mask from the SAM 2 model and saves to a plot figure and an image
            plt.figure(figsize=(10.24, 10.24))
            plt.imshow(out_masks[0])
            plt.axis('off')
            plt.savefig(output_path + 'mask_1.png')
            segmented = out_masks[0].astype(int)
            cv2.imwrite(output_path + output_mask_1, segmented)
            plt.show()

            # Setting the number of rows and columns for the subplot figures
            rows = 1
            columns = 3

            # Creates the subplot figure
            fig = plt.figure(figsize=(10, 7))
            fig.add_subplot(rows, columns, 1)
            plt.imshow(image)
            plt.axis('off')

            # Adds a subplot at the 2nd position
            fig.add_subplot(rows, columns, 2)

            # Segmentation mask subplot
            plt.imshow(out_masks[0])
            plt.axis('off')

            # Adds a subplot at the 3rd position
            fig.add_subplot(rows, columns, 3)

            # Segmentation mask over image subplot
            image0 = cv2.imread(output_path + 'output_image_1.png')
            image_temp = cv2.cvtColor(image0, cv2.COLOR_BGR2RGB)
            image0 = cv2.resize(image_temp, (1000, 1000), interpolation=cv2.INTER_LINEAR)
            plt.imshow(image0)
            plt.axis('off')

            # Saves the Matplotlib figure to the outputs file directory
            plt.savefig(output_path + 'output_image_all_1.png')

            # Displays the subplot figure to the screen
            plt.show()

            # Prints got to the end of the prediction iteration to the screen
            print("\nSAM 2 Model Predictions Complete")

            # Defines the Metrics and HelperFunctions Python objects
            metric = Metrics()

            # Stores the input image, segmentation masks and ground truth masks as NumPy arrays
            image = cv2.imread(dir_images)
            mask = cv2.imread(output_path + output_mask_1, 0)
            gt = cv2.imread(gt_name, 0)

            # Stores the mask and ground truth as a Pytorch tensor from the to_tensor helper function
            tensor_mask, tensor_gt = helper.to_tensor(mask, gt)

            # Reshape the PyTroch tensors
            tensor_mask = tensor_mask.reshape(1, 1, args.width, args.height)
            tensor_gt = tensor_gt.reshape(1, 1, args.width, args.height)

            # Find maximum value and its index and replaces with the new value
            mask_max_value, mask_max_index = tensor_mask.max(), tensor_mask.argmax()
            new_value = 1

            # Replace the maximum value in the tensor
            tensor_mask[tensor_mask == mask_max_value] = new_value

            # Find the maximum pixel value and its index
            gt_max_value, gt_max_index = tensor_gt.max(), tensor_gt.argmax()
            new_value = 1

            # Replace the maximum pixel value in the tensor
            tensor_gt[tensor_gt == gt_max_value] = new_value

            # Segemntation metric calculations
            iou = metric.iou(tensor_mask, tensor_gt.long())
            ds = metric.dsc(tensor_mask, tensor_gt.long())

            # Prints out the segmentation metric results to the screen
            print('\nDice Score Coefficient (DCE): ', ds.item())
            print('Intersection Over Union (IoU): ', iou.item())

            # Display the model segmentation mask on the screen and save it as an image
            figure_size = 10.24
            plt.figure(figsize=(figure_size, figure_size))
            plt.imshow(mask)
            plt.title('Segmented Image'), plt.xticks([]), plt.yticks([])
            plt.axis('off')
            plt.savefig(output_path + 'mask_only_1.png')
            plt.show()

            # Display the ground truth mask on the screen and save it as an image
            figure_size = 10.24
            plt.figure(figsize=(figure_size, figure_size))
            plt.imshow(gt)
            plt.title('Ground Truth'), plt.xticks([]), plt.yticks([])
            plt.savefig(output_path + 'gt_only_1.png')
            plt.show()

# Initialize SAM 2 model
def load_sam2_model(checkpoint_path, model_cfg="sam2_hiera_l.yaml"):
    """
    Load SAM 2 model
    checkpoint_path: path to model checkpoint
    model_cfg: model configuration file
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam2_model = build_sam2(model_cfg, checkpoint_path, device=device)
    predictor = SAM2ImagePredictor(sam2_model)
    return predictor


def load_pascal_voc_mask(mask_path, class_id=None):
    """
    Load Pascal VOC format segmentation mask

    Args:
        mask_path: path to Pascal VOC mask (PNG with indexed colors)
        class_id: specific class ID to extract (1, 2, 3, etc.)
                 If None, uses all non-background pixels (!=0, !=255)

    Returns:
        Binary mask (H, W) as float32
    """
    # Load mask - convert to grayscale/single channel
    mask = Image.open(mask_path)

    # Ensure single channel
    if mask.mode == 'RGB' or mask.mode == 'RGBA':
        mask = mask.convert('L')

    mask_np = np.array(mask)

    # Ensure 2D array
    if len(mask_np.shape) == 3:
        mask_np = mask_np[:, :, 0]

    if class_id is not None:
        # Extract specific class
        binary_mask = (mask_np == class_id).astype(np.float32)
    else:
        # Use all object pixels (ignore background=0 and boundary=255)
        binary_mask = ((mask_np > 0) & (mask_np < 255)).astype(np.float32)

    return binary_mask


def segment_with_pascal_voc_mask(predictor, image_path, mask_path, class_id=None):
    """
    Perform segmentation using a Pascal VOC format mask as prompt

    Args:
        predictor: SAM2ImagePredictor instance
        image_path: path to input image
        mask_path: path to Pascal VOC segmentation mask
        class_id: specific class to use as prompt (optional)
    """
    # Load image
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)

    # Set image in predictor
    predictor.set_image(image_np)

    # Load Pascal VOC mask
    mask_prompt = load_pascal_voc_mask(mask_path, class_id)

    # Resize mask to match image dimensions first
    if mask_prompt.shape[0] != image_np.shape[0] or mask_prompt.shape[1] != image_np.shape[1]:
        mask_prompt_img = Image.fromarray((mask_prompt * 255).astype(np.uint8), mode='L')
        mask_prompt_img = mask_prompt_img.resize(
            (image_np.shape[1], image_np.shape[0]),  # (width, height)
            Image.Resampling.NEAREST
        )
        mask_prompt = np.array(mask_prompt_img).astype(np.float32) / 255.0

    # SAM 2 expects low-resolution mask input (256x256)
    # Resize mask to low resolution
    mask_prompt_img = Image.fromarray((mask_prompt * 255).astype(np.uint8), mode='L')
    mask_prompt_lowres = mask_prompt_img.resize((256, 256), Image.Resampling.BILINEAR)
    mask_prompt_lowres = np.array(mask_prompt_lowres).astype(np.float32) / 255.0

    # Convert to tensor with shape [1, 1, 256, 256]
    mask_input = torch.from_numpy(mask_prompt_lowres).float()
    mask_input = mask_input.unsqueeze(0).unsqueeze(0)

    print(f"Low-res mask input shape: {mask_input.shape}")

    # Move to same device as model
    device = next(predictor.model.parameters()).device
    mask_input = mask_input.to(device)

    # Predict with mask prompt
    masks, scores, logits = predictor.predict(
        point_coords=None,
        point_labels=None,
        mask_input=mask_input,
        multimask_output=True,
    )

    return masks, scores, logits


# Example usage
if __name__ == "__main__":
    
    main()
