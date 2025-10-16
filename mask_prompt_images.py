import torch

import numpy as np
from PIL import Image
import os

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


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
    # Paths
    checkpoint = 'C:/model_weights/sam-2/sam2_hiera_base_plus.pt'
    model_cfg = 'C:/Users/r02sw23/PycharmProjects/pythonProject1/.venv/PANet-master-borebreen-sam2/sam2_configs/sam2_hiera_b+.yaml'
    output_path = 'C:/Users/r02sw23/PycharmProjects/pythonProject1/.venv/A18-SAM-2-model-distributed-3GPUs/'
    tot_image_no = 10

    os.makedirs(output_path, exist_ok=True)

    for i in range(1, tot_image_no+1):
        image_path = f"C:/Users/r02sw23/Documents/borebreen-drone-image-data/masks/borebreen_crop_drone_{i}.png"
        voc_mask_path = f'C:/Users/r02sw23/PycharmProjects/pythonProject1/.venv/A13_Supervised_LR_FNN_LSTM_borebreen/LR/test_results/borebreen_crop_drone_{i}.png'
    
        # Load model
        predictor = load_sam2_model(checkpoint, model_cfg)

        # Option 1: Use all object pixels as prompt
        masks, scores, logits = segment_with_pascal_voc_mask(
            predictor, image_path, voc_mask_path
        )

        # Option 2: Use specific class ID as prompt (e.g., class 1)
        # masks, scores, logits = segment_with_pascal_voc_mask(
        #     predictor, image_path, voc_mask_path, class_id=1
        # )

        # Select best mask
        best_mask_idx = np.argmax(scores)
        best_mask = masks[best_mask_idx]

        print(f"Generated {len(masks)} masks")
        print(f"Scores: {scores}")
        print(f"Best mask index: {best_mask_idx}")

        # Save result
        result_mask = Image.fromarray((best_mask * 255).astype(np.uint8))
        result_mask.save(output_path + f"output_refined_mask_{i}.png")

        # Optionally: Save all masks
        for i, (mask, score) in enumerate(zip(masks, scores)):
            mask_img = Image.fromarray((mask * 255).astype(np.uint8))

            mask_img.save(output_path + f"output_mask_{i}_score_{score:.3f}.png")
