import torch
import numpy as np
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

def apply_gradcam(model, input_tensor):
    """Apply Grad-CAM++ for visualization and extract visual features."""
    # For EfficientNet-B0, target the last convolutional layer in the blocks
    # EfficientNet has a 'blocks' attribute, which is a list of sequential blocks
    target_layers = [model.backbone.blocks[-1][-1].conv_pw]  # Last block's pointwise conv
    
    cam = GradCAMPlusPlus(model=model, target_layers=target_layers)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        target_class = outputs.argmax().item()
    
    targets = [ClassifierOutputTarget(target_class)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
    
    input_image = input_tensor[0].permute(1, 2, 0).numpy()
    input_image = (input_image - input_image.min()) / (input_image.max() - input_image.min())
    visualization = show_cam_on_image(input_image, grayscale_cam, use_rgb=True)
    
    # Extract simple visual features from heatmap
    if grayscale_cam.max() > 0.5:
        visual_features = "Significant opacity in highlighted regions"
    else:
        visual_features = "Mild changes in highlighted regions"
    
    return visualization, visual_features