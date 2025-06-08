# utils/gradcam.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import os
import timm

# --- Helper Class to Register Hooks ---
class ActivationsAndGradients:
    """
    Class for extracting activations and
    registering gradients from targetted intermediate layers
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        self.hook_a = None
        self.hook_g = None

        # Register forward hook to save activations
        if isinstance(target_layer, nn.Module):
             self.hook_a = target_layer.register_forward_hook(self.save_activations)
             # Register backward hook to save gradients
             # Use register_full_backward_hook for newer PyTorch versions as warned
             if hasattr(target_layer, "register_full_backward_hook"):
                  self.hook_g = target_layer.register_full_backward_hook(self.save_gradients)
                  print(f"Registered full backward hook on layer: {type(target_layer).__name__}")
             else:
                  self.hook_g = target_layer.register_backward_hook(self.save_gradients)
                  print(f"Registered backward hook on layer: {type(target_layer).__name__}")

        else:
             print(f"Warning: Target layer is not a valid nn.Module: {type(target_layer).__name__}. Hooks not registered.")


    def save_activations(self, module, input, output):
        self.activations = output.detach()

    # Modified save_gradients signature for register_full_backward_hook
    def save_gradients(self, module, grad_input, grad_output):
        # grad_output is a tuple containing gradients for output tensors
        # We need the gradient w.r.t. the layer's *output*, which is usually the first element
        # It might also be a tuple for layers with multiple outputs, take the first one
        if isinstance(grad_output, tuple):
            self.gradients = grad_output[0].detach()
        else:
            self.gradients = grad_output.detach()


    def remove_hooks(self):
        """Remove the registered hooks."""
        if self.hook_a is not None:
            self.hook_a.remove()
            self.hook_a = None
        if self.hook_g is not None:
            self.hook_g.remove()
            self.hook_g = None


# --- Core Grad-CAM Calculation ---
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval() # Ensure model is on evaluation mode
        self.target_layer = target_layer
        # ActivationsAndGradients instance is created here, hooks are registered
        self.activations_and_grads = ActivationsAndGradients(model, target_layer)

    def get_cam_weights(self, target_class_gradients):
        return torch.mean(target_class_gradients, dim=(2, 3), keepdim=True)

    def get_loss(self, output, target_class):
        # Compute the score for the target class logit/score
        if isinstance(target_class, int):
            if output.ndim == 1:
                 target_class_score = output[target_class]
            else:
                 target_class_score = output[:, target_class]
        else:
             if target_class.size(0) != output.size(0):
                  print(f"Warning: Target class batch size mismatch ({target_class.size(0)} vs {output.size(0)}). Assuming batch size is 1.")
                  if target_class.ndim == 1 and target_class.size(0) > 0:
                       target_class_tensor = target_class[0].unsqueeze(0)
                  else:
                       print("Error: Cannot determine target class index from provided tensor.")
                       return None
             else:
                  target_class_tensor = target_class

             target_class_score = torch.gather(output, 1, target_class_tensor.unsqueeze(1)).squeeze(1)

        return target_class_score # Maximize the score for the target class

    def __call__(self, input_tensor, target_class):
        """
        Generates Grad-CAM heatmap.

        Args:
            input_tensor (torch.Tensor): The preprocessed and normalized input image tensor (1xCxHxW).
            target_class (int or torch.Tensor): The index of the target class for Grad-CAM.

        Returns:
            torch.Tensor: The generated Grad-CAM heatmap (1x1xHxW), scaled to [0, 1].
                          Returns None if generation fails.
        """
        if input_tensor.ndim != 4 or input_tensor.size(0) != 1:
             print(f"Error generating Grad-CAM: Input tensor must have shape (1xCxHxW), got {input_tensor.shape}")
             self.activations_and_grads.remove_hooks() # Clean up hooks on error
             return None

        if isinstance(target_class, int):
             target_class_tensor = torch.tensor([target_class], device=input_tensor.device)
        elif torch.is_tensor(target_class):
             target_class_tensor = target_class.to(input_tensor.device)
             if target_class_tensor.ndim == 0:
                 target_class_tensor = target_class_tensor.unsqueeze(0)
             elif target_class_tensor.ndim == 1 and target_class_tensor.size(0) == 1:
                 pass
             else:
                 print(f"Error: target_class tensor must be scalar or (1,), got {target_class_tensor.shape}")
                 self.activations_and_grads.remove_hooks()
                 return None
        else:
             print(f"Error: target_class must be an integer or a tensor, got {type(target_class)}")
             self.activations_and_grads.remove_hooks()
             return None


        self.model.zero_grad()

        model_device = next(self.model.parameters()).device
        input_tensor_on_device = input_tensor.to(model_device)
        target_class_tensor_on_device = target_class_tensor.to(model_device)


        try:
             output = self.model(input_tensor_on_device)
        except Exception as e:
             print(f"Error during model forward pass for Grad-CAM: {e}")
             self.activations_and_grads.remove_hooks()
             return None

        loss = self.get_loss(output, target_class_tensor_on_device)
        if loss is None:
             print("Error computing loss for target class.")
             self.activations_and_grads.remove_hooks()
             return None


        try:
             loss.backward(retain_graph=True) # retain_graph=True often needed if model output is used later
        except Exception as e:
             print(f"Error during backward pass for Grad-CAM: {e}")
             self.activations_and_grads.remove_hooks()
             return None

        gradients = self.activations_and_grads.gradients
        activations = self.activations_and_grads.activations

        # Hooks were removed inside ActivationsAndGradients.__call__ after retrieving


        if gradients is None or activations is None:
             print("Error: Could not capture activations or gradients after backward pass.")
             return None

        if gradients.size(0) != activations.size(0) or gradients.size(1) != activations.size(1):
             print(f"Warning: Gradient and activation shapes mismatch (Batch x Channel): {gradients.shape[:2]} vs {activations.shape[:2]}.")
             print("This indicates an issue with target_layer selection or hook registration.")
             return None


        weights = self.get_cam_weights(gradients)

        cam = weights * activations
        cam = torch.sum(cam, dim=1, keepdim=True)

        cam = F.relu(cam)

        cam = F.interpolate(cam, size=(input_tensor.size(2), input_tensor.size(3)), mode='bilinear', align_corners=False)

        cam_min = cam.min()
        cam_max = cam.max()
        if cam_max == cam_min:
             print("Warning: Grad-CAM map is constant (min == max). Setting map to zeros.")
             cam = torch.zeros_like(cam)
        else:
             cam = (cam - cam_min) / (cam_max - cam_min)

        return cam

# --- Function to Superimpose Heatmap (Keep as is) ---
def show_cam_on_image(img: np.ndarray,
                      cam: np.ndarray,
                      use_rgb: bool = False,
                      colormap: int = cv2.COLORMAP_JET,
                      alpha: float = 0.4):
    # ... (keep the existing code for show_cam_on_image) ...
    """
    Heatmap visualization.
    Modified from https://github.com/jacobgil/pytorch-grad-cam
    ... (rest of docstring) ...
    """
    cam_scaled = cam * 255
    cam_scaled = cam_scaled.astype(np.uint8)

    colored_cam = cv2.applyColorMap(cam_scaled, colormap)

    # Resize the colormapped heatmap to match the original image dimensions
    colored_cam_resized = cv2.resize(colored_cam, (img.shape[1], img.shape[0]))

    img = img.astype(np.uint8)
    if use_rgb:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    overlay = cv2.addWeighted(img, 1 - alpha, colored_cam_resized, alpha, 0)

    return overlay

# --- Helper to Find Target Layer (REFINED for DenseNet) ---
def find_target_layer(model):
    """
    Attempts to find a suitable target layer for Grad-CAM in a timm model.
    Prioritizes known stable targets like DenseNet's norm5 or ResNet's layer4.
    """
    # Try specific known targets first
    # DenseNet: features.norm5
    if hasattr(model, 'features') and hasattr(model.features, 'norm5') and isinstance(model.features.norm5, nn.Module):
         print("Found DenseNet-like features. Targetting features.norm5.")
         return model.features.norm5

    # ResNet variants: layer4 (Sequential)
    if hasattr(model, 'layer4') and isinstance(model.layer4, nn.Sequential):
         print("Found ResNet-like layer4 (Sequential). Targetting layer4.")
         return model.layer4

    # EfficientNet variants: blocks[-1] (Sequential)
    if hasattr(model, 'blocks') and isinstance(model.blocks, nn.Sequential):
         print("Found EfficientNet-like blocks (Sequential). Targetting blocks[-1].")
         return model.blocks[-1]

    # RegNet variants: trunk_output.block4 (Sequential)
    if hasattr(model, 'trunk_output') and hasattr(model.trunk_output, 'block4') and isinstance(model.trunk_output.block4, nn.Sequential):
         print("Found RegNet-like trunk_output. Targetting trunk_output.block4.")
         return model.trunk_output.block4


    # Fallback: If specific targets not found, try inspecting all named modules for last Conv2d
    print("Attempting to find a target layer by inspecting named modules (fallback)...")
    target_layer = None
    for name, module in reversed(list(model.named_modules())):
         if isinstance(module, nn.Conv2d):
             # Avoid targeting the very first conv layer
             if name != 'conv1' and not name.endswith('.conv1'):
                  print(f"  Found last Conv2d module: {name}. Returning it as target.")
                  target_layer = module
                  break # Found a likely candidate

    if target_layer:
         return target_layer


    print(f"Error: Could not automatically find a suitable target layer for Grad-CAM for model type {type(model).__name__}.")
    print("Please manually inspect the model architecture (print(model)) and specify the target layer.")
    # Example manual return: return model.layer4[-1].conv3 for ResNet Bottleneck
    # Example manual return for DenseNet: return model.features.norm5


    return None # Indicate that no suitable layer was found


# --- Main Function to Generate Grad-CAM Overlay (Keep as is) ---
def generate_gradcam_overlay(model, image_tensor, original_image_np, target_class_idx):
    # ... (keep the existing code for generate_gradcam_overlay) ...
    """
    Generates and overlays a Grad-CAM heatmap on an image.
    ... (rest of docstring) ...
    """
    if image_tensor.ndim != 4 or image_tensor.size(0) != 1:
         print(f"Error generating Grad-CAM: Input tensor must have shape (1xCxHxW), got {image_tensor.shape}")
         return None # Hooks were removed inside GradCAM.__call__ if created


    if original_image_np.ndim != 3 or original_image_np.shape[2] != 3:
         print(f"Error generating Grad-CAM: Original image must have shape (H x W x 3), got {original_image_np.shape}")
         return None # Hooks were removed inside GradCAM.__call__ if created


    if not isinstance(target_class_idx, int) or target_class_idx < 0:
         print(f"Error generating Grad-CAM: target_class_idx must be a non-negative integer, got {target_class_idx}")
         return None # Hooks were removed inside GradCAM.__call__ if created


    target_layer = find_target_layer(model)
    if target_layer is None:
        print(f"Error generating Grad-CAM: Could not find a suitable target layer for model {type(model).__name__}.")
        return None # No hooks were registered if target_layer is None

    # Create GradCAM instance (Hooks are registered in __init__)
    grad_cam_instance = GradCAM(model, target_layer) # Hooks are registered here

    # Generate the heatmap tensor
    grad_cam_map_tensor = grad_cam_instance(image_tensor, target_class_idx)

    # IMPORTANT: Remove hooks immediately after generating the CAM
    # This prevents memory leaks and unintended behavior
    grad_cam_instance.activations_and_grads.remove_hooks()


    if grad_cam_map_tensor is None:
         print("Error: Grad-CAM map tensor generation failed.")
         return None


    heatmap_np = grad_cam_map_tensor.squeeze().cpu().numpy()

    gradcam_overlay_np = show_cam_on_image(original_image_np, heatmap_np, use_rgb=True)

    return gradcam_overlay_np

# --- End of utils/gradcam.py ---