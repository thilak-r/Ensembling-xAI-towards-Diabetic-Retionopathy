# utils/preprocessing.py
import cv2
import numpy as np
import random
import os

# Define target image size - Make sure this matches your training size
IMAGE_SIZE = 256

# --- Helper Functions (crop_black, circle_crop, random_crop - keep as before) ---
# ... (Your existing crop_black, circle_crop, random_crop function definitions here) ...

def crop_black(img, tol=7):
    '''Perform automatic crop of black areas'''
    if img is None: return None
    if img.ndim == 2:
        mask = img > tol
        if mask.any(): return img[np.ix_(mask.any(1),mask.any(0))]
        else: return img
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol
        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
        if (check_shape == 0): return img
        else:
            img1 = img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
            img2 = img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
            img3 = img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
            img  = np.stack([img1, img2, img3], axis = -1)
            return img
    return img

def circle_crop(img, sigmaX=10): # sigmaX param is unused here, kept for signature match
    '''Perform circular crop around image center'''
    if img is None: return None
    height, width, depth = img.shape
    if height != width:
        largest_side = np.max((height, width))
        img = cv2.resize(img, (largest_side, largest_side))
        height, width, depth = img.shape
    x = int(width / 2)
    y = int(height / 2)
    r = np.amin((x,y))
    circle_img = np.zeros((height, width), np.uint8)
    cv2.circle(circle_img, (x,y), int(r), (1), thickness = -1)
    img = cv2.bitwise_and(img, img, mask = circle_img)
    return img

def random_crop(img, size=(0.9, 1)):
    '''Random crop (Data Augmentation)'''
    if img is None: return None
    height, width, depth = img.shape
    if height <= 0 or width <= 0: return img # Handle potentially empty images
    cut = 1 - random.uniform(size[0], size[1])
    target_h = int((1 - cut) * height)
    target_w = int((1 - cut) * width)
    if target_h <= 0 or target_w <= 0: return img
    i = random.randint(0, height - target_h)
    j = random.randint(0, width - target_w)
    h = i + target_h
    w = j + target_w
    img = img[i:h, j:w, :]
    return img


# --- Main Preprocessing Function for Inference (keep as before) ---
def preprocess_image_inference(image_path, sigmaX=10, image_size=IMAGE_SIZE):
    '''
    Loads image and applies custom preprocessing for inference.
    do_random_crop is set to False for inference.
    '''
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        print(f"Error: cv2.imread failed to load {image_path}")
        return None

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = crop_black(image, tol=7)
    if image is None: return None

    # No random_crop during inference

    image = cv2.resize(image, (image_size, image_size))

    # Apply the Ben Graham style enhancement
    image = image.astype(np.float32) # Convert to float before addWeighted
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), sigmaX), -4, 128)
    image = np.clip(image, 0, 255).astype(np.uint8) # Clip and convert back

    image = circle_crop(image, sigmaX=sigmaX) # sigmaX ignored in circle_crop
    if image is None: return None

    return image # Return as RGB NumPy array


# --- Function for the Third Display Image (replacing denoising) ---
# We keep the function name as 'denoise_image' as requested
def denoise_image(image_path, image_size=IMAGE_SIZE, clipLimit=2.0, tileGridSize=(8,8)):
    '''
    Loads image, extracts green channel, applies CLAHE, and resizes.
    (Used for visual comparison, not necessarily denoising).
    '''
    # Load the original image again (or could pass the original numpy array)
    img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        print(f"Error: cv2.imread failed to load {image_path} for alternative view")
        return None

    # Convert to grayscale (or just get the green channel)
    # Let's get the green channel as it's often best for vessels
    img_green = img_bgr[:,:,1] # Green channel is index 1 in BGR

    # Apply CLAHE to the green channel
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    img_clahe_green = clahe.apply(img_green)

    # CLAHE result is a single channel (grayscale).
    # To display it using cv2.imwrite with 3 channels later, we need to convert it.
    # We can convert it back to a 3-channel BGR image where all channels are the CLAHE result.
    img_clahe_green_bgr = cv2.cvtColor(img_clahe_green, cv2.COLOR_GRAY2BGR)
    # Or convert to RGB if saving and reading as RGB later
    img_clahe_green_rgb = cv2.cvtColor(img_clahe_green, cv2.COLOR_GRAY2RGB)


    # Resize to the target size (should be same as model input size)
    img_resized = cv2.resize(img_clahe_green_rgb, (image_size, image_size))


    return img_resized # Return as RGB NumPy array