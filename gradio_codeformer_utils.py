import torch
import cv2
import numpy as np
from PIL import Image
from torchvision.transforms.functional import normalize
from basicsr.utils import img2tensor, tensor2img
from basicsr.utils.download_util import load_file_from_url
from basicsr.utils.misc import get_device
from basicsr.utils.registry import ARCH_REGISTRY


def enhance_face_pil(pil_image, fidelity_weight=0.5):
    """
    Enhance a PIL image containing an aligned face
    
    Args:
        pil_image: PIL Image object (should be aligned face)
        fidelity_weight: Balance quality and fidelity (0-1, default 0.5)
        
    Returns:
        PIL Image object (enhanced face)
    """
    device = get_device()
    
    # Load CodeFormer model
    net = ARCH_REGISTRY.get('CodeFormer')(
        dim_embd=512, codebook_size=1024, n_head=8, n_layers=9, 
        connect_list=['32', '64', '128', '256']
    ).to(device)
    
    pretrain_model_url = 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth'
    ckpt_path = load_file_from_url(
        url=pretrain_model_url, 
        model_dir='weights/CodeFormer', 
        progress=True, 
        file_name=None
    )
    checkpoint = torch.load(ckpt_path)['params_ema']
    net.load_state_dict(checkpoint)
    net.eval()
    
    # Convert PIL to OpenCV format
    img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    
    # Resize to 512x512 (CodeFormer input size)
    img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
    
    # Prepare tensor
    cropped_face_t = img2tensor(img / 255., bgr2rgb=True, float32=True)
    normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
    cropped_face_t = cropped_face_t.unsqueeze(0).to(device)
    
    try:
        with torch.no_grad():
            output = net(cropped_face_t, w=fidelity_weight, adain=True)[0]
            restored_face = tensor2img(output, rgb2bgr=True, min_max=(-1, 1))
        del output
        torch.cuda.empty_cache()
    except Exception as error:
        print(f'Failed inference for CodeFormer: {error}')
        restored_face = tensor2img(cropped_face_t, rgb2bgr=True, min_max=(-1, 1))
    
    # Convert back to PIL
    restored_face = restored_face.astype('uint8')
    restored_face_rgb = cv2.cvtColor(restored_face, cv2.COLOR_BGR2RGB)
    return Image.fromarray(restored_face_rgb)

# Usage:
# enhanced_image = enhance_face_pil(your_pil_image, fidelity_weight=0.5)

def inpaint_face_pil(pil_image):
    """
    Inpaint a PIL image containing a masked face
    
    Args:
        pil_image: PIL Image object (should be 512x512 aligned face with mask)
        
    Returns:
        PIL Image object (inpainted face)
    """
    device = get_device()
    
    # Load CodeFormer inpainting model
    net = ARCH_REGISTRY.get('CodeFormer')(
        dim_embd=512, codebook_size=512, n_head=8, n_layers=9, 
        connect_list=['32', '64', '128']
    ).to(device)
    
    pretrain_model_url = 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer_inpainting.pth'
    ckpt_path = load_file_from_url(
        url=pretrain_model_url, 
        model_dir='weights/CodeFormer', 
        progress=True, 
        file_name=None
    )
    checkpoint = torch.load(ckpt_path)['params_ema']
    net.load_state_dict(checkpoint)
    net.eval()
    
    # Convert PIL to OpenCV format
    input_face = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    
    # Ensure 512x512 resolution
    if input_face.shape[:2] != (512, 512):
        input_face = cv2.resize(input_face, (512, 512), interpolation=cv2.INTER_LINEAR)
    
    # Prepare tensor
    input_face = img2tensor(input_face / 255., bgr2rgb=True, float32=True)
    normalize(input_face, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
    input_face = input_face.unsqueeze(0).to(device)
    
    try:
        with torch.no_grad():
            # Create mask based on white pixels (value 3 after normalization)
            mask = torch.zeros(512, 512)
            m_ind = torch.sum(input_face[0], dim=0)
            mask[m_ind == 3] = 1.0
            mask = mask.view(1, 1, 512, 512).to(device)
            
            # w is fixed to 1, adain=False for inpainting
            output_face = net(input_face, w=1, adain=False)[0]
            output_face = (1 - mask) * input_face + mask * output_face
            save_face = tensor2img(output_face, rgb2bgr=True, min_max=(-1, 1))
        del output_face
        torch.cuda.empty_cache()
    except Exception as error:
        print(f'Failed inference for CodeFormer: {error}')
        save_face = tensor2img(input_face, rgb2bgr=True, min_max=(-1, 1))
    
    # Convert back to PIL
    save_face = save_face.astype('uint8')
    save_face_rgb = cv2.cvtColor(save_face, cv2.COLOR_BGR2RGB)
    return Image.fromarray(save_face_rgb)

# Usage:
# inpainted_image = inpaint_face_pil(your_masked_pil_image)