# face_reconstruction.py

import os
import sys
import cv2
import numpy as np
import torch
import tempfile
import datetime
from PIL import Image
import shutil

class FaceReconstructor:
    def __init__(self, device='cuda'):
        """
        Initialize the 3D face reconstruction model.

        Args:
            device: Device to run the model on ('cuda' or 'cpu')
        """
        # Add parent directory to path for DECA imports
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
        from decalib.deca import DECA
        from decalib.utils.config import cfg as deca_cfg

        # Configure DECA
        deca_cfg.model.use_tex = False
        deca_cfg.rasterizer_type = 'standard'
        deca_cfg.model.extract_tex = True

        # Initialize DECA
        self.device = device
        self.deca = DECA(config=deca_cfg, device=device)

    def reconstruct_from_image(self, input_image, save_folder='output',
                               save_depth=False, save_obj=True, save_vis=True,
                               detector='fan', is_crop=True):
        """
        Reconstructs a 3D face model from a single image.

        Args:
            input_image: PIL Image or numpy array
            save_folder: Directory to save the results
            save_depth: Whether to save depth image
            save_obj: Whether to save OBJ file
            save_vis: Whether to save visualization
            detector: Face detector to use ('fan', 'mtcnn', etc.)
            is_crop: Whether to crop the face from the image

        Returns:
            dict: Dictionary containing paths to generated files
        """
        from decalib.datasets import datasets
        from decalib.utils import util

        # Create timestamp-based name
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"face_{timestamp}"

        # Create temporary directory for the input image
        temp_dir = tempfile.mkdtemp()
        temp_img_path = os.path.join(temp_dir, f"{name}.png")

        # Save the input image to the temporary directory
        if isinstance(input_image, Image.Image):
            input_image.save(temp_img_path)
        else:
            # Handle numpy array
            if isinstance(input_image, np.ndarray):
                if len(input_image.shape) == 3 and input_image.shape[2] == 4:  # RGBA
                    input_image = input_image[:, :, :3]  # Convert to RGB
                cv2.imwrite(temp_img_path, cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR))
            else:
                raise ValueError("Input image must be a PIL Image or numpy array")

        # Create output directory
        os.makedirs(save_folder, exist_ok=True)

        # Process the image using TestData
        testdata = datasets.TestData(temp_dir, iscrop=is_crop, face_detector=detector)

        # Check if TestData found any images
        if len(testdata) == 0:
            shutil.rmtree(temp_dir)
            raise ValueError("No face detected in the image.")

        # Process the image
        data = testdata[0]
        image_name = data['imagename']
        images = data['image'].to(self.device)[None,...]

        # Create folder for this specific image in the save folder
        image_save_folder = os.path.join(save_folder, image_name)
        os.makedirs(image_save_folder, exist_ok=True)

        # Process with DECA
        with torch.no_grad():
            codedict = self.deca.encode(images)
            opdict, visdict = self.deca.decode(codedict)

        # Initialize result paths
        result_paths = {}

        # Save outputs
        if save_obj:
            obj_path = os.path.join(image_save_folder, f"{image_name}.obj")
            self.deca.save_obj(obj_path, opdict)
            result_paths['obj_path'] = obj_path

        if save_depth:
            depth_image = self.deca.render.render_depth(opdict['trans_verts']).repeat(1, 3, 1, 1)
            visdict['depth_images'] = depth_image
            depth_path = os.path.join(image_save_folder, f"{image_name}_depth.jpg")
            cv2.imwrite(depth_path, util.tensor2image(depth_image[0]))
            result_paths['depth_path'] = depth_path

        if save_vis:
            vis_path = os.path.join(image_save_folder, f"{image_name}_vis.jpg")
            cv2.imwrite(vis_path, self.deca.visualize(visdict))
            result_paths['vis_path'] = vis_path

        # Clean up temporary directory
        shutil.rmtree(temp_dir)

        return result_paths


# Standalone function version for easier integration
def reconstruct_3d_face(input_image, save_folder='output', device='cuda',
                        save_depth=False, save_obj=True, save_vis=True):
    """
    Reconstructs a 3D face model from a single image.

    Args:
        input_image: PIL Image or numpy array
        save_folder: Directory to save the results
        device: Device to run the model on ('cuda' or 'cpu')
        save_depth: Whether to save depth image
        save_obj: Whether to save OBJ file
        save_vis: Whether to save visualization

    Returns:
        dict: Dictionary containing paths to generated files
    """
    reconstructor = FaceReconstructor(device=device)
    return reconstructor.reconstruct_from_image(
        input_image=input_image,
        save_folder=save_folder,
        save_depth=save_depth,
        save_obj=save_obj,
        save_vis=save_vis
    )