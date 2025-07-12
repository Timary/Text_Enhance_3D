import gradio as gr
import os
import time
from PIL import Image
import datetime
import tkinter as tk
from tkinter import filedialog
import torch
import shutil
from face_reconstruction import reconstruct_3d_face
from scripts.crop_align_face_with_return import align_face
from codeformer_script_test import run_codeformer_inference, run_inference_inpainting
import numpy as np


class ImageGeneratorGUI:
    def __init__(self, api_handler):
        self.api_handler = api_handler
        self.DALLE3_RESOLUTIONS = ["1024x1024", "1792x1024", "1024x1792"]
        self.GPT_IMAGE_RESOLUTIONS = ["1024x1024", "1536x1024", "1024x1536", "auto"]
        self.DALLE2_RESOLUTIONS = ["256x256", "512x512", "1024x1024"]
        # Initialize output directory
        self.output_dir = os.path.join(os.getcwd(), "generated_images")
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # Store the last generated image for 3D reconstruction
        self.last_generated_image = None

    def create_interface(self):
        with gr.Blocks(title="OpenAI Image Generator") as app:
            gr.Markdown("# OpenAI Image Generator")

            # Add a notice about GPT-Image-1 verification
            gr.Markdown("""
            > **Note:** Using GPT-Image-1 requires organization verification.
            > If you encounter verification errors, please go to
            > [OpenAI Organization Settings](https://platform.openai.com/settings/organization/general)
            > and click on Verify Organization.
            """)

            # Single column layout
            api_key = gr.Textbox(label="OpenAI API Key", type="password")

            # Mode selection including test mode as an option
            mode = gr.Radio(
                ["text2img", "img2img", "test mode (for gui testing)"],
                label="Mode",
                value="text2img"
            )

            # Model selection
            model_dropdown = gr.Dropdown(
                choices=["DALL-E-2", "DALL-E-3", "GPT-Image-1"],
                label="Model",
                value="DALL-E-2"
            )

            resolution = gr.Dropdown(
                choices=self.DALLE2_RESOLUTIONS,
                label="Resolution",
                value=self.DALLE2_RESOLUTIONS[0]
            )

            prompt = gr.Textbox(label="Prompt", lines=3)

            # Input image only visible for img2img mode
            input_image = gr.Image(label="Input Image (for Image-to-Image)", type="pil", visible=False)

            # Test image for test mode
            test_image_input = gr.Image(label="Test Image (for Test Mode)", type="pil", visible=False)

            generate_btn = gr.Button("Generate Image")

            progress = gr.Slider(0, 100, value=0, label="Progress")

            # Status message area
            status_msg = gr.Textbox(label="Status", interactive=False)

            # Create tabs for organizing the interface
            with gr.Tabs():
                with gr.TabItem("Image Generation"):
                    # Output image
                    output_image = gr.Image(label="Generated Image", type="pil")

                    # Output folder selection
                    with gr.Row():
                        output_folder = gr.Textbox(
                            label="Output Folder",
                            value=self.output_dir,
                            interactive=True
                        )
                        browse_folder_btn = gr.Button("Browse...")

                    # Filename input
                    filename_input = gr.Textbox(
                        label="Filename (without extension)",
                        value=f"generated_image_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    )

                    # Save options
                    with gr.Row():
                        save_btn = gr.Button("Save Image")
                        download_btn = gr.Button("Download Image", visible=True)

                    # File download component
                    download_file = gr.File(label="Download Image", visible=False)

                # --- NEW TAB ---
                with gr.TabItem("CodeFormer Enhancement"):
                    # Crop and Align Face button
                    crop_and_align_btn = gr.Button("Crop and Align Generated Face")
                    
                    # Image input with mask drawing utility
                    codeformer_cropped_aligned_image = gr.Image(
                        label="Generated Image after Crop and Align",
                        type="pil",
                    )
                    
                    # Enhance Face button
                    enhance_face_button = gr.Button("Enhance Generated Face using CodeFormer")
                    
                    # Image input with mask drawing utility
                    codeformer_enhanced_image = gr.Image(
                        label="CodeFormer Enhanced Image (Draw Mask to Select Face Region)",
                        tool="sketch",  # enables drawing/masking
                        type="pil",
                        brush_color="white"  # sets the sketch color to white
                    )
                    
                    # Inpaint Face button
                    inpaint_face_button = gr.Button("Inpaint Face using CodeFormer")
                    
                    # Image input with mask drawing utility
                    inpainted_face_image = gr.Image(
                        label="CodeFormer Inpainted Image",
                        type="pil",
                    )

                    # Output folder selection
                    with gr.Row():
                        codeformer_output_folder = gr.Textbox(
                            label="Output Folder",
                            value=self.output_dir,
                            interactive=True
                        )
                        codeformer_browse_folder_btn = gr.Button("Browse...")

                    # Filename input
                    codeformer_filename_input = gr.Textbox(
                        label="Filename (without extension)",
                        value=f"restored_image_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    )

                    # Save and Download buttons
                    with gr.Row():
                        codeformer_save_btn = gr.Button("Save Image")
                        codeformer_download_btn = gr.Button("Download Image", visible=True)

                    codeformer_download_file = gr.File(label="Download Image", visible=False)

                    # --- 3D Reconstruct Button MOVED HERE ---
                    codeformer_reconstruct_3d_btn = gr.Button("Reconstruct 3D Face")
                
                with gr.TabItem("3D Reconstruction (DECA)"):
                    # Visualization image from reconstruction
                    vis_image = gr.Image(label="3D Reconstruction Visualization", type="filepath")

                    # 3D viewer for .obj files
                    output_3d_viewer = gr.Model3D(label="3D Reconstruction")

                    # Output folder selection
                    with gr.Row():
                        output_folder_3d = gr.Textbox(
                            label="Output Folder",
                            value=self.output_dir,
                            interactive=True
                        )
                        browse_folder_btn_3d = gr.Button("Browse...")

                    # Filename input
                    filename_input_3d = gr.Textbox(
                        label="Filename (without extension)",
                        value=f"reconstructed_3d_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    )

                    # Save options
                    with gr.Row():
                        save_btn_3d = gr.Button("Save 3D Model")
                        download_btn_3d = gr.Button("Download 3D Model", visible=True)

                    # File download component
                    download_file_3d = gr.File(label="Download 3D Model", visible=False)

            # Event handlers
            # Event handlers for 3D Reconstruction tab
            browse_folder_btn_3d.click(
                self.browse_output_folder,
                inputs=None,
                outputs=output_folder_3d
            )

            save_btn_3d.click(
                self.save_3d_model_to_folder,
                inputs=[output_3d_viewer, output_folder_3d, filename_input_3d],
                outputs=status_msg
            )

            download_btn_3d.click(
                self.prepare_3d_download,
                inputs=[output_3d_viewer, filename_input_3d],
                outputs=[download_file_3d, status_msg]
            )

            model_dropdown.change(
                self.update_resolution_options,
                inputs=model_dropdown,
                outputs=resolution
            )

            mode.change(
                self.update_visibility,
                inputs=mode,
                outputs=[model_dropdown, input_image, test_image_input]
            )

            generate_btn.click(
                self.Image_Generation,
                inputs=[api_key, mode, model_dropdown, resolution, prompt, input_image, test_image_input],
                outputs=[progress, output_image, status_msg, filename_input]
            )

            browse_folder_btn.click(
                self.browse_output_folder,
                inputs=None,
                outputs=output_folder
            )

            save_btn.click(
                self.save_image_to_folder,
                inputs=[output_image, output_folder, filename_input],
                outputs=status_msg
            )

            download_btn.click(
                self.prepare_download,
                inputs=[output_image, filename_input],
                outputs=[download_file, status_msg]
            )
            
            codeformer_browse_folder_btn.click(
                self.browse_output_folder,
                inputs=None,
                outputs=codeformer_output_folder
            )

            codeformer_save_btn.click(
                self.save_image_to_folder,
                inputs=[inpainted_face_image, codeformer_output_folder, codeformer_filename_input],
                outputs=status_msg
            )

            codeformer_download_btn.click(
                self.prepare_download,
                inputs=[inpainted_face_image, codeformer_filename_input],
                outputs=[codeformer_download_file, status_msg]
            )

            # 3D reconstruct event (moved here)
            codeformer_reconstruct_3d_btn.click(
                self.reconstruct_3d_face,
                inputs=[inpainted_face_image, output_folder_3d],
                outputs=[progress, status_msg, output_3d_viewer, vis_image, filename_input_3d]
            )
            
            crop_and_align_btn.click(
                self.codeformer_crop_and_align_face,
                inputs=[output_image],
                outputs=[codeformer_cropped_aligned_image]
            )
            
            enhance_face_button.click(
                self.codeformer_enhance_face,
                inputs=[codeformer_cropped_aligned_image],
                outputs=[codeformer_enhanced_image]
            )
            
            inpaint_face_button.click(
                self.codeformer_inpaint_face,
                inputs=[codeformer_enhanced_image],
                outputs=[inpainted_face_image]
            )
            

        return app

    def update_resolution_options(self, model):
        if model == "DALL-E-3":
            return gr.update(choices=self.DALLE3_RESOLUTIONS, value=self.DALLE3_RESOLUTIONS[0])
        elif model == "DALL-E-2":
            return gr.update(choices=self.DALLE2_RESOLUTIONS, value=self.DALLE2_RESOLUTIONS[2])
        else:  # GPT-Image-1
            return gr.update(choices=self.GPT_IMAGE_RESOLUTIONS, value=self.GPT_IMAGE_RESOLUTIONS[0])

    def update_visibility(self, mode):
        # Update visibility based on mode selection
        if mode == "text2img":
            # For text2img, show all models
            model_choices = ["DALL-E-2", "DALL-E-3", "GPT-Image-1"]
            model_value = "DALL-E-2"
            input_image_visible = False
            test_image_visible = False
        elif mode == "img2img":
            # For img2img, only GPT-Image-1 is valid
            model_choices = ["DALL-E-2", "GPT-Image-1"]
            model_value = "DALL-E-2"
            input_image_visible = True
            test_image_visible = False
        else:  # test mode
            model_choices = ["DALL-E-2", "DALL-E-3", "GPT-Image-1"]
            model_value = "DALL-E-2"
            input_image_visible = False
            test_image_visible = True

        return [
            gr.update(choices=model_choices, value=model_value),
            gr.update(visible=input_image_visible),
            gr.update(visible=test_image_visible)
        ]

    def browse_output_folder(self):
        try:
            # Create and hide root window
            root = tk.Tk()
            root.withdraw()
            root.attributes('-topmost', True)

            # Open folder selection dialog
            folder_path = filedialog.askdirectory(
                title="Select Output Folder",
                initialdir=self.output_dir
            )

            root.destroy()

            if folder_path:
                self.output_dir = folder_path
                return folder_path
            else:
                return self.output_dir
        except Exception as e:
            print(f"Error in folder selection: {e}")
            return self.output_dir

    def Image_Generation(self, api_key, mode, model, resolution, prompt, input_image=None, test_image=None):
        is_test_mode = mode.startswith("test mode")

        # Generate a new filename based on timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        new_filename = f"generated_image_{timestamp}"

        if is_test_mode:
            # Simulate API call with delay
            for i in range(10):
                time.sleep(0.5)
                yield i*10, test_image if test_image else None, "Test mode: Processing...", new_filename

            # Return test image or a placeholder
            if test_image:
                self.last_generated_image = test_image
                yield 100, test_image, "Test mode: Completed with test image", new_filename
            else:
                # Create a simple placeholder image with text
                img = Image.new('RGB', (512, 512), color=(73, 109, 137))
                self.last_generated_image = img
                yield 100, img, "Test mode: Completed with placeholder image", new_filename
            return

        try:
            # Set API key for handler
            self.api_handler.api_key = api_key
            self.api_handler.headers["Authorization"] = f"Bearer {api_key}"

            # Initialize progress
            yield 10, None, "Starting image generation...", new_filename

            if mode == "text2img":
                # Text to image generation
                yield 30, None, f"Generating image with {model}...", new_filename
                image = self.api_handler.text_to_image(model, resolution, prompt)
                yield 70, None, "Processing generated image...", new_filename
                self.last_generated_image = image
                yield 100, image, f"Image successfully generated with {model}", new_filename

            elif mode == "img2img":
                # Image to image editing
                yield 30, None, "Processing input image...", new_filename
                yield 50, None, "Applying edits to image...", new_filename

                try:
                    model_used = model
                    image = self.api_handler.image_to_image(model, resolution, prompt, input_image)
                except ValueError as e:
                    if "verified" in str(e).lower():
                        # If verification error, try with DALL-E-2 fallback
                        yield 60, None, "GPT-Image-1 requires verification. Falling back to DALL-E-2...", new_filename
                        image = self.api_handler._fallback_dalle2_edit(resolution, prompt, input_image)
                        model_used = "DALL-E-2 (fallback)"
                    else:
                        raise

                yield 70, None, "Finalizing image...", new_filename
                self.last_generated_image = image
                yield 100, image, f"Image successfully edited with {model_used}", new_filename

        except Exception as e:
            yield 100, None, f"Error: {str(e)}", new_filename
            raise gr.Error(f"Error: {str(e)}")

    def save_image_to_folder(self, img, folder_path, filename):
        if img is None:
            return "No image to save"

        try:
            # Ensure folder exists
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            # Clean filename and add extension
            filename = os.path.basename(filename)
            if not filename:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"generated_image_{timestamp}"

            if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                filename += ".png"

            # Full path to save
            full_path = os.path.join(folder_path, filename)

            # Save the image
            img.save(full_path)
            return f"Image saved to {full_path}"
        except Exception as e:
            return f"Error saving image: {str(e)}"

    def prepare_download(self, img, filename):
        if img is None:
            return gr.update(visible=False), "No image to save"

        try:
            # Clean filename and add extension
            filename = os.path.basename(filename)
            if not filename:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"generated_image_{timestamp}"

            if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                filename += ".png"

            # Save the image to a temporary file
            temp_path = os.path.join(self.output_dir, filename)
            img.save(temp_path)

            # Make the download component visible and return the file path
            return gr.update(value=temp_path, visible=True), f"Click the download button above to save {filename}"
        except Exception as e:
            return gr.update(visible=False), f"Error preparing download: {str(e)}"

    def reconstruct_3d_face(self, input_image, output_folder):
        """
        Process the input image to create a 3D face reconstruction
        """
        if input_image is None:
            return 0, "No image to process", None, None, ""

        try:
            # Generate a new filename based on timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            new_filename = f"face_{timestamp}"

            # Update progress
            yield 10, "Starting 3D face reconstruction...", None, None, new_filename

            # Ensure output folder exists
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

            # Update progress
            yield 30, "Detecting face in image...", None, None, new_filename

            # Determine device
            # device = 'cuda' if torch.cuda.is_available() else 'cpu'
            # yield 40, f"Using {device} for processing...", None, None, new_filename

            # Process the image
            yield 50, "Reconstructing 3D face model...", None, None, new_filename

            # Call the face reconstruction function
            result_paths = reconstruct_3d_face(
                input_image=input_image,
                save_folder=output_folder,
                device='cuda',
                save_depth=False,
                save_obj=True,
                save_vis=True
            )
            
            # result_paths = {
            #     'obj_path' : 'None',
            #     'vis_path' : 'None',
            # }

            yield 80, "Processing complete, loading results...", None, None, new_filename

            # Get the paths from the result
            obj_path = result_paths.get('obj_path')
            vis_path = result_paths.get('vis_path')

            # Update the filename input with the actual filename
            if obj_path:
                base_filename = os.path.basename(obj_path)
                new_filename = os.path.splitext(base_filename)[0]

            # Return the results
            yield 100, "3D reconstruction completed successfully!", obj_path, vis_path, new_filename

        except Exception as e:
            yield 100, f"Error in 3D reconstruction: {str(e)}", None, None, new_filename

    def save_3d_model_to_folder(self, model_path, folder_path, filename):
        if model_path is None:
            return "No 3D model to save"

        try:
            # Ensure folder exists
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            # Clean filename and add extension
            filename = os.path.basename(filename)
            if not filename:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"reconstructed_3d_{timestamp}"

            if not filename.lower().endswith('.obj'):
                filename += ".obj"

            # Full path to save
            full_path = os.path.join(folder_path, filename)

            # Copy the model file to the destination
            if os.path.exists(model_path):
                shutil.copy2(model_path, full_path)
                return f"3D model saved to {full_path}"
            else:
                return f"Error: Source model file not found at {model_path}"
        except Exception as e:
            return f"Error saving 3D model: {str(e)}"

    def prepare_3d_download(self, model_path, filename):
        if model_path is None:
            return gr.update(visible=False), "No 3D model to save"

        try:
            # Clean filename and add extension
            filename = os.path.basename(filename)
            if not filename:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"reconstructed_3d_{timestamp}"

            if not filename.lower().endswith('.obj'):
                filename += ".obj"

            # Make the download component visible and return the file path
            return gr.update(value=model_path, visible=True), f"Click the download button above to save {filename}"
        except Exception as e:
            return gr.update(visible=False), f"Error preparing download: {str(e)}"
        
    def codeformer_crop_and_align_face(self, img):
        return align_face(img)
    
    def codeformer_enhance_face(self, img):
        img.save('temp.png')
        return_code, output, error = run_codeformer_inference('temp.png', 0.5, True)
        return Image.open('results/test_img_0.5/restored_faces/temp.png')
    
    def codeformer_inpaint_face(self, dict_img_mask):
        
        original = dict_img_mask["image"]  # PIL Image
        mask = dict_img_mask["mask"]       # PIL Image

        # Convert both to numpy arrays
        original_np = np.array(original)
        mask_np = np.array(mask.convert("L"))  # Ensure mask is grayscale

        # Create a boolean mask: True where mask is white (255)
        white_mask = mask_np == 255

        # If the image is RGB
        if original_np.ndim == 3:
            # Set masked pixels to white
            original_np[white_mask] = [255, 255, 255]
        else:
            # For grayscale images
            original_np[white_mask] = 255

        # Convert back to PIL Image
        img = Image.fromarray(original_np)
        img.save('temp.png')
        return_code, output, error = run_inference_inpainting('temp.png')
        return Image.open('results/test_inpainting_img/temp.png')
    