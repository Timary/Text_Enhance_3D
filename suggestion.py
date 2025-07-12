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

            # --- NEW TAB ---
            with gr.TabItem("CodeFormer Enhancement"):
                # Image input with mask drawing utility
                codeformer_image = gr.Image(
                    label="Input Image (Draw Mask to Select Face Region)",
                    tool="sketch",  # enables drawing/masking
                    type="pil"
                )

                # Restore Face button
                restore_face_btn = gr.Button("Restore Face")

                # Output image after restoration
                codeformer_output_image = gr.Image(label="Restored Image", type="pil")

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

                # Event handlers for CodeFormer tab
                codeformer_browse_folder_btn.click(
                    self.browse_output_folder,
                    inputs=None,
                    outputs=codeformer_output_folder
                )

                codeformer_save_btn.click(
                    self.save_image_to_folder,
                    inputs=[codeformer_output_image, codeformer_output_folder, codeformer_filename_input],
                    outputs=self.status_msg  # or a dedicated status textbox
                )

                codeformer_download_btn.click(
                    self.prepare_download,
                    inputs=[codeformer_output_image, codeformer_filename_input],
                    outputs=[codeformer_download_file, self.status_msg]
                )

                # Restore face event
                restore_face_btn.click(
                    self.restore_face,  # You need to implement this method
                    inputs=[codeformer_image],
                    outputs=codeformer_output_image
                )

                # 3D reconstruct event (moved here)
                codeformer_reconstruct_3d_btn.click(
                    self.reconstruct_3d_face,
                    inputs=[codeformer_output_image, codeformer_output_folder],
                    outputs=[self.progress, self.status_msg, self.output_3d_viewer, self.vis_image, codeformer_filename_input]
                )

    return app