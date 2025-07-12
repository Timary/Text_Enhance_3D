import requests
import io
from PIL import Image
from base64 import b64decode

# Constants
DALLE3_RESOLUTIONS = ["1024x1024", "1792x1024", "1024x1792"]
GPT_IMAGE_RESOLUTIONS = ["1024x1024", "1536x1024", "1024x1536", "auto"]
DALLE2_RESOLUTIONS = ["256x256", "512x512", "1024x1024"]

class OpenAIImageAPI:
    def __init__(self, api_key):
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {api_key}"
        }

    def validate_api_key(self):
        if not self.api_key:
            raise ValueError("API key is required")

    def text_to_image(self, model, resolution, prompt):
        """Generate an image from text prompt using OpenAI API"""
        self.validate_api_key()

        endpoint = "https://api.openai.com/v1/images/generations"

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        # Determine which model to use
        if model == "DALL-E-3":
            model_name = "dall-e-3"
        elif model == "DALL-E-2":
            model_name = "dall-e-2"
            # Ensure resolution is compatible with DALL-E-2
            if resolution not in DALLE2_RESOLUTIONS:
                resolution = "1024x1024"  # Default to highest resolution
        else:  # GPT-Image-1
            model_name = "gpt-image-1"

        # Try with the selected model
        try:
            payload = {
                "model": model_name,
                "prompt": prompt,
                "n": 1,
                "size": resolution,
                "response_format": "b64_json"
            }

            response = requests.post(endpoint, headers=headers, json=payload)
            response.raise_for_status()

            # Process response
            response_data = response.json()
            image_data = b64decode(response_data["data"][0]["b64_json"])
            return Image.open(io.BytesIO(image_data))

        except requests.exceptions.HTTPError as e:
            # Check if it's a verification error for GPT-Image-1
            if model_name == "gpt-image-1" and "verified" in str(e).lower():
                raise ValueError(
                    "Your organization must be verified to use GPT-Image-1. "
                    "Please go to https://platform.openai.com/settings/organization/general "
                    "and click on Verify Organization. If you just verified, it can take up to 15 minutes for access to propagate."
                )
            else:
                raise

    def image_to_image(self, model, resolution, prompt, input_image):
        """Edit an image based on a prompt using OpenAI API"""
        self.validate_api_key()

        if input_image is None:
            raise ValueError("Input image is required for image-to-image mode")

        # The endpoint for image edits
        endpoint = "https://api.openai.com/v1/images/edits"

        # Prepare the image
        img_byte_arr = io.BytesIO()
        input_image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)

        # Try with GPT-Image-1 first
        try:
            # Create multipart form data with a single image
            files = {
                'image': ('image.png', img_byte_arr, 'image/png'),
            }

            data = {
                'prompt': prompt,
                'model': model,
                'n': 1,
                'size' : resolution,
            }

            # Make API request
            response = requests.post(
                endpoint,
                headers=self.headers,
                files=files,
                data=data
            )

            # Check for verification error
            if response.status_code == 400 and "verified" in response.text.lower():
                raise ValueError(
                    "Your organization must be verified to use GPT-Image-1. "
                    "Please go to https://platform.openai.com/settings/organization/general "
                    "and click on Verify Organization. If you just verified, it can take up to 15 minutes for access to propagate."
                )

            response.raise_for_status()

            # Process response
            response_data = response.json()
            image_data = b64decode(response_data["data"][0]["b64_json"])
            return Image.open(io.BytesIO(image_data))

        except ValueError as e:
            # Re-raise verification errors
            if "verified" in str(e).lower():
                raise
            else:
                # For other errors, try with DALL-E-2 as fallback
                return self._fallback_dalle2_edit(resolution, prompt, input_image)

        except Exception as e:
            # For any other exception, try with DALL-E-2 as fallback
            return self._fallback_dalle2_edit(resolution, prompt, input_image)

    def _fallback_dalle2_edit(self, resolution, prompt, input_image):
        """Fallback to DALL-E-2 for image editing"""
        endpoint = "https://api.openai.com/v1/images/edits"

        # Reset the image byte array
        img_byte_arr = io.BytesIO()
        input_image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)

        # Create a transparent mask (required for DALL-E-2)
        mask = Image.new("RGBA", input_image.size, (0, 0, 0, 0))
        mask_byte_arr = io.BytesIO()
        mask.save(mask_byte_arr, format='PNG')
        mask_byte_arr.seek(0)

        # Adjust resolution for DALL-E-2
        if resolution not in DALLE2_RESOLUTIONS:
            # Default to closest DALL-E-2 resolution
            resolution = "1024x1024"

        files = {
            'image': ('image.png', img_byte_arr, 'image/png'),
            'mask': ('mask.png', mask_byte_arr, 'image/png'),
        }

        data = {
            'prompt': prompt,
            'n': 1,
            'size': resolution,
            'response_format': 'b64_json',
            # DALL-E-2 is the default model for this endpoint
        }

        response = requests.post(
            endpoint,
            headers=self.headers,
            files=files,
            data=data
        )

        # Raise detailed error for debugging
        if response.status_code != 200:
            error_detail = response.json() if response.text else "No error details"
            raise ValueError(f"API Error ({response.status_code}): {error_detail}")

        # Process response
        response_data = response.json()
        image_data = b64decode(response_data["data"][0]["b64_json"])
        return Image.open(io.BytesIO(image_data))