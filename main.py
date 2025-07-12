from openai_api_wraper import *
from gradio_app import *

def main():
    # Initialize the API handler
    api_handler = OpenAIImageAPI(api_key="")

    # Initialize the GUI with the API handler
    gui = ImageGeneratorGUI(api_handler)

    # Create and launch the interface
    app = gui.create_interface()
    app.queue()
    app.launch(share=True)

if __name__ == "__main__":
    main()