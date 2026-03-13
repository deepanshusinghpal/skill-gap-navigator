import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from gradio_app import app

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860, show_api=False)