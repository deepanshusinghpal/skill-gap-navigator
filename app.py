import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from gradio_app import app

if __name__ == "__main__":
    # Fetch the port Render assigns, default to 7860 locally
    port = int(os.environ.get("PORT", 7860))
    app.launch(server_name="0.0.0.0", server_port=port)