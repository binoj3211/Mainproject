from flask import Flask, render_template, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from diffusers import StableDiffusionPipeline
import torch
import os
import uuid
from datetime import datetime
from flask_cors import CORS
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppresses INFO, WARNING, and ERROR messages
 # Suppresses most TensorFlow warnings

# Ensure 'static' directory exists for saving images
os.makedirs("static", exist_ok=True)
os.environ['XLA_FLAGS'] = '--xla_gpu_triton_gemm_disable=true'

# Disable Triton optimizations on Windows
os.environ["XFORMERS_FORCE_DISABLE_TRITON"] = "1"

app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)  # Enable CORS for all routes

# SQLite Database Configuration
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///prompts.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)

# Database Model for Storing Prompts
class Prompt(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    prompt_text = db.Column(db.String(500), nullable=False)
    image_url = db.Column(db.String(200), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

# Initialize database inside the app context
with app.app_context():
    db.create_all()

# Lazy loading of the Stable Diffusion model
model = None

def load_model():
    """Loads and initializes the Stable Diffusion model locally."""
    global model
    if model is not None:
        return model

    try:
        local_model_path = "model"  # Update this path
        device = "cuda" if torch.cuda.is_available() else "cpu"

        model = StableDiffusionPipeline.from_pretrained(
            local_model_path,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            use_safetensors=True
        ).to(device)

        try:
            model.enable_xformers_memory_efficient_attention()
        except Exception as e:
            print(f"Xformers not available: {e}")

        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

@app.route('/')
def index():
    """Renders the main webpage."""
    return render_template("index.html")

@app.route('/generate', methods=['POST'])
def generate():
    """Handles image generation from text prompts and saves them."""
    global model
    if model is None:
        model = load_model()  # Load model on first request

    prompt = request.json.get('prompt', '').strip()
    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400

    try:
        with torch.no_grad():
            image = model(prompt).images[0]

        # Generate a unique filename
        image_filename = f"static/{uuid.uuid4().hex}.png"
        image.save(image_filename)

        # Save the prompt and image to the database
        new_prompt = Prompt(prompt_text=prompt, image_url=image_filename)
        db.session.add(new_prompt)
        db.session.commit()

        return jsonify({"prompt": prompt, "image_url": image_filename})
    except Exception as e:
        print(f"Error generating image: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/view-prompts', methods=['GET'])
def view_prompts():
    """Fetches all saved prompts from the database."""
    prompts = Prompt.query.order_by(Prompt.timestamp.desc()).all()
    data = [{
        "id": p.id,
        "prompt_text": p.prompt_text,
        "image_url": p.image_url,
        "timestamp": p.timestamp.isoformat()
    } for p in prompts]
    return jsonify(data)

@app.route('/delete-prompt/<int:prompt_id>', methods=['DELETE'])
def delete_prompt(prompt_id):
    """Deletes a single prompt and its associated image."""
    try:
        prompt = Prompt.query.get_or_404(prompt_id)

        # Delete the associated image file
        if os.path.exists(prompt.image_url):
            os.remove(prompt.image_url)

        # Delete the prompt from the database
        db.session.delete(prompt)
        db.session.commit()

        return jsonify({"message": "Prompt and image deleted successfully"})
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500

@app.route('/delete-all-prompts', methods=['DELETE'])
def delete_all_prompts():
    """Deletes all prompts and their associated images."""
    try:
        prompts = Prompt.query.all()
        for prompt in prompts:
            if os.path.exists(prompt.image_url):
                os.remove(prompt.image_url)

        num_deleted = Prompt.query.delete()
        db.session.commit()

        return jsonify({"message": f"Deleted {num_deleted} prompts and their images successfully"})
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500

@app.route('/shutdown', methods=['POST'])
def shutdown():
    """Gracefully shuts down the server."""
    global model
    if model:
        del model  # Free GPU memory
        torch.cuda.empty_cache()
    return jsonify({"message": "Server shutting down..."})

if __name__ == '__main__':
    app.run(debug=True)