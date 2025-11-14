# app.py (Updated with Landing Page)
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, current_app, session, flash
import os
import uuid
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import markdown2
import datetime

# Import database
from database import db, init_db, close_db

# Import preprocessing functions
try:
    from utils.preprocessing import preprocess_image_inference, denoise_image, IMAGE_SIZE
except ImportError:
    print("Error: Could not import preprocessing functions from utils.preprocessing.")
    def preprocess_image_inference(image_path, **kwargs): return None
    def denoise_image(image_path, **kwargs): return None
    IMAGE_SIZE = 256

# Import model loading and ensemble prediction functions
try:
    from utils.models import initialize_models, predict_ensemble
except ImportError:
    print("Error: Could not import model functions from utils.models.")
    def initialize_models(): print("Model functions not available.")
    def predict_ensemble(image_tensor, loaded_models): return None

# Import Grad-CAM functions
try:
    from utils.gradcam import generate_gradcam_overlay
except ImportError:
     print("Error: Could not import Grad-CAM functions from utils.gradcam.")
     def generate_gradcam_overlay(model, image_tensor, original_image_np, target_class_idx):
          print("Grad-CAM function not available.")
          return None

# Import Gemini utility functions
try:
    from utils.gemini_utils import generate_report_from_results, is_fundus_image_with_gemini
except ImportError:
    print("Error: Could not import gemini_utils functions from utils.gemini_utils.")
    def generate_report_from_results(model_results_text, patient_name=None, patient_age=None, analysis_datetime_str=None):
         return {'comprehensive': 'Gemini API not available.', 'synthetic': 'Gemini API not available.'}
    def is_fundus_image_with_gemini(image_path): return False, "Gemini validation unavailable.", "Validation function not available."


app = Flask(__name__)

# Configuration
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your-secret-key-change-this-in-production')
app.config['MONGODB_URI'] = os.getenv('MONGODB_URI', 'mongodb://127.0.0.1:27017/')
app.config['MONGODB_DB_NAME'] = os.getenv('MONGODB_DB_NAME', 'thilak')

# Configuration for uploaded files
UPLOAD_FOLDER = 'static/uploaded_images'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# --- Define Normalization Stats ---
NORMALIZATION_MEAN = [0.4247607406554276, 0.41477363903977665, 0.40914320244558566]
NORMALIZATION_STD = [0.261673335762141, 0.2545360808997545, 0.2384415815175597]

inference_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=NORMALIZATION_MEAN, std=NORMALIZATION_STD)
])

# --- Define DR Stage Labels ---
DR_STAGE_LABELS = {
    0: "No Diabetic Retinopathy",
    1: "Mild Non-Proliferative Diabetic Retinopathy (NPDR)",
    2: "Moderate Non-Proliferative Diabetic Retinopathy (NPDR)",
    3: "Severe Non-Proliferative Diabetic Retinopathy (NPDR)",
    4: "Proliferative Diabetic Retinopathy (PDR)"
}


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def login_required(f):
    """Decorator to require login for routes"""
    from functools import wraps
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'username' not in session:
            flash('Please login to access this page.', 'warning')
            return redirect(url_for('index'))
        return f(*args, **kwargs)
    return decorated_function


@app.route('/')
def index():
    """Landing page"""
    return render_template('landing.html')


@app.route('/upload')
@login_required
def upload():
    """Upload page - requires login"""
    return render_template('index.html')


@app.route('/signup', methods=['POST'])
def signup():
    """Handle user signup"""
    username = request.form.get('username')
    password = request.form.get('password')
    email = request.form.get('email', None)
    
    if not username or not password:
        flash('Username and password are required.', 'error')
        return redirect(url_for('index'))
    
    # Hash the password
    password_hash = generate_password_hash(password)
    
    # Create user in database
    user = db.create_user(username, password_hash, email)
    
    if user:
        flash('Account created successfully! Please login.', 'success')
    else:
        flash('Username already exists. Please choose a different username.', 'error')
    
    return redirect(url_for('index'))


@app.route('/login', methods=['POST'])
def login():
    """Handle user login"""
    username = request.form.get('username')
    password = request.form.get('password')
    
    if not username or not password:
        flash('Username and password are required.', 'error')
        return redirect(url_for('index'))
    
    # Find user in database
    user = db.find_user(username)
    
    if user and check_password_hash(user['password_hash'], password):
        session['username'] = username
        db.update_last_login(username)
        flash(f'Welcome back, {username}!', 'success')
        return redirect(url_for('upload'))
    else:
        flash('Invalid username or password.', 'error')
    
    return redirect(url_for('index'))


@app.route('/logout')
def logout():
    """Handle user logout"""
    session.pop('username', None)
    flash('You have been logged out.', 'info')
    return redirect(url_for('index'))


@app.route('/process', methods=['POST'])
@login_required
def process_images():
    loaded_models = current_app.config.get('LOADED_MODELS')

    if loaded_models is None or not isinstance(loaded_models, dict) or not any(model is not None for model in loaded_models.values()):
         print("FATAL ERROR: Models are not available in app.config.")
         return render_template('index.html', error_message="FATAL ERROR: Models not available.")

    # --- Get Patient Info from Form ---
    patient_name = request.form.get('patient_name', 'N/A')
    patient_age = request.form.get('patient_age', 'N/A')
    analysis_datetime_obj = datetime.datetime.now()
    analysis_datetime_str = analysis_datetime_obj.strftime("%Y-%m-%d %H:%M:%S")

    # Get current username from session
    username = session.get('username', 'anonymous')

    # --- Handle single image file upload ---
    image_file = request.files.get('fundus_image')

    if not image_file or image_file.filename == '':
         print("Image file is missing or empty.")
         return render_template('index.html', error_message="Please upload an image file.")

    # Initialize variables to None
    original_url = filtered_url = denoised_url = None
    model_results_text = None
    reports = None
    gradcam_data = []
    original_path = None
    filtered_path = None

    # --- Process the single image ---
    if image_file and allowed_file(image_file.filename):
        original_filename = secure_filename(image_file.filename)
        file_extension = os.path.splitext(original_filename)[1]
        unique_filename_base = str(uuid.uuid4())
        original_filename_saved = f"{unique_filename_base}_original_{original_filename}"
        original_path = os.path.join(app.config['UPLOAD_FOLDER'], original_filename_saved)
        image_file.save(original_path)
        original_url = url_for('static', filename=f'uploaded_images/{original_filename_saved}')
        print(f"Saved original image: {original_filename_saved}")

        # --- STEP 1: Validate image type with Gemini ---
        is_fundus, gemini_validation_response, validation_error = is_fundus_image_with_gemini(original_path)

        if not is_fundus:
             print(f"Image validation failed: {validation_error or gemini_validation_response}")
             if os.path.exists(original_path):
                 os.remove(original_path)
             return render_template('index.html', error_message=f"Uploaded image is not a fundus image or validation failed.")

        print("Image validated as fundus image. Proceeding with analysis.")

        # Load original image numpy array
        original_image_np = cv2.imread(original_path, cv2.IMREAD_COLOR)
        if original_image_np is not None:
             original_image_np = cv2.cvtColor(original_image_np, cv2.COLOR_BGR2RGB)

        # --- STEP 2: Apply filters and display ---
        filtered_image_np = preprocess_image_inference(original_path, image_size=IMAGE_SIZE)
        if filtered_image_np is not None:
            filtered_filename_saved = f"{unique_filename_base}_filtered_{original_filename}"
            filtered_path = os.path.join(app.config['UPLOAD_FOLDER'], filtered_filename_saved)
            filtered_image_bgr = cv2.cvtColor(filtered_image_np, cv2.COLOR_RGB2BGR)
            cv2.imwrite(filtered_path, filtered_image_bgr)
            filtered_url = url_for('static', filename=f'uploaded_images/{filtered_filename_saved}')
            print(f"Saved filtered image: {filtered_filename_saved}")

            # --- Prepare Filtered Image for Model Inference ---
            filtered_tensor = inference_transform(filtered_image_np)
            filtered_tensor = filtered_tensor.unsqueeze(0)

            # --- STEP 3 & 5: Model Evaluation & Overall Result ---
            ensemble_results = predict_ensemble(filtered_tensor, loaded_models)

            if ensemble_results is not None:
                # --- Format Model Results Text with Labels ---
                model_results_text = "Individual Model Results:\n"
                md_lines = []
                md_lines.append("### Individual Model Results")
        
                if ensemble_results['individual_results']:
                    for name, res in ensemble_results['individual_results'].items():
                        if 'error' in res:
                            model_results_text += f"- {name}: Error during inference - {res['error']}\n"
                            md_lines.append(f"| **{name}** | ⚠️ | *Error* | — |")
                        else:
                            stage_label = DR_STAGE_LABELS.get(res['predicted_class'], "Unknown Stage")
                            model_results_text += f"- {name}: Stage {res['predicted_class']} (Confidence: {res['confidence']:.4f}) - **{stage_label}**\n"
                            md_lines.append(
                                f"- **{name}**: {stage_label} - Stage {res['predicted_class']} **(Confidence: {res['confidence']:.4f})**"
                            )
                else:
                    model_results_text += "No individual model results available.\n"
                    md_lines.append("*No individual model results available.*")
        
                model_results_text += f"\nEnsemble Prediction (Majority Vote):\n"
                md_lines.append("")
                md_lines.append("### Ensemble Prediction (Majority Vote)")
                ensemble_prediction = ensemble_results.get('ensemble_prediction')
                ensemble_confidence = ensemble_results.get('ensemble_confidence_fraction')

                if ensemble_prediction is not None:
                    ensemble_stage_label = DR_STAGE_LABELS.get(ensemble_prediction, "Unknown Stage")
                    model_results_text += f"- Predicted Stage: {ensemble_prediction} (Confidence: {ensemble_confidence:.4f} of successful models) - **{ensemble_stage_label}**\n"
                    md_lines.append(
                        f"- **Predicted Stage:** **{ensemble_prediction}** ({ensemble_stage_label})"
                    )
                    md_lines.append(
                        f"- **Confidence:** **{ensemble_confidence:.4f}** of successful models"
                    )
                elif ensemble_results['individual_results']:
                    model_results_text += "- Could not determine (no clear majority or all failed).\n"
                    md_lines.append("- Could not determine (no clear majority or all failed).")
                else:
                    model_results_text += "- No models were available for prediction.\n"
                    md_lines.append("- No models were available for prediction.")
        
                model_results_md = "\n".join(md_lines)
                model_results_html = markdown2.markdown(model_results_md)

                # --- STEP 4: Generate GradCAM ---
                gradcam_paths = []
                if ensemble_prediction is not None and ensemble_prediction != 0 and original_image_np is not None:
                    print(f"Generating Grad-CAMs for Stage {ensemble_prediction}...")
                    target_class_idx = ensemble_prediction

                    for model_name, model in loaded_models.items():
                        if model is not None:
                            print(f"  Generating Grad-CAM for {model_name}...")
                            gradcam_overlay_np = generate_gradcam_overlay(
                                model,
                                filtered_tensor,
                                original_image_np,
                                target_class_idx
                            )

                            if gradcam_overlay_np is not None:
                                gradcam_filename_saved = f"{unique_filename_base}_gradcam_{model_name}_stage{target_class_idx}_{original_filename}"
                                gradcam_path = os.path.join(app.config['UPLOAD_FOLDER'], gradcam_filename_saved)
                                cv2.imwrite(gradcam_path, gradcam_overlay_np)
                                gradcam_data.append({
                                    'url': url_for('static', filename=f'uploaded_images/{gradcam_filename_saved}'),
                                    'model_name': model_name
                                })
                                gradcam_paths.append(gradcam_filename_saved)
                                print(f"  Saved Grad-CAM for {model_name}: {gradcam_filename_saved}")
                
                elif ensemble_prediction == 0:
                    print("Skipping Grad-CAM generation because ensemble prediction is Stage 0 (No DR).")

                # --- STEP 6: Generate Reports ---
                reports = generate_report_from_results(
                    model_results_text,
                    patient_name=patient_name,
                    patient_age=patient_age,
                    analysis_datetime_str=analysis_datetime_str
                )

                # --- SAVE TO DATABASE ---
                analysis_data = {
                    'original_image_path': original_filename_saved,
                    'filtered_image_path': filtered_filename_saved if filtered_path else None,
                    'denoised_image_path': None,  # Will be updated below
                    'ensemble_prediction': ensemble_prediction,
                    'ensemble_confidence': ensemble_confidence,
                    'individual_results': ensemble_results.get('individual_results', {}),
                    'comprehensive_report': reports.get('comprehensive', ''),
                    'synthetic_report': reports.get('synthetic', ''),
                    'gradcam_paths': gradcam_paths
                }

                # Save to database
                analysis_id = db.save_analysis(
                    username=username,
                    patient_name=patient_name,
                    patient_age=patient_age,
                    analysis_data=analysis_data
                )

                if analysis_id:
                    print(f"✓ Analysis saved to database with ID: {analysis_id}")

            else:
                model_results_text = "Model inference failed."
                reports = {'comprehensive': 'Model inference failed.', 'synthetic': 'Model inference failed.'}

        # Apply Denoising (CLAHE Green Channel)
        if original_image_np is not None:
            denoised_image_np = denoise_image(original_path, image_size=IMAGE_SIZE)
            if denoised_image_np is not None:
                denoised_filename_saved = f"{unique_filename_base}_clahe_{original_filename}"
                denoised_path = os.path.join(app.config['UPLOAD_FOLDER'], denoised_filename_saved)
                denoised_image_bgr = cv2.cvtColor(denoised_image_np, cv2.COLOR_RGB2BGR)
                cv2.imwrite(denoised_path, denoised_image_bgr)
                denoised_url = url_for('static', filename=f'uploaded_images/{denoised_filename_saved}')
                print(f"Saved CLAHE Green: {denoised_filename_saved}")

    # --- Render Results ---
    reports_html = {k: markdown2.markdown(v) for k, v in reports.items()} if reports else None

    return render_template('results.html',
                           original_url=original_url,
                           filtered_url=filtered_url,
                           denoised_url=denoised_url,
                           model_results_text=model_results_html,
                           gradcam_data=gradcam_data,
                           reports_html=reports_html,
                           patient_name=patient_name,
                           patient_age=patient_age,
                           analysis_datetime=analysis_datetime_str
                           )


@app.route('/history')
@login_required
def history():
    """Show user's analysis history"""
    username = session.get('username')
    analyses = db.get_user_analyses(username, limit=20)
    return render_template('history.html', analyses=analyses)


if __name__ == '__main__':
    print("Loading models on startup...")
    loaded_models_dict = initialize_models()

    if not loaded_models_dict or not isinstance(loaded_models_dict, dict) or not any(model is not None for model in loaded_models_dict.values()):
        print("FATAL ERROR: Models failed to load on startup.")
        app.config['LOADED_MODELS'] = {}
    else:
        app.config['LOADED_MODELS'] = loaded_models_dict
        print("Models stored on app.config.")

    # Initialize database
    print("\nInitializing database connection...")
    if init_db(app):
        print("✓ Application ready to start")
    else:
        print("⚠ Warning: Database connection failed. Some features may not work.")

    try:
        app.run(debug=True)
    finally:
        # Close database connection when app shuts down
        close_db()