# utils/gemini_utils.py
import os
import google.generativeai as genai
from dotenv import load_dotenv
import cv2
import numpy as np
import json

# Load environment variables from .env file
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    print("Error: GEMINI_API_KEY not found in .env file.")
    raise ValueError("GEMINI_API_KEY not set in .env file")

try:
    genai.configure(api_key=GEMINI_API_KEY)
    print("Gemini API configured from utils.gemini_utils.")
except Exception as e:
    print(f"Error configuring Gemini API in utils.gemini_utils: {e}")
    raise e

# Use a current, stable multimodal model like 'gemini-1.5-flash-latest' or 'gemini-pro-vision'
MULTIMODAL_MODEL_NAME = "gemini-2.5-flash-lite"


def get_gemini_model(model_name):
    """Helper to get a GenerativeModel instance."""
    try:
        model = genai.GenerativeModel(model_name)
        return model
    except Exception as e:
        print(f"Error getting Gemini model {model_name}: {e}")
        return None


# --- Function to Generate Text Report from Model Results (Corrected Prompt Variable) ---
def generate_report_from_results(model_results_text, patient_name=None, patient_age=None, analysis_datetime_str=None):
    """
    Generates a textual report based on provided model results text and patient info.

    Args:
        model_results_text (str): A formatted string summarizing model predictions.
        patient_name (str, optional): Patient's name. Defaults to None.
        patient_age (str, optional): Patient's age. Defaults to None.
        analysis_datetime_str (str, optional): Formatted string of analysis date/time. Defaults to None.

    Returns:
        dict: A dictionary containing the comprehensive and synthetic report text, or error messages.
    """
    model = get_gemini_model(MULTIMODAL_MODEL_NAME)
    if model is None:
        return {
            'comprehensive': 'Error: Gemini model not available for report generation.',
            'synthetic': 'Error: Gemini model not available for report generation.'
        }

    # Format patient info and date/time for the report
    patient_info = ""
    if patient_name != 'N/A' and patient_name is not None and patient_name.strip() != '':
        patient_info += f"Patient Name: {patient_name.strip()}\n" # Use strip() to remove leading/trailing whitespace
    if patient_age != 'N/A' and patient_age is not None and patient_age.strip() != '':
        patient_info += f"Patient Age: {patient_age.strip()}\n" # Use strip()
    if analysis_datetime_str is not None and analysis_datetime_str.strip() != '':
        patient_info += f"Analysis Date/Time: {analysis_datetime_str.strip()}\n" # Use strip()

    if patient_info: # Only add the header if there is info to display
        patient_info = "Patient Information:\n" + patient_info + "\n"


    # Design your prompts, INCLUDING patient info and date
    prompt_comprehensive = f"""
    Generate a detailed Diabetic Retinopathy (DR) analysis report for a patient based on the following AI model results and patient information.
    Format the report as if you were an AI assistant presenting findings to a clinician.
    Include patient information and the analysis date/time at the beginning.
    Discuss the individual model predictions, ensemble prediction, and confidence scores.
    Explain the potential DR stage indicated by these results according to the standard 0-4 grading scale (0: No DR, 1: Mild NPDR, 2: Moderate NPDR, 3: Severe NPDR, 4: PDR).
    If DR is detected (stage > 0), briefly mention the typical characteristics of that stage.
    Keep the tone professional and informative.

    {patient_info}

    Model Results:
    {model_results_text}

    Generated Report:
    """

    prompt_synthetic = f"""
    Generate a concise, synthetic report based on the following AI model results and patient information,
    representing a potential quick summary or impression from an ophthalmologist reviewing the AI findings.
    Include patient name and age if available, and the analysis date/time.
    State the most likely DR stage indicated by the ensemble result and a brief, high-level summary.
    Keep it very short and easy to understand.

    Patient Name: {patient_name if patient_name != 'N/A' and patient_name is not None and patient_name.strip() != '' else 'N/A'}
    Patient Age: {patient_age if patient_age != 'N/A' and patient_age is not None and patient_age.strip() != '' else 'N/A'}
    Analysis Time: {analysis_datetime_str if analysis_datetime_str is not None and analysis_datetime_str.strip() != '' else 'N/A'}

    Model Results:
    {model_results_text}

    Synthetic Impression:
    """

    # --- Call Gemini API ---
    try:
        print("Calling Gemini API for comprehensive report...")
        response_comprehensive = model.generate_content([prompt_comprehensive])
        comprehensive_report = response_comprehensive.text
        print("Comprehensive report generated.")
    except Exception as e:
        print(f"Error generating comprehensive report with Gemini: {e}")
        comprehensive_report = f"Error generating comprehensive report: {e}"

    # --- Call Gemini API for Synthetic Report ---
    try:
        print("Calling Gemini API for synthetic report...")
        response_synthetic = model.generate_content([prompt_synthetic])
        synthetic_report = response_synthetic.text
        print("Synthetic report generated.")
    except Exception as e:
        print(f"Error generating synthetic report with Gemini: {e}")
        synthetic_report = f"Error generating synthetic report: {e}"

    return {
        'comprehensive': comprehensive_report,
        'synthetic': synthetic_report
    }


# --- is_fundus_image_with_gemini function (Keep as is) ---
def is_fundus_image_with_gemini(image_path):
    # ... (Keep the existing code for is_fundus_image_with_gemini) ...
    """
    Checks if the uploaded image is a fundus image using Gemini API.
    ... (rest of the function) ...
    """
    model = get_gemini_model(MULTIMODAL_MODEL_NAME)
    if model is None:
        return False, "Gemini model not available for validation.", "Gemini model not available."

    try:
        img_np = cv2.imread(image_path)
        if img_np is None:
            print(f"Validation Error: Could not read image file {image_path} for Gemini validation.")
            return False, "Could not read image file.", "Failed to read uploaded image."

        is_success, buffer = cv2.imencode(".jpg", img_np)
        if not is_success:
                print(f"Validation Error: Could not encode image {image_path} to bytes for Gemini validation.")
                return False, "Could not encode image.", "Failed to process image for validation."

        image_part = {
            'mime_type': 'image/jpeg',
            'data': buffer.tobytes()
        }

    except Exception as e:
        print(f"Validation Error: Error preparing image {image_path} for Gemini API: {e}")
        return False, f"Error preparing image: {e}", "Internal error during image validation."

    prompt_parts = [
        image_part,
        "Is this a medical image of an eye fundus? Answer only 'Yes' or 'No'."
    ]

    try:
        print(f"Calling Gemini API to validate image type: {os.path.basename(image_path)}...")
        response = model.generate_content(prompt_parts)
        gemini_response_text = response.text.strip().lower()
        print(f"Gemini validation response: '{gemini_response_text}'")

        is_fundus = gemini_response_text == 'yes'

        return is_fundus, response.text.strip(), None

    except Exception as e:
        print(f"Validation Error: Error calling Gemini API for validation: {e}")
        if hasattr(e, 'response') and hasattr(e.response, 'prompt_feedback') and hasattr(e.response.prompt_feedback, 'block_reason'):
                block_reason = e.response.prompt_feedback.block_reason
                print(f"Validation Error: Prompt was blocked for reason: {block_reason}")
                return False, f"Validation blocked: {block_reason}", f"Image validation failed (blocked): {block_reason}"
        return False, f"API call error: {e}", "Internal error during image validation API call."