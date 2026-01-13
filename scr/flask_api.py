"""
Flask REST API for E-commerce Product Classification
Serves the trained MobileNetV2 model for batch processing
"""

from flask import Flask, request, jsonify
from tensorflow import keras
import numpy as np
from PIL import Image
import json
import os
from datetime import datetime
import logging
import traceback
from threading import Thread

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(script_dir, 'models/mobilenetv2_ecommerce.h5')
CLASS_INDICES_PATH = os.path.join(script_dir,'models/class_indices.json')
MODEL_INFO_PATH = os.path.join(script_dir,'models/model_info.json')
PREDICTIONS_FILE = os.path.join(script_dir, 'Data/predictions.json')
IMG_SIZE = (224, 224)


# Global variables
model = None
class_names = None
idx_to_class = None
mongo_client = None
db = None
def load_model_and_classes():
    """Load the trained model and class mappings"""
    global model, class_names, idx_to_class
    
    try:
        logger.info(f"Loading model from {MODEL_PATH}")
        print(f"Model path: {MODEL_PATH}")
        print(f"File exists: {os.path.exists(MODEL_PATH)}")

        model = keras.models.load_model(MODEL_PATH)
        logger.info("Model loaded successfully")
        
        # Load class indices
        with open(CLASS_INDICES_PATH, 'r') as f:
            class_to_idx = json.load(f)
        
        # Create reverse mapping
        idx_to_class = {v: k for k, v in class_to_idx.items()}
        class_names = list(class_to_idx.keys())
        
        logger.info(f"Loaded {len(class_names)} product categories")
        
        # Load model info
        if os.path.exists(MODEL_INFO_PATH):
            with open(MODEL_INFO_PATH, 'r') as f:
                model_info = json.load(f)
            logger.info(f"Model trained on: {model_info['trained_date']}")
            logger.info(f"Validation accuracy: {model_info['final_val_accuracy']:.4f}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        print(f"DETAILED ERROR: {type(e).__name__}: {str(e)}")  # Add this
        traceback.print_exc()  # Add this too
        return False

def save_prediction(prediction_result):
    """Save prediction to local JSON file, avoiding duplicates"""
    try:
        # Load existing predictions
        with open(PREDICTIONS_FILE, 'r') as f:
            predictions = json.load(f)
        
        # Check if this file_path already exists
        file_path = prediction_result.get('file_path')
        
        # Search for existing prediction with same file_path
        existing_idx = None
        for idx, pred in enumerate(predictions):
            if pred.get('file_path') == file_path:
                existing_idx = idx
                break
        
        # If found, update it; otherwise add new
        if existing_idx is not None:
            logger.info(f"Updating prediction for {file_path}")
            predictions[existing_idx] = prediction_result
        else:
            logger.info(f"Adding new prediction for {file_path}")
            predictions.append(prediction_result)
        
        # Save back
        with open(PREDICTIONS_FILE, 'w') as f:
            json.dump(predictions, f, indent=2)
        
        logger.info(f"Saved prediction for {file_path}")
        
    except Exception as e:
        logger.error(f"Error saving prediction: {str(e)}")

def preprocess_image(image_path):
    """
    Preprocess image for model prediction
    
    Args:
        image_path
        
    Returns:
        Preprocessed image array
    """
    try:
        with open(image_path) as f:
            image_data = Image.open(image_path)
        # Resize
        image = image_data.resize(IMG_SIZE)
        
        # Convert to array and normalize
        img_array = np.array(image) / 255.0
        
        # Resize to fit model
        img_array = img_array.reshape(1,224,224,3)

        return img_array
        
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        raise

def predict_image(img_array, return_top_k=3):
    """
    Make prediction on preprocessed image
    
    Args:
        img_array: Preprocessed image array
        return_top_k: Number of top predictions to return
        
    Returns:
        Dictionary with prediction results
    """
    try:
        # Get predictions
        predictions = model.predict(img_array, verbose=0)[0]
        
        # Get top-k predictions
        top_k_indices = np.argsort(predictions)[-return_top_k:][::-1]
        
        results = {
            'predictions': [],
            'top_class': idx_to_class[top_k_indices[0]],
            'top_confidence': float(predictions[top_k_indices[0]])
        }
        
        for idx in top_k_indices:
            results['predictions'].append({
                'class': idx_to_class[idx],
                'percentage': f"{predictions[idx] * 100:.2f}%"
            })
        
        return results
        
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        raise
@app.route('/')
def home():
    """Home endpoint - API info"""
    return jsonify({
        'name': 'E-Commerce Product Classification API',
        'version': '1.0',
        'endpoints': {
            'health': 'GET /health',
            'classes': 'GET /classes',
            'stats': 'GET /stats',
            'predict': 'POST /predict',
            'predict_batch': 'POST /predict/batch'
        }
    }), 200

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    status = {
        'status': 'healthy',
        'model_loaded': model is not None,
        'classes_loaded': class_names is not None,
        'num_classes': len(class_names) if class_names else 0,
        'storage': 'local_json',
        'timestamp': datetime.utcnow().isoformat()
    }
    return jsonify(status), 200

@app.route('/predict', methods=['POST'])
def predict():
    """
    Single image prediction endpoint
    
    Use:
    curl -X POST http://localhost:5005/predict \
    -H "Content-Type: application/json" \
    -d '{"file_path": "link_to_image_in_directory"}' 
    """
    try:
        if not request.json:
            return jsonify({'error': 'No JSON data provided'}), 400
                
        if 'file_path' in request.json:
            file_path = request.json['file_path']
            logger.info(f"Loading from file path: {file_path}")
            img_array = preprocess_image(file_path)
            filename = os.path.basename(file_path)
        
        # Option 2: Serialised base64 image (NEW)
        elif 'image' in request.json:
            image_data = request.json['image']
            logger.info("Loading from base64 serialized data")
            
            # Decode base64 to bytes
            image_bytes = base64.b64decode(image_data)
            
            # Convert to PIL Image
            img = Image.open(io.BytesIO(image_bytes))
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img = img.resize(IMG_SIZE)
            
            img_array = np.array(img, dtype='float32') / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            filename = request.json.get('filename', 'serialized_image')

        # Make prediction
        result = predict_image(img_array)
        result['file_path'] = filename
        result['timestamp'] = datetime.utcnow().isoformat()
        
        save_prediction(result)
        
        logger.info(f"Prediction: {result['top_class']} ({result['top_confidence']:.2%})")
        
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Error in predict endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """
    Batch prediction endpoint
    
    Expected JSON format:
    {
        "file_paths": [
            "/path/to/image1.jpg",
            "/path/to/image2.jpg",
            "/path/to/image3.jpg"
        ]
    }
    
    Use:
    curl -X POST http://localhost:5005/predict/batch \
    -H "Content-Type: application/json" \
    -d '{"file_paths": ["/path/to/img1.jpg", "/path/to/img2.jpg"]}'
    """
    try:
        if not request.json:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        if 'file_paths' not in request.json:
            return jsonify({'error': 'No file_paths array provided'}), 400
        
        file_paths = request.json['file_paths']
        
        if not isinstance(file_paths, list):
            return jsonify({'error': 'file_paths must be an array'}), 400
        
        logger.info(f"Processing batch of {len(file_paths)} images")
        
        results = []
        errors = []
        
        for idx, file_path in enumerate(file_paths):
            try:
                if not isinstance(file_path, str):
                    errors.append({
                        'index': idx,
                        'error': 'file_path must be a string'
                    })
                    continue
                
                filename = os.path.basename(file_path)
                logger.info(f"[{idx+1}/{len(file_paths)}] Processing: {filename}")
                
                # Preprocess and predict
                img_array = preprocess_image(file_path)
                result = predict_image(img_array)
                
                # Add metadata
                result['file_path'] = filename
                result['full_path'] = file_path
                result['timestamp'] = datetime.utcnow().isoformat()
                result['batch_index'] = idx
                
                # Save to local file
                save_prediction(result)
                
                results.append(result)
                logger.info(f"Prediction: {result['top_class']} ({result['top_confidence']:.2%})")
                
            except Exception as e:
                logger.error(f"Error processing {file_path}: {str(e)}")
                errors.append({
                    'index': idx,
                    'file_path': file_path,
                    'error': str(e)
                })
        
        response = {
            'total_images': len(file_paths),
            'successful': len(results),
            'failed': len(errors),
            'results': results,
            'errors': errors if errors else None
        }
        
        logger.info(f"Batch complete: {len(results)}/{len(file_paths)} successful")
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Error in batch predict endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/classes', methods=['GET'])
def get_classes():
    """Get list of available product classes"""
    if class_names is None:
        return jsonify({'error': 'Classes not loaded'}), 500
    
    return jsonify({
        'num_classes': len(class_names),
        'classes': class_names
    }), 200

@app.route('/stats', methods=['GET'])
def get_stats():
    """Get prediction statistics from local file"""
    try:
        if not os.path.exists(PREDICTIONS_FILE):
            return jsonify({
                'total_predictions': 0,
                'class_distribution': [],
                'recent_predictions': []
            }), 200
        
        # Load predictions
        with open(PREDICTIONS_FILE, 'r') as f:
            predictions = json.load(f)
        
        # Calculate class distribution
        class_counts = {}
        for pred in predictions:
            cls = pred.get('top_class', 'unknown')
            class_counts[cls] = class_counts.get(cls, 0) + 1
        
        class_distribution = [
            {'_id': cls, 'count': count}
            for cls, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
        ]
        
        # Get recent predictions (last 10)
        recent = predictions[-10:] if len(predictions) > 10 else predictions
        recent.reverse()  # Most recent first
        
        stats = {
            'total_predictions': len(predictions),
            'class_distribution': class_distribution,
            'recent_predictions': recent
        }
        
        return jsonify(stats), 200
        
    except Exception as e:
        logger.error(f"Error getting stats: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.before_request
def before_request():
    """Log all requests"""
    logger.info(f"{request.method} {request.path}")

@app.after_request
def after_request(response):
    """Log all responses"""
    logger.info(f"Response: {response.status_code}")
    return response

def main():
    """Initialize and run the Flask app"""
    logger.info("="*70)
    logger.info("E-COMMERCE PRODUCT CLASSIFICATION API")
    logger.info("="*70)
    
    # Load model
    if not load_model_and_classes():
        logger.error("Failed to load model. Exiting.")
        return
    
    logger.info(f"Predictions will be saved to: {PREDICTIONS_FILE}")
    
    # Run app
    logger.info("\nStarting Flask server...")
    logger.info("Available endpoints:")
    logger.info("  GET  /health          - Health check")
    logger.info("  GET  /classes         - List product classes")
    logger.info("  GET  /stats           - Get prediction statistics")
    logger.info("  POST /predict         - Single image prediction")
    logger.info("  POST /predict/batch   - Batch image prediction")
    logger.info("="*70)
    
    thread = Thread(target=lambda: app.run(host='0.0.0.0', port=5005, debug=False))
    thread.daemon = True
    thread.start()


if __name__ == '__main__':
    main()
