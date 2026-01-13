"""
Batch prediction script for GitHub Actions
Processes all images in a directory and appends results to predictions.json
"""

import os
import sys
import json
import logging
import shutil
import argparse
from pathlib import Path
from datetime import datetime

# Import functions from your Flask API
from flask_api import (
    load_model_and_classes, 
    preprocess_image, 
    predict_image,
    save_prediction
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main batch prediction function"""
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Batch image classification')
    parser.add_argument('--source-dir', default='Data/pre_batch', help='Source image directory')
    parser.add_argument('--output-dir', default='Data/post_batch', help='Output directory for processed images')
    
    args = parser.parse_args()
    
    # Use environment variables if provided
    source_dir = os.environ.get('SOURCE', args.source_dir)
    output_dir = os.environ.get('PROCESSED', args.output_dir)
    
    logger.info("="*70)
    logger.info("BATCH IMAGE CLASSIFICATION")
    logger.info("="*70)
    logger.info(f"Source directory: {source_dir}")
    logger.info(f"Output directory: {output_dir}")
    
    # Load model
    logger.info("Loading model...")
    model_path = 'models/mobilenetv2_ecommerce.h5'
    class_indices_path = 'models/class_indices.json'
    
    if not load_model_and_classes():
        logger.error("Failed to load model. Exiting.")
        return False
    
    # Find all images
    source_path = Path(source_dir)
    if not source_path.exists():
        logger.error(f"Source directory not found: {source_dir}")
        return False
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif'}
    image_files = [f for f in source_path.rglob('*') if f.suffix.lower() in image_extensions]
    
    if not image_files:
        logger.warning(f"No images found in {source_dir}")
        return True
    
    logger.info(f"Found {len(image_files)} images to process")
    
    # Process images
    results_count = 0
    errors_count = 0
    errors = []
    
    for idx, image_path in enumerate(image_files, 1):
        try:
            logger.info(f"[{idx}/{len(image_files)}] Processing: {image_path.name}")
            
            # Preprocess and predict
            img_array = preprocess_image(str(image_path))
            result = predict_image(img_array)
            
            # Add metadata
            result['file_path'] = image_path.name
            result['full_path'] = str(image_path)
            result['timestamp'] = datetime.utcnow().isoformat()
            result['source'] = 'batch'  # Tag as batch prediction
            
            # Save to predictions.json (uses your existing save_prediction function)
            save_prediction(result)
            
            logger.info(f"  → Predicted: {result['top_class']} ({result['top_confidence']:.2%})")
            
            # Move processed image
            output_path = Path(output_dir) / image_path.name
            shutil.move(str(image_path), str(output_path))
            logger.info(f"  → Moved to: {output_path}")
            
            results_count += 1
            
        except Exception as e:
            logger.error(f"Error processing {image_path.name}: {str(e)}")
            errors_count += 1
            errors.append({
                'file': image_path.name,
                'error': str(e)
            })
    
    # Log summary
    logger.info("="*70)
    logger.info("BATCH COMPLETE")
    logger.info(f"Processed: {results_count}/{len(image_files)}")
    logger.info(f"Errors: {errors_count}")
    
    if errors:
        logger.error("Failed images:")
        for error in errors:
            logger.error(f"  - {error['file']}: {error['error']}")
    
    logger.info("="*70)
    
    return errors_count == 0

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
