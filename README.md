# E-Commerce Product Classification API

Automated image classification system for refund department categorisation. This project uses a deep learning model (MobileNetV2) to classify product images into categories and provides both a REST API for on-demand predictions and scheduled batch processing via GitHub Actions.

## Overview

This system helps online shopping platforms automatically categorize returned/refunded items by analyzing product images. Instead of manual sorting, the system:
- Provides a REST API for real-time predictions
- Processes batches of images every night automatically
- Stores all predictions for tracking and analysis
- Reduces manual workforce needed for categorization

##  Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Data Ingestion Layer                     │
│  - New product images uploaded to Data/pre_batch/           │
│  - Images ready for processing                              │
└──────────────────────┬──────────────────────────────────────┘
                       │
        ┌──────────────┴──────────────┐
        │                             │
┌───────▼──────────┐        ┌────────▼────────┐
│   Flask REST API │        │ Batch Processor │
│  (On-demand)     │        │ (Scheduled)     │
│  POST /predict   │        │ GitHub Actions  │
│  POST /predict/  │        │ (Every night)   │
│        batch     │        │                 │
└────────┬─────────┘        └────────┬────────┘
         │                           │
         └───────────────┬───────────┘
                         │
            ┌────────────▼────────────┐
            │   MobileNetV2 Model     │
            │   (Trained classifier)  │
            └────────────┬────────────┘
                         │
            ┌────────────▼───────────┐
            │  Prediction Storage    │
            │  predictions.json      │
            │  + Processed Images    │
            │  Data/post_batch/      │
            └────────────────────────┘
```

## Features

- **REST API Endpoints:**
  - `GET /health` - Health check
  - `GET /classes` - List product categories
  - `GET /stats` - Prediction statistics
  - `POST /predict` - Single image prediction
  - `POST /predict/batch` - Batch predictions

- **Automated Batch Processing:**
  - Scheduled nightly execution (3 AM UTC)
  - Processes all images in `Data/pre_batch/`
  - Moves processed images to `Data/post_batch/`
  - Appends results to predictions.json

- **Prediction Tracking:**
  - JSON-based storage of all predictions
  - Avoids duplicate predictions for same image
  - Timestamps and metadata for all predictions

- **Dataset Pull**
  - Source datasets can be automatically loaded into the Data folder using Kaggle
  - Check, val and train datasets are made available through python script
  - Check can be used to test automated batch processing and REST API

## Quick Start

### Prerequisites
- Python 3.9+
- pip or conda
- Git

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/YOUR_USERNAME/automated_image_processing.git
cd automated_image_processing
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Verify model files exist:**
```bash
ls models/
# Should show:
# - mobilenetv2_ecommerce.h5
# - class_indices.json
```

## 📖 Usage

### Option 1: REST API (On-Demand)

**Start the Flask server:**
```bash
python scr/flask_api.py
```

Server runs at `http://localhost:5001`

**Single prediction:**
```bash
curl -X POST http://localhost:5001/predict \
  -H "Content-Type: application/json" \
  -d '{"file_path": "/path/to/image.jpg"}'
```

**Batch prediction:**
```bash
curl -X POST http://localhost:5001/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "file_paths": [
      "/path/to/image1.jpg",
      "/path/to/image2.jpg"
    ]
  }'
```

**Python client:**
```python
import requests

response = requests.post(
    'http://localhost:5001/predict',
    json={'file_path': '/path/to/image.jpg'}
)

result = response.json()
print(f"Predicted class: {result['top_class']}")
print(f"Confidence: {result['top_confidence']:.2%}")
```

### Option 2: Batch Processing (Automated)

Images are automatically processed every night at 3 AM UTC via GitHub Actions.

**Manual trigger:**
1. Go to repository → **Actions** tab
2. Select **Batch Image Classification** workflow
3. Click **Run workflow** → **Run**

**Add images to process:**
1. Put images in `Data/pre_batch/`
2. Workflow processes them overnight
3. Results moved to `Data/post_batch/`
4. Predictions appended to `predictions/predictions.json`

## Project Structure

```
automated_image_processing/
├── .github/
│   └── workflows/
│       └── batch_process.yml          # GitHub Actions workflow
├── scr/
│   ├── flask_api.py                   # REST API server
│   ├── batch_job.py                   # Batch processing script
│   └── model_training.py              # (Reference - already trained)
├── Data/
│   ├── pre_batch/                     # Input images for batch processing
│   ├── post_batch/                    # Processed images (output)
│   ├── ECOMMERCE_PRODUCT_IMAGES/      # Training data, gets added by running prepare_source_data.py
│   │   ├── train/
│   │   ├── val/
│   │   └── check/
│   └── predictions.json               # All prediction results
├── models/
│   ├── class_indices.json             # Trained model
│   ├── mobilenetv2_ecommerce.h5       # Class mappings
│   ├── model_info.json                # Training metadata
│   └── training_history.json          # Training history
├── requirements.txt                   # Python dependencies
└── README.md                           
```

## Classes

See `models/class_indices.json` for complete list

## Output Format

**Prediction result:**
```json
{
  "top_class": "BABY_PRODUCTS",
  "top_confidence": 0.95,
  "top_predictions": [
    {
      "rank": 1,
      "class": "BABY_PRODUCTS",
      "confidence": 0.95,
      "percentage": "95.00%"
    },
    {
      "rank": 2,
      "class": "CLOTHING",
      "confidence": 0.03,
      "percentage": "3.00%"
    },
    {
      "rank": 3,
      "class": "ELECTRONICS",
      "confidence": 0.02,
      "percentage": "2.00%"
    }
  ],
  "file_path": "image.jpg",
  "timestamp": "2026-01-11T15:30:45.123456"
}
```

## Configuration

### Change batch processing schedule

Edit `.github/workflows/batch_process.yml`:
```yaml
on:
  schedule:
    - cron: '0 3 * * *'  # Change this (3 AM UTC = "0 3")
    # Examples:
    # '0 2 * * *'  = 2 AM UTC
    # '0 0 * * *'  = Midnight UTC
    # '0 22 * * *' = 10 PM UTC
```

### Change image input/output directories

Edit `.github/workflows/batch_process.yml`:
```yaml
env:
  SOURCE_DIR: Data/pre_batch    # Where to read images
  PROCESSED_DIR: Data/post_batch  # Where to move processed images
```

## Testing

### Test Flask API locally:
```bash
# Terminal 1: Start server
python scr/flask_api.py

# Terminal 2: Test prediction
curl -X POST http://localhost:5001/health
```

### Test batch processor locally:
```bash
python scr/batch_job.py
```

### Test GitHub Actions:
1. Go to **Actions** tab
2. Click **Batch Image Classification**
3. Click **Run workflow**
4. Monitor execution in real-time

## Monitoring

**Check prediction statistics:**
```bash
curl http://localhost:5001/stats
```

**View all predictions:**
```bash
cat Data/predictions.json
```

**Check GitHub Actions runs:**
1. Repository → **Actions** tab
2. Select workflow run
3. View logs and output

## ⚙️ Technical Details

- **Model:** MobileNetV2 (transfer learning from ImageNet)
- **Input size:** 224×224 pixels
- **Normalization:** Rescaling to [0, 1]
- **Classes:** 9 product categories
- **Framework:** TensorFlow/Keras
- **API:** Flask
- **Automation:** GitHub Actions
- **Storage:** JSON (predictions), local filesystem (images)

## Troubleshooting

**Flask API won't start:**
```bash
# Check if port 5001 is in use
lsof -i :5001
# Kill if needed
kill -9 <PID>
```

**Batch processor fails:**
- Check `Data/pre_batch/` exists and has images
- Verify `models/` files exist
- Check GitHub Actions logs for details
