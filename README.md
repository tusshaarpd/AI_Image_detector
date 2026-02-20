# AI Image Detector

A tool that detects whether an image was made by an AI or taken as a real photograph.
It uses four different detection methods and combines their results to give you a final answer.

---

## Table of Contents

1. [What This Project Does](#what-this-project-does)
2. [How It Works (Simple Overview)](#how-it-works-simple-overview)
3. [Project Structure](#project-structure)
4. [Step-by-Step Setup](#step-by-step-setup)
5. [Running the App](#running-the-app)
6. [How Detection Works](#how-detection-works)
7. [API Endpoints](#api-endpoints)
8. [Configuration Settings](#configuration-settings)
9. [Training Your Own Model](#training-your-own-model)
10. [Running Tests](#running-tests)
11. [Docker Deployment](#docker-deployment)
12. [Understanding the Results](#understanding-the-results)

---

## What This Project Does

When you upload an image, the app runs four separate checks on it:

1. **Deep Learning (CNN)** - A neural network trained to spot AI images
2. **Frequency Analysis** - Checks hidden patterns in the image that AI tools leave behind
3. **Metadata Check** - Reads the image file's hidden info to see if an AI tool created it
4. **Texture Analysis** - Checks if the textures in the image look too smooth or perfect

All four results are combined to produce a single confidence score (0% to 100%), a verdict (AI or Real), and a risk level (Low / Medium / High).

---

## How It Works (Simple Overview)

```
You upload an image
        |
        v
Image is validated (correct format? not too large?)
        |
        v
Four checks run at the same time:
  [CNN Model]  [Frequency]  [Metadata]  [Texture]
        |
        v
Results are combined using weighted scoring:
  CNN = 45%,  Frequency = 20%,  Metadata = 20%,  Texture = 15%
        |
        v
Final answer: "AI Generated" or "Real Photo" + confidence score
```

---

## Project Structure

```
AI_Image_detector/
|
|-- streamlit_app.py        # Web interface (drag & drop image upload)
|-- requirements.txt        # Python packages needed
|-- Dockerfile              # For building a Docker container
|-- docker-compose.yml      # To run both web app and API together
|-- IMPLEMENTATION.md       # Detailed technical documentation
|
|-- app/
|   |-- main.py             # Starts the REST API server
|   |
|   |-- core/
|   |   |-- config.py       # App settings (model name, weights, limits)
|   |   |-- preprocessing.py# Validates and prepares uploaded images
|   |   |-- detector_service.py # Runs all four checks and returns results
|   |   |-- logging.py      # Structured logging setup
|   |   |-- metrics.py      # Prometheus monitoring metrics
|   |   |-- security.py     # API key authentication
|   |
|   |-- detectors/
|   |   |-- frequency.py    # Frequency/FFT analysis detector
|   |   |-- metadata.py     # EXIF metadata detector
|   |   |-- texture.py      # Texture analysis detector
|   |   |-- ensemble.py     # Combines all detector results
|   |
|   |-- models/
|   |   |-- cnn_classifier.py # Deep learning model (EfficientNet-B4)
|   |
|   |-- schemas/
|   |   |-- detection.py    # Data structures for requests and responses
|   |
|   |-- api/
|       |-- routes.py       # REST API endpoint definitions
|       |-- web.py          # Web UI route handling
|
|-- scripts/
|   |-- train.py            # Script to train a custom model
|
|-- tests/
|   |-- test_detectors.py   # Tests for individual detectors
|   |-- test_ensemble.py    # Tests for the combining logic
|   |-- test_preprocessing.py # Tests for image validation
|   |-- test_cnn_classifier.py # Tests for the CNN model
|
|-- models/
    |-- weights/            # Folder where trained model files are saved (.pth)
```

---

## Step-by-Step Setup

### Step 1: Check Your Python Version

You need Python 3.10 or newer.

```bash
python --version
```

If your version is too old, download a newer version from [python.org](https://python.org).

---

### Step 2: Create a Virtual Environment

A virtual environment keeps this project's packages separate from other projects.

```bash
# Create the virtual environment
python -m venv venv

# Activate it (Linux / Mac)
source venv/bin/activate

# Activate it (Windows)
venv\Scripts\activate
```

You will see `(venv)` in your terminal. This means it is active.

---

### Step 3: Install Required Packages

```bash
pip install -r requirements.txt
```

This installs everything the project needs, including:
- PyTorch (deep learning)
- FastAPI (REST API server)
- Streamlit (web interface)
- OpenCV and Pillow (image processing)
- And more...

If you have a GPU and want faster processing, install the GPU version of PyTorch:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

---

### Step 4: Set Up Configuration (Optional)

Create a `.env` file in the project root to customize settings:

```bash
# .env file
MODEL_NAME=efficientnet_b4
DEVICE=cpu
MAX_IMAGE_SIZE_MB=10
CONFIDENCE_THRESHOLD=0.5
API_KEY=your-secret-key-here
LOG_LEVEL=INFO
```

If you skip this step, the app will use sensible default values.

---

### Step 5: Verify Installation

Run a quick test to make sure everything is installed correctly:

```bash
pytest tests/ -v
```

You should see most tests passing. Some CNN tests may be skipped if you have not trained a model yet.

---

## Running the App

### Option A: Web Interface (Streamlit)

This gives you a visual drag-and-drop interface in your browser.

```bash
streamlit run streamlit_app.py
```

Then open your browser and go to: `http://localhost:8501`

**What you can do in the web interface:**
- Upload any image (JPEG, PNG, or WEBP)
- See a preview of your image
- View the confidence score and verdict
- See what each of the four detectors found
- Adjust the detection weights using sliders
- Download the raw result as JSON

---

### Option B: REST API (FastAPI)

This is for developers who want to integrate detection into their own programs.

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Then open your browser and go to: `http://localhost:8000/docs`

This shows you interactive API documentation where you can test the endpoints directly.

---

## How Detection Works

### Detector 1: CNN Model (45% of final score)

**What it does:**
Uses a deep learning model (EfficientNet-B4) that has been trained to tell the difference between AI images and real photos.

**How it works:**
1. The image is resized to 224x224 pixels
2. It is passed through a neural network with millions of learned parameters
3. The network outputs a probability (0 to 1) that the image is AI-generated
4. A score near 1.0 means very likely AI; near 0.0 means very likely real

**Files involved:**
- `app/models/cnn_classifier.py` - The model code
- `models/weights/` - Where trained model files are stored

---

### Detector 2: Frequency Analysis (20% of final score)

**What it does:**
Uses math (Fast Fourier Transform / FFT) to look at patterns in the image that are invisible to the eye but reveal how the image was made.

**How it works:**
1. The image is converted to a frequency map (like a spectrogram)
2. Three things are checked:
   - **Spectral drop-off:** AI images tend to have fewer fine details. The frequency energy drops off too steeply.
   - **GAN fingerprints:** Some AI models leave repeating grid-like patterns in the frequency domain.
   - **High-frequency energy:** Real photos have more natural noise in fine details. AI images are often too smooth.

**Files involved:**
- `app/detectors/frequency.py`

---

### Detector 3: Metadata Check (20% of final score)

**What it does:**
Reads the hidden information stored inside the image file (called EXIF data) to look for clues about where the image came from.

**How it works:**
1. Reads EXIF tags from the file
2. Checks for:
   - **AI tool signatures:** Known names like "Stable Diffusion", "Midjourney", "DALL-E" appear in over 18 detected tools
   - **Camera metadata:** Real photos usually contain camera model, shutter speed, ISO, GPS, etc. AI images typically lack these.
   - **Software tags:** Checks if software like Photoshop was used
   - **Suspicious dimensions:** AI image generators often produce images at sizes like 512x512 or 1024x1024 (powers of 2)

**Files involved:**
- `app/detectors/metadata.py`

---

### Detector 4: Texture Analysis (15% of final score)

**What it does:**
Checks how the surface textures look across the whole image. AI images often have textures that are too smooth or too uniform.

**How it works:**
Four texture properties are measured:

1. **Texture Variance:** The image is divided into 32x32 blocks. In real photos, each block looks different. In AI images, they look more similar to each other.

2. **Blur Consistency:** Real photos have natural blur (especially in the background). AI images sometimes have inconsistent or unusually uniform blur.

3. **Edge Coherence:** Checks the direction of edges (lines, borders, shapes) in the image. Real images have varied edge directions. AI images can be too orderly.

4. **Color Correlation:** Looks at how the Red, Green, and Blue channels relate to each other. Unusually high correlation between channels can indicate AI generation.

**Files involved:**
- `app/detectors/texture.py`

---

### How Results Are Combined (Ensemble Engine)

**File:** `app/detectors/ensemble.py`

The four detector scores are combined like this:

```
Final Score = (CNN × 0.45) + (Frequency × 0.20) + (Metadata × 0.20) + (Texture × 0.15)
```

The final score is a number between 0 and 1:
- **0.00 to 0.40** → Low Risk (probably a real photo)
- **0.40 to 0.70** → Medium Risk (uncertain, could be either)
- **0.70 to 1.00** → High Risk (probably AI-generated)

If the score is **0.50 or higher**, the verdict is: **AI Generated**
If the score is **below 0.50**, the verdict is: **Real Photo**

---

## API Endpoints

### Upload an image file

```
POST /api/v1/detect/upload
```

**How to use:**
```bash
curl -X POST "http://localhost:8000/api/v1/detect/upload" \
  -H "X-API-Key: your-secret-key-here" \
  -F "file=@/path/to/your/image.jpg"
```

---

### Send a Base64-encoded image

```
POST /api/v1/detect
```

**How to use:**
```bash
# First encode your image
IMAGE_B64=$(base64 -w 0 /path/to/your/image.jpg)

curl -X POST "http://localhost:8000/api/v1/detect" \
  -H "X-API-Key: your-secret-key-here" \
  -H "Content-Type: application/json" \
  -d '{"image_base64": "'"$IMAGE_B64"'"}'
```

---

### Health check

```
GET /api/v1/health
```

Returns OK if the server is running.

---

### Example Response

```json
{
  "is_ai_generated": true,
  "confidence": 0.82,
  "risk_level": "high",
  "signals": [
    "CNN model indicates high probability of AI generation",
    "Abnormal spectral drop-off detected",
    "No camera metadata found",
    "Uniform texture patterns detected"
  ],
  "detector_scores": {
    "cnn": 0.91,
    "frequency": 0.75,
    "metadata": 0.80,
    "texture": 0.60
  },
  "processing_time_ms": 342
}
```

---

## Configuration Settings

All settings can be changed using environment variables or a `.env` file.

| Setting | Default | What it does |
|---------|---------|--------------|
| `MODEL_NAME` | `efficientnet_b4` | Which neural network to use. Options: `efficientnet_b4`, `resnet50` |
| `DEVICE` | `cpu` | Where to run the model. Use `cuda` if you have a GPU |
| `MAX_IMAGE_SIZE_MB` | `10` | Maximum file size allowed in megabytes |
| `CONFIDENCE_THRESHOLD` | `0.5` | Score above this = AI. Score below = Real |
| `WEIGHT_CNN` | `0.45` | How much the CNN detector counts toward the final score |
| `WEIGHT_FREQUENCY` | `0.20` | How much the frequency detector counts |
| `WEIGHT_METADATA` | `0.20` | How much the metadata detector counts |
| `WEIGHT_TEXTURE` | `0.15` | How much the texture detector counts |
| `API_KEY` | (none) | Secret key required to use the REST API |
| `LOG_LEVEL` | `INFO` | Logging detail. Options: `DEBUG`, `INFO`, `WARNING`, `ERROR` |

---

## Training Your Own Model

If you have your own dataset of AI and real images, you can train a custom model to improve accuracy.

### Step 1: Prepare Your Dataset

Organize your images into folders like this:

```
data/
  train/
    ai/       # AI-generated images for training
    real/     # Real photos for training
  val/
    ai/       # AI-generated images for validation
    real/     # Real photos for validation
```

---

### Step 2: Run the Training Script

```bash
python scripts/train.py \
  --data_dir data/ \
  --epochs 20 \
  --batch_size 32 \
  --lr 1e-4
```

**What each option means:**
- `--data_dir` - Path to your dataset folder
- `--epochs` - How many times to train on all images (more = usually better)
- `--batch_size` - How many images to process at once (lower this if you run out of memory)
- `--lr` - Learning rate (how fast the model learns; 0.0001 is a good default)

---

### Step 3: Training Process

The script will:
1. Load and augment your images (flip, rotate, adjust colors)
2. Train the model epoch by epoch
3. Check accuracy on the validation set after each epoch
4. Save the best model checkpoint automatically
5. Save training history to a JSON file

---

### Step 4: Use Your Trained Model

After training, the model weights are saved to `models/weights/`. The app will automatically use them on the next startup.

---

## Running Tests

Run all tests:

```bash
pytest tests/ -v
```

Run tests and see code coverage:

```bash
pytest tests/ -v --cov=app --cov-report=term-missing
```

Run only one test file:

```bash
pytest tests/test_detectors.py -v
```

**What each test file checks:**

| Test File | What it tests |
|-----------|---------------|
| `test_detectors.py` | Frequency, Metadata, and Texture detectors individually |
| `test_ensemble.py` | The logic that combines all four results |
| `test_preprocessing.py` | Image loading, validation, and Base64 decoding |
| `test_cnn_classifier.py` | The CNN model output format and behavior |

---

## Docker Deployment

Docker lets you run the app without installing Python or any packages on your machine.

### Step 1: Install Docker

Download and install Docker Desktop from [docker.com](https://docker.com).

---

### Step 2: Start the Services

**Run only the web interface:**

```bash
docker compose up streamlit
```

Then open: `http://localhost:8501`

---

**Run only the REST API:**

```bash
docker compose up api
```

Then open: `http://localhost:8000/docs`

---

**Run both at the same time:**

```bash
docker compose up
```

---

### Step 3: Stop the Services

```bash
docker compose down
```

---

### What Docker Does

The `Dockerfile` sets up:
- Python 3.11 environment
- All required system libraries (OpenGL for image processing)
- All Python packages from `requirements.txt`
- Model weights stored in a persistent volume (survives restarts)

The `docker-compose.yml` sets up:
- **Streamlit service** on port 8501
- **FastAPI service** on port 8000 with 4 worker processes
- 4 GB memory limit per service
- Shared volume for model weights

---

## Understanding the Results

### Confidence Score

The confidence score tells you how certain the system is about its decision.

| Score | Meaning |
|-------|---------|
| 0% - 30% | Very likely a real photo |
| 30% - 50% | Probably real, but some AI signals found |
| 50% - 70% | Uncertain - could be either |
| 70% - 90% | Probably AI-generated |
| 90% - 100% | Very likely AI-generated |

---

### Risk Levels

| Risk Level | Score Range | What it means |
|------------|-------------|----------------|
| Low | 0.00 - 0.40 | The image looks like a real photo |
| Medium | 0.40 - 0.70 | The system is not sure. Review manually. |
| High | 0.70 - 1.00 | The image shows strong signs of AI generation |

---

### Signals

The app explains its decision with plain-English signals like:

- "CNN model indicates high probability of AI generation"
- "Abnormal spectral drop-off detected" (frequency analysis finding)
- "No camera metadata found" (suspicious lack of EXIF data)
- "Image dimensions match common AI output sizes (1024x1024)"
- "Uniform texture patterns detected across regions"

These signals help you understand *why* the app made its decision.

---

### Limitations

No detection system is perfect. Keep these in mind:

- **Edited real photos** may sometimes be flagged as AI
- **High-quality AI images** may sometimes pass as real
- The system works best on uncompressed or lightly compressed images
- Heavily cropped or resized images may reduce accuracy
- Training your own model on a larger dataset will improve results

---

## Common Problems

### "No module named torch"

You did not install the requirements. Run:
```bash
pip install -r requirements.txt
```

### "File too large"

The default maximum file size is 10MB. You can increase it in `.env`:
```
MAX_IMAGE_SIZE_MB=20
```

### "401 Unauthorized" from API

You need to include your API key in the request header:
```
X-API-Key: your-secret-key-here
```

### Slow detection

If you have a GPU, set this in `.env`:
```
DEVICE=cuda
```

Otherwise, CPU detection takes a few seconds per image which is normal.

---

## Technology Summary

| Technology | Purpose |
|------------|---------|
| Python 3.10+ | Main programming language |
| PyTorch | Deep learning model training and inference |
| EfficientNet-B4 | CNN architecture for image classification |
| FastAPI | REST API server |
| Streamlit | Web user interface |
| OpenCV | Image processing and edge detection |
| Pillow | Image loading and format conversion |
| NumPy / SciPy | Math and FFT frequency analysis |
| Pydantic | Settings and data validation |
| Prometheus | Monitoring and metrics |
| structlog | Structured logging |
| Docker | Containerized deployment |
| pytest | Testing framework |
