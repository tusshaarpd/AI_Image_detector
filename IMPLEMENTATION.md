# AI Image Detector — How It Works

## What Does This Project Do?

This tool looks at any image you upload and answers one question:

> **"Was this image created by an AI tool, or is it a real photograph?"**

It works by running **four different checks** on the image at the same time, then combining all the results to give you a final answer along with a confidence score.

---

## How to Run It

### Option 1 — Run on your computer (simplest)

```bash
# Step 1: Install all required packages
pip install -r requirements.txt

# Step 2: Start the web app
streamlit run streamlit_app.py
```

Then open your browser and go to **http://localhost:8501**

You will see a page where you can drag and drop any image to analyse it.

### Option 2 — Run with Docker (no Python needed)

```bash
# Start the web app
docker compose up streamlit
```

Open **http://localhost:8501** in your browser.

---

## What You See in the App

When you upload an image, the app shows you:

| What | Meaning |
|---|---|
| **Verdict** | "Likely AI-Generated" or "Likely Authentic" |
| **Confidence** | How sure the system is (0% = completely unsure, 100% = very sure) |
| **Risk Level** | Low / Medium / High — how suspicious the image is |
| **Signals** | Plain-English reasons behind the decision |
| **Detector Breakdown** | Score from each of the 4 individual checks |

You can also adjust how much weight each check carries using the sliders on the left sidebar.

---

## The Four Checks Explained (in plain English)

### Check 1 — AI Brain Scan (CNN Classifier)
**Weight: 45% of the final score**

Think of this like showing the image to someone who has studied thousands of AI-generated images and real photos. The system uses a deep learning model called **EfficientNet-B4** that was pre-trained on millions of images. It looks at the overall visual "feel" of the image and gives a score from 0 (definitely real) to 1 (definitely AI).

- File: `app/models/cnn_classifier.py`

### Check 2 — Hidden Pattern Scan (Frequency Analysis)
**Weight: 20% of the final score**

Every AI image-generation tool leaves invisible mathematical "fingerprints" in the image. When AI creates an image, it builds it up in layers — and this process leaves patterns in the image's hidden frequency data (like how a radio signal has a particular frequency).

This check converts the image into its frequency components (using a technique called FFT) and looks for:
- Unusual drop-offs in detail (AI images often look "too smooth")
- Repeating patterns left by the AI generation process
- Lack of fine detail in high-frequency areas

- File: `app/detectors/frequency.py`

### Check 3 — Photo ID Card Check (Metadata Analysis)
**Weight: 20% of the final score**

Every real photo taken by a camera stores hidden information called **EXIF data** — like the camera brand, lens type, shutter speed, GPS location, and date taken. AI-generated images usually have none of this, or they contain signatures from the AI tool that made them.

This check looks for:
- Known AI tool names in the hidden data (Stable Diffusion, Midjourney, DALL-E, and 16 others)
- Missing or sparse camera information
- Square dimensions like 512×512 or 1024×1024 (typical of AI outputs)

- File: `app/detectors/metadata.py`

### Check 4 — Surface Texture Inspection (Texture Analysis)
**Weight: 15% of the final score**

Real photographs have natural randomness in their textures — skin has pores, grass blades vary, backgrounds have gradual blur. AI images tend to be too perfect, too smooth, or have inconsistencies that a human wouldn't notice but a computer can spot.

This check looks for:
- Textures that are unnaturally uniform across the image
- Blur that appears or disappears abruptly (real cameras blur gradually)
- Edge patterns that are too regular or too few
- Colour channels that are suspiciously well-correlated

- File: `app/detectors/texture.py`

---

## How the Final Decision Is Made

After all four checks run, a **combining engine** (called the Ensemble) merges the results:

```
Final Score = (CNN × 45%) + (Frequency × 20%) + (Metadata × 20%) + (Texture × 15%)
```

That combined score is then smoothed out (so it truly represents a probability) and compared to a threshold:

- **Score ≥ 0.50** → The image is flagged as AI-generated
- **Score 0.00–0.40** → Risk Level: **Low** (likely real)
- **Score 0.40–0.70** → Risk Level: **Medium** (uncertain)
- **Score 0.70–1.00** → Risk Level: **High** (likely AI)

All the "signals" (plain-English reasons) from every check are collected and shown to you.

- File: `app/detectors/ensemble.py`

---

## How an Image Flows Through the System

```
You upload an image
        ↓
Image is checked for size and format (max 10 MB, JPEG/PNG/WEBP only)
        ↓
    ┌───────────────────────────────────────┐
    │  All 4 checks run at the same time    │
    │                                       │
    │  ① CNN Brain Scan      → score 0–1   │
    │  ② Frequency Check     → score 0–1   │
    │  ③ Metadata Check      → score 0–1   │
    │  ④ Texture Check       → score 0–1   │
    └───────────────────────────────────────┘
        ↓
Combining Engine mixes the four scores using weights
        ↓
Final verdict: is_ai_generated + confidence + risk level + reasons
        ↓
Shown to you in the app
```

---

## Training Your Own Model (Optional)

The CNN check works out-of-the-box with general knowledge, but it becomes more accurate when trained specifically on AI-vs-real images. If you have your own dataset, here is how to train it:

### Step 1 — Organise your images like this

```
data/
  train/
    ai/       ← put AI-generated images here
    real/     ← put real photos here
  val/
    ai/       ← a smaller set for testing accuracy
    real/
```

### Step 2 — Run the training script

```bash
python scripts/train.py \
  --data_dir data/ \
  --epochs 20 \
  --batch_size 32
```

### What happens during training

1. Images are randomly flipped, rotated, and slightly colour-shifted to make the model more robust.
2. The model trains for the number of epochs (rounds) you specify.
3. After each round, accuracy is checked. The best version is automatically saved.
4. A history file is saved so you can see how training improved over time.
5. Once saved, the new weights are automatically used next time you start the app — no extra steps needed.

- File: `scripts/train.py`

---

## File Map — What Each File Does

```
AI_Image_detector/
│
├── streamlit_app.py          ← The web app you run and see in the browser
│
├── app/
│   ├── main.py               ← Starts the API server (alternative to Streamlit)
│   │
│   ├── core/
│   │   ├── config.py         ← All settings (image size limits, thresholds, etc.)
│   │   ├── preprocessing.py  ← Validates and prepares the image before checking
│   │   ├── detector_service.py  ← Runs the full pipeline (used by the API)
│   │   ├── logging.py        ← Records what the app is doing behind the scenes
│   │   ├── metrics.py        ← Tracks usage stats (response times, counts, etc.)
│   │   └── security.py       ← Protects the API with a key
│   │
│   ├── detectors/
│   │   ├── metadata.py       ← Check 3: reads EXIF data
│   │   ├── frequency.py      ← Check 2: looks for hidden AI patterns
│   │   ├── texture.py        ← Check 4: inspects surface texture quality
│   │   └── ensemble.py       ← Combining engine: merges all four scores
│   │
│   ├── models/
│   │   └── cnn_classifier.py ← Check 1: the AI brain scan model
│   │
│   └── schemas/
│       └── detection.py      ← Defines the structure of inputs and outputs
│
├── scripts/
│   └── train.py              ← Script to train the model on your own data
│
├── tests/                    ← Automated checks to make sure everything works
│   ├── test_detectors.py
│   ├── test_ensemble.py
│   ├── test_preprocessing.py
│   └── test_cnn_classifier.py
│
├── models/weights/           ← Where trained model files are saved (.pth files)
├── requirements.txt          ← List of Python packages needed
├── Dockerfile                ← Recipe to build a Docker container
└── docker-compose.yml        ← Runs the app in containers (Streamlit + API)
```

---

## Settings You Can Change

These can be set in a `.env` file in the project root, or as environment variables.

| Setting | Default | What it controls |
|---|---|---|
| `MODEL_NAME` | `efficientnet_b4` | Which AI model to use for the brain scan |
| `DEVICE` | `cpu` | Use `cuda` if you have a GPU for much faster processing |
| `MAX_IMAGE_SIZE_MB` | `10` | Largest image the app will accept |
| `CONFIDENCE_THRESHOLD` | `0.5` | Score cutoff for calling something AI-generated |
| `WEIGHT_CNN` | `0.45` | How much the brain scan matters in the final score |
| `WEIGHT_FREQUENCY` | `0.20` | How much the hidden pattern scan matters |
| `WEIGHT_METADATA` | `0.20` | How much the photo ID check matters |
| `WEIGHT_TEXTURE` | `0.15` | How much the texture check matters |
| `API_KEY` | `changeme-in-production` | Password for the REST API |
| `LOG_LEVEL` | `INFO` | How much detail to log (`DEBUG` for more, `WARNING` for less) |

---

## Running the Tests

To verify everything is working correctly:

```bash
# Run all tests and see detailed results
pytest tests/ -v

# Run tests and also show what percentage of the code is covered
pytest tests/ -v --cov=app --cov-report=term-missing
```

---

## Why Four Checks Instead of One?

No single method catches everything:

- The **CNN** is good at overall visual patterns but can be fooled by unusual real photos.
- **Frequency analysis** catches GAN fingerprints but doesn't help with diffusion models.
- **Metadata** is definitive when the AI tool left its name, but most images have no metadata.
- **Texture analysis** catches uniformity issues but can't see frequency artefacts.

By combining all four, the system covers the weaknesses of each individual method and gives a much more reliable result.

---

## REST API (For Developers)

If you want to use this programmatically (e.g. from another app):

```bash
# Start the API server
docker compose up api

# Send an image file
curl -X POST http://localhost:8000/api/v1/detect/upload \
  -H "X-API-Key: changeme-in-production" \
  -F "file=@photo.jpg"
```

Response:
```json
{
  "is_ai_generated": true,
  "confidence_score": 0.82,
  "risk_level": "High",
  "signals_detected": [
    "No camera metadata present",
    "Square dimensions typical of AI output: 1024x1024",
    "CNN classifier: high AI probability (0.91)"
  ],
  "model_version": "1.0.0"
}
```

Full API docs are available at **http://localhost:8000/docs**
