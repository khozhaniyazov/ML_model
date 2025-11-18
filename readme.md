# AI Document Inspector - YOLO Document Scanner

A modern Flask web application for detecting signatures, stamps, and QR codes on images or PDF files using a custom YOLO model.

## Features
- 🎨 Modern, responsive UI with animations and gradients
- 📄 Accepts images (JPG, PNG) and PDF files
- 🔍 Converts PDFs to images using PyMuPDF
- 🤖 Runs YOLO detection on each page
- 📊 Real-time statistics dashboard
- 💾 Saves annotated images with detection results
- 📦 Exports detection results as JSON
- 📥 Creates a ZIP archive with all JSON files
- ⚡ Per-page processing time tracking

## Local Installation & Setup

### Prerequisites
- Python 3.8 or higher
- Your YOLO model file (`my_model.pt`) in the project root

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Run the Application

```bash
python app.py
```

The server will start and you'll see:
```
Starting Flask server on http://127.0.0.1:5000
```

### Step 3: Access the Application

Open your web browser and navigate to:
- **Local access**: http://127.0.0.1:5000 or http://localhost:5000
- **Network access**: http://YOUR_IP_ADDRESS:5000 (accessible from other devices on your local network)

## Usage

1. **Upload a file**: Click the upload area or drag & drop an image/PDF file
2. **Scan**: Click "Scan Document" to process
3. **View results**: See detection results with bounding boxes, statistics, and processing times
4. **Download**: Download individual result images or the complete JSON results ZIP file

## Project Structure

```
├── app.py                 # Flask application
├── my_model.pt           # YOLO model file (required)
├── requirements.txt      # Python dependencies
├── templates/
│   └── index.html       # Frontend template
├── static/
│   ├── uploads/         # Uploaded files (auto-created)
│   └── results/         # Processed results (auto-created)
└── readme.md            # This file
```

## Configuration

The app is configured to run on:
- **Host**: `0.0.0.0` (accessible from all network interfaces)
- **Port**: `5000`
- **Debug Mode**: Enabled (for development)

To change these settings, edit the last lines in `app.py`:

```python
app.run(host="0.0.0.0", port=5000, debug=True)
```

## Notes

- The application creates `static/uploads/` and `static/results/` directories automatically
- Debug mode is enabled by default (disable for production)
- The model file (`my_model.pt`) must be in the project root directory
