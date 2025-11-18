# YOLO Document Scanner (Flask)

This is a Flask web application for detecting signatures, stamps, and QR codes on images or PDF files using a custom YOLO model.

## Features
- Accepts images and PDFs
- Converts PDFs to images using PyMuPDF
- Runs YOLO detection on each page
- Saves annotated images
- Exports detection results as JSON
- Creates a ZIP archive with all JSON files

