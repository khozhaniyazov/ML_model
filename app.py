
from flask import Flask, render_template, request, send_from_directory
from ultralytics import YOLO
from PIL import Image
import numpy as np
import os
import time
import zipfile
import cv2
import numpy as np

import json
import fitz  # PyMuPDF


def export_json(results, output_path_json):
    """
    Exports YOLO detections to a structured JSON file.
    Schema:
    {
        "detections": [
            {
                "class_id": 1,
                "class_name": "signature",
                "confidence": 0.94,
                "bbox": [x1, y1, x2, y2]
            }
        ]
    }
    """
    detections = []

    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(float)
        cls = int(box.cls)
        conf = float(box.conf)

        detections.append({
            "class_id": cls,
            "class_name": results[0].names[cls],
            "confidence": round(conf, 4),
            "bbox": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)]
        })

    data = {"detections": detections}

    with open(output_path_json, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)



def draw_pretty_boxes(image, results):
    """
    Pretty bounding boxes with BIG readable text.
    Text size auto-scales based on image resolution.
    """
    img = image.copy()

    # Color per class (signature, stamp, qr)
    class_colors = {
        0: (92, 180, 255),   # signature – light blue
        1: (80, 220, 120),   # stamp – greenish
        2: (255, 150, 60),   # qr – orange
    }

    # Dynamic scaling relative to image diagonal
    diag = np.sqrt(img.shape[0]**2 + img.shape[1]**2)
    font_scale = diag / 1500   # adjust if needed
    font_thickness = max(2, int(diag / 2000))

    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        cls = int(box.cls)
        conf = float(box.conf)

        color = class_colors.get(cls, (200, 200, 200))
        label = f"{results[0].names[cls].upper()}  {int(conf*100)}%"

        # ----- Draw main bounding box -----
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3, cv2.LINE_AA)

        # ----- Compute label box -----
        (text_w, text_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
        )

        label_y = y1 - 12 if y1 - 12 > 0 else y1 + text_h + 12

        # Label background (bigger, cleaner)
        cv2.rectangle(
            img,
            (x1, label_y - text_h - 10),
            (x1 + text_w + 20, label_y + 5),
            color,
            -1,
            cv2.LINE_AA
        )

        # ----- Put text -----
        cv2.putText(
            img,
            label,
            (x1 + 10, label_y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            font_thickness,
            cv2.LINE_AA
        )

    return img


app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

model = YOLO("my_model.pt")   # your model
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "image" not in request.files:
            return "No file uploaded"

        file = request.files["image"]
        if file.filename == "":
            return "Empty filename"

        input_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(input_path)
        
        # Detect if PDF or image
        filename_lower = file.filename.lower()
        images = []
        
        if filename_lower.endswith(".pdf"):
            # Convert PDF to images using PyMuPDF
            doc = fitz.open(input_path)
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                pix = page.get_pixmap()
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                images.append((img, page_num + 1))  # Keep track of page number
        else:
            # Regular image
            img = Image.open(input_path)
            images.append((img, 1))

        # Run YOLO on each image/page
        all_results = []
        output_paths = []
        json_results = []

        for img, page_number in images:
            img_np = np.array(img)
            results = model(img_np)
            annotated = results[0].plot()

            # Save output image
            output_filename = f"result_page{page_number}_" + os.path.splitext(file.filename)[0] + ".png"
            output_path = os.path.join(RESULT_FOLDER, output_filename)
            Image.fromarray(annotated).save(output_path)
            output_paths.append(output_path)

            # Save JSON
            json_filename = f"{os.path.splitext(file.filename)[0]}_page{page_number}.json"
            json_output_path = os.path.join(RESULT_FOLDER, json_filename)
            export_json(results, json_output_path)
            json_results.append(json_filename)

            all_results.append(results)

            zip_filename = os.path.splitext(file.filename)[0] + "_results.zip"
            zip_path = os.path.join(RESULT_FOLDER, zip_filename)

            with zipfile.ZipFile(zip_path, 'w') as zipf:
                for json_file in json_results:
                    zipf.write(os.path.join(RESULT_FOLDER, json_file), arcname=json_file)


            detection_times = []

            for img, page_number in images:
                img_np = np.array(img)

                start_time = time.time()        # начало таймера
                results = model(img_np)         # детекция
                end_time = time.time()          # конец таймера

                detection_time = round(end_time - start_time, 2)  # время в секундах
                detection_times.append(detection_time)
                
                annotated = results[0].plot()

                # Сохраняем изображение с результатами
                output_filename = f"result_page{page_number}_" + os.path.splitext(file.filename)[0] + ".png"
                output_path = os.path.join(RESULT_FOLDER, output_filename)
                Image.fromarray(annotated).save(output_path)
                output_paths.append(output_path)

                # JSON
                json_filename = f"{os.path.splitext(file.filename)[0]}_page{page_number}.json"
                json_output_path = os.path.join(RESULT_FOLDER, json_filename)
                export_json(results, json_output_path)
                json_results.append(json_filename)

                all_results.append(results)

            results_with_time = list(zip(output_paths, detection_times))

            return render_template(
                "index.html",
                input_image=input_path,
                output_image=output_paths,
                json_zip=zip_filename   # <-- single zip file
            )


    

    return render_template("index.html", input_image=None, output_image=None)

@app.route("/download/<path:filename>")
def download_file(filename):
    return send_from_directory(RESULT_FOLDER, filename, as_attachment=True)


# >>> REQUIRED FOR FLASK TO RUN <<<
if __name__ == "__main__":
    print("Starting Flask server on http://127.0.0.1:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)
