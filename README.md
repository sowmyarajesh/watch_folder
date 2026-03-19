# Image Processing Pipeline

Folder-watching FastAPI service with YOLOv8 object detection.

## Setup

```bash
# 1. Create virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the application (from project root)
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## Configuration

Override defaults via environment variables or a `.env` file:

| Variable              | Default         | Description                              |
|-----------------------|-----------------|------------------------------------------|
| `INPUT_FOLDER`        | `input_folder`  | Root folder to watch                     |
| `RESULTS_FOLDER`      | `results`       | Where annotated output images are saved  |
| `DB_PATH`             | `app_db.sqlite` | SQLite file location                     |
| `MAX_WORKER_THREADS`  | `4`             | Concurrent YOLOv8 inference threads      |
| `QUEUE_MAX_SIZE`      | `500`           | Back-pressure cap on prediction queue    |
| `YOLO_MODEL_NAME`     | `yolov8n.pt`    | Model size: n/s/m/l/x                    |
| `YOLO_CONFIDENCE`     | `0.25`          | Detection confidence threshold           |

## API Endpoints

| Method | Path                        | Description                         |
|--------|-----------------------------|-------------------------------------|
| GET    | `/health`                   | Liveness probe + queue depth        |
| GET    | `/queue/status`             | Queue depth and worker config       |
| GET    | `/categories`               | List all collection sessions        |
| GET    | `/categories/{category_id}` | Category detail + image breakdown   |
| GET    | `/images/{image_id}`        | Single image + prediction results   |
| GET    | `/docs`                     | Swagger UI                          |

## Folder Structure

```
input_folder/
└── <session-UID>/          ← drop any named folder here
    ├── img001.jpg
    ├── img002.jpg
    └── ...

results/
└── <category_id>/
    ├── img001_result.jpg   ← annotated bounding box images
    └── ...
```

## How it works

```
input_folder/               watchdog
  <UID>/                   ─────────▶  category_table (queued)
    img001.jpg             ─────────▶  image_table (queued)
                                            │
                                            ▼
                                   asyncio.Queue (bounded 500)
                                            │
                                            ▼
                                   ThreadPoolExecutor (4 threads)
                                            │
                                       YOLOv8 inference
                                            │
                                   image_table (completed)
                                   prediction_result (JSON)
                                   result_file_path (annotated jpg)
```

## Prediction Result Format

`image_table.prediction_result` is a JSON array:

```json
[
  {
    "label": "person",
    "confidence": 0.874,
    "bbox_xyxy": [120.5, 45.2, 380.1, 512.0]
  },
  {
    "label": "car",
    "confidence": 0.631,
    "bbox_xyxy": [500.0, 200.0, 720.3, 400.5]
  }
]
```


## Example dataset 

The sample dataset for this can be downloaded from the kaggle.  For this project, I used the brain tumor dataset from kaggle.