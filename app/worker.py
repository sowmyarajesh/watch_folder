"""
worker.py
─────────
Consumes items from prediction_queue and runs YOLOv8 inference in a
ThreadPoolExecutor so blocking inference doesn't stall the asyncio event loop.

Flow per image:
  1. Dequeue item
  2. Check cancellation → mark deleted if cancelled
  3. Mark image as 'processing' in DB
  4. Run YOLOv8 in thread pool
  5. Persist bounding boxes + labels as JSON in image_table
  6. Save annotated result image to results/
  7. Recompute category completed_percentage
  8. If all images done → mark category 'completed'
"""
import asyncio
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from sqlmodel import select

from app.config import settings
from app.database import get_session
from app.models import CategoryTable, ImageTable
from app.queue_manager import cancelled_categories, cancelled_images, is_cancelled, prediction_queue
from app.schemas import StatusEnum

logger = logging.getLogger(__name__)

# Single global YOLO model instance — loaded once, reused across all threads.
# ultralytics is thread-safe for inference when share=False (default).
_yolo_model = None


def _load_model():
    global _yolo_model
    if _yolo_model is None:
        from ultralytics import YOLO
        _yolo_model = YOLO(settings.yolo_model_name)
        logger.info(f"YOLOv8 model loaded: {settings.yolo_model_name}")
    return _yolo_model


def _run_inference(image_path: str, result_dir: Path) -> dict:
    """
    Blocking inference call — executed inside ThreadPoolExecutor.
    Returns dict with detection results.
    """
    model = _load_model()
    results = model(
        image_path,
        conf=settings.yolo_confidence,
        verbose=False,
    )
    result = results[0]  # single image

    # Build structured detection output
    detections = []
    for box in result.boxes:
        detections.append({
            "label": result.names[int(box.cls)],
            "confidence": round(float(box.conf), 4),
            "bbox_xyxy": [round(float(v), 2) for v in box.xyxy[0].tolist()],
        })

    # Save annotated image
    result_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(image_path).stem
    result_image_path = str(result_dir / f"{stem}_result.jpg")
    result.save(filename=result_image_path)

    return {
        "detections": detections,
        "result_image_path": result_image_path,
    }


def _update_category_progress(category_id: str) -> None:
    """Recompute completed_percentage and overall status for a category."""
    with get_session() as session:
        images = session.exec(
            select(ImageTable).where(ImageTable.category_id == category_id)
        ).all()

        if not images:
            return

        total = len(images)
        done = sum(
            1 for img in images
            if img.processing_status in (StatusEnum.completed, StatusEnum.deleted, StatusEnum.failed)
        )
        pct = round((done / total) * 100, 2)

        category = session.get(CategoryTable, category_id)
        if category is None:
            return

        category.completed_percentage = pct
        category.updated_at = datetime.utcnow()

        # Determine overall category status
        active = [
            img for img in images
            if img.processing_status not in (StatusEnum.deleted,)
        ]
        if all(img.processing_status == StatusEnum.completed for img in active) and active:
            category.status = StatusEnum.completed
        elif any(img.processing_status == StatusEnum.processing for img in active):
            category.status = StatusEnum.processing

        session.add(category)
        session.commit()


async def _process_item(item: dict, executor: ThreadPoolExecutor, loop: asyncio.AbstractEventLoop) -> None:
    image_id = item["image_id"]
    image_path = item["image_path"]
    category_id = item["category_id"]

    # ── Cancellation check (before touching DB) ──────────────────────────────
    if is_cancelled(image_id, category_id):
        logger.info(f"Skipping cancelled image {image_id}")
        return

    # ── Mark as processing ───────────────────────────────────────────────────
    with get_session() as session:
        image = session.get(ImageTable, image_id)
        if image is None or image.processing_status == StatusEnum.deleted:
            return
        image.processing_status = StatusEnum.processing
        image.updated_at = datetime.utcnow()
        session.add(image)
        session.commit()

    _update_category_progress(category_id)

    # ── Run YOLOv8 in thread pool ────────────────────────────────────────────
    result_dir = settings.results_folder / category_id
    try:
        inference_result = await loop.run_in_executor(
            executor,
            _run_inference,
            image_path,
            result_dir,
        )

        # ── Post-inference cancellation check ───────────────────────────────
        if is_cancelled(image_id, category_id):
            logger.info(f"Image {image_id} completed inference but was cancelled — marking deleted")
            with get_session() as session:
                image = session.get(ImageTable, image_id)
                if image:
                    image.processing_status = StatusEnum.deleted
                    image.updated_at = datetime.utcnow()
                    session.add(image)
                    session.commit()
            return

        # ── Persist results ──────────────────────────────────────────────────
        with get_session() as session:
            image = session.get(ImageTable, image_id)
            if image is None:
                return
            image.processing_status = StatusEnum.completed
            image.prediction_result = json.dumps(inference_result["detections"])
            image.result_file_path = inference_result["result_image_path"]
            image.updated_at = datetime.utcnow()
            session.add(image)
            session.commit()

        logger.info(f"✓ Completed {Path(image_path).name} — {len(inference_result['detections'])} detections")

    except Exception as exc:
        logger.exception(f"Inference failed for image {image_id}: {exc}")
        with get_session() as session:
            image = session.get(ImageTable, image_id)
            if image:
                image.processing_status = StatusEnum.failed
                image.updated_at = datetime.utcnow()
                session.add(image)
                session.commit()

    finally:
        _update_category_progress(category_id)


async def worker_loop(executor: ThreadPoolExecutor) -> None:
    """
    Runs forever as an asyncio Task.
    Spawns up to max_worker_threads concurrent inference jobs.
    """
    loop = asyncio.get_running_loop()
    semaphore = asyncio.Semaphore(settings.max_worker_threads)
    logger.info(f"Worker loop started — {settings.max_worker_threads} concurrent threads")

    async def bounded_process(item):
        async with semaphore:
            await _process_item(item, executor, loop)

    while True:
        item = await prediction_queue.get()
        if item is None:            # shutdown sentinel
            logger.info("Worker loop shutting down")
            break
        asyncio.create_task(bounded_process(item))
        prediction_queue.task_done()
