"""
main.py
───────
FastAPI application with:
  • Lifespan context — starts DB, watcher, worker on startup; cleans up on shutdown
  • GET /health             — liveness probe
  • GET /categories         — list all categories with status
  • GET /categories/{id}    — single category detail with images
  • GET /queue/status       — prediction queue depth
to run the app: uvicorn app.main:app --reload --port 3030
"""
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import JSONResponse
from sqlmodel import Session, select

from app.config import settings
from app.database import get_db_session, init_db
from app.models import CategoryTable, ImageTable
from app.queue_manager import prediction_queue
from app.watcher import start_watcher
from app.worker import worker_loop

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Application state (shared across lifespan) ────────────────────────────────
_executor: Optional[ThreadPoolExecutor] = None
_worker_task: Optional[asyncio.Task] = None
_observer = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup → yield → Shutdown."""
    global _executor, _worker_task, _observer

    # 1. Initialise DB (creates tables if missing)
    logger.info("Initialising database …")
    init_db()

    # 2. Start thread pool for YOLOv8 inference
    _executor = ThreadPoolExecutor(
        max_workers=settings.max_worker_threads,
        thread_name_prefix="yolo-worker",
    )

    # 3. Start async worker loop
    loop = asyncio.get_running_loop()
    _worker_task = asyncio.create_task(worker_loop(_executor))
    logger.info("Inference worker loop started")

    # 4. Start watchdog observer (daemon thread)
    _observer = start_watcher(loop)

    logger.info("🚀 Application ready")
    yield  # ─────────────── app is live ───────────────

    # ── Shutdown ──────────────────────────────────────────────────────────────
    logger.info("Shutting down …")

    # Stop watchdog
    if _observer:
        _observer.stop()
        _observer.join()

    # Signal worker to exit cleanly (sentinel None)
    await prediction_queue.put(None)
    if _worker_task:
        await asyncio.wait_for(_worker_task, timeout=10)

    # Shutdown thread pool (wait for in-flight inference to finish)
    if _executor:
        _executor.shutdown(wait=True)

    logger.info("Shutdown complete")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Image Processing Pipeline",
    description="Folder-watching + YOLOv8 inference pipeline",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["ops"])
def health_check():
    """Liveness probe — returns 200 when the app is up."""
    return {
        "status": "ok",
        "queue_depth": prediction_queue.qsize(),
        "watching": str(settings.input_folder.resolve()),
    }


# ── Queue ─────────────────────────────────────────────────────────────────────

@app.get("/queue/status", tags=["ops"])
def queue_status():
    return {
        "queue_depth": prediction_queue.qsize(),
        "max_size": settings.queue_max_size,
        "worker_threads": settings.max_worker_threads,
    }


# ── Categories ────────────────────────────────────────────────────────────────

@app.get("/categories", response_model=List[CategoryTable], tags=["data"])
def list_categories(session: Session = Depends(get_db_session)):
    """List all collection sessions."""
    return session.exec(select(CategoryTable)).all()


@app.get("/categories/{category_id}", tags=["data"])
def get_category(category_id: str, session: Session = Depends(get_db_session)):
    """Get a single category with all its images."""
    cat = session.get(CategoryTable, category_id)
    if cat is None:
        raise HTTPException(status_code=404, detail="Category not found")

    images = session.exec(
        select(ImageTable).where(ImageTable.category_id == category_id)
    ).all()

    return {
        "category": cat,
        "images": images,
        "total_images": len(images),
        "completed": sum(1 for i in images if i.processing_status == "completed"),
        "queued": sum(1 for i in images if i.processing_status == "queued"),
        "processing": sum(1 for i in images if i.processing_status == "processing"),
        "failed": sum(1 for i in images if i.processing_status == "failed"),
        "deleted": sum(1 for i in images if i.processing_status == "deleted"),
    }


# ── Images ────────────────────────────────────────────────────────────────────

@app.get("/images/{image_id}", tags=["data"])
def get_image(image_id: str, session: Session = Depends(get_db_session)):
    """Get a single image record with prediction results."""
    image = session.get(ImageTable, image_id)
    if image is None:
        raise HTTPException(status_code=404, detail="Image not found")
    return image