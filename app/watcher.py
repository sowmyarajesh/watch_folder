"""
watcher.py
──────────
Uses watchdog to monitor input_folder for:
  • New sub-folder  → register category, scan existing .jpg files, enqueue them
  • New .jpg file   → register image, enqueue for inference
  • Deleted .jpg    → mark image deleted, cancel in-flight inference
  • Deleted folder  → mark category + all images deleted, cancel all in-flight

watchdog callbacks run in a watchdog thread — all DB writes go through
the synchronous get_session() context manager (safe for threads).
Enqueuing to asyncio.Queue from a thread requires run_coroutine_threadsafe.
"""
import asyncio
import logging
import os
from datetime import datetime
from pathlib import Path

from sqlmodel import select
from watchdog.events import (
    DirCreatedEvent,
    DirDeletedEvent,
    FileCreatedEvent,
    FileDeletedEvent,
    FileSystemEventHandler,
)
from watchdog.observers import Observer

from app.config import settings
from app.database import get_session
from app.models import CategoryTable, ImageTable
from app.queue_manager import (
    cancel_category,
    cancel_image,
    prediction_queue,
)
from app.schemas import StatusEnum

logger = logging.getLogger(__name__)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _get_or_create_category(sub_folder_name: str, path: str) -> CategoryTable:
    """Return existing category row or create a new one."""
    with get_session() as session:
        cat = session.exec(
            select(CategoryTable).where(CategoryTable.sub_folder_name == sub_folder_name)
        ).first()
        if cat is None:
            cat = CategoryTable(
                sub_folder_name=sub_folder_name,
                path=path,
                status=StatusEnum.queued,
            )
            session.add(cat)
            session.commit()
            session.refresh(cat)
            logger.info(f"New category registered: {sub_folder_name} ({cat.category_id})")
        return cat


def _enqueue_image(image_id: str, image_path: str, category_id: str, loop: asyncio.AbstractEventLoop) -> None:
    """Thread-safe enqueue to asyncio.Queue."""
    item = {"image_id": image_id, "image_path": image_path, "category_id": category_id}
    asyncio.run_coroutine_threadsafe(prediction_queue.put(item), loop)


def _register_image(image_path: str, category_id: str, loop: asyncio.AbstractEventLoop) -> None:
    """Insert image row and enqueue for inference."""
    abs_path = str(Path(image_path).resolve())
    with get_session() as session:
        # Idempotent — skip if already registered
        existing = session.exec(
            select(ImageTable).where(ImageTable.image_path == abs_path)
        ).first()
        if existing is not None:
            return

        image = ImageTable(
            category_id=category_id,
            image_path=abs_path,
            processing_status=StatusEnum.queued,
        )
        session.add(image)
        session.commit()
        session.refresh(image)
        logger.info(f"Image queued: {Path(abs_path).name} → {image.image_id}")

    _enqueue_image(image.image_id, abs_path, category_id, loop)


def _scan_existing_images(folder_path: str, category_id: str, loop: asyncio.AbstractEventLoop) -> None:
    """When a folder is first detected, register all .jpg files already inside."""
    for jpg in Path(folder_path).glob("*.jpg"):
        _register_image(str(jpg), category_id, loop)


# ── Event Handler ─────────────────────────────────────────────────────────────

class InputFolderHandler(FileSystemEventHandler):
    def __init__(self, loop: asyncio.AbstractEventLoop):
        super().__init__()
        self.loop = loop
        self._input_root = settings.input_folder.resolve()

    def _is_direct_subfolder(self, path: str) -> bool:
        """Only react to immediate children of input_folder, not deeper nesting."""
        return Path(path).parent.resolve() == self._input_root

    def _is_jpg_in_subfolder(self, path: str) -> bool:
        p = Path(path)
        return (
            p.suffix.lower() == ".jpg"
            and p.parent.parent.resolve() == self._input_root
        )

    # ── Directory events ──────────────────────────────────────────────────────

    def on_created(self, event):
        if isinstance(event, DirCreatedEvent) and self._is_direct_subfolder(event.src_path):
            folder_path = event.src_path
            sub_folder_name = Path(folder_path).name
            cat = _get_or_create_category(sub_folder_name, str(Path(folder_path).resolve()))
            _scan_existing_images(folder_path, cat.category_id, self.loop)

    def on_deleted(self, event):
        if isinstance(event, DirDeletedEvent) and self._is_direct_subfolder(event.src_path):
            sub_folder_name = Path(event.src_path).name
            self._handle_folder_deleted(sub_folder_name)

        elif isinstance(event, FileDeletedEvent) and self._is_jpg_in_subfolder(event.src_path):
            self._handle_image_deleted(event.src_path)

    # ── File events ───────────────────────────────────────────────────────────

    def on_created(self, event):  # noqa: F811  (override covers both dir + file)
        if isinstance(event, DirCreatedEvent) and self._is_direct_subfolder(event.src_path):
            folder_path = event.src_path
            sub_folder_name = Path(folder_path).name
            cat = _get_or_create_category(sub_folder_name, str(Path(folder_path).resolve()))
            _scan_existing_images(folder_path, cat.category_id, self.loop)

        elif isinstance(event, FileCreatedEvent) and self._is_jpg_in_subfolder(event.src_path):
            sub_folder_name = Path(event.src_path).parent.name
            with get_session() as session:
                cat = session.exec(
                    select(CategoryTable).where(CategoryTable.sub_folder_name == sub_folder_name)
                ).first()
            if cat:
                _register_image(event.src_path, cat.category_id, self.loop)
            else:
                # Race: file arrived before DirCreated fired — create category now
                folder_path = str(Path(event.src_path).parent)
                cat = _get_or_create_category(sub_folder_name, str(Path(folder_path).resolve()))
                _register_image(event.src_path, cat.category_id, self.loop)

    # ── Deletion handlers ─────────────────────────────────────────────────────

    def _handle_folder_deleted(self, sub_folder_name: str) -> None:
        logger.warning(f"Folder deleted: {sub_folder_name}")
        with get_session() as session:
            cat = session.exec(
                select(CategoryTable).where(CategoryTable.sub_folder_name == sub_folder_name)
            ).first()
            if cat is None:
                return

            cancel_category(cat.category_id)

            # Mark all images deleted
            images = session.exec(
                select(ImageTable).where(ImageTable.category_id == cat.category_id)
            ).all()
            for img in images:
                if img.processing_status not in (StatusEnum.completed,):
                    img.processing_status = StatusEnum.deleted
                    img.updated_at = datetime.utcnow()
                    session.add(img)
                cancel_image(img.image_id)

            cat.status = StatusEnum.deleted
            cat.updated_at = datetime.utcnow()
            session.add(cat)
            session.commit()
            logger.info(f"Category {cat.category_id} and {len(images)} images marked deleted")

    def _handle_image_deleted(self, image_path: str) -> None:
        abs_path = str(Path(image_path).resolve())
        logger.warning(f"Image deleted: {abs_path}")
        with get_session() as session:
            image = session.exec(
                select(ImageTable).where(ImageTable.image_path == abs_path)
            ).first()
            if image is None:
                return
            cancel_image(image.image_id)
            if image.processing_status != StatusEnum.completed:
                image.processing_status = StatusEnum.deleted
                image.updated_at = datetime.utcnow()
                session.add(image)
                session.commit()


# ── Observer bootstrap ────────────────────────────────────────────────────────

def start_watcher(loop: asyncio.AbstractEventLoop) -> Observer:
    """
    Starts the watchdog Observer in its own daemon thread.
    Returns the Observer so main.py can stop it on shutdown.
    """
    # Bootstrap: register any sub-folders that already exist before app start
    _bootstrap_existing_folders(loop)

    handler = InputFolderHandler(loop)
    observer = Observer()
    observer.schedule(handler, str(settings.input_folder.resolve()), recursive=True)
    observer.start()
    logger.info(f"Watching: {settings.input_folder.resolve()}")
    return observer


def _bootstrap_existing_folders(loop: asyncio.AbstractEventLoop) -> None:
    """On startup, pick up any folders/images already present in input_folder."""
    input_root = settings.input_folder.resolve()
    for sub in input_root.iterdir():
        if sub.is_dir():
            cat = _get_or_create_category(sub.name, str(sub))
            _scan_existing_images(str(sub), cat.category_id, loop)
