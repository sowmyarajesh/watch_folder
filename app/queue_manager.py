"""
queue_manager.py
────────────────
Central asyncio.Queue that connects:
  watcher  ──(put)──▶  queue  ──(get)──▶  worker

Each item in the queue is a dict:
{
    "image_id"   : str,
    "image_path" : str,
    "category_id": str,
}

A separate cancelled_images set allows the worker to skip
processing for images/categories deleted before inference starts.
"""
import asyncio
from typing import Set

from app.config import settings

# Bounded queue — blocks producers when full, providing natural back-pressure
# for large (1000+) image sessions.
prediction_queue: asyncio.Queue = asyncio.Queue(maxsize=settings.queue_max_size)

# Thread-safe sets for cancellation signalling
cancelled_images: Set[str] = set()      # image_ids
cancelled_categories: Set[str] = set()  # category_ids


def cancel_image(image_id: str) -> None:
    cancelled_images.add(image_id)


def cancel_category(category_id: str) -> None:
    cancelled_categories.add(category_id)


def is_cancelled(image_id: str, category_id: str) -> bool:
    return image_id in cancelled_images or category_id in cancelled_categories
