import uuid
from datetime import datetime
from typing import Optional, List
from sqlmodel import SQLModel, Field, Relationship

from app.schemas import StatusEnum


def generate_uuid() -> str:
    return str(uuid.uuid4())


class CategoryTable(SQLModel, table=True):
    """
    One row per watched sub-folder (collection session).
    category_id  : internal UUID primary key
    sub_folder_name : the UID folder name (e.g. "session_abc123")
    path         : absolute path to the sub-folder
    status       : lifecycle state of the collection
    completed_percentage : 0-100 float derived from image statuses
    """
    __tablename__ = "category_table"

    category_id: str = Field(default_factory=generate_uuid, primary_key=True)
    sub_folder_name: str = Field(index=True, unique=True)
    path: str
    status: StatusEnum = Field(default=StatusEnum.queued)
    completed_percentage: float = Field(default=0.0)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    # Relationship — not a real column, used for ORM joins
    images: List["ImageTable"] = Relationship(back_populates="category")


class ImageTable(SQLModel, table=True):
    """
    One row per .jpg file discovered in a sub-folder.
    image_id         : internal UUID primary key
    category_id      : FK → category_table
    processing_status: lifecycle state of this image's inference
    prediction_result: JSON string of bounding boxes + labels (set after completion)
    image_path       : absolute path to source .jpg
    result_file_path : absolute path to annotated result image (set after completion)
    """
    __tablename__ = "image_table"

    image_id: str = Field(default_factory=generate_uuid, primary_key=True)
    category_id: str = Field(foreign_key="category_table.category_id", index=True)
    processing_status: StatusEnum = Field(default=StatusEnum.queued)
    prediction_result: Optional[str] = Field(default=None)   # JSON string
    image_path: str
    result_file_path: Optional[str] = Field(default=None)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    # Relationship
    category: Optional[CategoryTable] = Relationship(back_populates="images")
