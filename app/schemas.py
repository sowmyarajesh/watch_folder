from enum import Enum


class StatusEnum(str, Enum):
    queued = "queued"
    processing = "processing"
    completed = "completed"
    deleted = "deleted"
    failed = "failed"
