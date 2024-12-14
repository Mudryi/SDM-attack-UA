from pydantic import BaseModel
from typing import List


class ClassificationItem(BaseModel):
    text: str = ""
    words: List[str] = []
    label: int = 0
