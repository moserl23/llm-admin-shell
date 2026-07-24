"""Core data containers for ML experiments.

This module defines the lightweight example representation shared across
training, evaluation, and analysis code.
"""

from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)
class Example:
    """Immutable text example used throughout the ML pipeline.

    Besides the text and target label, examples may carry optional grouping
    and provenance metadata for split control, debugging, and error analysis.
    """

    text: str
    label: str
    group: Optional[str] = None

    # Optional provenance fields are kept on the example so downstream code can
    # inspect errors without reloading the original source record.
    log_type: Optional[str] = None
    path: Optional[str] = None
    line_no: Optional[int] = None

