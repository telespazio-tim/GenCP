"""Shared pipeline configuration base.

Both pipelines are configured around an input directory and an output directory;
this base captures that shared surface and coerces the paths. Pipeline-specific
configuration (bands, chip size, styling, tiling, thresholds) lives in each
pipeline's own config subclass.
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class BaseConfig:
    """Configuration common to the HR and VHR data-preparation pipelines."""

    input_dir: Path = field(default_factory=lambda: Path("."))
    output_dir: Path = field(default_factory=lambda: Path("."))

    def __post_init__(self):
        self.input_dir = Path(self.input_dir)
        self.output_dir = Path(self.output_dir)
