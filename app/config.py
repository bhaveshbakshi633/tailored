"""
GeomCalc configuration.

All tunable parameters live here so the measurement pipeline
is fully configurable without touching algorithmic code.
"""

from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field


class MeshProcessingConfig(BaseSettings):
    """Parameters governing mesh cleaning and alignment."""

    # Laplacian smoothing
    smoothing_iterations: int = 3
    smoothing_lambda: float = 0.5

    # Floor removal
    floor_percentile: float = 2.0  # lowest N% of vertices used for floor detection
    floor_margin_m: float = 0.02  # meters above detected floor to cut

    # Component filtering
    min_component_ratio: float = 0.05  # components smaller than this fraction are removed


class LandmarkConfig(BaseSettings):
    """Height-percentage priors for anatomical landmarks (fraction of total height)."""

    neck_min: float = 0.79
    neck_max: float = 0.87
    shoulder_height: float = 0.81
    chest_min: float = 0.68
    chest_max: float = 0.76
    waist_min: float = 0.57
    waist_max: float = 0.65
    hip_min: float = 0.48
    hip_max: float = 0.55
    crotch_min: float = 0.43
    crotch_max: float = 0.50
    knee_height: float = 0.27
    ankle_height: float = 0.04

    # Curvature-based profile parameters
    profile_n_slices: int = 300
    profile_smooth_sigma: float = 5.0  # Gaussian smoothing sigma (in slice units)


class MeasurementConfig(BaseSettings):
    """Multi-slice averaging and search parameters."""

    n_averaging_slices: int = 7
    averaging_band_width_m: float = 0.02  # ±1 cm band around target height
    search_resolution: int = 60  # number of slices when searching for extremum
    min_contour_points: int = 20  # minimum points for a valid cross-section


class ConfidenceConfig(BaseSettings):
    """Weights and thresholds for the confidence model."""

    w_slice_consistency: float = 2.0
    w_symmetry: float = 1.5
    w_density: float = 1.0
    w_smoothness: float = 1.0
    min_density_triangles: int = 40  # minimum triangle intersections per slice


class SizingConfig(BaseSettings):
    """Size recommendation parameters."""

    default_fit: str = "regular"  # tight | regular | loose
    fit_offset_tight_cm: float = -2.0
    fit_offset_loose_cm: float = 3.0
    default_stretch_factor: float = 1.0  # 1.0 = no stretch


class StorageConfig(BaseSettings):
    """File storage paths."""

    upload_dir: Path = Path("/tmp/geomcalc/uploads")
    result_dir: Path = Path("/tmp/geomcalc/results")
    max_upload_size_mb: int = 100


class AppConfig(BaseSettings):
    """Top-level application configuration."""

    app_name: str = "GeomCalc"
    version: str = "0.1.0"
    debug: bool = False
    api_key_header: str = "X-API-Key"
    cors_origins: list[str] = ["*"]

    mesh: MeshProcessingConfig = Field(default_factory=MeshProcessingConfig)
    landmarks: LandmarkConfig = Field(default_factory=LandmarkConfig)
    measurement: MeasurementConfig = Field(default_factory=MeasurementConfig)
    confidence: ConfidenceConfig = Field(default_factory=ConfidenceConfig)
    sizing: SizingConfig = Field(default_factory=SizingConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)


config = AppConfig()
