from __future__ import annotations

from typing import Any

TASK_DESCRIPTIONS: dict[str, str] = {
    "spontaneous": "Spontaneous activity without explicit visual stimulation.",
    "checkerboard": (
        "Visual stimulation using a flickering checkerboard pattern "
        "(visual-stimulation subtype)."
    ),
    "kalatsky": (
        "Visual stimulation using moving-bar sweeps (Kalatsky retinotopy subtype)."
    ),
}

STATIC_METADATA: dict[str, Any] = {
    "manufacturer": "Verasonics",
    "manufacturers_model_name": "Vantage 128",
    "software_version": "Alan Urban Technology & Consulting (AUTC)",
    "probe_manufacturer": "Vermon",
    "probe_type": "linear",
    "probe_model": "L22-XTech",
    "probe_central_frequency": 15e6,
    "probe_number_of_elements": 128,
    "probe_pitch": 0.1,
    "probe_focal_width": 0.4,
    "probe_focal_depth": 8.0,
    "power_doppler_integration_duration": 0.3,
    "power_doppler_integration_stride": 0.3,
    "clutter_filter_window_duration": 0.4,
    "clutter_filter_window_stride": 0.3,
    "clutter_filters": [
        "highpass:15Hz",
        "svd:remove_first_15_components",
    ],
}
