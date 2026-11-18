"""RGB colours for face values 0-5 (U R F D L B). A viewer concern, not a cube one."""

import numpy as np

# Same scheme as text_viewer.DEFAULT_PALETTE: U white, R red, F green, D yellow, L orange, B blue.
DEFAULT_RGB = np.array(
    [
        [1.00, 1.00, 1.00],
        [0.80, 0.05, 0.05],
        [0.00, 0.62, 0.28],
        [1.00, 0.84, 0.00],
        [1.00, 0.45, 0.00],
        [0.00, 0.30, 0.85],
    ]
)

BODY_RGB = np.array([0.05, 0.05, 0.05])
BACKGROUND_RGB = np.array([0.13, 0.14, 0.16])
