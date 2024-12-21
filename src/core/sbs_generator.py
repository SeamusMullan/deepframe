"""Side-by-side stereo image generator from depth maps."""

from dataclasses import dataclass

import cv2
import numpy as np

from ..utils.config import FillMode, SBSLayout


@dataclass
class SBSSettings:
    """Settings for SBS generation."""

    depth_strength: float = 0.5  # 0.0 - 1.0, how much parallax
    eye_separation: int = 63  # Pixels, virtual eye distance
    depth_focus: float = 0.5  # 0.0 (near) - 1.0 (far), focal plane
    fill_mode: FillMode = FillMode.INPAINT
    layout: SBSLayout = SBSLayout.HALF


class SBSGenerator:
    """Generate side-by-side stereo images from RGB + depth."""

    def __init__(self, settings: SBSSettings | None = None) -> None:
        """
        Initialize the SBS generator.

        Args:
            settings: SBS generation settings.
        """
        self.settings = settings or SBSSettings()

    def generate(
        self,
        image: np.ndarray,
        depth: np.ndarray,
        settings: SBSSettings | None = None,
    ) -> np.ndarray:
        """
        Generate side-by-side stereo image.

        Args:
            image: RGB image (H, W, 3), uint8.
            depth: Depth map (H, W), float32, 0-1 range (0=near, 1=far).
            settings: Optional settings override.

        Returns:
            SBS image. Shape depends on layout:
            - FULL: (H, W*2, 3)
            - HALF: (H, W, 3)
        """
        s = settings or self.settings
        h, w = image.shape[:2]

        # Calculate disparity from depth
        # Disparity is inversely related to depth
        # Objects at focus plane have 0 disparity
        disparity = self._depth_to_disparity(depth, s)

        # Generate left and right views
        left = self._warp_image(image, disparity, is_left=True, fill_mode=s.fill_mode)
        right = self._warp_image(image, disparity, is_left=False, fill_mode=s.fill_mode)

        # Combine into SBS format
        if s.layout == SBSLayout.FULL:
            # Full resolution: just concatenate
            return np.hstack([left, right])
        else:
            # Half resolution: resize each eye to half width
            half_w = w // 2
            left_half = cv2.resize(left, (half_w, h), interpolation=cv2.INTER_AREA)
            right_half = cv2.resize(right, (half_w, h), interpolation=cv2.INTER_AREA)
            return np.hstack([left_half, right_half])

    def generate_anaglyph(
        self,
        image: np.ndarray,
        depth: np.ndarray,
        settings: SBSSettings | None = None,
    ) -> np.ndarray:
        """
        Generate red/cyan anaglyph image for preview.

        Args:
            image: RGB image (H, W, 3), uint8.
            depth: Depth map (H, W), float32, 0-1 range.
            settings: Optional settings override.

        Returns:
            Anaglyph RGB image (H, W, 3).
        """
        s = settings or self.settings

        # Generate stereo pair
        disparity = self._depth_to_disparity(depth, s)
        left = self._warp_image(image, disparity, is_left=True, fill_mode=s.fill_mode)
        right = self._warp_image(image, disparity, is_left=False, fill_mode=s.fill_mode)

        # Create anaglyph: red channel from left, cyan (green+blue) from right
        anaglyph = np.zeros_like(image)
        anaglyph[:, :, 0] = left[:, :, 0]  # Red from left
        anaglyph[:, :, 1] = right[:, :, 1]  # Green from right
        anaglyph[:, :, 2] = right[:, :, 2]  # Blue from right

        return anaglyph

    def _depth_to_disparity(self, depth: np.ndarray, settings: SBSSettings) -> np.ndarray:
        """
        Convert depth map to disparity map.

        Args:
            depth: Normalized depth (0=near, 1=far).
            settings: SBS settings.

        Returns:
            Disparity map in pixels.
        """
        # Shift depth so focus plane is at 0
        centered_depth = depth - settings.depth_focus

        # Scale by eye separation and depth strength
        # Positive disparity = shift right for left eye
        max_disparity = settings.eye_separation * settings.depth_strength

        # Near objects (negative centered_depth) get positive disparity (pop out)
        # Far objects (positive centered_depth) get negative disparity (recede)
        disparity = -centered_depth * max_disparity

        return disparity.astype(np.float32)

    def _warp_image(
        self,
        image: np.ndarray,
        disparity: np.ndarray,
        is_left: bool,
        fill_mode: FillMode,
    ) -> np.ndarray:
        """
        Warp image based on disparity to create stereo view.

        Args:
            image: Original RGB image.
            disparity: Disparity map in pixels.
            is_left: True for left eye, False for right eye.
            fill_mode: How to fill disoccluded regions.

        Returns:
            Warped image for the specified eye.
        """
        h, w = image.shape[:2]

        # For left eye, shift by +disparity/2
        # For right eye, shift by -disparity/2
        shift_factor = 0.5 if is_left else -0.5
        pixel_shift = disparity * shift_factor

        # Create sampling grid
        x_coords = np.arange(w, dtype=np.float32)
        y_coords = np.arange(h, dtype=np.float32)
        xx, yy = np.meshgrid(x_coords, y_coords)

        # Apply shift
        src_x = xx - pixel_shift
        src_y = yy  # No vertical shift

        # Remap image
        warped = cv2.remap(
            image,
            src_x,
            src_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

        # Handle disoccluded regions
        if fill_mode != FillMode.BLACK:
            # Create mask of invalid pixels (outside original image bounds)
            invalid_mask = (src_x < 0) | (src_x >= w)

            if fill_mode == FillMode.INPAINT and np.any(invalid_mask):
                # Use inpainting to fill holes
                mask = invalid_mask.astype(np.uint8) * 255
                warped = cv2.inpaint(warped, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

            elif fill_mode == FillMode.STRETCH:
                # Already handled by BORDER_REPLICATE
                pass

        return warped


def create_depth_visualization(
    depth: np.ndarray,
    colormap: int = cv2.COLORMAP_INFERNO,
) -> np.ndarray:
    """
    Create a colorized visualization of a depth map.

    Args:
        depth: Normalized depth map (0-1).
        colormap: OpenCV colormap to use.

    Returns:
        RGB visualization image.
    """
    # Convert to 8-bit
    depth_uint8 = (depth * 255).astype(np.uint8)

    # Apply colormap
    colored = cv2.applyColorMap(depth_uint8, colormap)

    # Convert BGR to RGB
    return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
