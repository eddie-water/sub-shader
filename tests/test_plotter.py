"""Tests for plotter validation behavior (QUAL-01, D-01, D-02)."""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch


class TestValidateTextureData:
    """Test that _validate_texture_data raises on invalid input."""

    def _make_renderer_with_mock_texture(self, texture_size=(256, 120)):
        """Create a minimal Renderer-like object for testing validation."""
        renderer = MagicMock()
        renderer.texture = MagicMock()
        renderer.texture.size = texture_size

        # Bind the real method to our mock
        from subshader.viz.plotter import Renderer
        renderer._validate_texture_data = Renderer._validate_texture_data.__get__(renderer)
        return renderer

    def test_raises_on_none(self):
        plotter = self._make_renderer_with_mock_texture()
        with pytest.raises(ValueError, match="Texture data is None"):
            plotter._validate_texture_data(None)

    def test_raises_on_nan(self):
        plotter = self._make_renderer_with_mock_texture(texture_size=(4, 3))
        data = np.ones((3, 4), dtype=np.float32)
        data[0, 0] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            plotter._validate_texture_data(data)

    def test_raises_on_wrong_dimensions(self):
        plotter = self._make_renderer_with_mock_texture()
        data = np.ones((10,), dtype=np.float32)  # 1D instead of 2D
        with pytest.raises(ValueError, match="Expected 2D"):
            plotter._validate_texture_data(data)

    def test_raises_on_no_shape(self):
        plotter = self._make_renderer_with_mock_texture()
        with pytest.raises(ValueError, match="no shape"):
            plotter._validate_texture_data("not an array")


class TestRenderGraphic:
    """Test that render_graphic re-raises exceptions."""

    def test_render_graphic_reraises_exceptions(self):
        """render_graphic should re-raise exceptions instead of swallowing them."""
        from subshader.viz.plotter import Renderer

        # Create a minimal renderer with mocked components
        renderer = MagicMock(spec=Renderer)
        renderer.ctx = MagicMock()
        renderer._check_gl_error = MagicMock(return_value=True)
        renderer.texture = MagicMock()
        renderer.vao = MagicMock()
        renderer.vao.render = MagicMock(side_effect=RuntimeError("GPU error"))
        renderer.TEXTURE_SLOT = 0

        # Bind the real render_graphic method
        renderer.render_graphic = Renderer.render_graphic.__get__(renderer)

        with pytest.raises(RuntimeError, match="GPU error"):
            renderer.render_graphic()
