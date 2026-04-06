"""Renderer module — OpenGL-based real-time visualization."""

from .renderer import Renderer

# DEPRECATED aliases — remove after all callers migrated
ShaderPlot = Renderer

__all__ = ["Renderer", "ShaderPlot"]
