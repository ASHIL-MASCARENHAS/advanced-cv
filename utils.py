"""Utilities module (lightweight): re-export HandDetector from HandTrackingModule.

This keeps backward compatibility for modules importing from `utils`.
"""

from HandTrackingModule import HandDetector

__all__ = ["HandDetector"]