"""Locate the tweet screenshots that the VLM scripts take as input.

The screenshots are distributed separately (see the README) and are expected
under ``display/``. The ``display_link`` field of an instance holds the file
name of its screenshot.
"""
import base64
import os

DISPLAY_DIR = 'display'


def display_path(instance):
    """Return the local path of the screenshot of an instance."""
    return os.path.join(DISPLAY_DIR, os.path.basename(instance['display_link']))


def display_data_url(instance):
    """Return the screenshot of an instance as a base64 data URL (for API calls)."""
    with open(display_path(instance), 'rb') as image_file:
        encoded = base64.b64encode(image_file.read()).decode('ascii')
    return f'data:image/png;base64,{encoded}'
