"""
ASGI entrypoint for Vercel.

Vercel detects `app` as the ASGI handler. We import the FastAPI
application instance from the backend package so routing and logic
remain in `backend/denoise.py`.
"""

from backend.denoise import app  # noqa: F401

