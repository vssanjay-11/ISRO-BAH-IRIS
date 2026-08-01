"""
streamlit_app.py — Root entry point for Streamlit Community Cloud deployment.

Streamlit Cloud requires the main file at the repo root.
This module delegates immediately to the actual application in frontend/app.py.
"""

import runpy
import os
import sys

# Ensure repo root is on path so all imports (config, backend, utils) resolve
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Auto-download weights on first cold start (Real-ESRGAN + FastSAM)
try:
    from startup import main as _startup
    _startup()
except Exception:
    pass  # Non-fatal — app will still launch; weights error shown in UI

# Run the actual Streamlit app
runpy.run_path(os.path.join(ROOT, "frontend", "app.py"), run_name="__main__")
