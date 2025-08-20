#!/usr/bin/env bash
# Create virtual environment
python -m venv neuralcanvas_env
source neuralcanvas_env/bin/activate  # On Windows: neuralcanvas_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
python main.py
