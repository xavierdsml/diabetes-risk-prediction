#!/bin/bash
echo "Installing dependencies..."
pip install -r requirements.txt

echo "Training ML models..."
python -m ml.train

echo "Build complete!"
