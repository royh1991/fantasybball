#!/bin/bash
# Setup conda environment for H-scoring

echo "Creating conda environment 'h_scoring'..."
conda env create -f environment.yml

echo ""
echo "Environment created successfully!"
echo ""
echo "To activate the environment, run:"
echo "  conda activate h_scoring"
echo ""
echo "Then test the data collector with:"
echo "  python test_data_collector.py"