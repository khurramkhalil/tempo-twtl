#!/bin/bash
# Quick setup script for the SELP-TWTL project

# Create the package structure
mkdir -p selp_twtl/twtl
mkdir -p selp_twtl/translator
mkdir -p selp_twtl/planner
mkdir -p selp_twtl/monitoring
mkdir -p examples

# Create __init__.py files
echo '"""SELP-TWTL package."""' > selp_twtl/__init__.py
echo '"""TWTL module."""' > selp_twtl/twtl/__init__.py
echo '"""Translator module."""' > selp_twtl/translator/__init__.py
echo '"""Planner module."""' > selp_twtl/planner/__init__.py
echo '"""Monitoring module."""' > selp_twtl/monitoring/__init__.py

# Copy source files to the right locations
cp twtl/grammar.py selp_twtl/twtl/
cp twtl/automaton.py selp_twtl/twtl/
cp translator/nl_to_twtl.py selp_twtl/translator/
cp planner/constrained_decoding.py selp_twtl/planner/
cp monitoring/execution_monitor.py selp_twtl/monitoring/

# Make sure the example is in the right place
cp -f examples/twtl_drone_example.py examples/

# Install the package in development mode
pip install -e .

echo "Setup complete! You can now run the example with:"
echo "python examples/twtl_drone_example.py"