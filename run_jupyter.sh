#!/bin/bash

set -e # exit script if an error occurs

source myenv/bin/activate # Install with ./setup.sh
cd code
python -m jupyter notebook demo.ipynb # Start jupyter using the python from the virtual environment
