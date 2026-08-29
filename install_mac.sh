#!/bin/bash
# Install system dependencies via Homebrew
brew install gdal spatialindex p7zip proj python-tk imagemagick

# Determine Homebrew prefix (Apple Silicon vs Intel)
if [[ $(uname -m) == "arm64" ]]; then
    BREW_PREFIX="/opt/homebrew"
else
    BREW_PREFIX="/usr/local"
fi

# Set environment variables for building Python extensions (GDAL, PROJ, etc.)
export LDFLAGS="-L$BREW_PREFIX/lib"
export CPPFLAGS="-I$BREW_PREFIX/include -DABS(x)=((x)<0?-(x):(x))"
export C_INCLUDE_PATH="$BREW_PREFIX/include"
export CPLUS_INCLUDE_PATH="$BREW_PREFIX/include"
export PROJ_DIR="$BREW_PREFIX"

# Create and activate virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies from requirements.txt
# Note: GDAL needs to match the system version
pip install -r requirements.txt

echo "Installation complete. Use 'source .venv/bin/activate' before running Ortho4XP.py"
