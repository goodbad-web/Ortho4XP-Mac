#!/bin/bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$script_dir"
cd "$repo_root"

if ! command -v brew >/dev/null 2>&1; then
    echo "Homebrew is required to install Ortho4XP dependencies." >&2
    exit 1
fi

# Install system dependencies via Homebrew.
brew install gdal spatialindex p7zip proj python-tk imagemagick

# Use the Python installed by Homebrew instead of whichever python3 happens to
# win PATH resolution (for example, a pyenv interpreter).
BREW_PREFIX="$(brew --prefix)"
PYTHON_BIN="$BREW_PREFIX/bin/python3"
if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Homebrew Python was not found at $PYTHON_BIN." >&2
    exit 1
fi

# Set environment variables for building Python extensions (GDAL, PROJ, etc.)
export LDFLAGS="-L$BREW_PREFIX/lib"
export CPPFLAGS="-I$BREW_PREFIX/include -DABS(x)=((x)<0?-(x):(x))"
export C_INCLUDE_PATH="$BREW_PREFIX/include"
export CPLUS_INCLUDE_PATH="$BREW_PREFIX/include"
export PROJ_DIR="$BREW_PREFIX"

# Recreate an existing environment when its interpreter is missing, not a
# virtual environment, or no longer belongs to the current Homebrew Python.
venv_dir="$repo_root/.venv"
venv_python="$venv_dir/bin/python"
expected_base_executable="$("$PYTHON_BIN" -c 'import os, sys; print(os.path.realpath(getattr(sys, "_base_executable", sys.executable)))')"
recreate_venv=0

if [[ -e "$venv_dir" || -L "$venv_dir" ]]; then
    if [[ ! -x "$venv_python" ]]; then
        recreate_venv=1
    else
        actual_base_executable=""
        if ! actual_base_executable="$("$venv_python" -c 'import os, sys; sys.exit(1) if sys.prefix == sys.base_prefix else None; print(os.path.realpath(getattr(sys, "_base_executable", sys.executable)))')"; then
            recreate_venv=1
        elif [[ "$actual_base_executable" != "$expected_base_executable" ]]; then
            recreate_venv=1
        fi
    fi
fi

if (( recreate_venv )); then
    backup_dir="$repo_root/.venv-broken-backup-$(date +%Y%m%d-%H%M%S)"
    backup_index=0
    while [[ -e "$backup_dir" || -L "$backup_dir" ]]; do
        backup_index=$((backup_index + 1))
        backup_dir="$repo_root/.venv-broken-backup-$(date +%Y%m%d-%H%M%S)-$backup_index"
    done
    mv "$venv_dir" "$backup_dir"
    echo "Backed up invalid .venv to $backup_dir"
fi

if [[ ! -d "$venv_dir" ]]; then
    "$PYTHON_BIN" -m venv "$venv_dir"
fi

venv_python="$venv_dir/bin/python"
if [[ ! -x "$venv_python" ]]; then
    echo "The virtual environment Python was not created at $venv_python." >&2
    exit 1
fi

# Upgrade pip and install dependencies using the exact environment Python.
"$venv_python" -m pip install --upgrade pip

# Install dependencies from requirements.txt
# Note: GDAL needs to match the system version
"$venv_python" -m pip install -r requirements.txt

# Fail setup if the installed environment is internally inconsistent or cannot
# import the application's complete startup dependency graph.
"$venv_python" -m pip check
"$venv_python" -c 'import Ortho4XP'

echo "Installation complete. Run '.venv/bin/python Ortho4XP.py' to start Ortho4XP."
