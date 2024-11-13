#!/bin/bash
set -e

# Install pyenv if not already installed
if [ ! -d "$HOME/.pyenv" ]; then
    curl https://pyenv.run | bash
fi

# Add pyenv to PATH
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv init -)"

# Install Python 3.11.4 if not already installed
if [ ! -d "$PYENV_ROOT/versions/3.11.4" ]; then
    pyenv install 3.11.4
fi

# Set as global version
pyenv global 3.11.4

# Install common packages
pip install --upgrade pip
pip install pytest black flake8 mypy

# Print versions for verification
echo "Python version:"
python --version
echo "Pip version:"
pip --version