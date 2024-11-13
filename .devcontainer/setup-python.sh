#!/bin/bash

# Error handling
set -e  # Exit on error
trap 'echo "Error on line $LINENO. Exit code: $?"' ERR

# Function for timestamped logging
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Function to check command success
check_step() {
    if [ $? -eq 0 ]; then
        log "✓ $1 completed successfully"
    else
        log "✗ Error: $1 failed"
        exit 1
    fi
}

log "Starting Python environment setup..."

# Update package lists
log "Updating package lists..."
sudo apt-get update
check_step "Package list update"

# Install Python tools if needed
log "Installing Python development tools..."
sudo apt-get install -y python3-pip python3-venv
check_step "Python tools installation"

# Set up virtual environment
log "Creating virtual environment..."
python -m venv .venv
check_step "Virtual environment creation"

# Activate virtual environment
log "Activating virtual environment..."
source .venv/bin/activate
check_step "Virtual environment activation"

# Upgrade pip
log "Upgrading pip..."
python -m pip install --upgrade pip
check_step "Pip upgrade"

# Install Python dependencies if requirements.txt exists
if [ -f "requirements.txt" ]; then
    log "Installing Python packages from requirements.txt..."
    pip install -r requirements.txt
    check_step "Python packages installation"
else
    log "No requirements.txt found. Installing core packages..."
    pip install playwright pytest pytest-playwright
    check_step "Core packages installation"
fi

# Install Playwright
log "Installing Playwright browsers..."
playwright install
check_step "Playwright browsers installation"

log "Setup completed successfully!"

# Print environment info
log "Environment Information:"
python --version
pip --version
playwright --version

log "You can now start developing!"