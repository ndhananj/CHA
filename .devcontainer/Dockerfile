FROM mcr.microsoft.com/devcontainers/python:3.11

# Install necessary dependencies for Playwright
RUN apt-get update && apt-get install -y \
    libegl1 \
    libopus0 \
    libwoff1 \
    libharfbuzz-icu0 \
    libgstreamer-plugins-base1.0-0 \
    libgstreamer1.0-0 \
    libhyphen0 \
    libx264-dev \
    libenchant-2-2 \
    libwebpdemux2 \
    libflite1 \
    libgudev-1.0-0 \
    libevdev2 \
    libgl1 \
    libgl1-mesa-glx \
    libgles2 \
    libgtk-3-0 \
    libpangocairo-1.0-0 \
    libcairo2 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libxkbcommon0 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxrandr2 \
    libgbm1 \
    libpango-1.0-0 \
    libfontconfig1 \
    libasound2 \
    && rm -rf /var/lib/apt/lists/*

# Install additional dependencies that might be needed
RUN apt-get update && apt-get install -y \
    libgtk-3-0 \
    libpangocairo-1.0-0 \
    libcairo2 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libxkbcommon0 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxrandr2 \
    libgbm1 \
    libpango-1.0-0 \
    libfontconfig1 \
    libasound2 \
    && rm -rf /var/lib/apt/lists/*

# Install Playwright system dependencies
RUN apt-get update && \
    npx playwright install-deps && \
    rm -rf /var/lib/apt/lists/*