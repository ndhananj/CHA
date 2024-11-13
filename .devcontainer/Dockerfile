FROM mcr.microsoft.com/devcontainers/python:3.11

# Install Node.js for npx commands
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
    && apt-get install -y nodejs

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

# Install Playwright system dependencies
RUN npm install -g playwright && \
    npx playwright install-deps && \
    rm -rf /var/lib/apt/lists/*