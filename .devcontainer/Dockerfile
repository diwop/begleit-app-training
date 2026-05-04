FROM axolotlai/axolotl:main-latest

# Install uv just in case it is needed for fast dependency management
RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
    && cp /root/.local/bin/uv /usr/local/bin/uv || cp /root/.cargo/bin/uv /usr/local/bin/uv || true \
    && cp /root/.local/bin/uvx /usr/local/bin/uvx || cp /root/.cargo/bin/uvx /usr/local/bin/uvx || true
