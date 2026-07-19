# Production Deployment Manual

This document details production deployment steps, Docker configurations, environment values, and reverse-proxy setups.

---

## 1. Environment Configurations

Make sure to populate a production-grade `.env` file at the root. The checklist of configurations is:

*   **API Tokens**:
    *   `OPENAI_API_KEY`: Authentication key for GPT model runs.
    *   `GEMINI_API_KEY` / `ANTHROPIC_API_KEY`: API keys for multi-provider fallbacks.
*   **Database**:
    *   `DATABASE_URL`: Production SQL connection string (SQLite file, or PostgreSQL connection).
*   **System Layouts**:
    *   `WORKSPACE_DIR`: Absolute or relative path to workspace.
    *   `LOG_LEVEL`: Production recommended level `WARNING` or `INFO`.

---

## 2. Deploying with Docker Compose

Deploy the container stack in daemon (background) mode:

```bash
# Build and launch production containers
docker compose -f docker-compose.prod.yml up --build -d

# Verify containers are running
docker compose -f docker-compose.prod.yml ps
```

This starts two services:
1.  `streamlit`: Runs the analytical web dashboard on internal port `8501`.
2.  `nginx`: Listens on public port `80`, handling client connections, routing to Streamlit, websocket upgrades, and enforcing security headers.

---

## 3. Nginx Reverse Proxy Details

Our [nginx.conf](file:///d:/Final%20Project/capstone-multi-agent-analyst/nginx.conf) is optimized for high performance and heavy analytical uploads:

*   **Client Upload Support**: `client_max_body_size 100M` allows uploading large files without trigger Nginx entity-too-large errors.
*   **Websocket Routing**:
    ```nginx
    proxy_http_version 1.1;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "upgrade";
    ```
    This is required to maintain Streamlit's connection channel.
*   **Timeouts**: Connect, send, and read timeouts are set to `600s` to support long-running agent pipelines.

---

## 4. Volume Mounts & File Permissions

The production compose configuration mounts the local directory `./workspace` into `/app/workspace` in the container.
*   **Ownership**: The container runs under a non-root user `appuser` (UID/GID `10001`).
*   **Permissions**: Ensure the host `./workspace` folder is writable by UID 10001:
    ```bash
    chown -R 10001:10001 ./workspace
    ```
