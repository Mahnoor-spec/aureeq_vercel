## 🐳 Shared Environments (Advanced)
If your VPS already has other containers running (e.g., Nginx Proxy Manager, Traefik, or other websites):

1.  **Avoid Port Conflicts**: In `docker-compose.vps.yml`, if port `80` or `443` is already taken, you can either:
    *   Stop the existing service temporarily.
    *   OR (Recommended) Disable the `aureeq-proxy` service in the compose file and point your existing external proxy to `aureeq-frontend:80` and `aureeq-backend:8001`.
2.  **Isolated Network**: AUREEQ uses a private network (`aureeq-net`), so it will not interfere with your other containers' internal traffic.

## 🚀 Step-by-Step Launch (Standard)

**NOTE:** To avoid conflicts on shared VPS environments, AUREEQ now runs on high-range ports: **12080** (HTTP) and **12443** (HTTPS).

1.  **Transfer Files**: Copy all files to your VPS folder.
2.  **Add Certificates**: Place `fullchain.pem` and `privkey.pem` in the `./certs` folder.
3.  **Set Environment Variables**: Create a `.env` file OR add the `OPENAI_API_KEY` in your VPS GUI (Hostinger Panel).
4.  **Launch Services**:
    ```bash
    docker-compose up -d --build
    ```
5.  **Access**: Open `http://your-vps-ip:12080` in your browser.

## 🛠 Troubleshooting
*   **"Port is already allocated"**: I have moved the ports to 12080/12443 to fix this.
*   **GPU Error**: GPU reservations have been removed. AUREEQ will run on your CPU.
