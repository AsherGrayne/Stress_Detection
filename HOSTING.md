# Hosting the demo publicly

Anyone can use the app if you put **Flutter Web** on a public URL and **Spring + Python inference** on public HTTPS APIs, with CORS allowing your web origin.

## What goes where

| Piece | What it is | Typical hosting |
|--------|------------|-----------------|
| **Flutter Web** | Static HTML/JS/CSS after `flutter build web` | Firebase Hosting, Netlify, Cloudflare Pages, GitHub Pages, Vercel |
| **Spring Boot** | REST API | Same VM as Docker, or **Cloud Run / Fly.io / Railway / Render** (container) |
| **Python inference** | ML service | **Second container** on the same platform, or second Cloud Run service |
| **MongoDB** | You already use **Atlas** | Allow access from your host IPs or `0.0.0.0/0` for a wide-open demo (not for real PHI) |

## Architecture (public demo)

```
User browser → https://your-app.web.app  (Flutter Web)
                    ↓ API calls
              https://api.yourdomain.com  (Spring on 8090)
                    ↓ internal
              http://inference:8081       (Python — private network only on most platforms)
```

On **single VM** (DigitalOcean droplet, EC2, etc.): `docker compose up` + **Caddy** or **nginx** for TLS (Let’s Encrypt) in front of `8090`. Publish **only** `8090` (and `443`); keep **8081** on Docker’s internal network so only Spring talks to inference.

On **Cloud Run / Fly.io** (two services): Spring gets a public URL; set `STRESS_INFERENCE_BASE_URL` to the **internal** or **private** URL of the inference service if the platform supports service-to-service auth; otherwise inference may need a public URL too (lock it down with a shared secret header if needed).

## Things you must change for a public web app

1. **API base URL in Flutter** — build with your real API:
   ```bash
   flutter build web --dart-define=API_BASE=https://api.yourdomain.com
   ```
2. **CORS on Spring** — allow your web origin (e.g. `https://your-app.web.app`), not only `localhost`. Update `WebConfig.java` `allowedOriginPatterns` or set from env if you add that.
3. **HTTPS** — browsers block mixed content; the API should be `https://`.
4. **Atlas Network Access** — add the **egress IPs** of your host, or temporarily `0.0.0.0/0` for a throwaway demo DB user with least privilege.
5. **Cost & abuse** — inference is CPU-heavy; use a small demo CSV window, rate limits, or HTTP auth for anything beyond a class demo.

## “Easiest” paths (high level)

- **One cheap VPS + Docker Compose + Caddy** — one place to run `docker compose`, automatic TLS, point DNS at the VPS. Good for “anyone with the link.”
- **Managed containers** — **Railway**, **Render**, **Fly.io**, **Google Cloud Run**: deploy `docker-compose`-like as two services; more setup, scales better.
- **Flutter only on CDN** — always pair with a reachable API; the web build has no secrets if `API_BASE` is only your public API URL.

## Feasibility

**Yes — it can be hosted online** so people can see it working. The ML model ships inside the **inference** Docker image; you do not expose the raw `Job/` folder to browsers, only the Spring API.

If you tell me your preferred host (**VPS**, **Railway**, **Render**, **Cloud Run**, etc.), we can add concrete steps or config files for that target.
