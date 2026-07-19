# Hosting the demo publicly

Anyone can use the app if you host the **Node.js dashboard/API** on a public HTTPS URL and keep the **Python inference** service reachable from it.

## What goes where

| Piece | What it is | Typical hosting |
|--------|------------|-----------------|
| **Node.js app** | Express API plus static browser dashboard | Same VM as Docker, or **Cloud Run / Fly.io / Railway / Render** (container) |
| **Python inference** | ML service | **Second container** on the same platform, or second Cloud Run service |
| **MongoDB** | You already use **Atlas** | Allow access from your host IPs or `0.0.0.0/0` for a wide-open demo (not for real PHI) |

## Architecture (public demo)

```
User browser -> https://yourdomain.com      (Node dashboard/API on 8090)
                    |
                    v
              http://inference:8081         (Python, private network preferred)
```

On **single VM** (DigitalOcean droplet, EC2, etc.): `docker compose up` + **Caddy** or **nginx** for TLS in front of `8090`. Publish **only** `8090` (and `443`); keep **8081** on Docker's internal network so only Node talks to inference.

On **Cloud Run / Fly.io** (two services): Node gets a public URL; set `STRESS_INFERENCE_BASE_URL` to the **internal** or **private** URL of the inference service if the platform supports service-to-service auth; otherwise inference may need a public URL too.

## Things you must change for a public web app

1. **HTTPS** - browsers should access the Node dashboard through `https://`.
2. **Inference networking** - keep port `8081` private if possible and point `STRESS_INFERENCE_BASE_URL` at the internal inference URL.
3. **Atlas Network Access** - add the egress IPs of your host, or temporarily `0.0.0.0/0` for a throwaway demo DB user with least privilege.
4. **Cost and abuse** - inference is CPU-heavy; use a small demo CSV window, rate limits, or HTTP auth for anything beyond a class demo.

## “Easiest” paths (high level)

- **One cheap VPS + Docker Compose + Caddy** — one place to run `docker compose`, automatic TLS, point DNS at the VPS. Good for “anyone with the link.”
- **Managed containers** — **Railway**, **Render**, **Fly.io**, **Google Cloud Run**: deploy `docker-compose`-like as two services; more setup, scales better.

## Feasibility

**Yes, it can be hosted online** so people can see it working. The ML model ships inside the **inference** Docker image; browsers only talk to the Node app.

If you tell me your preferred host (**VPS**, **Railway**, **Render**, **Cloud Run**, etc.), we can add concrete steps or config files for that target.
