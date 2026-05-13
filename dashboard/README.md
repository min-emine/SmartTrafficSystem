# Smart Traffic Dashboard

Next.js dashboard for the AI-driven smart traffic management system.

## Scripts

```bash
npm install
npm run dev
npm run build
npm run typecheck
```

`npm run build` exports the static dashboard to `out/`.

The dashboard now reads `public/traffic-state.json` when it exists. The Python runtime writes that file on each frame, so the UI mirrors the live detector state without needing a separate backend.

If the file is unavailable, the dashboard falls back to the bundled demo data and GIF assets.
