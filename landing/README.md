# Landing Page

A single self-contained static HTML landing page for the **AI Research Assistant**
project — no build step, no dependencies beyond CDN-hosted Tailwind CSS and
Font Awesome.

## What it is

`index.html` is a marketing/overview page: problem statement, the 12-stage
pipeline diagram, feature grid, tech stack, screenshot placeholders, a
quickstart snippet, and links out to the GitHub repo and live demo. It does
**not** call the backend API — it is purely descriptive, safe to host
anywhere as a static file.

## Configure the links

Open `index.html` and edit the three constants near the bottom of the file:

```js
const GITHUB_URL = "https://github.com/YOUR_USERNAME/ai-research-assistant";
const LIVE_DEMO_URL = "https://your-deployed-url-here";
```

Every "GitHub" and "Live Demo" button/link on the page reads from these two
constants, so you only need to edit them once.

## Preview locally

```bash
cd ai-research-assistant/landing
python3 -m http.server 8080
# open http://localhost:8080
```

## Deploy options

- **GitHub Pages** (simplest, free): in the GitHub repo settings, enable
  Pages and point it at the `landing/` folder on the `main` branch (or copy
  `index.html` into a `docs/` folder if your GitHub Pages config requires
  that convention). The page will be served at
  `https://YOUR_USERNAME.github.io/ai-research-assistant/`.
- **Cloudflare Pages / Netlify / Vercel**: point any of these at the
  `landing/` directory as the site root — no build command needed.
- **Anywhere that serves static files**: it's one HTML file with zero
  server-side requirements.
