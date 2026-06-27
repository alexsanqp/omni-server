# Publishing this directory to `github.com/alexsanqp/omni-server`

This `omni-server/` tree is a **complete, standalone project** that was extracted
from the YoutubeAuto monorepo. It currently lives inside that repo only because
the automation that produced it could push to `alexsanqp/youtubeauto` but not to
`alexsanqp/omni-server`. Move it to its own repo with one of the methods below,
then delete `omni-server/` from YoutubeAuto.

## Method 1 — fresh repo (simplest, no history)

```bash
# from the YoutubeAuto checkout
cp -r omni-server /tmp/omni-server
cd /tmp/omni-server
git init -b main
git add -A
git commit -m "Initial import: standalone OmniParser v2 server"
git remote add origin git@github.com:alexsanqp/omni-server.git
git push -u origin main
```

## Method 2 — preserve history with `git subtree`

Run from the YoutubeAuto checkout (on the branch that contains `omni-server/`):

```bash
git subtree split --prefix=omni-server -b omni-server-export
git push git@github.com:alexsanqp/omni-server.git omni-server-export:main
```

## After publishing

1. In YoutubeAuto, remove the staging copy:
   ```bash
   git rm -r omni-server
   git commit -m "chore: omni-server moved to its own repo"
   ```
   (The YouTube backend already talks to it purely over HTTP via `OMNIPARSER_URL`
   — nothing else depends on this directory.)
2. On the GPU box, clone the new repo and follow its `README.md` (Docker or uv).
3. Point the client at it: set `OMNIPARSER_URL` (+ `OMNIPARSER_AUTH_TOKEN` if you
   enabled `OMNI_AUTH_TOKEN`) in the YouTube `.env`.

## Notes

- `LICENSE` is MIT with copyright "alexsanqp" — change the holder/year or the
  license to your preference before publishing.
- `vendor/` and `weights/` are git-ignored and never committed; they are
  re-created on the GPU box by `scripts/setup_vendor.py` and
  `scripts/download_weights.py`.
