# Project report: 2026-11-24

## Summary

Made the app testable end to end on Datalab (the Linux JupyterHub machine) with one command, reachable from any browser through a Cloudflare tunnel, and preloaded with a tiny demo dataset. Testing it on a real Android phone exposed a layout bug that hid the image completely; that is fixed. Then added a guided tutorial, a project-wide YOLO auto-label that saves its boxes, downloads of images with the annotations drawn on, a new dark gray/black theme with self-hosted fonts, and tool-switch animations.

Commits: `cb5e5ea`, `b3f5390`, `f33945b`, `cdfcab2`, `12ef1f4` (and this report).

## Why an early working prototype

We chose to build a small but complete, runnable prototype early and test it on the real target machine and devices, rather than spending longer comparing designs and tools on paper.

- **Errors show up immediately.** A running app fails in concrete, visible ways. A design document can look correct while the real system breaks on details nobody thought to write down.
- **We see the real output.** Masks, boxes, label tags and exported files can be inspected directly. Whether an annotation is "visible enough" or a layout "works on mobile" is answered by looking, not by guessing.
- **Fixes are cheap while the code is small.** Each problem was found and fixed within the same day, before more features were built on top of it.
- **It matches the project requirements.** The exam asks for an app that runs with one command, measures model quality and effort, and produces an exported dataset. All of that needs a working system, not just an analysis.

Concrete examples from today, each found only by running the app:

1. **Image invisible on Android.** On a phone the annotate page showed no image at all. Cause: the page was a fixed row layout, and the right-hand panel (320 px, not allowed to shrink) plus the toolbar left the canvas about 0 px wide. Nothing in the design or the desktop tests hinted at this; it appeared the first time the app was opened on a phone through the tunnel.
2. **No touch input.** The canvas listened only to mouse events, which the canvas library does not fire for touch, and touching the image scrolled the page instead of drawing. Found by trying to draw on the phone.
3. **Stale UI after an update.** After rebuilding, the browser kept the old `index.html` from its cache and loaded the old app, so a fix looked like it had not worked. Found while checking the mobile fix. The server now sends `Cache-Control: no-cache` for `index.html`.
4. **Tutorial card in the wrong place.** The tour's explanation card first appeared far below the screen. Cause: the page's fade-in animation uses a CSS `transform`, which makes `position: fixed` relative to that element instead of the window. Found on the first visual check; fixed by rendering the tour directly on `<body>`.
5. **Faint label names.** Label names drawn as plain coloured text were hard to read on busy photos. Seeing them on real COCO images led to solid coloured label tags with white text.

None of these would have been caught by comparing options in theory. Each took minutes to fix once it was visible.

## Done

**One-command test on Datalab: `./run.sh datalab`**
- Runs in order: setup (models, Python env, npm install, UI build), backend unit tests, model smoke test on `bus.jpg`, then an end-to-end HTTP check against a temporary server with its own data folder, so real data is untouched. If everything passes, it starts the app.
- New `backend/scripts/e2e_check.py` goes through the app the way a user does: health, UI page, project, label, upload, SAM click and box, YOLO per image and for the whole project, save and reload annotations, annotated-image download, effort session and events, stats, events CSV, and COCO / YOLO / annotated-images export.
- Options: `NO_SERVE=1` (checks only), `SKIP_MODELS=1` (skip AI steps), `E2E_PORT`.

**Cloudflare tunnel**
- After the checks, `./run.sh datalab` opens a Cloudflare quick tunnel (no account needed) and prints a public `https://*.trycloudflare.com` URL. `cloudflared` is downloaded to `~/.local/bin` if missing (no root).
- `TUNNEL=0` turns it off; `TUNNEL=1 ./run.sh` adds it to the normal launcher.
- Security note: the app has no login, so anyone with the URL can view and change data while the tunnel runs. The URL changes every run; stop with Ctrl+C when done.

**Demo dataset**
- `backend/scripts/load_demo.py` downloads Ultralytics **coco8** (8 images from COCO val2017, about 0.5 MB) once into `data/demo/` and creates the project "Demo: coco8" with 12 labels for the classes present. Loaded automatically by `./run.sh datalab`; skipped if already there.
- Ground-truth boxes are kept on disk (`data/demo/coco8/labels`) but not imported, so they can be used later for the mAP evaluation (M6).
- Licence: COCO annotations CC BY 4.0; images are Flickr photos under their own Creative Commons licences.

**Mobile fix**
- On narrow screens the toolbar, canvas and side panel stack vertically, and the header wraps. Desktop layout unchanged.
- Canvas uses pointer events (finger, pen, mouse), blocks page scrolling while drawing, and has − / + zoom buttons.
- Annotations are easier to see: thicker outlines, stronger fill, and a label-name tag on each shape.

**Guided tutorial**
- In-house tour with a circular spotlight that moves to each control and pulses, plus an explanation card. 17 stops in the editor (image, every tool, undo/redo, zoom, labels, auto-label, annotation list, mode, autosave, download, Mark done, navigation, help) and 8 on the project page.
- Opens on the first visit, can be skipped, and is replayed with the **?** button (or the `?` key in the editor). Works on desktop and mobile.

**Saving auto-labelled images**
- Already present: per-image "Auto-label this image" with accept / reject per proposal or all at once.
- New "Auto-label all images" on the project page (`POST /api/projects/{id}/auto-label`): runs YOLO on every image and saves the boxes as AI annotations with their scores. Labels are matched by name and created if missing. Images that already have annotations are skipped by default.
- New "⤓ Image" button in the editor (`GET /api/images/{id}/rendered`): downloads the image as a JPEG with its annotations drawn on.
- New "Annotated images (.zip)" export (`format=images`). COCO and YOLO zips still contain images plus labels.

**UI refresh**
- Near-black backgrounds, neutral grays, white primary buttons and active tool; cyan kept only for AI-related items.
- Self-hosted fonts bundled into the build (Inter, Space Grotesk, JetBrains Mono), so they work offline and through the tunnel.
- Tool switches pop the button and show a short card with the tool name and a tip; pages fade in. Animations are disabled when the device asks for reduced motion.

## Tests

- Backend: 11 tests pass (2 new: project auto-label with a fake YOLO model, and annotated-image rendering / export).
- End-to-end check: 16 checks pass locally without models (AI steps skipped). The AI steps run on Datalab.
- Browser checks at desktop and 375 px phone width: image visible, drawing a box saves, label tags readable, tour and tool-switch card render in the right place.

## Not yet verified

- `./run.sh datalab` end to end on Datalab, including the tunnel and the real SAM / YOLO steps.
- Real touch input on an Android phone (tested with mouse input at phone size).

## Risks and limits

- "Auto-label all images" runs as one request. Cloudflare quick tunnels cut requests off after about 100 seconds, so a project with hundreds of images could time out through the tunnel. The 8-image demo is fine.
- On a phone, Smart select cannot add background points (that needs right-click or Alt-click). Tapping the object and dragging a box both work.

## Next

- Run `./run.sh datalab` on Datalab and test from the phone.
- M6: evaluation scripts (click simulator NoC@85/90, mAP against the coco8 ground truth, user-study analysis).
