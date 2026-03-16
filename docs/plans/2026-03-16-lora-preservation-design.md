# LoRA Preservation: Auto-Upload Deleted Models to HuggingFace

## Goal

Detect when LoRA models are deleted from CivitAI and automatically upload them to HuggingFace as a public archive — serving as both personal backup and community preservation resource (structured for CivArchive and others to ingest).

## Data Model Changes

Add three columns to the SQLite `models` table:

- `hf_uploaded` INTEGER (0/1) — whether uploaded to HuggingFace
- `hf_repo_id` TEXT — e.g. `username/flux-lora-archive`
- `hf_uploaded_at` REAL — timestamp of upload

A model is a **preservation candidate** when `civitai_deleted=1 AND hf_uploaded=0`.

## HuggingFace Repo Structure

One repo per base model: `username/flux-lora-archive`, `username/sdxl-lora-archive`, etc.

```
flux-lora-archive/
├── README.md                          # Repo-level index of all preserved LoRAs
├── model-name-abc123/                 # Folder per LoRA (name + sha256[:6])
│   ├── README.md                      # Model card: name, trigger words, tags, description,
│   │                                  #   original CivitAI URL, creator, base model, license
│   ├── model.safetensors              # The actual LoRA file
│   ├── example_images/                # Creator's example images
│   │   ├── 001.webp
│   │   └── ...
│   ├── community_images/              # Community creations
│   │   ├── 001.webp
│   │   └── ...
│   └── metadata.json                  # Machine-readable: all CivitAI metadata, hashes,
│                                      #   trigger words, tags, stats, dates
```

Repos are auto-created via `HfApi.create_repo(exist_ok=True, private=False)`.

## Upload Pipeline

`HuggingFaceUploadService` handles uploads:

1. **Resolve repo** — Map base model to repo name, create repo if needed
2. **Build folder name** — `{model-name}-{sha256[:6]}`
3. **Generate metadata.json** — All cached CivitAI metadata, trigger words, tags, stats, hashes
4. **Generate model card README.md** — Human-readable with inline example images
5. **Upload files** — Via `huggingface_hub.HfApi.upload_folder()` (single commit per model):
   - model.safetensors, example_images/, community_images/, metadata.json, README.md
6. **Update repo README** — Append entry to repo-level index
7. **Mark in DB** — Set `hf_uploaded=1`, `hf_repo_id`, `hf_uploaded_at`

WebSocket broadcasts progress.

## Detection & Triggering

### Event-driven (immediate)

When `MetadataSyncService` detects a 404 from CivitAI, emit an event. `HuggingFaceUploadService` picks it up and queues a background upload. User sees a WebSocket notification.

### Periodic sweep (catch-up)

On server startup or manual trigger, query DB for `civitai_deleted=1 AND hf_uploaded=0`. Queue all matches. Catches: pre-existing deleted models, failed uploads, offline deletions.

### Manual trigger

- Per-model "Upload to HuggingFace" button on model cards (works on any model, not just deleted)
- Bulk action: "Upload all deleted models to HuggingFace"

### Guards

- Skip if no HF token configured (log warning)
- Skip if model file no longer on disk
- Skip if already uploaded unless force=true

## Settings & UI

### Settings additions

- `huggingface_token` — Write-access API token
- `huggingface_username` — Auto-detected from token on first use
- `hf_auto_upload_on_delete` — Toggle for auto-upload (default: true)

### UI additions

- **Model card:** HF icon/badge if uploaded (links to HF page), "Upload to HF" button if not
- **Deleted model indicator:** Visual badge "Deleted from CivitAI"
- **Settings page:** HF token input + connection test button
- **Bulk action:** "Upload deleted models to HuggingFace" button

No new pages — integrates into existing model cards and settings.

## Dependencies

- `huggingface_hub` Python package (add to requirements.txt)
