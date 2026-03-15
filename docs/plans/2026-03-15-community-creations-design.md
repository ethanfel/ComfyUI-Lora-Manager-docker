# Community Creations Feature — Design

## Goal

Add a "Community Creations" tab showing the top community-generated images for each installed LoRA, with full generation metadata (prompts, sampler settings). Images are fetched from CivitAI, downloaded locally for permanence, and displayed in a card grid grouped by LoRA.

## Architecture

New service + DB table + routes + frontend tab. Follows the same patterns as the CivitAI stats feature: separate table in `civitai_stats.db`, dedicated service class, bulk fetch with WebSocket progress, and a new UI tab.

### Data Flow

**Bulk fetch:**
1. User clicks "Fetch Community Images" button on the Community Creations tab.
2. For each LoRA with `civitai_model_id` in scanner cache:
   - Call `GET /api/v1/images?modelId={id}&sort=Most Reactions&limit=20`.
   - Filter out: images by the LoRA author (`civitai_creator_username`), images without `meta.meta.prompt`, prompts shorter than 20 characters.
   - Keep top 10 after filtering.
   - Download image files to `{example_images_path}/{sha256}/community/`.
   - Store metadata in `community_images` table.
3. WebSocket progress broadcasts throughout.
4. Rate limit: 1.5s between API calls.

**Piggyback on stats fetch:**
After stats fetch completes for a LoRA, if it has zero community images in DB, trigger image fetch for that model.

**Permanence:**
Once downloaded, images and metadata persist even if CivitAI removes the model. Images are only cleaned up when the user explicitly deletes a LoRA.

### DB Schema (in `civitai_stats.db`)

```sql
CREATE TABLE community_images (
    civitai_image_id INTEGER PRIMARY KEY,
    sha256 TEXT NOT NULL,
    civitai_model_id INTEGER,
    username TEXT,
    image_url TEXT,
    local_filename TEXT,
    width INTEGER,
    height INTEGER,
    prompt TEXT,
    negative_prompt TEXT,
    steps INTEGER,
    sampler TEXT,
    cfg_scale REAL,
    seed INTEGER,
    denoise REAL,
    base_model TEXT,
    like_count INTEGER DEFAULT 0,
    heart_count INTEGER DEFAULT 0,
    laugh_count INTEGER DEFAULT 0,
    comment_count INTEGER DEFAULT 0,
    created_at TEXT,
    fetched_at REAL
);
CREATE INDEX idx_community_sha256 ON community_images(sha256);
```

### API Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/api/lm/community-images/fetch` | Trigger bulk fetch |
| POST | `/api/lm/community-images/by-hashes` | Get images for given model hashes |
| GET | `/api/lm/community-images/status` | DB count and fetch status |

### Image Storage

- Path: `{example_images_path}/{sha256}/community/{civitai_image_id}.jpg`
- Served via existing `/example_images_static/` static route.
- URL pattern in frontend: `/example_images_static/{sha256}/community/{civitai_image_id}.jpg`

### Filtering Rules

1. Skip images by the LoRA author: compare image `username` against `civitai_creator_username` from scanner cache (case-insensitive).
2. Skip images without `meta.meta.prompt`.
3. Skip images with prompt length < 20 characters.
4. Keep top 10 per LoRA after filtering (API returns sorted by Most Reactions).

### Frontend

- New "Community Creations" tab in the main nav bar.
- Card grid layout: image thumbnail, prompt excerpt, generation params, reaction counts.
- Cards grouped under LoRA name headers (with link to the LoRA in the Loras tab).
- Click a card to expand: full prompt (copyable), all generation params, full-size image.
- "Fetch Community Images" button in toolbar with WebSocket progress bar.
- Sort options: Most Liked, Most Recent, by LoRA name.

### Files to Create/Modify

| File | Action |
|------|--------|
| `py/services/community_images_db.py` | Create — DB access class for `community_images` table |
| `py/services/community_images_service.py` | Create — Fetch service (API calls, filtering, image download) |
| `py/routes/community_images_routes.py` | Create — Route handlers |
| `py/lora_manager.py` | Modify — Register community images routes |
| `standalone.py` | Modify — Register community images routes |
| `static/js/community_creations.js` | Create — Frontend tab logic |
| `static/css/community_creations.css` | Create — Styles for the new tab |
| `templates/community_creations.html` | Create — HTML template |
| `py/services/civitai_stats_service.py` | Modify — Add piggyback hook for new LoRAs |
