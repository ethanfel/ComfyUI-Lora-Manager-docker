/**
 * Community Creations page — card grid of community images grouped by LoRA.
 */

// -- State ----------------------------------------------------------------
let _allImages = {};   // sha256 -> [{image}, ...]
let _modelNames = {};  // sha256 -> model_name
let _sortKey = "reactions:desc";

// -- Init -----------------------------------------------------------------
async function init() {
    setupFetchButton();
    setupSortSelect();
    await loadImages();
}

// -- Load images from API -------------------------------------------------
async function loadImages() {
    try {
        // Paginate through all lora list pages to collect hashes + names
        // (server caps page_size at 100)
        const hashes = [];
        let page = 1;
        while (true) {
            const listResp = await fetch(`/api/lm/loras/list?page=${page}&page_size=100`);
            const listData = await listResp.json();
            const items = listData.items || [];
            if (items.length === 0) break;

            for (const item of items) {
                if (item.sha256) {
                    hashes.push(item.sha256);
                    _modelNames[item.sha256] = item.model_name || item.file_name || "Unknown";
                }
            }

            // Stop if we got fewer than page_size (last page)
            if (items.length < 100) break;
            page++;
        }

        if (hashes.length === 0) {
            showEmpty();
            return;
        }

        // Fetch community images for all hashes
        const resp = await fetch("/api/lm/community-images/by-hashes", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ hashes }),
        });
        const data = await resp.json();
        if (data.success && data.images) {
            _allImages = data.images;
        }

        if (Object.keys(_allImages).length === 0) {
            showEmpty();
        } else {
            renderGrid();
        }
    } catch (err) {
        console.error("[Community] Failed to load images:", err);
        showEmpty();
    }
}

// -- Render grid ----------------------------------------------------------
function renderGrid() {
    const grid = document.getElementById("communityGrid");
    const empty = document.getElementById("communityEmpty");
    if (!grid) return;

    grid.innerHTML = "";
    empty.style.display = "none";

    // Build sorted groups
    const groups = buildSortedGroups();

    if (groups.length === 0) {
        showEmpty();
        return;
    }

    for (const group of groups) {
        const section = document.createElement("div");
        section.className = "community-lora-group";

        // Header
        const header = document.createElement("div");
        header.className = "community-lora-header";
        header.innerHTML = `<h3>${escapeHtml(group.name)}</h3>
            <span class="lora-link">${group.images.length} image${group.images.length !== 1 ? "s" : ""}</span>`;
        section.appendChild(header);

        // Cards
        const cardsDiv = document.createElement("div");
        cardsDiv.className = "community-cards";

        for (const img of group.images) {
            cardsDiv.appendChild(createCard(img, group.sha256));
        }

        section.appendChild(cardsDiv);
        grid.appendChild(section);
    }
}

function buildSortedGroups() {
    const groups = [];
    for (const [sha256, images] of Object.entries(_allImages)) {
        if (!images || images.length === 0) continue;
        const name = _modelNames[sha256] || "Unknown";

        // Sort images within group by reactions
        const sorted = [...images].sort((a, b) => {
            const ra = (a.like_count || 0) + (a.heart_count || 0);
            const rb = (b.like_count || 0) + (b.heart_count || 0);
            return rb - ra;
        });

        groups.push({ sha256, name, images: sorted });
    }

    // Sort groups
    const [key, dir] = _sortKey.split(":");
    const asc = dir === "asc" ? 1 : -1;

    if (key === "reactions") {
        groups.sort((a, b) => {
            const ra = a.images.reduce((s, i) => s + (i.like_count || 0) + (i.heart_count || 0), 0);
            const rb = b.images.reduce((s, i) => s + (i.like_count || 0) + (i.heart_count || 0), 0);
            return (rb - ra) * asc;
        });
    } else if (key === "recent") {
        groups.sort((a, b) => {
            const da = a.images[0]?.created_at || "";
            const db_ = b.images[0]?.created_at || "";
            return db_.localeCompare(da) * asc;
        });
    } else if (key === "lora") {
        groups.sort((a, b) => a.name.localeCompare(b.name) * asc);
    }

    return groups;
}

// -- Card creation --------------------------------------------------------
function createCard(img, sha256) {
    const card = document.createElement("div");
    card.className = "community-card";
    card.addEventListener("click", () => showDetail(img, sha256));

    // Image
    const imgUrl = img.local_filename
        ? `/example_images_static/${img.local_filename}`
        : img.image_url || "";

    card.innerHTML = `
        <img class="community-card-image" src="${escapeHtml(imgUrl)}"
             alt="Community creation" loading="lazy"
             onerror="this.style.display='none'">
        <div class="community-card-body">
            <div class="community-card-prompt">${escapeHtml(img.prompt || "")}</div>
            <div class="community-card-meta">
                ${img.sampler ? `<span class="community-meta-tag">${escapeHtml(img.sampler)}</span>` : ""}
                ${img.steps ? `<span class="community-meta-tag">${img.steps} steps</span>` : ""}
                ${img.cfg_scale ? `<span class="community-meta-tag">CFG ${img.cfg_scale}</span>` : ""}
                ${img.base_model ? `<span class="community-meta-tag">${escapeHtml(img.base_model)}</span>` : ""}
            </div>
            <div class="community-card-footer">
                <div class="community-card-reactions">
                    ${img.like_count ? `<span class="community-reaction"><i class="fas fa-thumbs-up"></i> ${img.like_count}</span>` : ""}
                    ${img.heart_count ? `<span class="community-reaction"><i class="fas fa-heart"></i> ${img.heart_count}</span>` : ""}
                    ${img.comment_count ? `<span class="community-reaction"><i class="fas fa-comment"></i> ${img.comment_count}</span>` : ""}
                </div>
                <span class="community-card-user">${escapeHtml(img.username || "")}</span>
            </div>
        </div>
    `;
    return card;
}

// -- Detail modal ---------------------------------------------------------
function showDetail(img, sha256) {
    // Remove existing overlay
    const existing = document.querySelector(".community-detail-overlay");
    if (existing) existing.remove();

    const imgUrl = img.local_filename
        ? `/example_images_static/${img.local_filename}`
        : img.image_url || "";

    const overlay = document.createElement("div");
    overlay.className = "community-detail-overlay";
    overlay.addEventListener("click", (e) => {
        if (e.target === overlay) overlay.remove();
    });

    overlay.innerHTML = `
        <div class="community-detail">
            <img class="community-detail-image" src="${escapeHtml(imgUrl)}" alt="Community creation">
            <div class="community-detail-info">
                <h4>Prompt</h4>
                <div class="community-detail-prompt">
                    <button class="copy-btn" title="Copy prompt"><i class="fas fa-copy"></i> Copy</button>
                    ${escapeHtml(img.prompt || "")}
                </div>
                ${img.negative_prompt ? `
                    <h4>Negative Prompt</h4>
                    <div class="community-detail-prompt">${escapeHtml(img.negative_prompt)}</div>
                ` : ""}
                <h4>Parameters</h4>
                <div class="community-detail-params">
                    ${img.steps ? `<div class="community-detail-param"><strong>Steps:</strong> ${img.steps}</div>` : ""}
                    ${img.sampler ? `<div class="community-detail-param"><strong>Sampler:</strong> ${escapeHtml(img.sampler)}</div>` : ""}
                    ${img.cfg_scale ? `<div class="community-detail-param"><strong>CFG Scale:</strong> ${img.cfg_scale}</div>` : ""}
                    ${img.seed ? `<div class="community-detail-param"><strong>Seed:</strong> ${img.seed}</div>` : ""}
                    ${img.denoise ? `<div class="community-detail-param"><strong>Denoise:</strong> ${img.denoise}</div>` : ""}
                    ${img.base_model ? `<div class="community-detail-param"><strong>Base Model:</strong> ${escapeHtml(img.base_model)}</div>` : ""}
                    ${img.width && img.height ? `<div class="community-detail-param"><strong>Size:</strong> ${img.width}x${img.height}</div>` : ""}
                </div>
                <div class="community-card-reactions" style="margin-top:12px;">
                    ${img.like_count ? `<span class="community-reaction"><i class="fas fa-thumbs-up"></i> ${img.like_count}</span>` : ""}
                    ${img.heart_count ? `<span class="community-reaction"><i class="fas fa-heart"></i> ${img.heart_count}</span>` : ""}
                    ${img.laugh_count ? `<span class="community-reaction"><i class="fas fa-laugh"></i> ${img.laugh_count}</span>` : ""}
                    ${img.comment_count ? `<span class="community-reaction"><i class="fas fa-comment"></i> ${img.comment_count}</span>` : ""}
                </div>
                <div class="community-card-user" style="margin-top:8px;">
                    by ${escapeHtml(img.username || "unknown")}
                    ${img.created_at ? ` &middot; ${new Date(img.created_at).toLocaleDateString()}` : ""}
                </div>
            </div>
        </div>
    `;

    // Copy button handler
    document.body.appendChild(overlay);
    const copyBtn = overlay.querySelector(".copy-btn");
    if (copyBtn) {
        copyBtn.addEventListener("click", (e) => {
            e.stopPropagation();
            navigator.clipboard.writeText(img.prompt || "").then(() => {
                copyBtn.innerHTML = '<i class="fas fa-check"></i> Copied';
                setTimeout(() => {
                    copyBtn.innerHTML = '<i class="fas fa-copy"></i> Copy';
                }, 2000);
            });
        });
    }

    // Close on Escape
    const escHandler = (e) => {
        if (e.key === "Escape") {
            overlay.remove();
            document.removeEventListener("keydown", escHandler);
        }
    };
    document.addEventListener("keydown", escHandler);
}

// -- Fetch button ---------------------------------------------------------
function setupFetchButton() {
    const btn = document.getElementById("fetchCommunityBtn");
    if (!btn) return;

    btn.addEventListener("click", async () => {
        btn.disabled = true;
        btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> <span>Fetching...</span>';

        try {
            const resp = await fetch("/api/lm/community-images/fetch", { method: "POST" });
            const data = await resp.json();
            if (data.success) {
                const count = data.stored || 0;
                btn.innerHTML = `<i class="fas fa-check"></i> <span>${count} images saved</span>`;
                setTimeout(() => {
                    btn.innerHTML = '<i class="fas fa-images"></i> <span>Fetch Community Images</span>';
                    btn.disabled = false;
                }, 3000);
                // Reload images
                await loadImages();
            } else {
                throw new Error(data.error || "Unknown error");
            }
        } catch (err) {
            btn.innerHTML = '<i class="fas fa-exclamation-triangle"></i> <span>Error</span>';
            btn.title = err.message || String(err);
            console.error("[Community] Fetch failed:", err);
            setTimeout(() => {
                btn.innerHTML = '<i class="fas fa-images"></i> <span>Fetch Community Images</span>';
                btn.title = "Fetch community images from CivitAI";
                btn.disabled = false;
            }, 5000);
        }
    });
}

// -- Sort select ----------------------------------------------------------
function setupSortSelect() {
    const select = document.getElementById("communitySortSelect");
    if (!select) return;
    select.addEventListener("change", () => {
        _sortKey = select.value;
        renderGrid();
    });
}

// -- Helpers --------------------------------------------------------------
function showEmpty() {
    const grid = document.getElementById("communityGrid");
    const empty = document.getElementById("communityEmpty");
    if (grid) grid.innerHTML = "";
    if (empty) empty.style.display = "";
}

function escapeHtml(str) {
    const div = document.createElement("div");
    div.textContent = str;
    return div.innerHTML;
}

// -- Start ----------------------------------------------------------------
if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
} else {
    init();
}
