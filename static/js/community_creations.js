/**
 * Community Creations page — card grid of community images grouped by LoRA,
 * paginated by model to avoid loading too much data at once.
 */

// -- State ----------------------------------------------------------------
let _sortKey = "reactions:desc";
let _currentPage = 1;
let _totalPages = 1;
const PAGE_SIZE = 10;

// -- Init -----------------------------------------------------------------
async function init() {
    setupFetchButton();
    setupSortSelect();
    await loadPage(1);
}

// -- Load a single page of models -----------------------------------------
async function loadPage(page) {
    _currentPage = page;
    const grid = document.getElementById("communityGrid");
    if (grid) grid.innerHTML = '<div class="community-loading"><i class="fas fa-spinner fa-spin"></i> Loading...</div>';

    try {
        const resp = await fetch(
            `/api/lm/community-images/by-models?page=${page}&page_size=${PAGE_SIZE}&sort=${encodeURIComponent(_sortKey)}`
        );
        const data = await resp.json();

        if (!data.success || !data.models || data.models.length === 0) {
            showEmpty();
            renderPagination(0, 0);
            return;
        }

        _totalPages = data.total_pages;
        renderGrid(data.models);
        renderPagination(data.page, data.total_pages);
        window.scrollTo({ top: 0, behavior: "smooth" });
    } catch (err) {
        console.error("[Community] Failed to load page:", err);
        showEmpty();
        renderPagination(0, 0);
    }
}

// -- Render grid ----------------------------------------------------------
function renderGrid(models) {
    const grid = document.getElementById("communityGrid");
    const empty = document.getElementById("communityEmpty");
    if (!grid) return;

    grid.innerHTML = "";
    if (empty) empty.style.display = "none";

    for (const model of models) {
        const section = document.createElement("div");
        section.className = "community-lora-group";

        // Header
        const header = document.createElement("div");
        header.className = "community-lora-header";
        header.innerHTML = `<h3>${escapeHtml(model.model_name)}</h3>
            <span class="lora-link">${model.image_count} image${model.image_count !== 1 ? "s" : ""}</span>`;
        section.appendChild(header);

        // Sort images within group by reactions
        const sorted = [...model.images].sort((a, b) => {
            const ra = (a.like_count || 0) + (a.heart_count || 0);
            const rb = (b.like_count || 0) + (b.heart_count || 0);
            return rb - ra;
        });

        // Cards
        const cardsDiv = document.createElement("div");
        cardsDiv.className = "community-cards";
        for (const img of sorted) {
            cardsDiv.appendChild(createCard(img, model.sha256));
        }
        section.appendChild(cardsDiv);
        grid.appendChild(section);
    }
}

// -- Pagination -----------------------------------------------------------
function renderPagination(currentPage, totalPages) {
    let pager = document.getElementById("communityPagination");
    if (!pager) {
        pager = document.createElement("div");
        pager.id = "communityPagination";
        pager.className = "community-pagination";
        const grid = document.getElementById("communityGrid");
        if (grid) grid.parentNode.insertBefore(pager, grid.nextSibling);
    }

    if (totalPages <= 1) {
        pager.innerHTML = "";
        return;
    }

    let html = "";

    // Prev button
    html += `<button class="page-btn" ${currentPage <= 1 ? "disabled" : ""} data-page="${currentPage - 1}">
        <i class="fas fa-chevron-left"></i>
    </button>`;

    // Page numbers — show up to 7 with ellipsis
    const pages = buildPageNumbers(currentPage, totalPages);
    for (const p of pages) {
        if (p === "...") {
            html += `<span class="page-ellipsis">&hellip;</span>`;
        } else {
            html += `<button class="page-btn ${p === currentPage ? "active" : ""}" data-page="${p}">${p}</button>`;
        }
    }

    // Next button
    html += `<button class="page-btn" ${currentPage >= totalPages ? "disabled" : ""} data-page="${currentPage + 1}">
        <i class="fas fa-chevron-right"></i>
    </button>`;

    pager.innerHTML = html;

    // Bind click handlers
    pager.querySelectorAll(".page-btn:not([disabled])").forEach((btn) => {
        btn.addEventListener("click", () => {
            const p = parseInt(btn.dataset.page, 10);
            if (p >= 1 && p <= totalPages) loadPage(p);
        });
    });
}

function buildPageNumbers(current, total) {
    if (total <= 7) {
        return Array.from({ length: total }, (_, i) => i + 1);
    }
    const pages = [];
    pages.push(1);
    if (current > 3) pages.push("...");
    for (let i = Math.max(2, current - 1); i <= Math.min(total - 1, current + 1); i++) {
        pages.push(i);
    }
    if (current < total - 2) pages.push("...");
    pages.push(total);
    return pages;
}

// -- Card creation --------------------------------------------------------
function createCard(img, sha256) {
    const card = document.createElement("div");
    card.className = "community-card";
    card.addEventListener("click", () => showDetail(img, sha256));

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
    const existing = document.querySelector(".community-detail-overlay");
    if (existing) existing.remove();

    const imgUrl = img.local_filename
        ? `/example_images_static/${img.local_filename}`
        : img.image_url || "";

    const overlay = document.createElement("div");
    overlay.className = "community-detail-overlay";
    const removeOverlay = () => {
        overlay.remove();
        document.removeEventListener("keydown", escHandler);
    };
    overlay.addEventListener("click", (e) => {
        if (e.target === overlay) removeOverlay();
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
                    ${img.seed != null ? `<div class="community-detail-param"><strong>Seed:</strong> ${img.seed}</div>` : ""}
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

    const escHandler = (e) => {
        if (e.key === "Escape") removeOverlay();
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

        let ws = null;
        try {
            const proto = location.protocol === "https:" ? "wss:" : "ws:";
            ws = new WebSocket(`${proto}//${location.host}/ws/fetch-progress`);
            ws.onmessage = (e) => {
                try {
                    const msg = JSON.parse(e.data);
                    if (msg.type === "community_images_progress") {
                        btn.innerHTML = `<i class="fas fa-spinner fa-spin"></i> <span>${msg.current}/${msg.total}</span>`;
                    }
                } catch {}
            };
        } catch {}

        try {
            const resp = await fetch("/api/lm/community-images/fetch", { method: "POST" });
            const data = await resp.json();
            if (ws) ws.close();
            if (data.success) {
                const count = data.stored || 0;
                btn.innerHTML = `<i class="fas fa-check"></i> <span>${count} images saved</span>`;
                setTimeout(() => {
                    btn.innerHTML = '<i class="fas fa-images"></i> <span>Fetch Community Images</span>';
                    btn.disabled = false;
                }, 3000);
                await loadPage(1);
            } else {
                throw new Error(data.error || "Unknown error");
            }
        } catch (err) {
            if (ws) ws.close();
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
        loadPage(1);
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
