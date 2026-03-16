/**
 * Community Creations page — card grid of community images grouped by LoRA,
 * filtered by base model tabs and paginated.
 */

// -- State ----------------------------------------------------------------
let _sortKey = "reactions:desc";
let _currentPage = 1;
let _totalPages = 1;
let _pageSize = 10;
let _baseModelFilter = "";  // "" = all
let _baseModelCounts = {};  // { "Flux.1 D": 12, "Pony": 8, ... }

// -- Init -----------------------------------------------------------------
async function init() {
    setupFetchButton();
    setupSortSelect();
    setupPageSizeSelect();
    await loadPage(1);
}

// -- Load a single page of models -----------------------------------------
async function loadPage(page) {
    _currentPage = page;
    const grid = document.getElementById("communityGrid");
    if (grid) grid.innerHTML = '<div class="community-loading"><i class="fas fa-spinner fa-spin"></i> Loading...</div>';

    try {
        let url = `/api/lm/community-images/by-models?page=${page}&page_size=${_pageSize}&sort=${encodeURIComponent(_sortKey)}`;
        if (_baseModelFilter) {
            url += `&base_model=${encodeURIComponent(_baseModelFilter)}`;
        }

        const resp = await fetch(url);
        const data = await resp.json();

        if (!data.success) {
            showEmpty();
            renderPagination(0, 0);
            return;
        }

        // Update base model tabs (always from full data, not filtered)
        if (data.base_models) {
            _baseModelCounts = data.base_models;
            renderBaseModelTabs();
        }

        if (!data.models || data.models.length === 0) {
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

// -- Base model tabs ------------------------------------------------------
function renderBaseModelTabs() {
    const container = document.getElementById("communityBaseModelTabs");
    if (!container) return;

    const entries = Object.entries(_baseModelCounts)
        .sort((a, b) => b[1] - a[1]);  // sort by count desc

    if (entries.length <= 1) {
        container.innerHTML = "";
        return;
    }

    // Total count across all base models
    const totalCount = entries.reduce((sum, [, count]) => sum + count, 0);

    let html = `<button class="base-model-tab ${!_baseModelFilter ? "active" : ""}" data-base-model="">All (${totalCount})</button>`;
    for (const [name, count] of entries) {
        const active = _baseModelFilter === name ? "active" : "";
        html += `<button class="base-model-tab ${active}" data-base-model="${escapeHtml(name)}">${escapeHtml(name)} (${count})</button>`;
    }

    container.innerHTML = html;

    container.querySelectorAll(".base-model-tab").forEach((btn) => {
        btn.addEventListener("click", () => {
            _baseModelFilter = btn.dataset.baseModel;
            loadPage(1);
        });
    });
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
        const baseTag = model.base_model
            ? `<span class="community-base-tag">${escapeHtml(model.base_model)}</span>`
            : "";
        header.innerHTML = `<h3>${escapeHtml(model.model_name)}</h3>
            ${baseTag}
            <a class="lora-link" href="/loras?search=${encodeURIComponent(model.model_name)}" title="View LoRA details"><i class="fas fa-external-link-alt"></i> View LoRA</a>
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
            cardsDiv.appendChild(createCard(img, model.sha256, model.model_name));
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

    // Page numbers
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
function createCard(img, sha256, modelName) {
    const card = document.createElement("div");
    card.className = "community-card";
    card.addEventListener("click", (e) => {
        if (e.target.closest("a")) return;
        showDetail(img, sha256, modelName);
    });

    const imgUrl = img.local_filename
        ? `/example_images_static/${img.local_filename}`
        : img.image_url || "";

    card.innerHTML = `
        <div class="community-card-image-wrap">
            <img class="community-card-image" src="${escapeHtml(imgUrl)}"
                 alt="Community creation" loading="lazy"
                 onerror="this.style.display='none'">
            ${img.has_workflow ? '<span class="community-workflow-badge" title="ComfyUI workflow available"><i class="fas fa-project-diagram"></i> Workflow</span>' : ""}
        </div>
        <div class="community-card-body">
            <div class="community-card-prompt">${escapeHtml(img.prompt || "")}</div>
            <div class="community-card-meta">
                ${img.sampler ? `<span class="community-meta-tag">${escapeHtml(img.sampler)}</span>` : ""}
                ${img.steps ? `<span class="community-meta-tag">${img.steps} steps</span>` : ""}
                ${img.cfg_scale ? `<span class="community-meta-tag">CFG ${img.cfg_scale}</span>` : ""}
                ${img.base_model ? `<span class="community-meta-tag">${escapeHtml(img.base_model)}</span>` : ""}
            </div>
            ${renderResourceTags(img.resources)}
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
function showDetail(img, sha256, modelName) {
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
                <div class="community-detail-lora-link">
                    <a href="/loras?search=${encodeURIComponent(modelName || '')}" title="View LoRA details">
                        <i class="fas fa-puzzle-piece"></i> ${escapeHtml(modelName || "View LoRA")}
                    </a>
                </div>
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
                ${renderResourceTags(img.resources)}
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
                <div class="community-detail-actions" style="margin-top:12px;">
                    ${img.civitai_image_id ? `
                    <a class="workflow-btn civitai-link" href="https://civitai.com/images/${img.civitai_image_id}" target="_blank" rel="noopener" title="View on CivitAI">
                        <i class="fas fa-external-link-alt"></i> View on CivitAI
                    </a>` : ""}
                    ${img.has_workflow ? `
                    <button class="workflow-btn workflow-download-btn" data-image-id="${img.civitai_image_id}" title="Download ComfyUI workflow">
                        <i class="fas fa-project-diagram"></i> Download Workflow
                    </button>` : ""}
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

    // Workflow download handler
    const workflowBtn = overlay.querySelector(".workflow-download-btn");
    if (workflowBtn) {
        workflowBtn.addEventListener("click", async (e) => {
            e.stopPropagation();
            const imageId = workflowBtn.dataset.imageId;
            try {
                const resp = await fetch(`/api/lm/community-images/workflow/${imageId}`);
                const data = await resp.json();
                if (data.success && data.data) {
                    // Download the workflow portion as JSON file
                    const workflow = data.data.workflow || data.data;
                    const blob = new Blob([JSON.stringify(workflow, null, 2)], { type: "application/json" });
                    const url = URL.createObjectURL(blob);
                    const a = document.createElement("a");
                    a.href = url;
                    a.download = `workflow_${imageId}.json`;
                    a.click();
                    URL.revokeObjectURL(url);
                } else {
                    alert("No workflow found for this image.");
                }
            } catch (err) {
                console.error("[Community] Failed to fetch workflow:", err);
                alert("Failed to download workflow.");
            }
        });
    }

    const escHandler = (e) => {
        if (e.key === "Escape") removeOverlay();
    };
    document.addEventListener("keydown", escHandler);
}

// -- Fetch buttons --------------------------------------------------------
function setupFetchButton() {
    const fetchBtn = document.getElementById("fetchCommunityBtn");
    const refetchBtn = document.getElementById("refetchCommunityBtn");
    const dropdownToggle = document.getElementById("fetchDropdownToggle");
    const dropdownMenu = document.getElementById("fetchDropdownMenu");

    if (fetchBtn) {
        fetchBtn.addEventListener("click", () => doFetch(fetchBtn, false));
    }

    // Dropdown toggle
    if (dropdownToggle && dropdownMenu) {
        dropdownToggle.addEventListener("click", (e) => {
            e.stopPropagation();
            const open = dropdownMenu.style.display !== "none";
            dropdownMenu.style.display = open ? "none" : "";
        });
        // Close on outside click
        document.addEventListener("click", () => {
            dropdownMenu.style.display = "none";
        });
        dropdownMenu.addEventListener("click", (e) => e.stopPropagation());
    }

    if (refetchBtn) {
        refetchBtn.addEventListener("click", () => {
            if (dropdownMenu) dropdownMenu.style.display = "none";
            if (!confirm("Re-fetch all community images? This will re-download and convert all images to WebP.")) return;
            doFetch(fetchBtn || refetchBtn, true);
        });
    }

    const cancelBtn = document.getElementById("cancelFetchBtn");
    if (cancelBtn) {
        cancelBtn.style.display = "none";
        cancelBtn.addEventListener("click", async () => {
            cancelBtn.disabled = true;
            cancelBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Stopping...';
            try {
                await fetch("/api/lm/community-images/cancel", { method: "POST" });
            } catch (err) {
                console.error("[Community] Cancel failed:", err);
            }
        });
    }
}

let _isFetching = false;

async function doFetch(btn, force) {
    if (_isFetching) return;

    const defaultIcon = "fa-images";
    const defaultLabel = "Fetch Community Images";

    _isFetching = true;
    btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> <span>Fetching...</span>';

    // Show cancel button
    const cancelBtn = document.getElementById("cancelFetchBtn");
    if (cancelBtn) cancelBtn.style.display = "";

    let ws = null;
    try {
        const proto = location.protocol === "https:" ? "wss:" : "ws:";
        ws = new WebSocket(`${proto}//${location.host}/ws/fetch-progress`);
        ws.onmessage = (e) => {
            try {
                const msg = JSON.parse(e.data);
                if (msg.type === "community_images_progress") {
                    btn.innerHTML = `<i class="fas fa-spinner fa-spin"></i> <span>${msg.current}/${msg.total}</span>`;
                } else if (msg.type === "community_images_warning") {
                    showWarning(msg.message);
                }
            } catch {}
        };
    } catch {}

    try {
        const fetchOpts = { method: "POST" };
        if (force) {
            fetchOpts.headers = { "Content-Type": "application/json" };
            fetchOpts.body = JSON.stringify({ force: true });
        }
        const resp = await fetch("/api/lm/community-images/fetch", fetchOpts);
        const data = await resp.json();
        if (ws) ws.close();
        if (data.success) {
            const count = data.stored || 0;
            const label = data.cancelled
                ? `Stopped — ${count} images saved`
                : `${count} images saved`;
            const icon = data.cancelled ? "fa-stop" : "fa-check";
            btn.innerHTML = `<i class="fas ${icon}"></i> <span>${label}</span>`;
            setTimeout(() => {
                btn.innerHTML = `<i class="fas ${defaultIcon}"></i> <span>${defaultLabel}</span>`;
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
            btn.innerHTML = `<i class="fas ${defaultIcon}"></i> <span>${defaultLabel}</span>`;
            btn.title = "";
        }, 5000);
    } finally {
        _isFetching = false;
        if (cancelBtn) cancelBtn.style.display = "none";
    }
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

// -- Page size select -----------------------------------------------------
function setupPageSizeSelect() {
    const select = document.getElementById("communityPageSizeSelect");
    if (!select) return;
    select.addEventListener("change", () => {
        _pageSize = parseInt(select.value, 10) || 10;
        loadPage(1);
    });
}

// -- Resource tags --------------------------------------------------------
function renderResourceTags(resources) {
    if (!resources || !resources.length) return "";
    const tags = resources.map(r => {
        const icon = r.type === "lora" ? "fa-puzzle-piece" : r.type === "checkpoint" ? "fa-cube" : "fa-box";
        const label = r.name || (r.type || "model");
        const weight = r.weight != null && r.type === "lora" ? ` (${escapeHtml(String(r.weight))})` : "";
        const title = escapeHtml(r.type || "");
        const content = `<i class="fas ${icon}"></i> ${escapeHtml(label)}${weight}`;
        if (r.modelId) {
            const url = `https://civitai.com/models/${r.modelId}`;
            return `<a class="community-resource-tag" href="${url}" target="_blank" rel="noopener" title="${title}" onclick="event.stopPropagation()">${content}</a>`;
        }
        return `<span class="community-resource-tag" title="${title}">${content}</span>`;
    });
    return `<div class="community-card-resources">${tags.join("")}</div>`;
}

// -- Helpers --------------------------------------------------------------
function showWarning(message) {
    // Show a dismissible warning banner above the grid
    let banner = document.getElementById("communityWarning");
    if (!banner) {
        banner = document.createElement("div");
        banner.id = "communityWarning";
        banner.className = "community-warning";
        const grid = document.getElementById("communityGrid");
        if (grid) grid.parentNode.insertBefore(banner, grid);
    }
    banner.innerHTML = `<i class="fas fa-exclamation-triangle"></i> ${escapeHtml(message)}
        <button class="warning-dismiss" title="Dismiss"><i class="fas fa-times"></i></button>`;
    banner.style.display = "";
    banner.querySelector(".warning-dismiss").addEventListener("click", () => {
        banner.style.display = "none";
    });
    // Auto-dismiss after 10s
    setTimeout(() => { banner.style.display = "none"; }, 10000);
}

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
