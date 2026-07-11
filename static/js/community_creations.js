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
let _searchQuery = "";
let _modelManagerModels = [];
let _modelManagerSelected = new Set();
let _modelManagerSearchQuery = "";
let _modelManagerLastFocus = null;
let _modelManagerLoading = false;
let _modelManagerLoadPromise = null;
let _modelManagerRefreshing = false;
let _modelManagerAutoRefreshStarted = false;
let _pageLoadRequestId = 0;
const _modelManagerVisibilityOverrides = new Map();
const _modelManagerVisibilityPending = new Set();
let _modelManagerVisibilityQueue = Promise.resolve();
const MODEL_MANAGER_RENDER_LIMIT = 300;
const MODEL_MANAGER_SELECTION_LIMIT = 5000;

// -- Lazy loading via IntersectionObserver --------------------------------
let _lazyObserver = null;

function initLazyObserver() {
    if (_lazyObserver) return;
    _lazyObserver = new IntersectionObserver((entries) => {
        for (const entry of entries) {
            if (!entry.isIntersecting) continue;
            const el = entry.target;
            const src = el.dataset.src;
            if (src) {
                el.onload = () => { el.style.opacity = "1"; };
                el.src = src;
                delete el.dataset.src;
            }
            _lazyObserver.unobserve(el);
        }
    }, { rootMargin: "200px" });  // start loading 200px before visible
}

function observeLazy(el) {
    if (_lazyObserver) _lazyObserver.observe(el);
}

// -- Init -----------------------------------------------------------------
async function init() {
    initLazyObserver();
    setupFetchButton();
    setupSearch();
    setupSortSelect();
    setupPageSizeSelect();
    setupModelManager();
    await loadPage(1);
}

// -- Load a single page of models -----------------------------------------
async function loadPage(page) {
    const requestId = ++_pageLoadRequestId;
    _currentPage = page;
    const grid = document.getElementById("communityGrid");
    if (grid) grid.innerHTML = '<div class="community-loading"><i class="fas fa-spinner fa-spin"></i> Loading...</div>';

    try {
        let url = `/api/lm/community-images/by-models?page=${page}&page_size=${_pageSize}&sort=${encodeURIComponent(_sortKey)}`;
        if (_baseModelFilter) {
            url += `&base_model=${encodeURIComponent(_baseModelFilter)}`;
        }
        if (_searchQuery) {
            url += `&search=${encodeURIComponent(_searchQuery)}`;
        }

        const resp = await fetch(url);
        const data = await resp.json();
        if (requestId !== _pageLoadRequestId) return;

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
            if (page > 1 && data.total_pages > 0 && page > data.total_pages) {
                await loadPage(data.total_pages);
                return;
            }
            showEmpty();
            renderPagination(0, 0);
            return;
        }

        _totalPages = data.total_pages;
        renderGrid(data.models);
        renderPagination(data.page, data.total_pages);
        window.scrollTo({ top: 0, behavior: "smooth" });
    } catch (err) {
        if (requestId !== _pageLoadRequestId) return;
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
            <span class="lora-link">${model.image_count} image${model.image_count !== 1 ? "s" : ""}</span>
            <button class="community-refresh-btn" data-sha256="${escapeHtml(model.sha256)}" title="Re-fetch community images for this LoRA">
                <i class="fas fa-sync-alt"></i>
            </button>
            <button class="community-hide-btn" data-sha256="${escapeHtml(model.sha256)}"
                    title="Hide this LoRA from the Community page"
                    aria-label="Hide this LoRA from the Community page">
                <i class="fas fa-eye-slash"></i>
            </button>`;
        section.appendChild(header);

        // Community-only hide handler. Hidden models remain available in the
        // model manager so this action is always reversible.
        const hideBtn = header.querySelector(".community-hide-btn");
        hideBtn.setAttribute("aria-label", `Hide ${model.model_name} from the Community page`);
        hideBtn.addEventListener("click", async (e) => {
            e.stopPropagation();
            const pageHideButtons = [...document.querySelectorAll(".community-hide-btn")];
            const buttonIndex = pageHideButtons.indexOf(hideBtn);
            const siblingButton = pageHideButtons[buttonIndex + 1]
                || pageHideButtons[buttonIndex - 1]
                || null;
            const siblingHash = siblingButton?.dataset.sha256 || "";
            hideBtn.disabled = true;
            hideBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i>';
            try {
                await queueModelVisibilityUpdate(model.sha256, true);
                _modelManagerVisibilityOverrides.set(
                    String(model.sha256).toLowerCase(),
                    true
                );
                await loadPage(_currentPage);
                const replacementButton = [...document.querySelectorAll(
                    ".community-hide-btn"
                )].find((button) => button.dataset.sha256 === siblingHash);
                const focusTarget = replacementButton
                    || document.getElementById("fetchCommunityBtn");
                focusTarget?.focus();
            } catch (err) {
                console.error("[Community] Failed to hide model:", err);
                showInlineHeaderError(hideBtn, err.message || "Failed to hide model");
                hideBtn.innerHTML = '<i class="fas fa-eye-slash"></i>';
                hideBtn.disabled = false;
            }
        });

        // Refresh button handler
        const refreshBtn = header.querySelector(".community-refresh-btn");
        refreshBtn.addEventListener("click", async (e) => {
            e.stopPropagation();
            refreshBtn.disabled = true;
            refreshBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i>';
            try {
                const resp = await fetch("/api/lm/community-images/refresh-model", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ sha256: model.sha256 }),
                });
                const data = await resp.json();
                if (data.success) {
                    // Re-render the cards in this group with updated data
                    const cardsContainer = section.querySelector(".community-cards");
                    if (cardsContainer && data.images) {
                        cardsContainer.innerHTML = "";
                        const sorted = [...data.images].sort((a, b) => {
                            const ra = (a.like_count || 0) + (a.heart_count || 0);
                            const rb = (b.like_count || 0) + (b.heart_count || 0);
                            return rb - ra;
                        });
                        for (const img of sorted) {
                            cardsContainer.appendChild(createCard(img, model.sha256, model.model_name));
                        }
                        // Observe lazy images in refreshed cards
                        cardsContainer.querySelectorAll("img[data-src]").forEach(observeLazy);
                    }
                    refreshBtn.innerHTML = '<i class="fas fa-check"></i>';
                    setTimeout(() => {
                        refreshBtn.innerHTML = '<i class="fas fa-sync-alt"></i>';
                        refreshBtn.disabled = false;
                    }, 2000);
                } else {
                    throw new Error(data.error || "Refresh failed");
                }
            } catch (err) {
                console.error("[Community] Refresh model failed:", err);
                const msg = err.message || String(err);
                refreshBtn.innerHTML = '<i class="fas fa-exclamation-triangle"></i>';
                refreshBtn.title = msg;
                // Also show a brief inline message next to the button
                const errSpan = document.createElement("span");
                errSpan.className = "community-refresh-error";
                errSpan.textContent = msg;
                refreshBtn.parentNode.insertBefore(errSpan, refreshBtn.nextSibling);
                setTimeout(() => {
                    refreshBtn.innerHTML = '<i class="fas fa-sync-alt"></i>';
                    refreshBtn.title = "Re-fetch community images for this LoRA";
                    refreshBtn.disabled = false;
                    errSpan.remove();
                }, 5000);
            }
        });

        // Sort images within group by reactions
        const sorted = [...model.images].sort((a, b) => {
            const ra = (a.like_count || 0) + (a.heart_count || 0);
            const rb = (b.like_count || 0) + (b.heart_count || 0);
            return rb - ra;
        });

        // Cards — show all (lazy loading handles performance)
        const cardsDiv = document.createElement("div");
        cardsDiv.className = "community-cards";
        for (const img of sorted) {
            cardsDiv.appendChild(createCard(img, model.sha256, model.model_name));
        }
        section.appendChild(cardsDiv);
        grid.appendChild(section);

        // Observe all lazy images (IntersectionObserver only loads visible ones)
        section.querySelectorAll("img[data-src]").forEach(observeLazy);
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

    // Use thumbnail for grid cards (smaller, faster), full image for detail modal
    const thumbUrl = img.thumbnail_url || img.preview_url || img.image_url || "";
    const mediaUrl = img.preview_url || img.image_url || "";
    const isVideo = img.media_type === "video";

    card.innerHTML = `
        <button class="community-card-detail-trigger" type="button" aria-haspopup="dialog"></button>
        <div class="community-card-image-wrap">
            ${isVideo
                ? `<video class="community-card-image" src="${escapeHtml(mediaUrl)}#t=1" muted playsinline preload="metadata"
                    onerror="this.outerHTML='<div class=\\'community-card-placeholder\\'>Video unavailable</div>'"></video>
                   <span class="community-video-badge" title="Video"><i class="fas fa-play"></i></span>`
                : `<img class="community-card-image" data-src="${escapeHtml(thumbUrl)}" alt="Community creation"
                    onerror="this.outerHTML='<div class=\\'community-card-placeholder\\'>Image unavailable</div>'">`
            }
            ${img.has_workflow ? '<span class="community-workflow-badge" title="ComfyUI workflow available"><i class="fas fa-project-diagram"></i> Workflow</span>' : ""}
        </div>
        <div class="community-card-body">
            <div class="community-card-prompt">${escapeHtml(img.prompt || "")}</div>
            <div class="community-card-meta">
                ${img.sampler ? `<span class="community-meta-tag">${escapeHtml(img.sampler)}</span>` : ""}
                ${img.steps ? `<span class="community-meta-tag">${escapeHtml(String(img.steps))} steps</span>` : ""}
                ${img.cfg_scale ? `<span class="community-meta-tag">CFG ${escapeHtml(String(img.cfg_scale))}</span>` : ""}
                ${img.base_model ? `<span class="community-meta-tag">${escapeHtml(img.base_model)}</span>` : ""}
            </div>
            ${renderResourceTags(img.resources)}
            <div class="community-card-footer">
                <div class="community-card-reactions">
                    ${img.like_count ? `<span class="community-reaction"><i class="fas fa-thumbs-up"></i> ${escapeHtml(String(img.like_count))}</span>` : ""}
                    ${img.heart_count ? `<span class="community-reaction"><i class="fas fa-heart"></i> ${escapeHtml(String(img.heart_count))}</span>` : ""}
                    ${img.comment_count ? `<span class="community-reaction"><i class="fas fa-comment"></i> ${escapeHtml(String(img.comment_count))}</span>` : ""}
                </div>
                <span class="community-card-user">${escapeHtml(img.username || "")}</span>
            </div>
        </div>
    `;

    const detailTrigger = card.querySelector(".community-card-detail-trigger");
    detailTrigger.setAttribute(
        "aria-label",
        `View community creation${modelName ? ` for ${modelName}` : ""}${img.username ? ` by ${img.username}` : ""}`
    );
    detailTrigger.addEventListener("click", () => {
        showDetail(img, sha256, modelName, detailTrigger);
    });

    // Hover-to-play for video cards
    if (isVideo) {
        const video = card.querySelector("video");
        if (video) {
            card.addEventListener("mouseenter", () => video.play().catch(() => {}));
            card.addEventListener("mouseleave", () => { video.pause(); video.currentTime = 0; });
        }
    }

    return card;
}

// -- Detail modal ---------------------------------------------------------
function showDetail(img, sha256, modelName, trigger = null) {
    const existing = document.querySelector(".community-detail-overlay");
    if (existing) {
        if (typeof existing._closeCommunityDetail === "function") {
            existing._closeCommunityDetail(false);
        } else {
            existing.remove();
        }
    }

    const mediaUrl = img.preview_url || img.image_url || "";
    const isVideo = img.media_type === "video";
    const previousFocus = trigger || document.activeElement;

    const overlay = document.createElement("div");
    overlay.className = "community-detail-overlay";
    const removeOverlay = (restoreFocus = true) => {
        if (!overlay.isConnected) return;
        overlay.remove();
        document.body.classList.remove("community-detail-open");
        setCommunityPageInert(false);
        if (
            restoreFocus
            && previousFocus
            && previousFocus.isConnected
            && typeof previousFocus.focus === "function"
        ) {
            previousFocus.focus();
        }
    };
    overlay._closeCommunityDetail = removeOverlay;
    overlay.addEventListener("click", (e) => {
        if (e.target === overlay) removeOverlay();
    });

    overlay.innerHTML = `
        <section class="community-detail" role="dialog" aria-modal="true" tabindex="-1">
            <button class="community-detail-close" type="button" aria-label="Close community creation details">
                <i class="fas fa-times" aria-hidden="true"></i>
            </button>
            ${isVideo
                ? `<video class="community-detail-image" src="${escapeHtml(mediaUrl)}#t=0.001" controls loop playsinline></video>`
                : `<img class="community-detail-image" src="${escapeHtml(mediaUrl)}" alt="Community creation">`
            }
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
                    ${img.steps ? `<div class="community-detail-param"><strong>Steps:</strong> ${escapeHtml(String(img.steps))}</div>` : ""}
                    ${img.sampler ? `<div class="community-detail-param"><strong>Sampler:</strong> ${escapeHtml(img.sampler)}</div>` : ""}
                    ${img.cfg_scale ? `<div class="community-detail-param"><strong>CFG Scale:</strong> ${escapeHtml(String(img.cfg_scale))}</div>` : ""}
                    ${img.seed != null ? `<div class="community-detail-param"><strong>Seed:</strong> ${escapeHtml(String(img.seed))}</div>` : ""}
                    ${img.denoise ? `<div class="community-detail-param"><strong>Denoise:</strong> ${escapeHtml(String(img.denoise))}</div>` : ""}
                    ${img.base_model ? `<div class="community-detail-param"><strong>Base Model:</strong> ${escapeHtml(img.base_model)}</div>` : ""}
                    ${img.width && img.height ? `<div class="community-detail-param"><strong>Size:</strong> ${escapeHtml(String(img.width))}x${escapeHtml(String(img.height))}</div>` : ""}
                </div>
                ${renderResourceTags(img.resources)}
                <div class="community-card-reactions" style="margin-top:12px;">
                    ${img.like_count ? `<span class="community-reaction"><i class="fas fa-thumbs-up"></i> ${escapeHtml(String(img.like_count))}</span>` : ""}
                    ${img.heart_count ? `<span class="community-reaction"><i class="fas fa-heart"></i> ${escapeHtml(String(img.heart_count))}</span>` : ""}
                    ${img.laugh_count ? `<span class="community-reaction"><i class="fas fa-laugh"></i> ${escapeHtml(String(img.laugh_count))}</span>` : ""}
                    ${img.comment_count ? `<span class="community-reaction"><i class="fas fa-comment"></i> ${escapeHtml(String(img.comment_count))}</span>` : ""}
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
        </section>
    `;

    document.body.appendChild(overlay);
    const dialog = overlay.querySelector(".community-detail");
    dialog?.setAttribute(
        "aria-label",
        `Community creation details${modelName ? ` for ${modelName}` : ""}`
    );
    document.body.classList.add("community-detail-open");
    setCommunityPageInert(true);

    const closeBtn = overlay.querySelector(".community-detail-close");
    closeBtn?.addEventListener("click", () => removeOverlay());
    closeBtn?.focus();

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
                const resp = await fetch(
                    `/api/lm/community-images/workflow/${imageId}?sha256=${encodeURIComponent(sha256)}`
                );
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

    overlay.addEventListener("keydown", (e) => {
        if (e.key === "Escape") {
            e.preventDefault();
            removeOverlay();
            return;
        }
        if (e.key !== "Tab") return;

        const focusable = [...overlay.querySelectorAll(
            'button:not([disabled]), input:not([disabled]), [href], [tabindex]:not([tabindex="-1"])'
        )].filter((element) => !element.hidden);
        if (!focusable.length) {
            e.preventDefault();
            dialog?.focus();
            return;
        }
        const first = focusable[0];
        const last = focusable[focusable.length - 1];
        if (e.shiftKey && document.activeElement === first) {
            e.preventDefault();
            last.focus();
        } else if (!e.shiftKey && document.activeElement === last) {
            e.preventDefault();
            first.focus();
        }
    });
}

// -- Fetch buttons --------------------------------------------------------
function setupFetchButton() {
    const fetchBtn = document.getElementById("fetchCommunityBtn");
    const fetchMissingBtn = document.getElementById("fetchMissingCommunityBtn");
    const refetchBtn = document.getElementById("refetchCommunityBtn");
    const dropdownToggle = document.getElementById("fetchDropdownToggle");
    const dropdownMenu = document.getElementById("fetchDropdownMenu");

    if (fetchBtn) {
        fetchBtn.addEventListener("click", () => openModelManager());
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

    if (fetchMissingBtn) {
        fetchMissingBtn.addEventListener("click", () => {
            if (dropdownMenu) dropdownMenu.style.display = "none";
            if (_isFetching) return;
            // The backend reconciles the scanner, excludes hidden models, and
            // tops up every model with fewer than ten stored images.
            doFetch(fetchBtn || fetchMissingBtn, false);
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

async function doFetch(btn, force, hashes = null) {
    if (_isFetching) {
        return { success: false, error: "A community fetch is already running." };
    }
    const defaultIcon = "fa-images";
    const defaultLabel = "Fetch Community Images";
    const result = { success: false, error: "" };

    _isFetching = true;
    btn.disabled = true;
    btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> <span>Fetching...</span>';

    // Show cancel button
    const cancelBtn = document.getElementById("cancelFetchBtn");
    if (cancelBtn) {
        cancelBtn.disabled = false;
        cancelBtn.innerHTML = '<i class="fas fa-stop"></i> Stop';
        cancelBtn.style.display = "";
    }

    let ws = null;
    try {
        const proto = location.protocol === "https:" ? "wss:" : "ws:";
        ws = new WebSocket(`${proto}//${location.host}/ws/fetch-progress`);
        ws.onmessage = (e) => {
            try {
                const msg = JSON.parse(e.data);
                if (msg.type === "community_images_progress") {
                    const storedInfo = msg.stored ? ` (${msg.stored} img)` : "";
                    btn.innerHTML = `<i class="fas fa-spinner fa-spin"></i> <span>${msg.current}/${msg.total}${storedInfo}</span>`;
                } else if (msg.type === "community_images_warning") {
                    showWarning(msg.message);
                }
            } catch {}
        };
    } catch {}

    try {
        const fetchOpts = { method: "POST" };
        if (force || Array.isArray(hashes)) {
            const body = {};
            if (force) body.force = true;
            if (Array.isArray(hashes)) body.hashes = hashes;
            fetchOpts.headers = { "Content-Type": "application/json" };
            fetchOpts.body = JSON.stringify(body);
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
            result.success = true;
        } else {
            throw new Error(data.error || "Unknown error");
        }
    } catch (err) {
        if (ws) ws.close();
        btn.innerHTML = '<i class="fas fa-exclamation-triangle"></i> <span>Error</span>';
        btn.title = err.message || String(err);
        result.error = err.message || String(err);
        showWarning(`Community fetch failed: ${result.error}`);
        console.error("[Community] Fetch failed:", err);
        setTimeout(() => {
            btn.innerHTML = `<i class="fas ${defaultIcon}"></i> <span>${defaultLabel}</span>`;
            btn.title = "";
        }, 5000);
    } finally {
        _isFetching = false;
        btn.disabled = false;
        if (cancelBtn) cancelBtn.style.display = "none";
    }
    return result;
}

// -- Model manager --------------------------------------------------------
function setupModelManager() {
    const overlay = document.getElementById("communityModelManager");
    const list = document.getElementById("communityModelManagerList");
    const search = document.getElementById("communityModelManagerSearch");
    if (!overlay || !list || !search) return;

    // The page container creates a low z-index stacking context. Mount the
    // fixed dialog at the document root so it also covers the fixed header.
    if (overlay.parentElement !== document.body) {
        document.body.appendChild(overlay);
    }

    document.getElementById("communityModelManagerClose")?.addEventListener("click", closeModelManager);
    document.getElementById("communityModelManagerCancel")?.addEventListener("click", closeModelManager);

    overlay.addEventListener("click", (event) => {
        if (event.target === overlay) closeModelManager();
    });
    overlay.addEventListener("keydown", handleModelManagerKeydown);

    search.addEventListener("input", () => {
        _modelManagerSearchQuery = search.value.trim().toLowerCase();
        renderModelManagerModels();
    });

    document.getElementById("communitySelectNewBtn")?.addEventListener("click", () => {
        selectModelManagerModels((model) => (
            !model.hidden && model.fetchable && model.image_count === 0
        ));
    });

    document.getElementById("communitySelectVisibleBtn")?.addEventListener("click", () => {
        selectModelManagerModels((model) => !model.hidden && model.fetchable);
    });

    document.getElementById("communityClearSelectionBtn")?.addEventListener("click", () => {
        _modelManagerSelected.clear();
        renderModelManagerModels();
    });

    document.getElementById("communityRefreshModelsBtn")?.addEventListener("click", () => {
        refreshModelManagerModels();
    });

    document.getElementById("communityFetchSelectedBtn")?.addEventListener("click", async () => {
        const allowed = new Set(
            _modelManagerModels
                .filter((model) => model.fetchable && !model.hidden)
                .map((model) => model.sha256)
        );
        const hashes = [..._modelManagerSelected].filter((hash) => allowed.has(hash));
        if (!hashes.length) {
            setModelManagerStatus("Select at least one available LoRA to fetch.", "error");
            return;
        }
        if (hashes.length > MODEL_MANAGER_SELECTION_LIMIT) {
            setModelManagerStatus(
                `Select no more than ${MODEL_MANAGER_SELECTION_LIMIT} LoRAs at once.`,
                "error"
            );
            return;
        }

        closeModelManager();
        const fetchBtn = document.getElementById("fetchCommunityBtn");
        const result = await doFetch(
            fetchBtn || document.getElementById("communityFetchSelectedBtn"),
            true,
            hashes
        );
        if (!result.success) {
            reopenModelManagerAfterFetchError(result.error);
        }
    });

    list.addEventListener("change", (event) => {
        const checkbox = event.target.closest(".community-manager-checkbox");
        if (!checkbox) return;
        if (checkbox.checked) {
            if (_modelManagerSelected.size >= MODEL_MANAGER_SELECTION_LIMIT) {
                checkbox.checked = false;
                setModelManagerStatus(
                    `Selection is limited to ${MODEL_MANAGER_SELECTION_LIMIT} LoRAs.`,
                    "error"
                );
                return;
            }
            _modelManagerSelected.add(checkbox.value);
        } else {
            _modelManagerSelected.delete(checkbox.value);
        }
        updateModelManagerSelectionUI();
    });

    list.addEventListener("click", (event) => {
        const button = event.target.closest(".community-manager-visibility");
        if (button) handleModelManagerVisibility(button);
    });
}

function selectModelManagerModels(predicate) {
    let truncated = false;
    for (const model of getFilteredModelManagerModels()) {
        if (!predicate(model) || _modelManagerSelected.has(model.sha256)) continue;
        if (_modelManagerSelected.size >= MODEL_MANAGER_SELECTION_LIMIT) {
            truncated = true;
            break;
        }
        _modelManagerSelected.add(model.sha256);
    }
    renderModelManagerModels();
    if (truncated) {
        setModelManagerStatus(
            `Selected the first ${MODEL_MANAGER_SELECTION_LIMIT} matching LoRAs. Narrow the search to choose a different set.`,
            "error"
        );
    }
}

async function openModelManager() {
    if (_isFetching) return;
    const overlay = document.getElementById("communityModelManager");
    const search = document.getElementById("communityModelManagerSearch");
    if (!overlay || !search) return;

    _modelManagerLastFocus = document.activeElement;
    _modelManagerSelected.clear();
    _modelManagerSearchQuery = "";
    search.value = "";
    overlay.hidden = false;
    overlay.setAttribute("aria-hidden", "false");
    document.body.classList.add("community-model-manager-open");
    setCommunityPageInert(true);
    search.focus();

    await loadModelManagerModels({
        refresh: false,
        preserveSelection: false,
    });
    if (!_modelManagerAutoRefreshStarted) {
        _modelManagerAutoRefreshStarted = true;
        refreshModelManagerModels();
    }
}

function closeModelManager() {
    const overlay = document.getElementById("communityModelManager");
    if (!overlay || overlay.hidden) return;
    overlay.hidden = true;
    overlay.setAttribute("aria-hidden", "true");
    document.body.classList.remove("community-model-manager-open");
    setCommunityPageInert(false);
    if (_modelManagerLastFocus && typeof _modelManagerLastFocus.focus === "function") {
        _modelManagerLastFocus.focus();
    }
    _modelManagerLastFocus = null;
}

function reopenModelManagerAfterFetchError(message) {
    const overlay = document.getElementById("communityModelManager");
    const search = document.getElementById("communityModelManagerSearch");
    if (!overlay || !search) return;
    _modelManagerLastFocus = document.getElementById("fetchCommunityBtn");
    overlay.hidden = false;
    overlay.setAttribute("aria-hidden", "false");
    document.body.classList.add("community-model-manager-open");
    setCommunityPageInert(true);
    renderModelManagerModels();
    setModelManagerStatus(message || "Community fetch failed.", "error");
    search.focus();
}

function setCommunityPageInert(inert) {
    document.querySelectorAll(".page-content, .app-header").forEach((element) => {
        element.toggleAttribute("inert", inert);
    });
}

function handleModelManagerKeydown(event) {
    if (event.key === "Escape") {
        event.preventDefault();
        closeModelManager();
        return;
    }
    if (event.key !== "Tab") return;

    const overlay = document.getElementById("communityModelManager");
    if (!overlay) return;
    const focusable = [...overlay.querySelectorAll(
        'button:not([disabled]), input:not([disabled]), [href], [tabindex]:not([tabindex="-1"])'
    )].filter((element) => !element.hidden);
    if (!focusable.length) return;

    const first = focusable[0];
    const last = focusable[focusable.length - 1];
    if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
    } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
    }
}

async function fetchModelInventory(refresh = false) {
    const suffix = refresh ? "?refresh=true" : "";
    const response = await fetch(`/api/lm/community-images/models${suffix}`);
    let data = null;
    try {
        data = await response.json();
    } catch {
        throw new Error("The model inventory returned an invalid response.");
    }
    if (!response.ok || !data.success) {
        throw new Error(data.error || "Failed to load the Community model inventory.");
    }

    return (Array.isArray(data.models) ? data.models : [])
        .filter((model) => model && model.sha256)
        .map((model) => ({
            sha256: String(model.sha256),
            model_name: String(model.model_name || model.sha256).trim(),
            base_model: String(model.base_model || "").trim(),
            image_count: Math.max(0, Number.parseInt(model.image_count, 10) || 0),
            hidden: Boolean(model.hidden),
            fetchable: Boolean(model.fetchable),
            unavailable_reason: String(model.unavailable_reason || "").trim(),
        }));
}

function loadModelManagerModels(options = {}) {
    if (_modelManagerLoadPromise) return _modelManagerLoadPromise;

    const request = performModelManagerLoad(options);
    _modelManagerLoadPromise = request;
    request.finally(() => {
        if (_modelManagerLoadPromise === request) {
            _modelManagerLoadPromise = null;
        }
    });
    return request;
}

async function performModelManagerLoad({
    refresh = false,
    preserveSelection = true,
} = {}) {
    const list = document.getElementById("communityModelManagerList");
    const refreshBtn = document.getElementById("communityRefreshModelsBtn");
    if (!list) return;

    _modelManagerLoading = true;
    if (refreshBtn) refreshBtn.disabled = true;
    list.innerHTML = '<div class="community-manager-loading"><i class="fas fa-spinner fa-spin" aria-hidden="true"></i> Loading models...</div>';
    setModelManagerStatus(refresh ? "Refreshing the installed LoRA list..." : "Loading models...");

    try {
        const models = await fetchModelInventory(refresh);
        applyModelManagerInventory(models, preserveSelection);
        renderModelManagerModels();
        setModelManagerInventoryStatus();
    } catch (err) {
        console.error("[Community] Failed to load model manager:", err);
        _modelManagerModels = [];
        _modelManagerSelected.clear();
        list.innerHTML = `<div class="community-manager-empty"><i class="fas fa-exclamation-triangle" aria-hidden="true"></i><p>${escapeHtml(err.message || "Failed to load models.")}</p></div>`;
        setModelManagerStatus(err.message || "Failed to load models.", "error");
        updateModelManagerSelectionUI();
    } finally {
        _modelManagerLoading = false;
        if (refreshBtn) refreshBtn.disabled = false;
        updateModelManagerSelectionUI();
    }
}

function applyModelManagerInventory(models, preserveSelection = true) {
    const previousSelection = preserveSelection ? _modelManagerSelected : new Set();
    for (const model of models) {
        const key = String(model.sha256).toLowerCase();
        if (!_modelManagerVisibilityOverrides.has(key)) continue;
        model.hidden = _modelManagerVisibilityOverrides.get(key);
    }
    _modelManagerModels = models;
    const selectable = new Set(
        _modelManagerModels
            .filter((model) => model.fetchable && !model.hidden)
            .map((model) => model.sha256)
    );
    _modelManagerSelected = new Set(
        [...previousSelection].filter((hash) => selectable.has(hash))
    );
}

function setModelManagerInventoryStatus() {
    const hiddenCount = _modelManagerModels.filter((model) => model.hidden).length;
    const unavailableCount = _modelManagerModels.filter((model) => !model.fetchable).length;
    const details = [
        `${_modelManagerModels.length} LoRA${_modelManagerModels.length === 1 ? "" : "s"}`,
        `${hiddenCount} hidden`,
    ];
    if (unavailableCount) details.push(`${unavailableCount} unavailable`);
    setModelManagerStatus(details.join(" · "));
}

async function refreshModelManagerModels() {
    if (_modelManagerLoading || _modelManagerRefreshing) return;
    const refreshBtn = document.getElementById("communityRefreshModelsBtn");
    _modelManagerRefreshing = true;
    if (refreshBtn) refreshBtn.disabled = true;
    setModelManagerStatus(
        "Scanning for newly added LoRAs in the background. You can keep using the current list."
    );

    try {
        const models = await fetchModelInventory(true);
        const focusState = captureModelManagerFocus();
        applyModelManagerInventory(models, true);
        renderModelManagerModels();
        restoreModelManagerFocus(focusState);
        setModelManagerInventoryStatus();
    } catch (err) {
        console.error("[Community] Failed to refresh model manager:", err);
        setModelManagerStatus(err.message || "Failed to refresh models.", "error");
    } finally {
        _modelManagerRefreshing = false;
        if (refreshBtn) refreshBtn.disabled = false;
    }
}

function captureModelManagerFocus() {
    const active = document.activeElement;
    if (active?.classList.contains("community-manager-checkbox")) {
        return { type: "checkbox", sha256: active.value };
    }
    if (active?.classList.contains("community-manager-visibility")) {
        return { type: "visibility", sha256: active.dataset.sha256 };
    }
    if (active?.id) return { type: "id", id: active.id };
    return null;
}

function restoreModelManagerFocus(focusState) {
    const overlay = document.getElementById("communityModelManager");
    if (!focusState || !overlay || overlay.hidden) return;

    let target = null;
    if (focusState.type === "checkbox") {
        target = [...overlay.querySelectorAll(".community-manager-checkbox")]
            .find((element) => element.value === focusState.sha256);
    } else if (focusState.type === "visibility") {
        target = [...overlay.querySelectorAll(".community-manager-visibility")]
            .find((element) => element.dataset.sha256 === focusState.sha256);
    } else if (focusState.type === "id") {
        const candidate = document.getElementById(focusState.id);
        if (candidate && overlay.contains(candidate)) target = candidate;
    }
    (target || document.getElementById("communityModelManagerSearch"))?.focus();
}

function getFilteredModelManagerModels() {
    if (!_modelManagerSearchQuery) return _modelManagerModels;
    return _modelManagerModels.filter((model) => {
        const haystack = `${model.model_name} ${model.base_model} ${model.unavailable_reason}`.toLowerCase();
        return haystack.includes(_modelManagerSearchQuery);
    });
}

function renderModelManagerModels() {
    const list = document.getElementById("communityModelManagerList");
    if (!list) return;
    const models = getFilteredModelManagerModels();
    list.innerHTML = "";

    if (!models.length) {
        const empty = document.createElement("div");
        empty.className = "community-manager-empty";
        empty.innerHTML = '<i class="fas fa-search" aria-hidden="true"></i><p>No LoRAs match this search.</p>';
        list.appendChild(empty);
        updateModelManagerSelectionUI();
        return;
    }

    const fragment = document.createDocumentFragment();
    models.slice(0, MODEL_MANAGER_RENDER_LIMIT).forEach((model) => {
        const row = document.createElement("div");
        row.className = "community-manager-row";
        if (model.hidden) row.classList.add("is-hidden");
        if (!model.fetchable) row.classList.add("is-unavailable");
        if (model.image_count === 0) row.classList.add("is-new");
        row.setAttribute("role", "listitem");

        const selectLabel = document.createElement("label");
        selectLabel.className = "community-manager-select";
        const checkbox = document.createElement("input");
        checkbox.type = "checkbox";
        checkbox.className = "community-manager-checkbox";
        checkbox.value = model.sha256;
        checkbox.checked = _modelManagerSelected.has(model.sha256);
        checkbox.disabled = model.hidden || !model.fetchable;
        checkbox.setAttribute("aria-label", `Select ${model.model_name}`);
        const checkmark = document.createElement("span");
        checkmark.className = "community-manager-checkmark";
        checkmark.setAttribute("aria-hidden", "true");
        selectLabel.append(checkbox, checkmark);

        const info = document.createElement("div");
        info.className = "community-manager-model-info";
        const titleLine = document.createElement("div");
        titleLine.className = "community-manager-title-line";
        const title = document.createElement("strong");
        title.className = "community-manager-model-name";
        title.textContent = model.model_name;
        titleLine.appendChild(title);
        if (model.base_model) {
            const baseModel = document.createElement("span");
            baseModel.className = "community-manager-base-model";
            baseModel.textContent = model.base_model;
            titleLine.appendChild(baseModel);
        }
        info.appendChild(titleLine);

        const status = document.createElement("div");
        status.className = "community-manager-model-status";
        const count = document.createElement("span");
        count.className = model.image_count === 0 ? "is-new" : "";
        count.textContent = model.image_count === 0
            ? "New · no images fetched"
            : `${model.image_count} community image${model.image_count === 1 ? "" : "s"}`;
        status.appendChild(count);
        if (model.hidden) {
            const hidden = document.createElement("span");
            hidden.className = "is-hidden";
            hidden.textContent = "Hidden from gallery";
            status.appendChild(hidden);
        }
        if (!model.fetchable) {
            const unavailable = document.createElement("span");
            unavailable.className = "is-unavailable";
            unavailable.textContent = model.unavailable_reason || "Community images unavailable";
            status.appendChild(unavailable);
        }
        info.appendChild(status);

        const visibility = document.createElement("button");
        visibility.type = "button";
        visibility.className = "community-manager-visibility";
        visibility.dataset.sha256 = model.sha256;
        visibility.setAttribute(
            "aria-label",
            `${model.hidden ? "Unhide" : "Hide"} ${model.model_name} on the Community page`
        );
        const visibilityPending = _modelManagerVisibilityPending.has(
            String(model.sha256).toLowerCase()
        );
        visibility.disabled = visibilityPending;
        visibility.innerHTML = visibilityPending
            ? '<i class="fas fa-spinner fa-spin" aria-hidden="true"></i><span>Saving...</span>'
            : model.hidden
                ? '<i class="fas fa-eye" aria-hidden="true"></i><span>Unhide</span>'
                : '<i class="fas fa-eye-slash" aria-hidden="true"></i><span>Hide</span>';

        row.append(selectLabel, info, visibility);
        fragment.appendChild(row);
    });
    if (models.length > MODEL_MANAGER_RENDER_LIMIT) {
        const limitNotice = document.createElement("div");
        limitNotice.className = "community-manager-result-limit";
        limitNotice.setAttribute("role", "listitem");
        limitNotice.textContent = (
            `Showing the first ${MODEL_MANAGER_RENDER_LIMIT} of ${models.length} LoRAs. `
            + "Use search to narrow the list."
        );
        fragment.appendChild(limitNotice);
    }
    list.appendChild(fragment);
    updateModelManagerSelectionUI();
}

function updateModelManagerSelectionUI() {
    const count = _modelManagerSelected.size;
    const countElement = document.getElementById("communitySelectedCount");
    const fetchButton = document.getElementById("communityFetchSelectedBtn");
    if (countElement) {
        countElement.textContent = `${count} selected`;
    }
    if (fetchButton) {
        fetchButton.disabled = count === 0 || _modelManagerLoading;
        fetchButton.innerHTML = `<i class="fas fa-images" aria-hidden="true"></i> Fetch selected${count ? ` (${count})` : ""}`;
    }
}

async function handleModelManagerVisibility(button) {
    const model = _modelManagerModels.find((item) => item.sha256 === button.dataset.sha256);
    if (!model) return;
    const sha256 = model.sha256;
    const key = String(sha256).toLowerCase();
    if (_modelManagerVisibilityPending.has(key)) return;
    const hidden = !model.hidden;
    _modelManagerVisibilityPending.add(key);
    button.disabled = true;
    button.innerHTML = '<i class="fas fa-spinner fa-spin" aria-hidden="true"></i><span>Saving...</span>';

    try {
        await queueModelVisibilityUpdate(sha256, hidden);
        _modelManagerVisibilityOverrides.set(key, hidden);
        const currentModel = _modelManagerModels.find(
            (item) => String(item.sha256).toLowerCase() === key
        );
        if (currentModel) currentModel.hidden = hidden;
        if (hidden) _modelManagerSelected.delete(sha256);
        _modelManagerVisibilityPending.delete(key);
        renderModelManagerModels();
        await loadPage(_currentPage);
        const replacementButton = [...document.querySelectorAll(
            ".community-manager-visibility"
        )].find((element) => String(element.dataset.sha256).toLowerCase() === key);
        const focusTarget = replacementButton
            || document.getElementById("communityModelManagerSearch");
        focusTarget?.focus();
        setModelManagerStatus(
            `${model.model_name} is now ${hidden ? "hidden from" : "visible on"} the Community page.`
        );
    } catch (err) {
        _modelManagerVisibilityPending.delete(key);
        console.error("[Community] Failed to update model visibility:", err);
        renderModelManagerModels();
        const replacementButton = [...document.querySelectorAll(
            ".community-manager-visibility"
        )].find((element) => String(element.dataset.sha256).toLowerCase() === key);
        (replacementButton || document.getElementById("communityModelManagerSearch"))?.focus();
        setModelManagerStatus(err.message || "Failed to update model visibility.", "error");
    }
}

function queueModelVisibilityUpdate(sha256, hidden) {
    const request = _modelManagerVisibilityQueue
        .catch(() => {})
        .then(() => updateModelVisibility(sha256, hidden));
    _modelManagerVisibilityQueue = request.catch(() => {});
    return request;
}

async function updateModelVisibility(sha256, hidden) {
    const response = await fetch("/api/lm/community-images/visibility", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sha256, hidden }),
    });
    let data = null;
    try {
        data = await response.json();
    } catch {
        throw new Error("The visibility update returned an invalid response.");
    }
    if (!response.ok || !data.success) {
        throw new Error(data.error || "Failed to update model visibility.");
    }
    return data;
}

function setModelManagerStatus(message, type = "") {
    const status = document.getElementById("communityModelManagerStatus");
    if (!status) return;
    status.className = `community-model-manager-status${type ? ` is-${type}` : ""}`;
    status.textContent = message || "";
}

function showInlineHeaderError(button, message) {
    const existing = button.parentNode?.querySelector(".community-refresh-error");
    if (existing) existing.remove();
    const error = document.createElement("span");
    error.className = "community-refresh-error";
    error.textContent = message;
    button.parentNode?.insertBefore(error, button.nextSibling);
    setTimeout(() => error.remove(), 5000);
}

// -- Search ---------------------------------------------------------------
function setupSearch() {
    const input = document.getElementById("communitySearch");
    if (!input) return;
    let debounce = null;
    input.addEventListener("input", () => {
        clearTimeout(debounce);
        debounce = setTimeout(() => {
            _searchQuery = input.value.trim();
            loadPage(1);
        }, 300);
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
