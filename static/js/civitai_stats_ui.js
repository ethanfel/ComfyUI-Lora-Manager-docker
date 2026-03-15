// static/js/civitai_stats_ui.js
/**
 * CivitAI Stats UI — card badges, sort dropdown options, fetch button.
 *
 * Self-contained script that patches model cards with CivitAI community
 * stats (downloads, rating, likes). Fetches stats from the local
 * /api/lm/civitai-stats/* endpoints.
 */
(function () {
    "use strict";

    // ── Compact number formatting ──────────────────────────────────
    function formatCompact(n) {
        if (n == null) return null;
        if (n >= 1_000_000) return (n / 1_000_000).toFixed(1).replace(/\.0$/, "") + "M";
        if (n >= 1_000) return (n / 1_000).toFixed(1).replace(/\.0$/, "") + "k";
        return String(n);
    }

    // ── Badge creation ─────────────────────────────────────────────
    function createStatBadge(icon, value, title) {
        if (value == null) return null;
        const badge = document.createElement("span");
        badge.className = "lm-stat-badge";
        badge.title = title;
        badge.innerHTML = `<i class="fas fa-${icon}"></i> ${formatCompact(value)}`;
        return badge;
    }

    // ── Inject CSS ─────────────────────────────────────────────────
    function injectStyles() {
        if (document.getElementById("lm-civitai-stats-styles")) return;
        const style = document.createElement("style");
        style.id = "lm-civitai-stats-styles";
        style.textContent = `
            .lm-stat-badges {
                display: flex;
                gap: 6px;
                flex-wrap: wrap;
                padding: 2px 6px;
            }
            .lm-stat-badge {
                display: inline-flex;
                align-items: center;
                gap: 3px;
                font-size: 11px;
                padding: 1px 5px;
                border-radius: 4px;
                background: rgba(255,255,255,0.1);
                color: rgba(255,255,255,0.8);
                white-space: nowrap;
            }
            .lm-stat-badge i {
                font-size: 10px;
                opacity: 0.7;
            }
        `;
        document.head.appendChild(style);
    }

    // ── Stats cache ────────────────────────────────────────────────
    const _statsMap = {};

    // Fetch stats for a batch of hashes from local DB
    async function fetchStatsForHashes(hashes) {
        if (!hashes.length) return;
        try {
            const resp = await _origFetch("/api/lm/civitai-stats/by-hashes", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ hashes }),
            });
            const data = await resp.json();
            if (data.success && data.stats) {
                Object.assign(_statsMap, data.stats);
            }
        } catch (e) {
            console.debug("[CivitAI Stats] Failed to fetch stats:", e);
        }
    }

    // ── Intercept list API to collect hashes ───────────────────────
    const _origFetch = window.fetch;
    window.fetch = async function (...args) {
        const response = await _origFetch.apply(this, args);
        const url = typeof args[0] === "string" ? args[0] : args[0]?.url;
        if (url && url.includes("/api/lm/") && url.includes("/list")) {
            try {
                const clone = response.clone();
                const data = await clone.json();
                const hashes = (data.items || [])
                    .map((item) => item.sha256)
                    .filter((h) => h && !_statsMap[h]);
                if (hashes.length > 0) {
                    // Fire and forget — cards will be patched when data arrives
                    fetchStatsForHashes(hashes).then(() => patchCards());
                }
            } catch (e) { /* ignore parse errors */ }
        }
        return response;
    };

    // ── Patch model cards ──────────────────────────────────────────
    function patchCards() {
        const cards = document.querySelectorAll(".model-card:not([data-stats-patched])");
        cards.forEach((card) => {
            const sha = card.dataset.sha256;
            if (!sha || !_statsMap[sha]) return;

            card.setAttribute("data-stats-patched", "1");

            const stats = _statsMap[sha];
            const container = document.createElement("div");
            container.className = "lm-stat-badges";

            const dlBadge = createStatBadge("download", stats.download_count, "Downloads");
            const ratingBadge = createStatBadge("star",
                stats.rating ? Number(stats.rating.toFixed(1)) : null, "Rating");
            const thumbsBadge = createStatBadge("thumbs-up", stats.thumbs_up_count, "Likes");

            [dlBadge, ratingBadge, thumbsBadge].forEach((b) => {
                if (b) container.appendChild(b);
            });

            if (container.children.length > 0) {
                const modelInfo = card.querySelector(".model-info");
                if (modelInfo) {
                    modelInfo.appendChild(container);
                }
            }
        });
    }

    // ── Page type detection ────────────────────────────────────────
    function _getPageType() {
        const path = window.location.pathname;
        if (path.includes("checkpoints")) return "checkpoints";
        if (path.includes("embeddings")) return "embeddings";
        if (path.includes("loras") || path === "/") return "loras";
        return null;
    }

    // ── Stats sort keys ──────────────────────────────────────────
    const _STATS_SORT_KEYS = new Set(["downloads", "rating", "thumbsup"]);

    // ── Patch sort dropdown ────────────────────────────────────────
    function patchSortDropdown() {
        if (!_getPageType()) return;  // skip non-model pages (recipes, statistics)
        const select = document.getElementById("sortSelect");
        if (!select || select.querySelector('[value="downloads:desc"]')) return;

        const group = document.createElement("optgroup");
        group.label = "CivitAI Stats";

        const options = [
            ["downloads:desc", "Most downloaded"],
            ["downloads:asc", "Least downloaded"],
            ["rating:desc", "Highest rated"],
            ["rating:asc", "Lowest rated"],
            ["thumbsup:desc", "Most liked"],
            ["thumbsup:asc", "Least liked"],
        ];

        options.forEach(([value, label]) => {
            const opt = document.createElement("option");
            opt.value = value;
            opt.textContent = label;
            group.appendChild(opt);
        });

        select.appendChild(group);

        // Restore persisted stats sort — our options didn't exist when
        // PageControls.loadSortPreference() ran, so the saved value
        // was silently ignored. Re-apply it now.
        const pageType = _getPageType();
        const saved = localStorage.getItem("lora_manager_" + pageType + "_sort")
            || localStorage.getItem(pageType + "_sort");  // legacy fallback
        if (saved && _STATS_SORT_KEYS.has(saved.split(":")[0]) && select.value !== saved) {
            select.value = saved;
            select.dispatchEvent(new Event("change"));
        }
    }

    // ── Toolbar "Fetch Stats" button ───────────────────────────────
    function addFetchStatsButton() {
        if (!_getPageType()) return;  // skip non-model pages (recipes, statistics)
        const toolbar = document.querySelector(".action-buttons");
        if (!toolbar || document.getElementById("fetchStatsBtn")) return;

        const group = document.createElement("div");
        group.className = "control-group";
        group.innerHTML = `
            <button id="fetchStatsBtn" data-action="fetch-stats"
                    title="Fetch CivitAI stats (downloads, ratings, likes)">
                <i class="fas fa-chart-bar"></i> <span>Fetch Stats</span>
            </button>
        `;

        const bulkBtn = document.getElementById("bulkOperationsBtn");
        if (bulkBtn && bulkBtn.closest(".control-group")) {
            toolbar.insertBefore(group, bulkBtn.closest(".control-group"));
        } else {
            toolbar.appendChild(group);
        }

        group.querySelector("button").addEventListener("click", async () => {
            const btn = document.getElementById("fetchStatsBtn");
            btn.disabled = true;
            btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> <span>Fetching...</span>';

            try {
                const resp = await _origFetch("/api/lm/civitai-stats/fetch", { method: "POST" });
                const data = await resp.json();
                if (data.success) {
                    const count = parseInt(data.updated, 10) || 0;
                    btn.innerHTML = `<i class="fas fa-check"></i> <span>${count} updated</span>`;
                    setTimeout(() => {
                        btn.innerHTML = '<i class="fas fa-chart-bar"></i> <span>Fetch Stats</span>';
                        btn.disabled = false;
                    }, 3000);
                    // Trigger page reload to show new stats
                    const sortSelect = document.getElementById("sortSelect");
                    if (sortSelect) {
                        sortSelect.dispatchEvent(new Event("change"));
                    }
                } else {
                    throw new Error(data.error || "Unknown error");
                }
            } catch (err) {
                btn.innerHTML = '<i class="fas fa-exclamation-triangle"></i> <span>Error</span>';
                console.error("[CivitAI Stats] Fetch failed:", err);
                setTimeout(() => {
                    btn.innerHTML = '<i class="fas fa-chart-bar"></i> <span>Fetch Stats</span>';
                    btn.disabled = false;
                }, 3000);
            }
        });
    }

    // ── Observe DOM for card rendering ─────────────────────────────
    function startObserver() {
        injectStyles();
        patchSortDropdown();
        addFetchStatsButton();
        patchCards();

        let debounceTimer = null;
        const observer = new MutationObserver(() => {
            if (debounceTimer) return;
            debounceTimer = setTimeout(() => {
                debounceTimer = null;
                patchCards();
                patchSortDropdown();
                addFetchStatsButton();
            }, 200);
        });

        const grid = document.getElementById("modelGrid");
        observer.observe(grid || document.body, { childList: true, subtree: true });
    }

    // ── Init ───────────────────────────────────────────────────────
    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", startObserver);
    } else {
        startObserver();
    }
})();
