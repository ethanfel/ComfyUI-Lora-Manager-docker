import { beforeEach, afterEach, describe, expect, it, vi } from 'vitest';
import fs from 'node:fs';
import { renderTemplate } from '../utils/domFixtures.js';

const { appCoreInitializeMock } = vi.hoisted(() => ({
  appCoreInitializeMock: vi.fn(),
}));

vi.mock('../../../static/js/core.js', () => ({
  appCore: {
    initialize: appCoreInitializeMock,
  },
}));

const COMMUNITY_MODULE = new URL(
  '../../../static/js/community_creations.js',
  import.meta.url,
).pathname;
const COMMUNITY_CSS = 'static/css/community_creations.css';

const HASH_ALPHA = 'a'.repeat(64);
const HASH_BETA = 'b'.repeat(64);
const HASH_HIDDEN = 'c'.repeat(64);
const HASH_UNAVAILABLE = 'd'.repeat(64);

function jsonResponse(payload, { ok = true, status = 200 } = {}) {
  return {
    ok,
    status,
    json: vi.fn().mockResolvedValue(payload),
  };
}

function galleryPayload(models = []) {
  return {
    success: true,
    models,
    page: 1,
    page_size: 10,
    total_models: models.length,
    total_pages: models.length ? 1 : 0,
    base_models: {
      'SDXL 1.0': models.length ? 1 : 0,
      'Krea 2': 0,
    },
  };
}

function inventoryPayload(hiddenHashes = new Set(), extraModels = []) {
  const models = [
    {
      sha256: HASH_ALPHA,
      model_name: 'Alpha Style',
      base_model: 'SDXL 1.0',
      image_count: 4,
      hidden: hiddenHashes.has(HASH_ALPHA),
      fetchable: true,
      unavailable_reason: '',
    },
    {
      sha256: HASH_BETA,
      model_name: 'Brand New Beta',
      base_model: 'Flux.1 D',
      image_count: 0,
      hidden: false,
      fetchable: true,
      unavailable_reason: '',
    },
    {
      sha256: HASH_HIDDEN,
      model_name: 'Hidden Gamma',
      base_model: 'Pony',
      image_count: 2,
      hidden: hiddenHashes.has(HASH_HIDDEN),
      fetchable: true,
      unavailable_reason: '',
    },
    {
      sha256: HASH_UNAVAILABLE,
      model_name: 'Local Only',
      base_model: 'SD 1.5',
      image_count: 0,
      hidden: false,
      fetchable: false,
      unavailable_reason: 'No CivitAI metadata',
    },
    ...extraModels,
  ];
  return {
    success: true,
    total_models: models.length,
    hidden_count: hiddenHashes.size,
    models,
  };
}

function galleryModel() {
  return {
    sha256: HASH_ALPHA,
    model_name: 'Alpha Style',
    base_model: 'SDXL 1.0',
    image_count: 1,
    images: [{
      civitai_image_id: 101,
      prompt: 'A detailed community prompt',
      username: 'Creator',
      media_type: 'image',
      preview_url: '/preview/community-101.webp',
      thumbnail_url: '/preview/community-101-thumb.webp',
      resources: [{ type: 'lora', name: 'Linked Resource', modelId: 77 }],
    }],
  };
}

class MockIntersectionObserver {
  observe() {}
  unobserve() {}
  disconnect() {}
}

class MockWebSocket {
  constructor() {
    this.onmessage = null;
  }

  close() {}
}

describe('Community Creations model manager', () => {
  let fetchMock;
  let hiddenHashes;
  let inventoryExtras;
  let refreshedInventoryExtras;
  let fetchFailure;
  let holdRefreshInventory;
  let releaseRefreshInventory;
  let holdInitialInventory;
  let releaseInitialInventory;
  let holdFirstVisibility;
  let releaseFirstVisibility;
  let visibilityRequestOrder;

  beforeEach(async () => {
    vi.resetModules();
    vi.clearAllMocks();
    renderTemplate('community_creations.html');
    const pageContent = document.createElement('main');
    pageContent.className = 'page-content';
    for (const child of [...document.body.childNodes]) {
      pageContent.appendChild(child);
    }
    const appHeader = document.createElement('header');
    appHeader.className = 'app-header';
    document.body.append(appHeader, pageContent);

    hiddenHashes = new Set([HASH_HIDDEN]);
    inventoryExtras = [];
    refreshedInventoryExtras = [];
    fetchFailure = false;
    holdRefreshInventory = false;
    releaseRefreshInventory = null;
    holdInitialInventory = false;
    releaseInitialInventory = null;
    holdFirstVisibility = false;
    releaseFirstVisibility = null;
    visibilityRequestOrder = [];
    appCoreInitializeMock.mockResolvedValue(undefined);
    globalThis.IntersectionObserver = MockIntersectionObserver;
    globalThis.WebSocket = MockWebSocket;
    window.scrollTo = vi.fn();

    fetchMock = vi.fn(async (url, options = {}) => {
      const path = String(url);
      if (path.startsWith('/api/lm/community-images/by-models')) {
        const models = hiddenHashes.has(HASH_ALPHA) ? [] : [galleryModel()];
        return jsonResponse(galleryPayload(models));
      }
      if (path.startsWith('/api/lm/community-images/models')) {
        const extras = path.includes('refresh=true')
          ? refreshedInventoryExtras
          : inventoryExtras;
        const payload = inventoryPayload(new Set(hiddenHashes), extras);
        if (path.includes('refresh=true') && holdRefreshInventory) {
          return new Promise((resolve) => {
            releaseRefreshInventory = () => resolve(jsonResponse(payload));
          });
        }
        if (!path.includes('refresh=true') && holdInitialInventory) {
          return new Promise((resolve) => {
            releaseInitialInventory = () => resolve(jsonResponse(payload));
          });
        }
        return jsonResponse(payload);
      }
      if (path === '/api/lm/community-images/visibility') {
        const body = JSON.parse(options.body);
        visibilityRequestOrder.push(body.sha256);
        const respond = () => {
          if (body.hidden) hiddenHashes.add(body.sha256);
          else hiddenHashes.delete(body.sha256);
          return jsonResponse({
            success: true,
            ...body,
            hidden_count: hiddenHashes.size,
          });
        };
        if (holdFirstVisibility && visibilityRequestOrder.length === 1) {
          return new Promise((resolve) => {
            releaseFirstVisibility = () => resolve(respond());
          });
        }
        return respond();
      }
      if (path === '/api/lm/community-images/fetch') {
        if (fetchFailure) {
          return jsonResponse(
            { success: false, error: 'Selected fetch was rejected' },
            { ok: false, status: 409 },
          );
        }
        return jsonResponse({ success: true, stored: 3, total: 1, skipped: 0 });
      }
      throw new Error(`Unexpected request: ${path}`);
    });
    globalThis.fetch = fetchMock;
    window.fetch = fetchMock;

    await import(COMMUNITY_MODULE);
    expect(appCoreInitializeMock).toHaveBeenCalledTimes(1);
    await vi.waitFor(() => {
      expect(fetchMock).toHaveBeenCalledWith(
        expect.stringContaining('/api/lm/community-images/by-models'),
      );
    });
  });

  afterEach(() => {
    delete globalThis.IntersectionObserver;
    delete globalThis.WebSocket;
    vi.restoreAllMocks();
  });

  it('lists new, hidden, and unavailable LoRAs and fetches only the selection', async () => {
    document.getElementById('fetchCommunityBtn').click();

    await vi.waitFor(() => {
      expect(document.querySelectorAll('.community-manager-row')).toHaveLength(4);
    });

    const overlay = document.getElementById('communityModelManager');
    expect(overlay.hidden).toBe(false);
    expect(overlay.parentElement).toBe(document.body);
    expect(document.getElementById('communityModelManagerList').textContent).toContain(
      'New · no images fetched',
    );
    expect(document.getElementById('communityModelManagerList').textContent).toContain(
      'No CivitAI metadata',
    );

    const betaCheckbox = document.querySelector(
      `.community-manager-checkbox[value="${HASH_BETA}"]`,
    );
    const hiddenCheckbox = document.querySelector(
      `.community-manager-checkbox[value="${HASH_HIDDEN}"]`,
    );
    const unavailableCheckbox = document.querySelector(
      `.community-manager-checkbox[value="${HASH_UNAVAILABLE}"]`,
    );
    expect(betaCheckbox.checked).toBe(false);
    expect(hiddenCheckbox.disabled).toBe(true);
    expect(unavailableCheckbox.disabled).toBe(true);

    document.getElementById('communitySelectNewBtn').click();
    expect(document.querySelector(
      `.community-manager-checkbox[value="${HASH_BETA}"]`,
    ).checked).toBe(true);
    document.getElementById('communityClearSelectionBtn').click();
    const alphaCheckbox = document.querySelector(
      `.community-manager-checkbox[value="${HASH_ALPHA}"]`,
    );
    alphaCheckbox.checked = true;
    alphaCheckbox.dispatchEvent(new Event('change', { bubbles: true }));
    document.getElementById('communityFetchSelectedBtn').click();

    await vi.waitFor(() => {
      const call = fetchMock.mock.calls.find(
        ([url]) => url === '/api/lm/community-images/fetch',
      );
      expect(call).toBeDefined();
      expect(JSON.parse(call[1].body)).toEqual({
        force: true,
        hashes: [HASH_ALPHA],
      });
    });
    expect(overlay.hidden).toBe(true);
  });

  it('shows installed base-model tabs before Community images are fetched', () => {
    const kreaTab = [...document.querySelectorAll('.base-model-tab')]
      .find((tab) => tab.dataset.baseModel === 'Krea 2');

    expect(kreaTab).not.toBeNull();
    expect(kreaTab.textContent).toBe('Krea 2 (0)');
  });

  it('joins a slow initial inventory load and still starts the automatic refresh after reopen', async () => {
    holdInitialInventory = true;
    const fetchButton = document.getElementById('fetchCommunityBtn');

    fetchButton.click();
    await vi.waitFor(() => {
      expect(releaseInitialInventory).toBeTypeOf('function');
    });

    document.getElementById('communityModelManagerClose').click();
    fetchButton.click();

    const inventoryCalls = () => fetchMock.mock.calls.filter(
      ([url]) => url === '/api/lm/community-images/models',
    );
    expect(inventoryCalls()).toHaveLength(1);
    expect(fetchMock).not.toHaveBeenCalledWith(
      '/api/lm/community-images/models?refresh=true',
    );

    releaseInitialInventory();
    await vi.waitFor(() => {
      expect(fetchMock).toHaveBeenCalledWith(
        '/api/lm/community-images/models?refresh=true',
      );
      expect(inventoryCalls()).toHaveLength(1);
    });
  });

  it('uses a native, non-nesting card trigger for the focus-managed detail dialog', async () => {
    await vi.waitFor(() => {
      expect(document.querySelector('.community-card')).not.toBeNull();
    });

    const card = document.querySelector('.community-card');
    const detailTrigger = card.querySelector('.community-card-detail-trigger');
    const resourceLink = card.querySelector('a.community-resource-tag');
    expect(card.hasAttribute('role')).toBe(false);
    expect(card.tabIndex).toBe(-1);
    expect(detailTrigger.tagName).toBe('BUTTON');
    expect(detailTrigger.getAttribute('aria-haspopup')).toBe('dialog');
    expect(resourceLink).not.toBeNull();
    expect(detailTrigger.contains(resourceLink)).toBe(false);
    resourceLink.addEventListener('click', (event) => event.preventDefault());
    resourceLink.click();
    expect(document.querySelector('.community-detail-overlay')).toBeNull();

    detailTrigger.focus();
    detailTrigger.click();

    const detailOverlay = document.querySelector('.community-detail-overlay');
    const dialog = detailOverlay.querySelector('[role="dialog"]');
    const closeButton = detailOverlay.querySelector('.community-detail-close');
    expect(dialog.getAttribute('aria-modal')).toBe('true');
    expect(closeButton).toBe(document.activeElement);
    expect(document.querySelector('.page-content').hasAttribute('inert')).toBe(true);
    expect(document.querySelector('.app-header').hasAttribute('inert')).toBe(true);
    expect(document.body.classList.contains('community-detail-open')).toBe(true);

    closeButton.click();

    expect(document.querySelector('.community-detail-overlay')).toBeNull();
    expect(document.querySelector('.page-content').hasAttribute('inert')).toBe(false);
    expect(document.querySelector('.app-header').hasAttribute('inert')).toBe(false);
    expect(document.body.classList.contains('community-detail-open')).toBe(false);
    expect(document.activeElement).toBe(detailTrigger);
  });

  it('provides an explicit visible focus ring for keyboard-focused cards', () => {
    const css = fs.readFileSync(COMMUNITY_CSS, 'utf8');
    const focusRule = css.slice(css.indexOf(
      '.community-card:has(.community-card-detail-trigger:focus-visible)',
    ));

    expect(focusRule).toContain('outline: 3px solid');
    expect(focusRule).toContain('outline-offset: 3px');
  });

  it('persists gallery hide and allows the model manager to unhide it', async () => {
    hiddenHashes.delete(HASH_ALPHA);
    await vi.waitFor(() => {
      expect(document.querySelector('.community-hide-btn')).not.toBeNull();
    });

    const galleryHideButton = document.querySelector('.community-hide-btn');
    galleryHideButton.focus();
    galleryHideButton.click();
    await vi.waitFor(() => {
      const visibilityCalls = fetchMock.mock.calls.filter(
        ([url]) => url === '/api/lm/community-images/visibility',
      );
      expect(JSON.parse(visibilityCalls[0][1].body)).toEqual({
        sha256: HASH_ALPHA,
        hidden: true,
      });
      expect(document.activeElement).toBe(
        document.getElementById('fetchCommunityBtn'),
      );
    });

    document.getElementById('fetchCommunityBtn').click();
    await vi.waitFor(() => {
      expect(document.querySelector('.community-manager-visibility')).not.toBeNull();
    });

    const alphaRow = [...document.querySelectorAll('.community-manager-row')]
      .find((row) => row.textContent.includes('Alpha Style'));
    expect(alphaRow.textContent).toContain('Hidden from gallery');
    alphaRow.querySelector('.community-manager-visibility').click();

    await vi.waitFor(() => {
      const visibilityCalls = fetchMock.mock.calls.filter(
        ([url]) => url === '/api/lm/community-images/visibility',
      );
      expect(visibilityCalls).toHaveLength(2);
      expect(JSON.parse(visibilityCalls[1][1].body)).toEqual({
        sha256: HASH_ALPHA,
        hidden: false,
      });
      expect(document.activeElement.classList.contains(
        'community-manager-visibility',
      )).toBe(true);
      expect(document.activeElement.dataset.sha256).toBe(HASH_ALPHA);
    });
  });

  it('serializes visibility writes and preserves them across a stale background refresh', async () => {
    hiddenHashes.delete(HASH_ALPHA);
    holdRefreshInventory = true;
    holdFirstVisibility = true;

    document.getElementById('fetchCommunityBtn').click();
    await vi.waitFor(() => {
      expect(releaseRefreshInventory).toBeTypeOf('function');
      expect(document.querySelectorAll('.community-manager-row')).toHaveLength(4);
    });

    const rowFor = (name) => [...document.querySelectorAll('.community-manager-row')]
      .find((row) => row.textContent.includes(name));
    rowFor('Alpha Style').querySelector('.community-manager-visibility').click();
    rowFor('Brand New Beta').querySelector('.community-manager-visibility').click();

    await vi.waitFor(() => {
      expect(visibilityRequestOrder).toEqual([HASH_ALPHA]);
      expect(releaseFirstVisibility).toBeTypeOf('function');
    });

    releaseFirstVisibility();
    await vi.waitFor(() => {
      expect(visibilityRequestOrder).toEqual([HASH_ALPHA, HASH_BETA]);
      expect(hiddenHashes.has(HASH_ALPHA)).toBe(true);
      expect(hiddenHashes.has(HASH_BETA)).toBe(true);
      expect(rowFor('Alpha Style').querySelector(
        '.community-manager-visibility',
      ).getAttribute('aria-label')).toContain('Unhide');
      expect(rowFor('Brand New Beta').querySelector(
        '.community-manager-visibility',
      ).getAttribute('aria-label')).toContain('Unhide');
    });

    releaseRefreshInventory();
    await vi.waitFor(() => {
      expect(rowFor('Alpha Style').querySelector(
        '.community-manager-visibility',
      ).getAttribute('aria-label')).toContain('Unhide');
      expect(rowFor('Brand New Beta').querySelector(
        '.community-manager-visibility',
      ).getAttribute('aria-label')).toContain('Unhide');
    });
  });

  it('caps large-library rendering while search can reach every model', async () => {
    inventoryExtras = Array.from({ length: 5001 }, (_, index) => ({
      sha256: `${index}`.padStart(64, '0'),
      model_name: index === 5000 ? 'Needle Model' : `Extra Model ${index}`,
      base_model: 'SDXL 1.0',
      image_count: 1,
      hidden: false,
      fetchable: true,
      unavailable_reason: '',
    }));
    refreshedInventoryExtras = inventoryExtras;

    document.getElementById('fetchCommunityBtn').click();
    await vi.waitFor(() => {
      expect(document.querySelector('.community-manager-result-limit')).not.toBeNull();
    });
    expect(document.querySelectorAll('.community-manager-row')).toHaveLength(300);
    expect(document.getElementById('communityModelManagerList').textContent).not.toContain(
      'Needle Model',
    );

    document.getElementById('communitySelectVisibleBtn').click();
    expect(document.getElementById('communitySelectedCount').textContent).toBe(
      '5000 selected',
    );
    expect(document.getElementById('communityModelManagerStatus').textContent).toContain(
      'Selected the first 5000 matching LoRAs',
    );

    const search = document.getElementById('communityModelManagerSearch');
    search.value = 'Needle Model';
    search.dispatchEvent(new Event('input', { bubbles: true }));

    expect(document.querySelectorAll('.community-manager-row')).toHaveLength(1);
    expect(document.getElementById('communityModelManagerList').textContent).toContain(
      'Needle Model',
    );
  });

  it('reopens with the selection intact when a selected fetch fails', async () => {
    document.getElementById('fetchCommunityBtn').click();
    await vi.waitFor(() => {
      expect(document.querySelectorAll('.community-manager-row')).toHaveLength(4);
    });

    const alphaCheckbox = document.querySelector(
      `.community-manager-checkbox[value="${HASH_ALPHA}"]`,
    );
    alphaCheckbox.checked = true;
    alphaCheckbox.dispatchEvent(new Event('change', { bubbles: true }));
    fetchFailure = true;
    document.getElementById('communityFetchSelectedBtn').click();

    await vi.waitFor(() => {
      expect(document.getElementById('communityModelManager').hidden).toBe(false);
      expect(document.getElementById('communityModelManagerStatus').textContent).toContain(
        'Selected fetch was rejected',
      );
    });
    expect(document.querySelector(
      `.community-manager-checkbox[value="${HASH_ALPHA}"]`,
    ).checked).toBe(true);
  });

  it('adds newly discovered LoRAs after the background inventory refresh', async () => {
    refreshedInventoryExtras = [{
      sha256: 'f'.repeat(64),
      model_name: 'Just Installed',
      base_model: 'Flux.1 D',
      image_count: 0,
      hidden: false,
      fetchable: true,
      unavailable_reason: '',
    }];

    document.getElementById('fetchCommunityBtn').click();

    await vi.waitFor(() => {
      expect(fetchMock).toHaveBeenCalledWith(
        '/api/lm/community-images/models?refresh=true',
      );
      expect(document.getElementById('communityModelManagerList').textContent).toContain(
        'Just Installed',
      );
    });
  });

  it('provides a whole-dialog scrolling fallback for short viewports', () => {
    const css = fs.readFileSync(COMMUNITY_CSS, 'utf8');
    const shortViewportRules = css.slice(css.indexOf('@media (max-height: 560px)'));

    expect(shortViewportRules).toContain('overflow-y: auto');
    expect(shortViewportRules).toContain('height: auto');
    expect(shortViewportRules).toContain('max-height: none');
    expect(shortViewportRules).toContain('min-height: 96px');
    expect(shortViewportRules).toContain('max-height: 42vh');
  });
});
