import { beforeEach, describe, expect, it, vi } from 'vitest';

const pageState = { filters: {} };

vi.mock('../../../static/js/state/index.js', () => ({
  getCurrentPageState: vi.fn(() => pageState),
  state: {
    currentPageType: 'loras',
    loadingManager: {
      showSimpleLoading: vi.fn(),
      hide: vi.fn(),
    },
  },
}));

vi.mock('../../../static/js/utils/uiHelpers.js', () => ({
  showToast: vi.fn(),
  updatePanelPositions: vi.fn(),
}));

vi.mock('../../../static/js/api/modelApiFactory.js', () => ({
  getModelApiClient: vi.fn(() => ({
    loadMoreWithVirtualScroll: vi.fn().mockResolvedValue(undefined),
  })),
}));

vi.mock('../../../static/js/utils/storageHelpers.js', () => ({
  getStorageItem: vi.fn(),
  setStorageItem: vi.fn(),
  removeStorageItem: vi.fn(),
}));

vi.mock('../../../static/js/utils/i18nHelpers.js', () => ({
  translate: vi.fn((key, _params, fallback) => fallback || key),
}));

vi.mock('../../../static/js/managers/FilterPresetManager.js', () => ({
  FilterPresetManager: vi.fn().mockImplementation(() => ({
    renderPresets: vi.fn(),
    saveActivePreset: vi.fn(),
    restoreActivePreset: vi.fn(),
    updateAddButtonState: vi.fn(),
    hasEmptyWildcardResult: vi.fn(() => false),
  })),
  EMPTY_WILDCARD_MARKER: '__EMPTY_WILDCARD_RESULT__',
}));

import { FilterManager } from '../../../static/js/managers/FilterManager.js';
import { getStorageItem } from '../../../static/js/utils/storageHelpers.js';

describe('FilterManager workflow toolbar state', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    pageState.filters = {};
    document.body.innerHTML = `
      <button id="filterButton"></button>
      <span id="activeFiltersCount"></span>
      <div id="filterPanel" class="hidden"></div>
      <button id="workflowFilterBtn"></button>
    `;
  });

  it('restores the active toolbar state from storage', () => {
    getStorageItem.mockReturnValue({
      baseModel: [],
      tags: {},
      hasWorkflow: true,
    });

    const manager = new FilterManager({ page: 'loras' });

    expect(manager.filters.hasWorkflow).toBe(true);
    expect(document.getElementById('workflowFilterBtn').classList.contains('active')).toBe(true);
  });

  it('clears the active toolbar state with the filter state', async () => {
    getStorageItem.mockReturnValue({
      baseModel: [],
      tags: {},
      hasWorkflow: true,
    });
    const manager = new FilterManager({ page: 'loras' });

    await manager.clearFilters();

    expect(manager.filters.hasWorkflow).toBe(false);
    expect(document.getElementById('workflowFilterBtn').classList.contains('active')).toBe(false);
  });
});
