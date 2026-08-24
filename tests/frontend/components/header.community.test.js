import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const mocks = vi.hoisted(() => ({
  filterManager: vi.fn(),
  initPageState: vi.fn(),
  searchManager: vi.fn(),
  updateElementAttribute: vi.fn((element, attribute, _key, _params, fallback) => {
    element.setAttribute(attribute, fallback);
  }),
}));

vi.mock('../../../static/js/managers/UpdateService.js', () => ({
  updateService: { toggleUpdateModal: vi.fn() },
}));

vi.mock('../../../static/js/utils/uiHelpers.js', () => ({
  toggleTheme: vi.fn(),
  setPreset: vi.fn(),
  CYCLE_ORDER: [],
  PRESET_NAMES: {},
}));

vi.mock('../../../static/js/managers/SearchManager.js', () => ({
  SearchManager: mocks.searchManager,
}));

vi.mock('../../../static/js/managers/FilterManager.js', () => ({
  FilterManager: mocks.filterManager,
}));

vi.mock('../../../static/js/state/index.js', () => ({
  initPageState: mocks.initPageState,
}));

vi.mock('../../../static/js/utils/storageHelpers.js', () => ({
  getStorageItem: vi.fn(),
  setStorageItem: vi.fn(),
}));

vi.mock('../../../static/js/utils/i18nHelpers.js', () => ({
  updateElementAttribute: mocks.updateElementAttribute,
}));

vi.mock('../../../static/js/services/supportersService.js', () => ({
  renderSupporters: vi.fn().mockResolvedValue(undefined),
}));

import { HeaderManager } from '../../../static/js/components/Header.js';

describe('HeaderManager on the Community page', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    window.history.pushState({}, '', '/community');
    document.body.innerHTML = `
      <div id="headerSearch">
        <input id="searchInput">
        <button id="searchOptionsToggle"></button>
        <button id="filterButton"></button>
      </div>
    `;
  });

  afterEach(() => {
    window.history.pushState({}, '', '/loras');
  });

  it('keeps unavailable search disabled and skips model-page managers', () => {
    const manager = new HeaderManager();
    const headerSearch = document.getElementById('headerSearch');

    expect(manager.currentPage).toBe('community');
    expect(mocks.initPageState).not.toHaveBeenCalled();
    expect(mocks.searchManager).not.toHaveBeenCalled();
    expect(mocks.filterManager).not.toHaveBeenCalled();
    expect(headerSearch.classList.contains('disabled')).toBe(true);
    expect(document.getElementById('searchInput').disabled).toBe(true);
    headerSearch.querySelectorAll('button').forEach((button) => {
      expect(button.disabled).toBe(true);
    });
  });
});
