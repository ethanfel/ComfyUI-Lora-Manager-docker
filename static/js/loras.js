import { appCore } from './core.js';
import { state } from './state/index.js';
import { updateCardsForBulkMode } from './components/shared/ModelCard.js';
import { createPageControls } from './components/controls/index.js';
import { confirmDelete, closeDeleteModal, confirmExclude, closeExcludeModal } from './utils/modalUtils.js';
import { ModelDuplicatesManager } from './components/ModelDuplicatesManager.js';
import { showModelModal } from './components/shared/ModelModal.js';
import { MODEL_TYPES } from './state/constants.js';

// Initialize the LoRA page
export class LoraPageManager {
    constructor() {
        // Add bulk mode to state
        state.bulkMode = false;
        state.selectedLoras = new Set();
        
        // Initialize page controls
        this.pageControls = createPageControls('loras');
        
        // Initialize the ModelDuplicatesManager
        this.duplicatesManager = new ModelDuplicatesManager(this);
        
        // Expose necessary functions to the page that still need global access
        // These will be refactored in future updates
        this._exposeRequiredGlobalFunctions();
    }
    
    _exposeRequiredGlobalFunctions() {
        // Only expose what's still needed globally
        // Most functionality is now handled by the PageControls component
        window.confirmDelete = confirmDelete;
        window.closeDeleteModal = closeDeleteModal;
        window.confirmExclude = confirmExclude;
        window.closeExcludeModal = closeExcludeModal;
        
        // Expose duplicates manager
        window.modelDuplicatesManager = this.duplicatesManager;
    }
    
    async initialize() {
        // Initialize cards for current bulk mode state (should be false initially)
        updateCardsForBulkMode(state.bulkMode);

        // Initialize common page features (including context menus and virtual scroll)
        appCore.initializePageFeatures();
    }
}

async function openModelByHash(sha256) {
    // Wait for virtual scroller to have items loaded
    const scroller = state.virtualScroller;
    if (!scroller || !scroller.items.length) {
        // Retry after a short delay if items haven't loaded yet
        await new Promise(r => setTimeout(r, 1000));
    }
    const items = state.virtualScroller?.items || [];
    const model = items.find(m => m.sha256 === sha256);
    if (model) {
        await showModelModal(model, MODEL_TYPES.LORA);
        // Clear hash so refreshing doesn't re-open
        history.replaceState(null, '', location.pathname);
    }
}

export async function initializeLoraPage() {
    // Initialize core application
    await appCore.initialize();

    // Initialize page-specific functionality
    const loraPage = new LoraPageManager();
    await loraPage.initialize();

    // Deep link: open modal for specific LoRA via hash fragment
    const hashParams = new URLSearchParams(location.hash.slice(1));
    const targetSha256 = hashParams.get('sha256');
    if (targetSha256) {
        openModelByHash(targetSha256);
    }

    return loraPage;
}

// Initialize everything when DOM is ready
document.addEventListener('DOMContentLoaded', initializeLoraPage);