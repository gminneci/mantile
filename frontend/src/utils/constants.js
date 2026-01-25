/**
 * Color scheme for layer comparison visualizations.
 * 
 * Primary colors (teal) are used for the left/primary configuration panel.
 * Comparison colors (orange) are used for the right/comparison configuration panel.
 * 
 * Light/dark variants are used when multiple layers from the same panel are selected:
 * - First selection from a panel: uses light color
 * - Second selection from same panel: uses dark color
 */
export const CHART_COLORS = {
  primary: '#14B8A6',         // Teal - primary config (left panel)
  primaryDark: '#0A7566',     // Dark teal - second selection from primary
  comparison: '#f96c56',      // Orange - comparison config (right panel)
  comparisonDark: '#c43c2a'   // Dark orange - second selection from comparison
};

/**
 * Memory breakdown visualization colors.
 * 
 * Primary (green/emerald palette) for main configuration.
 * Comparison (orange palette) for comparison configuration.
 * Each has three shades for weight, activation, and KV cache memory.
 */
export const MEMORY_COLORS = {
  primary: {
    weight: '#059669',      // emerald-600 - darkest
    activation: '#10B981',  // emerald-500 - medium
    kv: '#6EE7B7'          // emerald-300 - lightest
  },
  comparison: {
    weight: '#ea580c',      // orange-600 - darkest
    activation: '#f96c56',  // custom orange - medium
    kv: '#fb923c'          // orange-400 - lightest
  }
};
