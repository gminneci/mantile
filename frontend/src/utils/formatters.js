/**
 * Format a number with comma separators and optional decimal places.
 * Handles integers, floats, null, undefined, and NaN.
 * 
 * @param {number} num - The number to format
 * @param {number} decimals - Number of decimal places (default: 2)
 * @returns {string} Formatted number string or 'N/A'
 */
export const formatNumber = (num, decimals = 2) => {
  if (num === null || num === undefined || isNaN(num)) return 'N/A';
  
  const fixed = num.toFixed(decimals);
  const [integer, decimal] = fixed.split('.');
  const formattedInteger = integer.replace(/\B(?=(\d{3})+(?!\d))/g, ',');
  
  return decimal !== undefined ? `${formattedInteger}.${decimal}` : formattedInteger;
};
