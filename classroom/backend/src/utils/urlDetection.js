/**
 * Backend URL Detection Utility
 * Single point of truth for backend URL across all deployment platforms
 */

const path = require('path');

/**
 * Get backend URL based on deployment platform or environment variables
 * @returns {string} Backend URL
 */
function getBackendUrl() {
  // Priority 1: Manual override (production deployment)
  if (process.env.BACKEND_URL) {
    return process.env.BACKEND_URL;
  }

  // Priority 2: Cloud platform auto-detection
  
  // Google Cloud Platform
  if (process.env.K_SERVICE && process.env.GOOGLE_CLOUD_PROJECT) {
    // Google Cloud Run
    const region = process.env.GOOGLE_CLOUD_REGION || 'us-central1';
    return `https://${process.env.K_SERVICE}-${process.env.GOOGLE_CLOUD_PROJECT}.${region}.run.app`;
  }
  
  if (process.env.GAE_SERVICE && process.env.GOOGLE_CLOUD_PROJECT) {
    // Google App Engine with service
    return `https://${process.env.GAE_SERVICE}-dot-${process.env.GOOGLE_CLOUD_PROJECT}.appspot.com`;
  }
  
  if (process.env.GOOGLE_CLOUD_PROJECT && process.env.GAE_ENV) {
    // Google App Engine default service
    return `https://${process.env.GOOGLE_CLOUD_PROJECT}.appspot.com`;
  }

  // Other Cloud Platforms
  if (process.env.VERCEL_URL) {
    return `https://${process.env.VERCEL_URL}`;
  }
  
  if (process.env.HEROKU_APP_NAME) {
    return `https://${process.env.HEROKU_APP_NAME}.herokuapp.com`;
  }
  
  if (process.env.RAILWAY_STATIC_URL) {
    return `https://${process.env.RAILWAY_STATIC_URL}`;
  }
  
  if (process.env.RENDER_EXTERNAL_URL) {
    return process.env.RENDER_EXTERNAL_URL;
  }

  // Priority 3: Local development fallback
  const port = process.env.PORT || 4000;
  return `http://localhost:${port}`;
}

/**
 * Generate HTTP URL for file access by ML API
 * @param {string} filePath - Absolute file path on server
 * @returns {string} HTTP URL accessible by external ML API
 */
function getFileUrl(filePath) {
  const backendUrl = getBackendUrl();
  const relativePath = path.relative(path.join(__dirname, '../../public'), filePath);
  return `${backendUrl}/${relativePath.replace(/\\/g, '/')}`;
}

module.exports = {
  getBackendUrl,
  getFileUrl
};
