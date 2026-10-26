/**
 * Production-Ready Health Check Module for Node.js / Express
 * Compatible with Render Free & Paid Tiers, Docker, and Kubernetes
 *
 * Usage in your main server file (e.g., app.js or server.js):
 *   const express = require('express');
 *   const { registerHealthChecks } = require('./scripts/express-healthcheck-template');
 *   const app = express();
 *   ...
 *   registerHealthChecks(app, {
 *     checkDatabase: async () => true // Replace with real db.query('SELECT 1') if needed
 *   });
 */

const os = require('os');

const startTime = Date.now();

/**
 * Register lightweight and detailed health check routes on the Express app.
 * @param {import('express').Express} app - Express application instance
 * @param {Object} options - Health check options
 * @param {Function} [options.checkDatabase] - Optional async database ping check
 */
function registerHealthChecks(app, options = {}) {
  const { checkDatabase } = options;

  /**
   * 1. Lightweight Liveness Endpoint: /health
   * Used by Render's healthCheckPath and Keep-Alive pingers.
   * MUST return HTTP 200 quickly without heavy computations or DB locks.
   */
  app.get('/health', (req, res) => {
    res.setHeader('Cache-Control', 'no-cache, no-store, must-revalidate');
    res.status(200).json({
      status: 'ok',
      uptime_seconds: Math.floor((Date.now() - startTime) / 1000),
      timestamp: new Date().toISOString()
    });
  });

  /**
   * 2. Detailed Diagnostic Endpoint: /health/detail
   * Useful for diagnosing Render 512MB RAM OOM limits and system load.
   */
  app.get('/health/detail', (req, res) => {
    const memory = process.memoryUsage();
    res.setHeader('Cache-Control', 'no-cache, no-store, must-revalidate');
    res.status(200).json({
      status: 'ok',
      service: process.env.RENDER_SERVICE_NAME || 'node-express-app',
      node_version: process.version,
      uptime_seconds: Math.floor((Date.now() - startTime) / 1000),
      timestamp: new Date().toISOString(),
      memory: {
        rss_mb: (memory.rss / 1024 / 1024).toFixed(2),
        heap_total_mb: (memory.heapTotal / 1024 / 1024).toFixed(2),
        heap_used_mb: (memory.heapUsed / 1024 / 1024).toFixed(2),
        external_mb: (memory.external / 1024 / 1024).toFixed(2),
        free_tier_512mb_pct: ((memory.rss / (512 * 1024 * 1024)) * 100).toFixed(1) + '%'
      },
      system: {
        free_mem_mb: (os.freemem() / 1024 / 1024).toFixed(2),
        total_mem_mb: (os.totalmem() / 1024 / 1024).toFixed(2),
        load_avg_1m: os.loadavg()[0]
      }
    });
  });

  /**
   * 3. Readiness Endpoint: /health/ready
   * Confirms downstream services (like PostgreSQL, MongoDB, Redis) are reachable.
   */
  app.get('/health/ready', async (req, res) => {
    let dbStatus = 'skipped';
    let isHealthy = true;

    if (typeof checkDatabase === 'function') {
      try {
        const ok = await checkDatabase();
        dbStatus = ok ? 'connected' : 'unreachable';
        if (!ok) isHealthy = false;
      } catch (err) {
        dbStatus = `error: ${err.message}`;
        isHealthy = false;
      }
    }

    const statusCode = isHealthy ? 200 : 503;
    res.status(statusCode).json({
      ready: isHealthy,
      database: dbStatus,
      timestamp: new Date().toISOString()
    });
  });
}

/**
 * Helper to attach graceful shutdown handlers (SIGTERM / SIGINT).
 * Render sends SIGTERM during deployments and restarts.
 */
function setupGracefulShutdown(server, onCleanup = async () => {}) {
  const shutdown = async (signal) => {
    console.log(`[Graceful Shutdown] Received ${signal}. Closing HTTP server...`);
    server.close(async () => {
      console.log('[Graceful Shutdown] HTTP server closed.');
      try {
        await onCleanup();
        console.log('[Graceful Shutdown] Cleanup completed. Exiting process.');
        process.exit(0);
      } catch (err) {
        console.error('[Graceful Shutdown] Error during cleanup:', err);
        process.exit(1);
      }
    });

    // Force exit if hanging after 10 seconds
    setTimeout(() => {
      console.error('[Graceful Shutdown] Forced exit after timeout.');
      process.exit(1);
    }, 10000);
  };

  process.on('SIGTERM', () => shutdown('SIGTERM'));
  process.on('SIGINT', () => shutdown('SIGINT'));
}

module.exports = {
  registerHealthChecks,
  setupGracefulShutdown
};
