// Simple runtime service worker for caching /pkg/ JS and wasm assets.
const CACHE_NAME = 'mcbaise-assets-runtime-v1';

// Debug flag controllable by the client page via postMessage.
let SW_DEBUG = false;

self.addEventListener('install', (e) => {
  self.skipWaiting();
});

self.addEventListener('activate', (e) => {
  e.waitUntil(self.clients.claim());
});

// Cache-first strategy for /pkg/ assets and any .wasm files.
self.addEventListener('fetch', (event) => {
  const url = new URL(event.request.url);
  // Only handle GETs
  if (event.request.method !== 'GET') return;

  // Cache /pkg/ files and .wasm responses. Normalize cache keys to the
  // pathname (ignore query params like `v=`) so different `v` tokens still
  // hit the same cached asset during local development.
  if (url.pathname.includes('/pkg/') || url.pathname.endsWith('.wasm')) {
    event.respondWith(
      caches.open(CACHE_NAME).then(async (cache) => {
        const cacheKey = new Request(url.pathname);
        try {
          if (SW_DEBUG) console.debug('mcbaise:sw:fetch', { url: event.request.url, cacheKey: cacheKey.url });
        } catch (_) {}
        try {
          const cached = await cache.match(cacheKey);
          if (cached) {
            try { if (SW_DEBUG) console.debug('mcbaise:sw:respond-from-cache', { url: event.request.url, cacheKey: cacheKey.url }); } catch (_) {}
            return cached;
          }
          const resp = await fetch(event.request);
          // Only cache successful responses
          if (resp && resp.status === 200) {
            // Store under the normalized pathname key so queries don't fragment the cache.
            cache.put(cacheKey, resp.clone()).catch(() => {});
            try { if (SW_DEBUG) console.debug('mcbaise:sw:caching', { url: event.request.url, cacheKey: cacheKey.url }); } catch (_) {}
          }
          return resp;
        } catch (err) {
          // fallback to a normalized cache entry if network fails
          try {
            const cached = await cache.match(cacheKey);
            if (cached) {
              try { if (SW_DEBUG) console.debug('mcbaise:sw:fallback-cache-hit', { url: event.request.url, cacheKey: cacheKey.url }); } catch (_) {}
              return cached;
            }
            return Promise.reject(err);
          } catch (_) {
            return Promise.reject(err);
          }
        }
      })
    );
  }
});

self.addEventListener('message', (ev) => {
  // allow client pages to enable/disable debug logs from the SW
  try {
    const d = ev.data || {};
    if (d && d.type === 'SW_DEBUG_ENABLE') {
      SW_DEBUG = true;
      try { ev.source?.postMessage?.({ type: 'SW_DEBUG_ACK', enabled: true }); } catch (_) {}
      return;
    }
    if (d && d.type === 'SW_DEBUG_DISABLE') {
      SW_DEBUG = false;
      try { ev.source?.postMessage?.({ type: 'SW_DEBUG_ACK', enabled: false }); } catch (_) {}
      return;
    }
  } catch (_) {}
  // allow clearing cache via client message if desired
  const data = ev.data || {};
  if (data && data.type === 'CLEAR_MCBAISE_CACHE') {
    caches.keys().then(keys => {
      return Promise.all(keys.map(k => { if (k.startsWith('mcbaise-assets')) return caches.delete(k); }));
    }).then(() => {
      ev.source?.postMessage?.({ type: 'MCBAISE_CACHE_CLEARED' });
    });
    return;
  }

  // Pre-cache a set of URLs provided by the client (e.g. parent page knows the v token).
  if (data && data.type === 'PRECACHE' && Array.isArray(data.urls)) {
    const urls = data.urls;
    caches.open(CACHE_NAME).then(cache => {
      return Promise.all(urls.map(u => fetch(u).then(r => { if (r && r.status === 200) return cache.put(u, r.clone()); }).catch(() => {})));
    }).then(() => {
      ev.source?.postMessage?.({ type: 'MCBAISE_PRECACHE_DONE', urls });
    }).catch(() => {
      ev.source?.postMessage?.({ type: 'MCBAISE_PRECACHE_FAILED', urls });
    });
    return;
  }
});
