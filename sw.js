const CACHE = 'beat-maker-v4';
const ASSETS = ['./', 'index.html', 'manifest.json', 'icon.svg', 'icon-192.png', 'icon-512.png', 'icon-180.png',
  ...['trap', 'boombap', 'pop', 'lofi', 'latin'].flatMap(k => ['kick', 'snare', 'clap', 'hat', 'ohat', 'bass'].map(i => `samples/${k}/${i}.wav`)),
  ...['keys', 'pad', 'lead'].flatMap(i => [48, 60, 72].map(r => `samples/inst/${i}-${r}.wav`))];

self.addEventListener('install', e => {
  e.waitUntil(caches.open(CACHE).then(c => c.addAll(ASSETS)).then(() => self.skipWaiting()));
});
self.addEventListener('activate', e => {
  e.waitUntil(caches.keys().then(ks => Promise.all(ks.filter(k => k !== CACHE).map(k => caches.delete(k)))).then(() => self.clients.claim()));
});
self.addEventListener('fetch', e => {
  const u = new URL(e.request.url);
  if (e.request.method !== 'GET' || u.origin !== location.origin) return;
  e.respondWith(
    fetch(e.request).then(r => { const c = r.clone(); caches.open(CACHE).then(x => x.put(e.request, c)); return r; })
      .catch(() => caches.match(e.request).then(r => r || caches.match('./')))
  );
});
