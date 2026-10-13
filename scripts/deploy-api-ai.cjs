#!/usr/bin/env node
/**
 * Déploiement du worker API de l'application (api-ai) — dépôt JeanHug/AI.
 *   cible : worker "api-ai"  →  https://api-ai.hugdu77777.workers.dev
 *   rôle  : /app (l'application) + /api/chat, /api/image, /api/agent/* (relais VM)
 *
 * ATTENTION (leçon apprise) : `keep_bindings` NE conserve PAS les liaisons de
 * service. La liaison de service VMAI doit être re-déclarée À CHAQUE PUT,
 * sinon toutes les routes /api/agent/* renvoient l'erreur Cloudflare 1042.
 *
 * Variables d'environnement (secrets GitHub Actions) :
 *   CLOUDFLARE_TOKEN, CLOUDFLARE_ID, ACCESS_CODE (vérification finale).
 * Fichiers lus : api-ai/worker.js et api-ai/index.html (référence du /app).
 */
const https = require('https');
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');

const accountId = process.env.CLOUDFLARE_ID || 'aeb92cddbb638d207038daf831a5aab2';
const apiToken = process.env.CLOUDFLARE_TOKEN;
const accessCode = process.env.ACCESS_CODE || '';
const wanted = (process.env.WORKER_NAME || 'api-ai').trim();

if (!apiToken) {
  console.error("❌ CLOUDFLARE_TOKEN manquant.");
  process.exit(1);
}

const workerPath = path.join(__dirname, '..', 'api-ai', 'worker.js');
const indexRef = path.join(__dirname, '..', 'api-ai', 'index.html');
if (!fs.existsSync(workerPath)) {
  console.error(`❌ ${workerPath} introuvable.`);
  process.exit(1);
}
const src = fs.readFileSync(workerPath, 'utf8');

// Garde-fous : c'est bien le worker de l'application, et il garde la liaison VMAI.
if (!src.includes('const APP_HTML =')) {
  console.error('❌ Sécurité : ce fichier ne contient pas APP_HTML — déploiement annulé.');
  process.exit(1);
}
if (!src.includes('vmai.hugdu77777.workers.dev')) {
  console.error('❌ Sécurité : ce worker ne cible pas vmai — déploiement annulé.');
  process.exit(1);
}

function api(method, apiPath, body, ctype) {
  return new Promise((resolve, reject) => {
    const headers = { Authorization: 'Bearer ' + apiToken };
    if (ctype) headers['Content-Type'] = ctype;
    if (body) headers['Content-Length'] = Buffer.byteLength(body);
    const req = https.request({ hostname: 'api.cloudflare.com', path: apiPath, method, headers }, (res) => {
      let data = '';
      res.on('data', (c) => (data += c));
      res.on('end', () => {
        try {
          resolve({ status: res.statusCode, json: JSON.parse(data) });
        } catch (e) {
          resolve({ status: res.statusCode, json: null, raw: data });
        }
      });
    });
    req.on('error', reject);
    if (body) req.write(body);
    req.end();
  });
}

function get(url, headers) {
  return new Promise((resolve, reject) => {
    https
      .get(url, { headers: headers || {}, timeout: 30000 }, (res) => {
        const chunks = [];
        res.on('data', (c) => chunks.push(c));
        res.on('end', () => resolve({ status: res.statusCode, buf: Buffer.concat(chunks) }));
      })
      .on('error', reject)
      .on('timeout', function () { this.destroy(new Error('timeout')); });
  });
}

(async () => {
  // 1) nom exact du script
  const list = await api('GET', `/client/v4/accounts/${accountId}/workers/scripts`);
  const names = (list.json && list.json.result ? list.json.result : []).map((s) => s.id);
  console.log('Scripts du compte :', names.join(', '));
  const exact = names.find((n) => n.toLowerCase() === wanted.toLowerCase());
  if (!exact) {
    console.error(`❌ Aucun worker "${wanted}" — déploiement annulé.`);
    process.exit(1);
  }

  // 2) PUT : metadata avec la liaison de service VMAI re-déclarée
  const metadata = {
    main_module: 'worker.js',
    bindings: [{ type: 'service', name: 'VMAI', service: 'vmai' }],
    keep_bindings: ['secret_text', 'plain_text'],
  };
  const boundary = '----CFWorkerBoundary' + Date.now();
  const body = Buffer.from(
    `--${boundary}\r\nContent-Disposition: form-data; name="metadata"; filename="metadata.json"\r\n` +
      `Content-Type: application/json\r\n\r\n${JSON.stringify(metadata)}\r\n` +
      `--${boundary}\r\nContent-Disposition: form-data; name="worker.js"; filename="worker.js"\r\n` +
      `Content-Type: application/javascript+module\r\n\r\n${src}\r\n` +
      `--${boundary}--\r\n`
  );
  console.log(`🚀 Déploiement du worker "${exact}"…`);
  const put = await api('PUT', `/client/v4/accounts/${accountId}/workers/scripts/${exact}`, body,
    'multipart/form-data; boundary=' + boundary);
  if (put.status !== 200 || !put.json || !put.json.success) {
    console.error('❌ Échec :', String(JSON.stringify(put.json || put.raw)).slice(0, 600));
    process.exit(1);
  }
  console.log(`✅ Worker "${exact}" mis à jour (HTTP ${put.status}), liaison VMAI re-déclarée.`);

  await api('POST', `/client/v4/accounts/${accountId}/workers/scripts/${exact}/subdomain`,
    JSON.stringify({ enabled: true }), 'application/json');

  // 3) Vérifications
  const host = process.env.WORKER_HOST || `${exact.toLowerCase()}.hugdu77777.workers.dev`;
  const models = await get(`https://${host}/api/models`);
  console.log(`🔎 /api/models → HTTP ${models.status}`);
  if (models.status !== 200) { console.error('❌ /api/models en échec.'); process.exit(1); }

  const app = await get(`https://${host}/app`);
  const md5 = (b) => crypto.createHash('md5').update(b).digest('hex');
  let refOk = null;
  if (fs.existsSync(indexRef)) {
    const ref = fs.readFileSync(indexRef);
    refOk = md5(app.buf) === md5(ref);
    console.log(`🔎 /app → HTTP ${app.status} · ${app.buf.length} octets · md5 ${md5(app.buf)}`);
    console.log(`   référence api-ai/index.html : md5 ${md5(ref)} → ${refOk ? 'IDENTIQUE ✅' : 'DIFFÉRENT ❌'}`);
    if (!refOk) { console.error('❌ /app ne correspond pas à api-ai/index.html.'); process.exit(1); }
  }

  if (accessCode) {
    const r = await get(`https://${host}/api/agent/file?path=user/agent.md`,
      { 'X-Access-Code': accessCode });
    let ok = false;
    try { ok = JSON.parse(r.buf.toString()).ok === true; } catch (e) {}
    console.log(`🔎 /api/agent/file (agent.md) → HTTP ${r.status} · ok=${ok}`);
    if (!ok) { console.error("❌ Relais agent/VM en échec (liaison VMAI ?)."); process.exit(1); }
  } else {
    console.log('ℹ️  ACCESS_CODE non fourni : vérification agent ignorée.');
  }

  console.log(`🎉 Application déployée : https://${host}/app`);
})().catch((e) => {
  console.error('Erreur fatale :', e && e.message ? e.message : e);
  process.exit(1);
});
