#!/usr/bin/env node
/**
 * Déploiement automatique du Cloudflare Worker de la VM DE L'AGENT (dépôt JeanHug/AI).
 * Worker cible : "vmai"  →  https://vmai.hugdu77777.workers.dev
 *
 * Lancement (GitHub Actions) :
 *   CLOUDFLARE_TOKEN=… CLOUDFLARE_ID=… node scripts/deploy-cloudflare-worker.cjs
 *
 * Garde-fous :
 *  - refuse de déployer si cloudflare-worker/worker.js ne cible pas JeanHug/AI ;
 *  - vérifie le nom exact du script auprès de l'API avant le PUT (insensible à la casse) :
 *    ne peut PAS écraser un autre worker par erreur ;
 *  - le binding secret PASS est conservé (keep_bindings) ; si la variable PASS est
 *    fournie, elle est réinjectée explicitement ;
 *  - contrôle final : le worker doit s'identifier comme "VMAI" (sinon le job échoue).
 */
const https = require('https');
const fs = require('fs');
const path = require('path');

const accountId = process.env.CLOUDFLARE_ID || 'aeb92cddbb638d207038daf831a5aab2';
const apiToken = process.env.CLOUDFLARE_TOKEN;
const pass = (process.env.PASS || process.env.VM_PASS || process.argv[2] || '').trim();
const wanted = (process.env.WORKER_NAME || 'vmai').trim();

if (!apiToken) {
  console.error("❌ Erreur : CLOUDFLARE_TOKEN manquant dans les variables d'environnement.");
  process.exit(1);
}

const workerFilePath = path.join(__dirname, '..', 'cloudflare-worker', 'worker.js');
if (!fs.existsSync(workerFilePath)) {
  console.error(`❌ Erreur : fichier ${workerFilePath} introuvable.`);
  process.exit(1);
}
const workerContent = fs.readFileSync(workerFilePath, 'utf8');

if (!workerContent.includes('raw.githubusercontent.com/JeanHug/AI/tunnel-url/')) {
  console.error("❌ Sécurité : ce worker.js ne cible pas le dépôt JeanHug/AI — déploiement annulé.");
  process.exit(1);
}
if (!workerContent.includes('worker: "VMAI"')) {
  console.error('❌ Sécurité : ce worker.js ne s\'identifie pas comme "VMAI" — déploiement annulé.');
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

function httpsGet(url, headers) {
  return new Promise((resolve, reject) => {
    https
      .get(url, { headers: headers || {} }, (res) => {
        let data = '';
        res.on('data', (c) => (data += c));
        res.on('end', () => resolve({ status: res.statusCode, body: data }));
      })
      .on('error', reject);
  });
}

(async () => {
  // 1) Nom exact du worker cible parmi les scripts du compte
  const list = await api('GET', `/client/v4/accounts/${accountId}/workers/scripts`);
  const names = (list.json && list.json.result ? list.json.result : []).map((s) => s.id);
  console.log('Scripts présents sur le compte :', names.join(', ') || '(aucun)');
  const exact = names.find((n) => n.toLowerCase() === wanted.toLowerCase());
  if (!exact) {
    console.error(`❌ Aucun worker nommé "${wanted}" sur ce compte — déploiement annulé.`);
    process.exit(1);
  }
  console.log(`🚀 Déploiement du Worker Cloudflare "${exact}" (compte ${accountId})…`);

  // 2) PUT multipart (metadata + module)
  const metadata = { main_module: 'worker.js' };
  if (pass) {
    metadata.bindings = [{ type: 'secret_text', name: 'PASS', text: pass }];
    console.log('🔑 Binding PASS réinjecté (variable PASS fournie).');
  } else {
    metadata.keep_bindings = ['secret_text', 'plain_text'];
    console.log('🔒 Binding secret PASS conservé (keep_bindings).');
  }
  const boundary = '----CFWorkerBoundary' + Date.now();
  const body = Buffer.from(
    `--${boundary}\r\nContent-Disposition: form-data; name="metadata"; filename="metadata.json"\r\n` +
      `Content-Type: application/json\r\n\r\n${JSON.stringify(metadata)}\r\n` +
      `--${boundary}\r\nContent-Disposition: form-data; name="worker.js"; filename="worker.js"\r\n` +
      `Content-Type: application/javascript+module\r\n\r\n${workerContent}\r\n` +
      `--${boundary}--\r\n`
  );
  const put = await api(
    'PUT',
    `/client/v4/accounts/${accountId}/workers/scripts/${exact}`,
    body,
    'multipart/form-data; boundary=' + boundary
  );
  if (put.status !== 200 || !put.json || !put.json.success) {
    console.error("❌ Échec de l'envoi du Worker :", String(JSON.stringify(put.json || put.raw)).slice(0, 600));
    process.exit(1);
  }
  console.log(`✅ Worker "${exact}" mis à jour (HTTP ${put.status}).`);

  // 3) Sous-domaine workers.dev (idempotent)
  const sub = await api(
    'POST',
    `/client/v4/accounts/${accountId}/workers/scripts/${exact}/subdomain`,
    JSON.stringify({ enabled: true }),
    'application/json'
  );
  console.log('🌐 Sous-domaine workers.dev :', sub.json && sub.json.success ? 'activé' : `HTTP ${sub.status}`);

  // 4) Contrôle final : le worker doit annoncer worker:"VMAI"
  const host = process.env.WORKER_HOST || `${exact.toLowerCase()}.hugdu77777.workers.dev`;
  const url = `https://${host}/api/health`;
  const h = await httpsGet(url);
  let j = null;
  try {
    j = JSON.parse(h.body);
  } catch (e) {}
  console.log(`🔎 ${url} → HTTP ${h.status} ${String(h.body).slice(0, 160).replace(/\s+/g, ' ')}`);
  if (!j || j.worker !== 'VMAI') {
    console.error('❌ Le worker déployé ne s\'identifie pas comme VMAI — vérification manuelle requise.');
    process.exit(1);
  }
  console.log(`🎉 Worker VMAI opérationnel : https://${host}`);
})().catch((err) => {
  console.error('Erreur fatale:', err && err.message ? err.message : err);
  process.exit(1);
});
