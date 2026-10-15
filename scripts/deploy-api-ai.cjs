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
 *   CLOUDFLARE_TOKEN, CLOUDFLARE_ID, ACCESS_CODE (vérification finale),
 *   RF_TOKEN_1, RF_TOKEN_2, RF_TOKEN_3 (jetons Roboflow supplémentaires de la
 *   même organisation, posés ici en secrets du worker — voir plus bas).
 *
 * Jetons Roboflow : le worker en fait tourner plusieurs (rotation) car
 * certaines erreurs rares viennent des limites d'un jeton et non du modèle.
 * GitHub Actions ne peut pas les poser tout seul (`keep_bindings` ne fait que
 * CONSERVER les secrets existants) : ce script les transmet donc explicitement
 * à Cloudflare via l'API « Patch multiple script secrets » :
 *   PATCH /accounts/{id}/workers/scripts/{nom}/secrets-bulk
 *   corps : { secrets: { NOM: { name, text, type: "secret_text" } } }
 * (Aucune valeur n'est jamais écrite dans le dépôt ni affichée dans les logs.)
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
/* Jetons supplémentaires : noms des secrets attendus, dans l'ordre de rotation. */
const tokenSecrets = ['RF_TOKEN_1', 'RF_TOKEN_2', 'RF_TOKEN_3']
  .map((n) => ({ name: n, text: String(process.env[n] || '').trim() }))
  .filter((t) => t.text.length > 0);
/* Autres secrets du worker, posés exactement de la même façon (aucune valeur
   n'est écrite dans le dépôt ni affichée). NVIDIA_NIM_API_KEY et
   MODELSCOPE_API_KEY : clés réservées à des modèles qui seront ajoutés plus
   tard (aucun modèle n'est activé par la simple présence de ces clés). */
const otherSecrets = ['NVIDIA_NIM_API_KEY', 'MODELSCOPE_API_KEY', 'FIREBASE_EMAIL', 'FIREBASE_PASSWORD', 'HF_TOKEN']
  .map((n) => ({ name: n, text: String(process.env[n] || '').trim() }))
  .filter((t) => t.text.length > 0);

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

// Garde-fous « chaîne ModelScope » : le worker doit porter toute la mécanique
// (tables de secours, appel quotidien, routes /api/ms/*), sans quoi l'appel
// quotidien de l'application et les modèles servis par ModelScope seraient muets.
const msParts = ['MS_TEXT', 'MS_IMG', 'MS_FIRST', 'MS_BASE', 'MS_SITE', 'msKey', 'msChat', 'msImage',
  'msDaily', 'msTextRelay', '"/api/ms/daily"', '"/api/ms/models"', '"/api/ms/state"'];
/* Garde-fou « canal Arena » : sans ces morceaux, le modèle « arena » de
   l'application ne pourrait ni écrire ni recevoir. */
const arenaParts = ['arenaToken', 'arenaWrite', 'arenaRead', 'arenaAppOnline', '"/api/arena/send"', '"/api/arena/poll"'];
const arenaMissing = arenaParts.filter((n) => !src.includes(n));
if (arenaMissing.length) {
  console.error('❌ Sécurité : canal Arena incomplet dans le worker (' + arenaMissing.join(', ') + ') — déploiement annulé.');
  process.exit(1);
}
const msMissing = msParts.filter((n) => !src.includes(n));
if (msMissing.length) {
  console.error('❌ Sécurité : chaîne ModelScope incomplète dans le worker (' + msMissing.join(', ') + ') — déploiement annulé.');
  process.exit(1);
}
// Parité application ↔ worker : chaque modèle servi par ModelScope (clé des
// tables MS_TEXT / MS_IMG) doit exister dans l'application (roster ou modèle de
// secours). Renommer un modèle d'un côté seulement est ainsi impossible.
if (fs.existsSync(indexRef)) {
  const app = fs.readFileSync(indexRef, 'utf8');
  const msKeys = [];
  for (const table of ['MS_TEXT', 'MS_IMG']) {
    const m = src.match(new RegExp('const ' + table + '\\s*=\\s*\\{([\\s\\S]*?)\\n\\};'));
    if (!m) continue;
    for (const km of m[1].matchAll(/^\s*"([a-z0-9][a-z0-9.\-]*)"\s*:/gm)) msKeys.push(km[1]);
  }
  const orphans = msKeys.filter((k) => app.indexOf('"' + k + '"') < 0);
  if (msKeys.length < 5 || orphans.length) {
    console.error('❌ Sécurité : parité ModelScope rompue — modèles absents de l\'application : ' + (orphans.join(', ') || '(tables illisibles)') + ' — déploiement annulé.');
    process.exit(1);
  }
  console.log('✅ Chaîne ModelScope : ' + msKeys.length + ' modèles servis, tous présents dans l\'application.');
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

  // 2) PUT : metadata avec la liaison de service VMAI re-déclarée, la liaison
  //    du Durable Object des missions, et sa migration.
  //
  //    Détail de l'API (erreur 10021 si on se trompe) : `migrations` est un
  //    OBJET à un pas — { new_tag, new_sqlite_classes } — et non le tableau de
  //    wrangler.toml. Le pas est aussi soumis à une précondition de tag : une
  //    fois « v1 » appliqué, le renvoyer tel quel est refusé. D'où :
  //      • marqueur .do-migration.json (committé) → on ne renvoie le pas
  //        qu'une seule fois ;
  //      • en cas de refus, on réessaie automatiquement de part et d'autre.
  const doClass = 'MissionJob';
  const doTag = 'v1';
  const marker = path.join(__dirname, '.do-migration.json');
  let appliedTag = null;
  try { appliedTag = JSON.parse(fs.readFileSync(marker, 'utf8')).tag || null; } catch (e) {}

  const baseBindings = [
    { type: 'service', name: 'VMAI', service: 'vmai' },
    { type: 'durable_object_namespace', name: 'MISSION', class_name: doClass },
  ];
  const upload = async (withMigration, oldTag) => {
    const metadata = {
      main_module: 'worker.js',
      bindings: baseBindings,
      compatibility_date: '2024-11-12',
      keep_bindings: ['secret_text', 'plain_text'],
    };
    if (withMigration) {
      metadata.migrations = { new_tag: doTag, new_sqlite_classes: [doClass] };
      if (oldTag) metadata.migrations.old_tag = oldTag;
    }
    const boundary = '----CFWorkerBoundary' + Date.now();
    const body = Buffer.from(
      `--${boundary}\r\nContent-Disposition: form-data; name="metadata"; filename="metadata.json"\r\n` +
        `Content-Type: application/json\r\n\r\n${JSON.stringify(metadata)}\r\n` +
        `--${boundary}\r\nContent-Disposition: form-data; name="worker.js"; filename="worker.js"\r\n` +
        `Content-Type: application/javascript+module\r\n\r\n${src}\r\n` +
        `--${boundary}--\r\n`
    );
    const r = await api('PUT', `/client/v4/accounts/${accountId}/workers/scripts/${exact}`, body,
      'multipart/form-data; boundary=' + boundary);
    return { r, txt: JSON.stringify(r.json || r.raw || {}) };
  };

  let put;
  if (appliedTag) {
    /* Le pas a déjà été appliqué (marqueur présent) : on n'en parle plus. */
    console.log(`ℹ️  Migration du Durable Object déjà appliquée (tag ${appliedTag}) — pas renvoyée.`);
    put = await upload(false);
    if ((put.r.status !== 200 || !put.r.json || !put.r.json.success) && /10061|not currently configured to implement Durable Objects/i.test(put.txt)) {
      console.log('↻ La classe n\'est pas encore configurée : on renvoie le pas de migration.');
      put = await upload(true);
    }
  } else {
    console.log(`🚀 Déploiement du worker "${exact}" (migration du Durable Object, tag ${doTag})…`);
    put = await upload(true);
    const prem = /precondition|already|tag/i.test(put.txt) && put.r.status !== 200;
    if (prem) {
      console.log('↻ Pas de migration refusé (tag déjà appliqué ?) : on réessaie sans.');
      put = await upload(false);
    }
  }
  if (put.r.status !== 200 || !put.r.json || !put.r.json.success) {
    console.error('❌ Échec :', String(put.txt).slice(0, 600));
    process.exit(1);
  }
  try { fs.writeFileSync(marker, JSON.stringify({ tag: doTag, class: doClass }, null, 2) + '\n'); } catch (e) {}
  console.log(`✅ Worker "${exact}" mis à jour (HTTP ${put.r.status}), liaisons VMAI et MISSION re-déclarées.`);

  await api('POST', `/client/v4/accounts/${accountId}/workers/scripts/${exact}/subdomain`,
    JSON.stringify({ enabled: true }), 'application/json');

  /* 2-bis) Jetons Roboflow supplémentaires : posés en secrets du worker par
     lots (une seule version créée). Les secrets absents ne sont pas touchés. */
  const allSecrets = tokenSecrets.concat(otherSecrets);
  if (allSecrets.length) {
    const secrets = {};
    for (const t of allSecrets) {
      secrets[t.name] = { name: t.name, text: t.text, type: 'secret_text' };
    }
    const r = await api('PATCH', `/client/v4/accounts/${accountId}/workers/scripts/${exact}/secrets-bulk`,
      JSON.stringify({ secrets }), 'application/json');
    if (r.status !== 200 || !r.json || !r.json.success) {
      console.error('❌ Échec de la pose des secrets :', String(JSON.stringify(r.json || r.raw || {})).slice(0, 400));
      process.exit(1);
    }
    console.log(`🔐 Secrets posés : ${allSecrets.map((t) => t.name).join(', ')} (valeurs jamais écrites ni affichées).`);
  } else {
    console.log('ℹ️  Aucun jeton Roboflow supplémentaire fourni (RF_TOKEN_1/2/3) : secrets existants conservés.');
  }

  // 3) Vérifications
  const host = process.env.WORKER_HOST || `${exact.toLowerCase()}.hugdu77777.workers.dev`;
  const models = await get(`https://${host}/api/models`);
  console.log(`🔎 /api/models → HTTP ${models.status}`);
  if (models.status !== 200) { console.error('❌ /api/models en échec.'); process.exit(1); }

  const md5 = (b) => crypto.createHash('md5').update(b).digest('hex');
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

  /* Vérification du /app : la propagation de la nouvelle version peut prendre
     quelques secondes — on réessaie au lieu d'échouer au premier essai. */
  if (fs.existsSync(indexRef)) {
    const ref = fs.readFileSync(indexRef);
    let seen = '';
    let ok = false;
    for (let i = 1; i <= 10; i++) {
      const app = await get(`https://${host}/app`);
      seen = md5(app.buf);
      ok = app.status === 200 && seen === md5(ref);
      console.log(`🔎 /app essai ${i}/10 → HTTP ${app.status} · ${app.buf.length} octets · md5 ${seen}` +
        ` ${ok ? '= référence ✅' : '(référence ' + md5(ref) + ')'}`);
      if (ok) break;
      await sleep(6000);
    }
    if (!ok) {
      console.error('❌ /app ne correspond toujours pas après 10 essais.');
      process.exit(1);
    }
  }

  /* Vérification de la rotation : le worker annonce combien de jetons il
     utilise (un nombre, jamais une valeur). */
  if (accessCode) {
    let seen = -1;
    for (let i = 1; i <= 5; i++) {
      const r = await new Promise((resolve, reject) => {
        const payload = JSON.stringify({ model: 'gemini-4-argon', prompt: 'vérification de déploiement',
          params: { _tokens: true } });
        const rq = https.request({ hostname: host, path: '/api/chat', method: 'POST',
          headers: { 'X-Access-Code': accessCode, 'Content-Type': 'application/json',
            'Content-Length': Buffer.byteLength(payload) } }, (res) => {
          let d = ''; res.on('data', (c) => (d += c)); res.on('end', () => resolve({ status: res.statusCode, body: d }));
        });
        rq.on('error', reject); rq.write(payload); rq.end();
      });
      try { seen = JSON.parse(r.body).tokens; } catch (e) {}
      if (typeof seen === 'number' && seen > 0) break;
      await sleep(4000);
    }
    const attendu = 1 + tokenSecrets.length;
    console.log(`🔎 rotation des jetons Roboflow → ${seen} jeton(s) vu(s) par le worker (attendu ≥ ${attendu})`);
    if (!(seen >= attendu)) {
      console.error('❌ Rotation incomplète : le worker ne voit pas tous les jetons (secrets non posés ?).');
      process.exit(1);
    }
  }

  if (accessCode) {
    let ok = false;
    for (let i = 1; i <= 3 && !ok; i++) {
      const r = await get(`https://${host}/api/agent/file?path=user/agent.md`,
        { 'X-Access-Code': accessCode });
      try { ok = JSON.parse(r.buf.toString()).ok === true; } catch (e) {}
      console.log(`🔎 /api/agent/file (agent.md) essai ${i}/3 → HTTP ${r.status} · ok=${ok}`);
      if (!ok) await sleep(4000);
    }
    if (!ok) { console.error("❌ Relais agent/VM en échec (liaison VMAI ?)."); process.exit(1); }
  } else {
    console.log('ℹ️  ACCESS_CODE non fourni : vérification agent ignorée.');
  }

  /* Vérification du moteur de missions durables : on démarre une mission
     minuscule, on vérifie qu'elle est bien portée par un Durable Object et
     qu'elle répond, puis on l'arrête (aucune commande réellement exécutée). */
  if (accessCode) {
    const H = { 'X-Access-Code': accessCode, 'Content-Type': 'application/json' };
    const post = (p2, body) => new Promise((resolve, reject) => {
      const payload = JSON.stringify(body);
      const rq = https.request({ hostname: host, path: p2, method: 'POST',
        headers: Object.assign({}, H, { 'Content-Length': Buffer.byteLength(payload) }) }, (res) => {
        let d = ''; res.on('data', (c) => (d += c)); res.on('end', () => resolve({ status: res.statusCode, body: d }));
      });
      rq.on('error', reject); rq.write(payload); rq.end();
    });
    let durOk = false;
    try {
      const r = await post('/api/agent/job/start', { model: 'gemini-4-argon', mission: 'vérification de déploiement (ne rien exécuter)',
        intro: 'test', rappel: 'test', hist: '', image: 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==', image_kind: 'none', params: {} });
      const j = JSON.parse(r.body || '{}');
      console.log(`🔎 /api/agent/job/start → HTTP ${r.status} · job=${j.job || '?'} · durable=${j.durable === true}`);
      durOk = r.status === 200 && j.durable === true;
      if (j.job) {
        const st = await get(`https://${host}/api/agent/job?id=${encodeURIComponent(j.job)}`, { 'X-Access-Code': accessCode });
        console.log(`🔎 /api/agent/job?id → HTTP ${st.status} (état de la mission retrouvé)`);
        durOk = durOk && st.status === 200;
        await post(`/api/agent/job/stop?id=${encodeURIComponent(j.job)}`, {});
      }
    } catch (e) { console.log('🔎 moteur de missions : ' + (e && e.message ? e.message : e)); }
    if (!durOk) {
      console.error('❌ Le moteur de missions durables ne répond pas (liaison MISSION / migration v1 ?).');
      process.exit(1);
    }
    console.log('✅ Missions durables : Durable Object « MissionJob » opérationnel.');
  }

  console.log(`🎉 Application déployée : https://${host}/app`);
})().catch((e) => {
  console.error('Erreur fatale :', e && e.message ? e.message : e);
  process.exit(1);
});
