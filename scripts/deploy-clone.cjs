#!/usr/bin/env node
/**
 * Déploiement d'un CLONE du worker de l'application, sous un autre nom.
 *
 *   source  : api-ai/archive/worker-2026-10-29.js  (P116, état du 28/10 15:03 UTC,
 *             c'est-à-dire le worker tel qu'il était il y a une semaine jour pour jour)
 *   cible   : le worker nommé par WORKER_NAME (par défaut « ezjkerrstg »)
 *
 * Le worker « api-ai » n'est JAMAIS touché : on écrit un script distinct, avec
 * ses propres liaisons (service VMAI, Durable Object MISSION) et ses propres
 * secrets. Servir de banc d'essai : si ce clone répond, la version d'il y a une
 * semaine fonctionne encore telle quelle.
 *
 * Variables d'environnement (secrets GitHub Actions, identiques à
 * deploy-api-ai.yml) : CLOUDFLARE_TOKEN, CLOUDFLARE_ID, ACCESS_CODE,
 * ROBOFLOW_API_KEY_1..8, RF_TOKEN_1..3, MODELSCOPE_API_KEY, NVIDIA_NIM_API_KEY,
 * IFM_API_KEY, INCEPTION_API_KEY, ANTHROPIC_API_KEY, OPENAI_API_KEY,
 * HUGGINGFACE_TOKEN, HF_TOKEN, FIREBASE_EMAIL, FIREBASE_PASSWORD,
 * POLLINATIONS_API_KEY, GROQ_API_KEY.
 * Aucune valeur n'est écrite dans le dépôt ni affichée.
 */
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');

const accountId = process.env.CLOUDFLARE_ID || 'aeb92cddbb638d207038daf831a5aab2';
const apiToken = process.env.CLOUDFLARE_TOKEN;
const wanted = (process.env.WORKER_NAME || 'ezjkerrstg').trim();
const accessCode = String(process.env.ACCESS_CODE || '').trim();

if (!apiToken) {
  console.error('❌ CLOUDFLARE_TOKEN manquant.');
  process.exit(1);
}

const srcPath = path.join(__dirname, '..', 'api-ai', 'archive', 'worker-2026-10-29.js');
if (!fs.existsSync(srcPath)) {
  console.error('❌ Archive introuvable : ' + srcPath);
  process.exit(1);
}
const src = fs.readFileSync(srcPath, 'utf8');

/* Garde-fous : c'est bien un worker d'application complet. */
if (!src.includes('const APP_HTML =')) {
  console.error('❌ Sécurité : APP_HTML absent de l\'archive — déploiement annulé.');
  process.exit(1);
}
if (!src.includes('vmai.hugdu77777.workers.dev')) {
  console.error('❌ Sécurité : ce worker ne cible pas vmai — déploiement annulé.');
  process.exit(1);
}

const api = async (method, p, body, ctype) => {
  const r = await fetch('https://api.cloudflare.com' + p, {
    method,
    headers: Object.assign({ Authorization: 'Bearer ' + apiToken }, ctype ? { 'Content-Type': ctype } : {}),
    body,
  });
  let json = null;
  const txt = await r.text();
  try { json = JSON.parse(txt); } catch (e) { /* pas du JSON */ }
  return { status: r.status, json, txt };
};

const SECRET_NAMES = []
  .concat(Array.from({ length: 8 }, (_, i) => 'ROBOFLOW_API_KEY_' + (i + 1)))
  .concat(['RF_TOKEN_1', 'RF_TOKEN_2', 'RF_TOKEN_3'])
  .concat(['MODELSCOPE_API_KEY', 'NVIDIA_NIM_API_KEY', 'IFM_API_KEY', 'INCEPTION_API_KEY',
    'ANTHROPIC_API_KEY', 'OPENAI_API_KEY', 'HUGGINGFACE_TOKEN', 'HF_TOKEN',
    'FIREBASE_EMAIL', 'FIREBASE_PASSWORD', 'POLLINATIONS_API_KEY', 'GROQ_API_KEY', 'ACCESS_CODE']);

(async () => {
  /* 1) Le script existe-t-il déjà ? (on ne fait que le lire, jamais le créer
        par mégarde sous le mauvais nom) */
  const list = await api('GET', `/client/v4/accounts/${accountId}/workers/scripts`);
  if (!list.json || !list.json.success) {
    console.error('❌ Listing des workers impossible :', String(list.txt).slice(0, 300));
    process.exit(1);
  }
  const names = (list.json.result || []).map((s) => s.id);
  const exact = names.find((n) => n.toLowerCase() === wanted.toLowerCase()) || wanted;
  console.log(`Scripts du compte : ${names.join(', ')}`);
  console.log(`Cible : « ${exact} » ${names.includes(exact) ? '(mise à jour)' : '(création)'}`);

  /* 2) Envoi du script : liaisons VMAI + MISSION, migration initiale. */
  const doClass = 'MissionJob';
  const meta = {
    main_module: 'worker.js',
    bindings: [
      { type: 'service', name: 'VMAI', service: 'vmai' },
      { type: 'durable_object_namespace', name: 'MISSION', class_name: doClass },
    ],
    compatibility_date: '2024-11-12',
    keep_bindings: ['secret_text', 'plain_text'],
    migrations: { new_tag: 'v1', new_sqlite_classes: [doClass] },
  };
  console.log(`🚀 Envoi de ${srcPath} (${src.length} octets) vers « ${exact} »…`);
  const form = new FormData();
  form.append('metadata', new Blob([JSON.stringify(meta)], { type: 'application/json' }), 'metadata.json');
  form.append('worker.js', new Blob([src], { type: 'application/javascript+module' }), 'worker.js');
  let put = await api('PUT', `/client/v4/accounts/${accountId}/workers/scripts/${exact}`, form);
  if ((put.status !== 200 || !put.json || !put.json.success) && /precondition|already|tag/i.test(put.txt)) {
    console.log('↻ Migration refusée (tag déjà appliqué ?) : on renvoie sans migration.');
    delete meta.migrations;
    const form2 = new FormData();
    form2.append('metadata', new Blob([JSON.stringify(meta)], { type: 'application/json' }), 'metadata.json');
    form2.append('worker.js', new Blob([src], { type: 'application/javascript+module' }), 'worker.js');
    put = await api('PUT', `/client/v4/accounts/${accountId}/workers/scripts/${exact}`, form2);
  }
  if (put.status !== 200 || !put.json || !put.json.success) {
    console.error('❌ Échec du déploiement :', String(put.txt).slice(0, 600));
    process.exit(1);
  }
  console.log(`✅ Worker « ${exact} » écrit (HTTP ${put.status}).`);

  await api('POST', `/client/v4/accounts/${accountId}/workers/scripts/${exact}/subdomain`,
    JSON.stringify({ enabled: true }), 'application/json');

  /* 3) Les secrets : les mêmes que « api-ai », sur le clone. */
  const secrets = {};
  for (const n of SECRET_NAMES) {
    const v = String(process.env[n] || '').trim();
    if (v) secrets[n] = { name: n, text: v, type: 'secret_text' };
  }
  const nb = Object.keys(secrets).length;
  if (nb) {
    const r = await api('PATCH', `/client/v4/accounts/${accountId}/workers/scripts/${exact}/secrets-bulk`,
      JSON.stringify({ secrets }), 'application/json');
    if (r.status !== 200 || !r.json || !r.json.success) {
      console.error('❌ Pose des secrets :', String(r.txt).slice(0, 400));
      process.exit(1);
    }
    console.log(`🔐 ${nb} secret(s) posés sur « ${exact} » : ${Object.keys(secrets).join(', ')}`);
  } else {
    console.log('⚠️  Aucun secret fourni : le clone démarrera sans clés.');
  }

  /* 4) Vérifications légères (le clone a sa propre URL). */
  const host = `${exact.toLowerCase()}.hugdu77777.workers.dev`;
  const get = async (p, h) => {
    const r = await fetch(`https://${host}${p}`, { headers: h || {} });
    const buf = Buffer.from(await r.arrayBuffer());
    return { status: r.status, buf };
  };
  let ok = false;
  for (let i = 1; i <= 8; i++) {
    const models = await get('/api/models');
    console.log(`🔎 /api/models essai ${i}/8 → HTTP ${models.status} · ${models.buf.length} octets`);
    if (models.status === 200) { ok = true; break; }
    await new Promise((r) => setTimeout(r, 5000));
  }
  if (!ok) { console.error('❌ /api/models ne répond pas sur le clone.'); process.exit(1); }

  const app = await get('/app');
  console.log(`🔎 /app → HTTP ${app.status} · md5 ${crypto.createHash('md5').update(app.buf).digest('hex')}`);

  if (accessCode) {
    const f = await get('/api/agent/file?path=user/agent.md', { 'X-Access-Code': accessCode });
    let a = false;
    try { a = JSON.parse(f.buf.toString()).ok === true; } catch (e) {}
    console.log(`🔎 /api/agent/file (agent.md) → HTTP ${f.status} · ok=${a}`);
  }

  console.log(`🎉 Clone déployé : https://${host}/app`);
})().catch((e) => {
  console.error('Erreur fatale :', e && e.message ? e.message : e);
  process.exit(1);
});
