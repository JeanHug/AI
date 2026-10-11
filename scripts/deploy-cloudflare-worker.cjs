#!/usr/bin/env node
/**
 * Script de déploiement automatique du Cloudflare Worker "vm"
 * Utilise l'API Cloudflare officielle avec le jeton CLOUDFLARE_TOKEN
 * et injecte le secret PASS dans le Worker.
 */

const https = require('https');
const fs = require('fs');
const path = require('path');

const accountId = process.env.CLOUDFLARE_ID || 'aeb92cddbb638d207038daf831a5aab2';
const apiToken = process.env.CLOUDFLARE_TOKEN;
const pass = process.env.PASS || process.env.VM_PASS || process.argv[2];
const scriptName = 'vm';

if (!apiToken) {
  console.error('❌ Erreur : CLOUDFLARE_TOKEN manquant dans les variables d\'environnement.');
  process.exit(1);
}

if (!pass) {
  console.error('❌ Erreur : Mot de passe manquant (fournir variable d\'environnement PASS ou argument CLI).');
  process.exit(1);
}

const workerFilePath = path.join(__dirname, '..', 'cloudflare-worker', 'worker.js');
if (!fs.existsSync(workerFilePath)) {
  console.error(`❌ Erreur : fichier ${workerFilePath} introuvable.`);
  process.exit(1);
}

const workerContent = fs.readFileSync(workerFilePath, 'utf8');

async function deploy() {
  console.log(`🚀 Déploiement du Worker Cloudflare "${scriptName}" pour le compte ${accountId}...`);

  const boundary = '----CFWorkerBoundary' + Date.now();
  const metadata = {
    main_module: 'worker.js',
    bindings: [
      {
        type: 'secret_text',
        name: 'PASS',
        text: pass
      }
    ]
  };

  const bodyParts = [
    `--${boundary}\r\nContent-Disposition: form-data; name="metadata"; filename="metadata.json"\r\nContent-Type: application/json\r\n\r\n${JSON.stringify(metadata)}\r\n`,
    `--${boundary}\r\nContent-Disposition: form-data; name="worker.js"; filename="worker.js"\r\nContent-Type: application/javascript+module\r\n\r\n${workerContent}\r\n`,
    `--${boundary}--\r\n`
  ];

  const bodyBuffer = Buffer.from(bodyParts.join(''));

  const putPromise = new Promise((resolve, reject) => {
    const req = https.request({
      hostname: 'api.cloudflare.com',
      path: `/client/v4/accounts/${accountId}/workers/scripts/${scriptName}`,
      method: 'PUT',
      headers: {
        'Authorization': 'Bearer ' + apiToken,
        'Content-Type': 'multipart/form-data; boundary=' + boundary,
        'Content-Length': bodyBuffer.length
      }
    }, res => {
      let data = '';
      res.on('data', chunk => data += chunk);
      res.on('end', () => {
        try {
          resolve({ status: res.statusCode, body: JSON.parse(data) });
        } catch (e) {
          resolve({ status: res.statusCode, raw: data });
        }
      });
    });
    req.on('error', reject);
    req.write(bodyBuffer);
    req.end();
  });

  const uploadResult = await putPromise;
  if (uploadResult.status !== 200 || !uploadResult.body?.success) {
    console.error('❌ Échec de l\'envoi du Worker:', uploadResult.body || uploadResult.raw);
    process.exit(1);
  }
  console.log('✅ Worker script "vm" uploadé avec succès !');

  // Activer le sous-domaine workers.dev
  const subdomainPromise = new Promise((resolve) => {
    const postData = JSON.stringify({ enabled: true });
    const req = https.request({
      hostname: 'api.cloudflare.com',
      path: `/client/v4/accounts/${accountId}/workers/scripts/${scriptName}/subdomain`,
      method: 'POST',
      headers: {
        'Authorization': 'Bearer ' + apiToken,
        'Content-Type': 'application/json',
        'Content-Length': Buffer.byteLength(postData)
      }
    }, res => {
      let data = '';
      res.on('data', chunk => data += chunk);
      res.on('end', () => {
        try {
          resolve({ status: res.statusCode, body: JSON.parse(data) });
        } catch (e) {
          resolve({ status: res.statusCode, raw: data });
        }
      });
    });
    req.on('error', err => resolve({ error: err.message }));
    req.write(postData);
    req.end();
  });

  const subResult = await subdomainPromise;
  console.log('🌐 Activation sous-domaine workers.dev:', subResult.body?.success ? 'Activé' : subResult.status);

  // Récupérer le sous-domaine
  const getSubdomainPromise = new Promise((resolve) => {
    https.get({
      hostname: 'api.cloudflare.com',
      path: `/client/v4/accounts/${accountId}/workers/subdomain`,
      headers: { 'Authorization': 'Bearer ' + apiToken }
    }, res => {
      let data = '';
      res.on('data', chunk => data += chunk);
      res.on('end', () => {
        try {
          resolve(JSON.parse(data)?.result?.subdomain);
        } catch (e) {
          resolve(null);
        }
      });
    }).on('error', () => resolve(null));
  });

  const subdomain = await getSubdomainPromise;
  const workerUrl = `https://${scriptName}.${subdomain || 'hugdu77777'}.workers.dev`;
  console.log(`🎉 Worker opérationnel et déployé à l'adresse : ${workerUrl}`);
}

deploy().catch(err => {
  console.error('Erreur fatale:', err);
  process.exit(1);
});
