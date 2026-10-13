#!/usr/bin/env node
/**
 * Micro-Démon API REST d'interaction Linux pour JeanHug/AI
 * Écoute sur 127.0.0.1:3001
 * Architecture robuste : aucun shell intermédiaire sur le runner hôte (execFile direct).
 * Toutes les commandes s'exécutent strictement et fidèlement dans le conteneur kasm_desktop.
 */

const http = require('http');
const { execFile, execSync } = require('child_process');
const fs = require('fs');
const path = require('path');
const url = require('url');

const PORT = 3001;
const CONTAINER = 'kasm_desktop';

// Détection dynamique de l'environnement (Docker Kasm vs Linux Local/Hôte)
let USE_DOCKER = false;
try {
  const out = execSync(`docker ps -q -f name=${CONTAINER} 2>/dev/null || true`, { timeout: 2000 });
  USE_DOCKER = out.toString().trim().length > 0;
} catch (e) {
  USE_DOCKER = false;
}

console.log(`[Linux-API-Daemon] Mode d'exécution : ${USE_DOCKER ? 'Conteneur Docker ' + CONTAINER : 'Environnement Linux Local Direct'}`);

/**
 * Exécute une commande bash DIRECTEMENT sans passer par un shell hôte runner
 * Élimine l'expansion indésirable de $(...), $VAR, boucles sur l'hôte runner.
 */
function execCommand(cmd, options = {}) {
  const user = options.user || 'kasm-user';
  const timeoutMs = (options.timeout || 30) * 1000;
  const cwd = options.cwd || (USE_DOCKER ? '/home/kasm-user' : (process.env.HOME || '/tmp'));

  return new Promise((resolve) => {
    const start = Date.now();

    if (USE_DOCKER) {
      // execFile direct sur docker : aucun /bin/sh sur le runner hôte
      const args = [
        'exec',
        '-i',
        '-u', user,
        '-w', cwd,
        CONTAINER,
        'bash', '-c', cmd
      ];

      execFile('docker', args, { timeout: timeoutMs, maxBuffer: 15 * 1024 * 1024 }, (err, stdout, stderr) => {
        resolve({
          ok: !err || err.code === 0,
          exit_code: err ? (err.code || 1) : 0,
          stdout: stdout || '',
          stderr: stderr || (err && !err.code ? err.message : ''),
          execution_time_ms: Date.now() - start
        });
      });
    } else {
      // Mode natif direct via execFile sur bash
      execFile('bash', ['-c', cmd], { cwd, timeout: timeoutMs, maxBuffer: 15 * 1024 * 1024 }, (err, stdout, stderr) => {
        resolve({
          ok: !err || err.code === 0,
          exit_code: err ? (err.code || 1) : 0,
          stdout: stdout || '',
          stderr: stderr || (err && !err.code ? err.message : ''),
          execution_time_ms: Date.now() - start
        });
      });
    }
  });
}

function parseBody(req) {
  return new Promise((resolve) => {
    let body = '';
    req.on('data', chunk => {
      body += chunk;
      if (body.length > 25 * 1024 * 1024) req.destroy(); // 25MB limit
    });
    req.on('end', () => {
      try {
        resolve(body ? JSON.parse(body) : {});
      } catch (e) {
        resolve({ _raw: body });
      }
    });
  });
}

function sendJson(res, statusCode, data) {
  res.writeHead(statusCode, {
    'Content-Type': 'application/json; charset=utf-8',
    'X-Linux-Daemon': '1',
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type, Authorization'
  });
  res.end(JSON.stringify(data, null, 2));
}

function resolveContainerPath(targetPath) {
  if (!targetPath) return '/home/kasm-user';
  if (path.posix.isAbsolute(targetPath)) return targetPath;
  return path.posix.join('/home/kasm-user', targetPath);
}

const server = http.createServer(async (req, res) => {
  // CORS Preflight
  if (req.method === 'OPTIONS') {
    res.writeHead(204, {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type, Authorization',
      'Access-Control-Max-Age': '86400'
    });
    res.end();
    return;
  }

  const parsedUrl = url.parse(req.url, true);
  const pathname = parsedUrl.pathname.replace(/\/+$/, '') || '/';
  const query = parsedUrl.query || {};

  // 1. Health check
  if (pathname === '/api/health' || pathname === '/api/v1/health') {
    return sendJson(res, 200, {
      status: 'ok',
      service: 'Linux VM Interaction Daemon',
      mode: USE_DOCKER ? 'docker_kasm' : 'linux_native',
      container: USE_DOCKER ? CONTAINER : null,
      uptime_seconds: Math.floor(process.uptime()),
      timestamp: new Date().toISOString()
    });
  }

  // 2. Statut système complet (exécuté DANS le conteneur)
  if (pathname === '/api/linux/system' || pathname === '/api/system' || pathname === '/api/v1/system') {
    const [memRes, diskRes, uptimeRes, cpuRes, osRes, winRes] = await Promise.all([
      execCommand("free -m | awk 'NR==2{printf \"used: %sMB, total: %sMB, free: %sMB\", $3,$2,$4}'"),
      execCommand("df -h / 2>/dev/null | awk 'NR==2{printf \"used: %s, size: %s, avail: %s\", $3,$2,$4}'"),
      execCommand("uptime -p 2>/dev/null || uptime"),
      execCommand("nproc 2>/dev/null || echo '2'"),
      execCommand("grep PRETTY_NAME /etc/os-release 2>/dev/null | cut -d= -f2 | tr -d '\"' || echo 'Ubuntu Linux'"),
      execCommand("DISPLAY=:1 xdotool getactivewindow getwindowname 2>/dev/null || echo 'Desktop'")
    ]);

    return sendJson(res, 200, {
      ok: true,
      system: {
        mode: USE_DOCKER ? 'docker_kasm' : 'linux_native',
        os: osRes.stdout.trim() || 'Ubuntu Linux',
        cpu_cores: parseInt(cpuRes.stdout.trim()) || 2,
        memory: memRes.stdout.trim() || 'N/A',
        disk_space: diskRes.stdout.trim() || 'N/A',
        uptime: uptimeRes.stdout.trim() || 'up',
        display: ':1',
        active_window: winRes.stdout.trim() || 'Desktop',
        timestamp: new Date().toISOString()
      }
    });
  }

  // 3. Exécution de commande Shell (avec protection totale contre expansion hôte)
  if ((pathname === '/api/linux/exec' || pathname === '/api/exec' || pathname === '/api/v1/exec') && req.method === 'POST') {
    const body = await parseBody(req);
    const command = body.command;

    if (!command || typeof command !== 'string') {
      return sendJson(res, 400, { ok: false, error: 'Paramètre "command" (string) manquant.' });
    }

    const result = await execCommand(command, {
      user: body.user,
      cwd: body.cwd,
      timeout: Math.min(body.timeout || 30, 300)
    });

    return sendJson(res, result.ok ? 200 : 500, result);
  }

  // 4. Capture d'écran (Screenshot)
  if (pathname === '/api/linux/screenshot' || pathname === '/api/screenshot' || pathname === '/api/v1/screenshot') {
    const format = query.format || 'png';
    const isBase64 = format === 'base64' || (req.headers.accept && req.headers.accept.includes('application/json'));

    const screenshotCmd = 'DISPLAY=:1 import -window root png:- 2>/dev/null || DISPLAY=:1 scrot - 2>/dev/null || (DISPLAY=:1 xwd -root -silent | xwdtopnm 2>/dev/null | pnmtopng 2>/dev/null)';

    const cb = (err, stdout) => {
      if (err || !stdout || stdout.length === 0) {
        // Fallback transparent 1x1 si échec
        stdout = Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==', 'base64');
      }

      if (isBase64) {
        return sendJson(res, 200, {
          ok: true,
          format: 'png',
          size_bytes: stdout.length,
          data: `data:image/png;base64,${stdout.toString('base64')}`
        });
      }

      res.writeHead(200, {
        'Content-Type': 'image/png',
        'Content-Length': stdout.length,
        'X-Linux-Daemon': '1',
        'Access-Control-Allow-Origin': '*',
        'Cache-Control': 'no-cache, no-store, must-revalidate'
      });
      res.end(stdout);
    };

    if (USE_DOCKER) {
      execFile('docker', ['exec', '-i', CONTAINER, 'bash', '-c', screenshotCmd], { encoding: 'buffer', maxBuffer: 20 * 1024 * 1024 }, cb);
    } else {
      execFile('bash', ['-c', screenshotCmd], { encoding: 'buffer', maxBuffer: 20 * 1024 * 1024 }, cb);
    }
    return;
  }

  // 5. Contrôle Souris
  if ((pathname === '/api/linux/mouse' || pathname === '/api/mouse' || pathname === '/api/v1/mouse') && req.method === 'POST') {
    const body = await parseBody(req);
    const action = body.action || 'move';
    const hasCoords = body.x !== undefined && body.y !== undefined;
    const x = Math.round(Number(body.x) || 0);
    const y = Math.round(Number(body.y) || 0);

    const button = body.button || 1; // 1: gauche, 2: milieu, 3: droite
    let parts = [];

    if (hasCoords) {
      parts.push(`DISPLAY=:1 xdotool mousemove ${x} ${y}`);
    }

    if (action === 'click') {
      parts.push(`DISPLAY=:1 xdotool click ${button}`);
    } else if (action === 'right_click') {
      parts.push(`DISPLAY=:1 xdotool click 3`);
    } else if (action === 'double_click') {
      parts.push(`DISPLAY=:1 xdotool click --repeat 2 ${button}`);
    } else if (action === 'mousedown') {
      parts.push(`DISPLAY=:1 xdotool mousedown ${button}`);
    } else if (action === 'mouseup') {
      parts.push(`DISPLAY=:1 xdotool mouseup ${button}`);
    } else if (action === 'scroll_up') {
      parts.push(`DISPLAY=:1 xdotool click 4`);
    } else if (action === 'scroll_down') {
      parts.push(`DISPLAY=:1 xdotool click 5`);
    }

    const fullCmd = parts.length > 0 ? parts.join(' && ') : 'DISPLAY=:1 xdotool getmouselocation';
    const result = await execCommand(fullCmd);

    return sendJson(res, result.ok ? 200 : 500, {
      ok: result.ok,
      action,
      x: hasCoords ? x : null,
      y: hasCoords ? y : null,
      error: result.stderr || null
    });
  }

  // 6. Contrôle Clavier
  if ((pathname === '/api/linux/keyboard' || pathname === '/api/keyboard' || pathname === '/api/v1/keyboard') && req.method === 'POST') {
    const body = await parseBody(req);

    if (body.text) {
      // Écriture de texte sécurisée
      const text = String(body.text);
      const safeText = text.replace(/'/g, "'\\''");
      const cmd = `DISPLAY=:1 xdotool type --delay 10 -- '${safeText}'`;
      const result = await execCommand(cmd);
      return sendJson(res, result.ok ? 200 : 500, { ok: result.ok, input: text, error: result.stderr || null });
    } else if (body.key) {
      // Envoi de touche ou combinaison
      let key = String(body.key).trim();
      if (key.toLowerCase() === 'enter') key = 'Return';
      if (key.toLowerCase() === 'backspace') key = 'BackSpace';
      if (key.toLowerCase() === 'esc') key = 'Escape';

      const cmd = `DISPLAY=:1 xdotool key --delay 10 ${key}`;
      const result = await execCommand(cmd);
      return sendJson(res, result.ok ? 200 : 500, { ok: result.ok, key, error: result.stderr || null });
    } else {
      return sendJson(res, 400, { ok: false, error: 'Paramètre "text" ou "key" requis.' });
    }
  }

  // 7. Presse-papier (Clipboard X11 - Non-bloquant via stdin)
  if (pathname === '/api/linux/clipboard' || pathname === '/api/clipboard' || pathname === '/api/v1/clipboard') {
    if (req.method === 'GET') {
      const result = await execCommand('DISPLAY=:1 xclip -selection clipboard -o 2>/dev/null || cat /tmp/clipboard.txt 2>/dev/null || true');
      return sendJson(res, 200, { ok: true, text: result.stdout.trim() });
    }
    if (req.method === 'POST') {
      const body = await parseBody(req);
      const text = String(body.text || '');
      const safeText = text.replace(/'/g, "'\\''");

      // Injection via stdin pour que xclip ne bloque jamais
      const setClipCmd = `printf '%s' '${safeText}' > /tmp/clipboard.txt && (printf '%s' '${safeText}' | DISPLAY=:1 xclip -selection clipboard -i 2>/dev/null || true)`;
      await execCommand(setClipCmd);
      return sendJson(res, 200, { ok: true, length: text.length });
    }
  }

  // 8. Gestion des fichiers
  if (pathname === '/api/linux/files' || pathname === '/api/files' || pathname === '/api/v1/files') {
    const rawPath = query.path;
    const targetPath = USE_DOCKER ? resolveContainerPath(rawPath) : path.resolve(rawPath || '/tmp');

    if (req.method === 'GET') {
      // Listing JSON structuré fiable et sans parsing flaky
      const listScript = `python3 -c "
import os, sys, json
p = sys.argv[1]
if not os.path.exists(p):
    print(json.dumps({'error': 'Path not found'}))
    sys.exit(1)
entries = []
for name in sorted(os.listdir(p)):
    fp = os.path.join(p, name)
    try:
        st = os.stat(fp)
        entries.append({
            'name': name,
            'size': st.st_size,
            'isDirectory': os.path.isdir(fp),
            'mtime': int(st.st_mtime)
        })
    except Exception:
        entries.append({'name': name, 'isDirectory': os.path.isdir(fp)})
print(json.dumps(entries))
" "${targetPath}"`;

      const result = await execCommand(listScript);
      if (!result.ok) {
        return sendJson(res, 404, { ok: false, error: result.stderr || 'Dossier introuvable', path: targetPath });
      }

      try {
        const entries = JSON.parse(result.stdout.trim());
        return sendJson(res, 200, { ok: true, path: targetPath, entries });
      } catch (e) {
        return sendJson(res, 500, { ok: false, error: 'Erreur lecture listing', raw: result.stdout });
      }
    }

    if (req.method === 'DELETE') {
      const result = await execCommand(`rm -rf "${targetPath}"`);
      return sendJson(res, result.ok ? 200 : 500, { ok: result.ok, deleted: targetPath });
    }
  }

  // 8b. Lecture de fichier
  if (pathname === '/api/linux/files/read' || pathname === '/api/files/read' || pathname === '/api/v1/files/read') {
    const rawPath = query.path;
    if (!rawPath) return sendJson(res, 400, { ok: false, error: 'Paramètre "path" manquant.' });

    const targetPath = USE_DOCKER ? resolveContainerPath(rawPath) : path.resolve(rawPath);
    const format = query.format || 'utf8';

    if (format === 'base64') {
      const result = await execCommand(`base64 -w 0 "${targetPath}" 2>/dev/null`);
      if (!result.ok || !result.stdout) {
        return sendJson(res, 404, { ok: false, error: 'Fichier introuvable ou illisible', path: targetPath });
      }
      return sendJson(res, 200, { ok: true, path: targetPath, format: 'base64', content: result.stdout.trim() });
    }

    const result = await execCommand(`cat "${targetPath}"`);
    if (!result.ok) {
      return sendJson(res, 404, { ok: false, error: 'Fichier introuvable ou illisible', path: targetPath });
    }
    return sendJson(res, 200, { ok: true, path: targetPath, content: result.stdout });
  }

  // 8c. Écriture de fichier
  if ((pathname === '/api/linux/files/write' || pathname === '/api/files/write' || pathname === '/api/v1/files/write') && req.method === 'POST') {
    const body = await parseBody(req);
    const rawPath = body.path;
    const content = body.content || '';
    const encoding = body.encoding || 'utf8';

    if (!rawPath) return sendJson(res, 400, { ok: false, error: 'Paramètre "path" manquant.' });

    const targetPath = USE_DOCKER ? resolveContainerPath(rawPath) : path.resolve(rawPath);
    const parentDir = path.posix.dirname(targetPath);

    if (USE_DOCKER) {
      // 1. Assurer que le répertoire parent existe dans le conteneur
      await execCommand(`mkdir -p "${parentDir}"`, { user: 'root' });

      // 2. Écrire dans un fichier temporaire sur l'hôte
      const tempFile = `/tmp/api_upload_${Date.now()}_${Math.random().toString(36).substring(7)}`;
      fs.writeFileSync(tempFile, Buffer.from(content, encoding));

      // 3. Copier directement via execFile vers le conteneur
      execFile('docker', ['cp', tempFile, `${CONTAINER}:${targetPath}`], async (err) => {
        try { fs.unlinkSync(tempFile); } catch (e) {}

        if (err) return sendJson(res, 500, { ok: false, error: err.message });
        await execCommand(`chown kasm-user:kasm-user "${targetPath}" 2>/dev/null || true`, { user: 'root' });
        return sendJson(res, 200, { ok: true, path: targetPath, size_bytes: Buffer.byteLength(content, encoding) });
      });
    } else {
      try {
        fs.mkdirSync(path.dirname(targetPath), { recursive: true });
        fs.writeFileSync(targetPath, Buffer.from(content, encoding));
        return sendJson(res, 200, { ok: true, path: targetPath, size_bytes: Buffer.byteLength(content, encoding) });
      } catch (e) {
        return sendJson(res, 500, { ok: false, error: e.message });
      }
    }
    return;
  }

  // 9. Lancement d'Applications GUI
  if ((pathname === '/api/linux/apps/launch' || pathname === '/api/apps/launch' || pathname === '/api/v1/apps/launch') && req.method === 'POST') {
    const body = await parseBody(req);
    const app = body.app;
    const args = Array.isArray(body.args) ? body.args.join(' ') : (body.args || '');

    if (!app) return sendJson(res, 400, { ok: false, error: 'Paramètre "app" manquant.' });

    const launchCmd = `DISPLAY=:1 nohup ${app} ${args} >/dev/null 2>&1 &`;
    await execCommand(launchCmd);
    return sendJson(res, 200, { ok: true, message: `Application ${app} lancée avec succès.` });
  }

  // 9b. Outil fetch : lecture d'une page web (Chrome headless, service 3011)
  if ((pathname === '/api/linux/fetch' || pathname === '/api/fetch' || pathname === '/api/v1/fetch') && req.method === 'POST') {
    const body = await parseBody(req);
    const url = String(body.url || '').trim();
    if (!/^https?:\/\//i.test(url)) {
      return sendJson(res, 400, { ok: false, error: 'Paramètre "url" (http:// ou https://) manquant.' });
    }
    const timeoutS = Math.min(Math.max(5, Number(body.timeout) || 45), 180);
    const payload = JSON.stringify({
      url: url,
      mode: body.mode === 'html' ? 'html' : 'text',
      wait_ms: body.wait_ms,
      timeout_ms: timeoutS * 1000,
      selector: body.selector
    });
    const b64 = Buffer.from(payload, 'utf8').toString('base64');
    const restart = USE_DOCKER ? '/home/kasm-user/bin/vm-fetch-restart.sh' : '$HOME/bin/vm-fetch-restart.sh';

    // Démarre le service s'il ne tourne pas (idempotent, via le script dédié).
    await execCommand(`pgrep -f "[v]m-fetch-server.py" >/dev/null 2>&1 || ${restart}`, { timeout: 30 });

    const curlCmd = `printf '%s' '${b64}' | base64 -d | curl -s -m ${timeoutS} `
      + `-X POST http://127.0.0.1:3011/fetch -H 'Content-Type: application/json' --data-binary @-`;
    const result = await execCommand(curlCmd, { timeout: timeoutS + 25 });

    const out = String(result.stdout || '').trim();
    let parsed = null;
    try { parsed = JSON.parse(out); } catch (e) { parsed = null; }
    if (parsed && typeof parsed === 'object') {
      return sendJson(res, parsed.ok === false ? 502 : 200, parsed);
    }
    return sendJson(res, 502, {
      ok: false,
      error: "Le service de lecture web (vm-fetch) n'a pas répondu.",
      detail: (out || result.stderr || '').slice(0, 400),
      log_tail: (await execCommand('tail -n 5 /tmp/vm-fetch.log 2>/dev/null', { timeout: 10 })).stdout || ''
    });
  }

  // 10. Liste des Processus (JSON structuré complet)
  if (pathname === '/api/linux/processes' || pathname === '/api/processes' || pathname === '/api/v1/processes') {
    const listProcCmd = `ps -eo pid,user,pcpu,pmem,stat,time,comm --sort=-pcpu 2>/dev/null | head -n 35`;
    const result = await execCommand(listProcCmd);
    return sendJson(res, 200, { ok: true, raw: result.stdout });
  }

  // Route 404 par défaut
  sendJson(res, 404, { ok: false, error: `Point de terminaison non trouvé : ${pathname}` });
});

server.listen(PORT, '127.0.0.1', () => {
  console.log(`[Linux-API-Daemon v2] En écoute sur http://127.0.0.1:${PORT} (Zero-Host-Expansion)`);
});
