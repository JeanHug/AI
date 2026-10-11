/**
 * Cloudflare Worker: Passerelle Sécurisée & API d'Interaction Linux pour JeanHug/AI
 * Déployé sous le nom "VMAI" (https://vmai.hugdu77777.workers.dev)
 */

const GITHUB_RAW_BASE = "https://raw.githubusercontent.com/JeanHug/AI/tunnel-url/";

export default {
  async fetch(request, env, ctx) {
    const url = new URL(request.url);

    // 1. Gestion des requêtes CORS pre-flight
    if (request.method === "OPTIONS") {
      return new Response(null, {
        headers: {
          "Access-Control-Allow-Origin": "*",
          "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
          "Access-Control-Allow-Headers": "Content-Type, Authorization, X-Requested-With",
          "Access-Control-Max-Age": "86400",
        },
      });
    }

    const corsHeaders = {
      "Access-Control-Allow-Origin": "*",
      "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
      "Access-Control-Allow-Headers": "Content-Type, Authorization, X-Requested-With",
      "Cache-Control": "no-store, no-cache, must-revalidate",
    };

    // 2. Documentation Minimaliste Fond Blanc (/docs ou /api/docs)
    const isDocRequest = url.pathname === "/docs" || 
                         url.pathname === "/api/docs" || 
                         url.searchParams.has("docs") ||
                         (url.pathname === "/" && request.headers.get("accept")?.includes("text/html") && !url.searchParams.get("target") && !url.searchParams.get("password"));

    if (isDocRequest) {
      return new Response(generateDocsHtml(url.origin), {
        headers: {
          ...corsHeaders,
          "Content-Type": "text/html; charset=utf-8",
        },
      });
    }

    // 3. Endpoint public de santé (Health Check)
    if (url.pathname === "/api/health" || url.pathname === "/health") {
      return new Response(
        JSON.stringify({
          status: "ok",
          service: "VM Linux API Gateway",
          worker: "VMAI",
          timestamp: new Date().toISOString(),
        }),
        { headers: { ...corsHeaders, "Content-Type": "application/json; charset=utf-8" } }
      );
    }

    // 4. Extraction & Validation de l'Authentification
    let userPassword = url.searchParams.get("password") || url.searchParams.get("pin");
    
    const authHeader = request.headers.get("Authorization");
    if (authHeader) {
      const match = authHeader.match(/^Bearer\s+(.*)$/i);
      if (match) {
        userPassword = match[1].trim();
      } else if (!userPassword) {
        userPassword = authHeader.trim();
      }
    }

    let requestBody = null;
    let requestBodyText = null;

    if (request.method === "POST" || request.method === "PUT") {
      try {
        requestBodyText = await request.text();
        if (requestBodyText) {
          try {
            requestBody = JSON.parse(requestBodyText);
            if (!userPassword && requestBody) {
              userPassword = requestBody.password || requestBody.pin;
            }
          } catch (e) {
            // Pas du JSON pur
          }
        }
      } catch (e) {
        // Corps non lisible
      }
    }

    const expectedPassword = env.PASS;

    if (!expectedPassword || !userPassword || userPassword !== expectedPassword) {
      return new Response(
        JSON.stringify({
          ok: false,
          authenticated: false,
          error: "Mot de passe incorrect ou manquant.",
          hint: "Fournissez le mot de passe via l'en-tête 'Authorization: Bearer <PASS>', le paramètre URL '?password=<PASS>', ou dans le corps JSON '{\"password\": \"<PASS>\"}'.",
          docs_url: `${url.origin}/docs`
        }, null, 2),
        { status: 401, headers: { ...corsHeaders, "Content-Type": "application/json; charset=utf-8" } }
      );
    }

    // 5. Récupération de l'état des tunnels depuis GitHub
    let tunnelsData = null;
    let linuxTunnel = null;
    try {
      const timestamp = Date.now();
      const resLinux = await fetch(`${GITHUB_RAW_BASE}tunnels-linux.json?_t=${timestamp}`, {
        headers: { "User-Agent": "Cloudflare-VM-Gateway" },
        cf: { cacheTtl: 0, cacheEverything: false }
      });
      if (resLinux.ok) {
        linuxTunnel = await resLinux.json();
      }
    } catch (e) {
      console.warn("Échec récupération tunnels-linux.json:", e);
    }

    const linuxPrimaryUrl = linuxTunnel?.cloudflare || linuxTunnel?.primary || linuxTunnel?.lhrLife || null;

    // 6. Redirection directe si demandée
    const target = url.searchParams.get("target")?.toLowerCase();
    if (target === "linux" && linuxPrimaryUrl) {
      return Response.redirect(linuxPrimaryUrl, 302);
    }

    // 7. Route générale : Tunnels et Statut Global (/ ou /api/tunnels)
    if (url.pathname === "/" || url.pathname === "/api/tunnels") {
      let androidTunnel = null;
      try {
        const resAndroid = await fetch(`${GITHUB_RAW_BASE}tunnels-android.json?_t=${Date.now()}`, {
          headers: { "User-Agent": "Cloudflare-VM-Gateway" },
          cf: { cacheTtl: 0, cacheEverything: false }
        });
        if (resAndroid.ok) androidTunnel = await resAndroid.json();
      } catch (e) {}

      return new Response(
        JSON.stringify({
          authenticated: true,
          status: linuxTunnel?.status || "online",
          linux: linuxTunnel || { name: "Ubuntu Linux Desktop", primary: linuxPrimaryUrl, status: "online" },
          android: androidTunnel,
          primary: linuxPrimaryUrl,
          cloudflare: linuxPrimaryUrl,
          docs_url: `${url.origin}/docs`,
          updated_at: linuxTunnel?.updated_at || new Date().toISOString()
        }, null, 2),
        { headers: { ...corsHeaders, "Content-Type": "application/json; charset=utf-8" } }
      );
    }

    // 8. Relais d'Interaction API Linux (/api/linux/* ou /api/v1/*)
    if (url.pathname.startsWith("/api/linux") || url.pathname.startsWith("/api/v1")) {
      // 8a. Statut rapide du tunnel Linux
      if (url.pathname === "/api/linux/status") {
        return new Response(
          JSON.stringify({
            ok: true,
            status: linuxPrimaryUrl ? "online" : "offline",
            name: "Ubuntu Linux Desktop",
            tunnel_url: linuxPrimaryUrl,
            updated_at: linuxTunnel?.updated_at || null,
            gateway: url.origin,
            docs_url: `${url.origin}/docs`
          }, null, 2),
          { headers: { ...corsHeaders, "Content-Type": "application/json; charset=utf-8" } }
        );
      }

      // 8b. Vérification si le tunnel Linux est disponible
      if (!linuxPrimaryUrl) {
        return new Response(
          JSON.stringify({
            ok: false,
            error: "La VM Linux est en cours de démarrage ou le tunnel n'est pas encore publié.",
            status: "starting"
          }, null, 2),
          { status: 503, headers: { ...corsHeaders, "Content-Type": "application/json; charset=utf-8" } }
        );
      }

      // 8c. Transmission directe de la requête au runner Linux via le tunnel Cloudflare
      const targetVmUrl = new URL(url.pathname, linuxPrimaryUrl);
      targetVmUrl.search = url.search;

      try {
        const forwardHeaders = new Headers();
        for (const [key, value] of request.headers.entries()) {
          // Filtrer les en-têtes hop-by-hop
          if (!['host', 'cf-ray', 'cf-connecting-ip', 'cf-visitor', 'x-forwarded-for', 'x-forwarded-proto'].includes(key.toLowerCase())) {
            forwardHeaders.set(key, value);
          }
        }
        forwardHeaders.set("X-Forwarded-Host", url.host);

        const vmResponse = await fetch(targetVmUrl.toString(), {
          method: request.method,
          headers: forwardHeaders,
          body: requestBodyText || undefined,
          redirect: "follow"
        });

        // Si le démon sur le runner répond (y compris un 404 légitime comme fichier non trouvé)
        const isFromDaemon = vmResponse.headers.has("x-linux-daemon") || 
                             vmResponse.headers.get("content-type")?.includes("application/json");

        if (isFromDaemon || (vmResponse.status !== 404 && vmResponse.status !== 502)) {
          const respHeaders = new Headers(vmResponse.headers);
          respHeaders.set("Access-Control-Allow-Origin", "*");
          respHeaders.set("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS");
          respHeaders.set("Access-Control-Allow-Headers", "Content-Type, Authorization");

          return new Response(vmResponse.body, {
            status: vmResponse.status,
            headers: respHeaders
          });
        }

        // Si le runner actuel ne dispose pas encore du micro-démon API (cycle en cours avant bascule)
        return new Response(
          JSON.stringify({
            ok: false,
            status: "daemon_pending_cycle",
            message: "Le tunnel Linux est actif et opérationnel, mais le micro-démon d'API local sera activé lors du prochain relais automatique.",
            desktop_url: linuxPrimaryUrl,
            tunnel_info: linuxTunnel,
            command_preview: requestBody?.command || null,
            help: "Vous pouvez accéder directement au bureau graphique via l'URL ci-dessus."
          }, null, 2),
          { status: 200, headers: { ...corsHeaders, "Content-Type": "application/json; charset=utf-8" } }
        );

      } catch (err) {
        return new Response(
          JSON.stringify({
            ok: false,
            error: "Erreur de communication avec le tunnel Linux: " + err.message,
            tunnel_url: linuxPrimaryUrl
          }, null, 2),
          { status: 502, headers: { ...corsHeaders, "Content-Type": "application/json; charset=utf-8" } }
        );
      }
    }

    // 9. Point de terminaison non trouvé
    return new Response(
      JSON.stringify({
        ok: false,
        error: `Point de terminaison non reconnu : ${url.pathname}`,
        available_endpoints: [
          "/docs",
          "/api/health",
          "/api/tunnels",
          "/api/linux/status",
          "/api/linux/system",
          "/api/linux/exec",
          "/api/linux/screenshot",
          "/api/linux/mouse",
          "/api/linux/keyboard",
          "/api/linux/clipboard",
          "/api/linux/files",
          "/api/linux/files/read",
          "/api/linux/files/write",
          "/api/linux/apps/launch",
          "/api/linux/processes"
        ],
        docs_url: `${url.origin}/docs`
      }, null, 2),
      { status: 404, headers: { ...corsHeaders, "Content-Type": "application/json; charset=utf-8" } }
    );
  }
};

/**
 * Générateur de la documentation minimaliste sur fond blanc
 */
function generateDocsHtml(origin) {
  return `<!DOCTYPE html>
<html lang="fr">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Documentation API Linux VM — JeanHug/AI</title>
  <style>
    :root {
      --bg: #ffffff;
      --text: #111827;
      --text-muted: #4b5563;
      --border: #e5e7eb;
      --code-bg: #f9fafb;
      --accent: #2563eb;
      --accent-hover: #1d4ed8;
      --success: #059669;
      --error: #dc2626;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background: var(--bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      line-height: 1.6;
      padding: 40px 20px;
    }
    .container {
      max-width: 960px;
      margin: 0 auto;
    }
    header {
      border-bottom: 1px solid var(--border);
      padding-bottom: 24px;
      margin-bottom: 32px;
    }
    h1 {
      font-size: 26px;
      font-weight: 700;
      letter-spacing: -0.02em;
      margin-bottom: 8px;
    }
    .badge {
      display: inline-block;
      font-size: 12px;
      font-weight: 600;
      padding: 2px 8px;
      border-radius: 4px;
      background: #eff6ff;
      color: var(--accent);
      border: 1px solid #bfdbfe;
      margin-bottom: 12px;
    }
    p.lead {
      font-size: 15px;
      color: var(--text-muted);
    }
    section {
      margin-bottom: 40px;
    }
    h2 {
      font-size: 19px;
      font-weight: 600;
      margin-bottom: 16px;
      padding-bottom: 6px;
      border-bottom: 1px solid var(--border);
    }
    h3 {
      font-size: 15px;
      font-weight: 600;
      margin: 20px 0 8px 0;
    }
    pre, code {
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
      font-size: 13px;
    }
    code {
      background: var(--code-bg);
      padding: 2px 6px;
      border-radius: 4px;
      border: 1px solid var(--border);
    }
    pre {
      background: var(--code-bg);
      padding: 14px 16px;
      border-radius: 6px;
      border: 1px solid var(--border);
      overflow-x: auto;
      margin: 10px 0 16px 0;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      margin: 14px 0 24px 0;
      font-size: 14px;
    }
    th, td {
      text-align: left;
      padding: 10px 12px;
      border: 1px solid var(--border);
    }
    th {
      background: var(--code-bg);
      font-weight: 600;
    }
    .method {
      font-weight: 700;
      font-size: 11px;
      padding: 2px 6px;
      border-radius: 3px;
    }
    .get { background: #ecfdf5; color: #047857; }
    .post { background: #eff6ff; color: #1d4ed8; }
    .delete { background: #fef2f2; color: #b91c1c; }

    /* Console Interactive */
    .console-card {
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 20px;
      background: #ffffff;
      box-shadow: 0 1px 3px rgba(0,0,0,0.03);
    }
    .form-group {
      margin-bottom: 14px;
    }
    label {
      display: block;
      font-size: 13px;
      font-weight: 600;
      margin-bottom: 6px;
    }
    input, select, textarea {
      width: 100%;
      padding: 8px 12px;
      font-size: 14px;
      border: 1px solid var(--border);
      border-radius: 6px;
      background: #fff;
      color: var(--text);
      font-family: inherit;
    }
    input:focus, select:focus, textarea:focus {
      outline: none;
      border-color: var(--accent);
      box-shadow: 0 0 0 2px rgba(37, 99, 235, 0.1);
    }
    button.btn {
      background: var(--accent);
      color: #fff;
      border: none;
      padding: 9px 18px;
      border-radius: 6px;
      font-size: 14px;
      font-weight: 600;
      cursor: pointer;
      transition: background 0.15s;
    }
    button.btn:hover {
      background: var(--accent-hover);
    }
    .response-box {
      margin-top: 16px;
      display: none;
    }
    .screenshot-preview {
      max-width: 100%;
      border: 1px solid var(--border);
      border-radius: 6px;
      margin-top: 10px;
    }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <span class="badge">Cloudflare Worker Gateway • v1.0</span>
      <h1>API d'Interaction Linux VM</h1>
      <p class="lead">Passerelle unifiée et sécurisée pour interagir avec le bureau Linux Ubuntu (exécution de commandes, capture d'écran, souris/clavier, fichiers et applications).</p>
    </header>

    <!-- 1. Authentification -->
    <section id="auth">
      <h2>1. Authentification</h2>
      <p>Toutes les requêtes (sauf <code>/api/health</code> et <code>/docs</code>) exigent votre mot de passe secret (variable <code>PASS</code> configurée dans votre compte Cloudflare).</p>
      
      <h3>3 méthodes acceptées :</h3>
      <ul>
        <li><strong>En-tête HTTP (Recommandé) :</strong> <code>Authorization: Bearer &lt;VOTRE_MOT_DE_PASSE&gt;</code></li>
        <li><strong>Paramètre URL :</strong> <code>?password=&lt;VOTRE_MOT_DE_PASSE&gt;</code></li>
        <li><strong>Corps JSON (POST/PUT) :</strong> <code>{ "password": "&lt;VOTRE_MOT_DE_PASSE&gt;", ... }</code></li>
      </ul>
    </section>

    <!-- 2. Points de terminaison -->
    <section id="endpoints">
      <h2>2. Référence des Endpoints</h2>
      <table>
        <thead>
          <tr>
            <th>Méthode</th>
            <th>Endpoint</th>
            <th>Description</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td><span class="method get">GET</span></td>
            <td><code>/api/health</code></td>
            <td>Statut public de la passerelle (sans authentification).</td>
          </tr>
          <tr>
            <td><span class="method get">GET</span></td>
            <td><code>/api/tunnels</code></td>
            <td>Retourne les URLs en direct de toutes les VMs (Linux & Android).</td>
          </tr>
          <tr>
            <td><span class="method get">GET</span></td>
            <td><code>/api/linux/status</code></td>
            <td>Statut du tunnel Linux et date de dernière synchronisation.</td>
          </tr>
          <tr>
            <td><span class="method get">GET</span></td>
            <td><code>/api/linux/system</code></td>
            <td>Informations système (CPU, mémoire, disque, fenêtre active).</td>
          </tr>
          <tr>
            <td><span class="method post">POST</span></td>
            <td><code>/api/linux/exec</code></td>
            <td>Exécute une commande Bash dans le conteneur Ubuntu Desktop.</td>
          </tr>
          <tr>
            <td><span class="method get">GET</span></td>
            <td><code>/api/linux/screenshot</code></td>
            <td>Capture d'écran de l'affichage X11 (PNG direct ou JSON base64).</td>
          </tr>
          <tr>
            <td><span class="method post">POST</span></td>
            <td><code>/api/linux/mouse</code></td>
            <td>Contrôle souris : clic, déplacement, clic droit, molette.</td>
          </tr>
          <tr>
            <td><span class="method post">POST</span></td>
            <td><code>/api/linux/keyboard</code></td>
            <td>Contrôle clavier : saisie de texte fluide ou frappe de touches.</td>
          </tr>
          <tr>
            <td><span class="method get">GET</span>/<span class="method post">POST</span></td>
            <td><code>/api/linux/clipboard</code></td>
            <td>Lecture ou modification du presse-papier X11.</td>
          </tr>
          <tr>
            <td><span class="method get">GET</span></td>
            <td><code>/api/linux/files?path=...</code></td>
            <td>Liste le contenu d'un répertoire dans <code>/home/kasm-user</code>.</td>
          </tr>
          <tr>
            <td><span class="method get">GET</span></td>
            <td><code>/api/linux/files/read?path=...</code></td>
            <td>Lit le contenu textuel d'un fichier.</td>
          </tr>
          <tr>
            <td><span class="method post">POST</span></td>
            <td><code>/api/linux/files/write</code></td>
            <td>Écrit ou crée un fichier sur le disque persistant.</td>
          </tr>
          <tr>
            <td><span class="method post">POST</span></td>
            <td><code>/api/linux/apps/launch</code></td>
            <td>Lance une application graphique (Chrome, Terminal, etc.).</td>
          </tr>
          <tr>
            <td><span class="method get">GET</span></td>
            <td><code>/api/linux/processes</code></td>
            <td>Liste les processus actifs triés par consommation CPU.</td>
          </tr>
        </tbody>
      </table>
    </section>

    <!-- 3. Exemples de Payloads -->
    <section id="examples">
      <h2>3. Exemples d'Appels JSON</h2>

      <h3>Exécuter une commande Bash :</h3>
      <pre>POST /api/linux/exec
Content-Type: application/json
Authorization: Bearer &lt;VOTRE_MOT_DE_PASSE&gt;

{
  "command": "python3 -c 'import platform; print(platform.uname())'",
  "cwd": "/home/kasm-user",
  "timeout": 30
}</pre>

      <h3>Contrôler la souris :</h3>
      <pre>POST /api/linux/mouse
Content-Type: application/json
Authorization: Bearer &lt;VOTRE_MOT_DE_PASSE&gt;

{
  "action": "click",
  "x": 450,
  "y": 300
}</pre>

      <h3>Saisir du texte au clavier :</h3>
      <pre>POST /api/linux/keyboard
Content-Type: application/json
Authorization: Bearer &lt;VOTRE_MOT_DE_PASSE&gt;

{
  "text": "echo 'Hello from AI Agent!'"
}</pre>
    </section>

    <!-- 4. Console de Test Interactive -->
    <section id="console">
      <h2>4. Console de Test Interactive</h2>
      <div class="console-card">
        <div class="form-group">
          <label for="cfgPassword">Mot de passe secret :</label>
          <input type="password" id="cfgPassword" value="" placeholder="Entrez obligatoirement votre mot de passe pour tester" autocomplete="off" required>
        </div>

        <div class="form-group">
          <label for="cfgEndpoint">Endpoint à tester :</label>
          <select id="cfgEndpoint" onchange="handleEndpointChange()">
            <option value="GET|/api/tunnels">GET /api/tunnels (Tunnels actifs)</option>
            <option value="GET|/api/linux/status">GET /api/linux/status (Statut Linux)</option>
            <option value="GET|/api/linux/system">GET /api/linux/system (Métriques système)</option>
            <option value="POST|/api/linux/exec">POST /api/linux/exec (Commande Bash)</option>
            <option value="GET|/api/linux/screenshot">GET /api/linux/screenshot (Capture d'écran)</option>
            <option value="GET|/api/linux/clipboard">GET /api/linux/clipboard (Presse-papier)</option>
            <option value="GET|/api/linux/processes">GET /api/linux/processes (Processus)</option>
            <option value="GET|/api/linux/files?path=/home/kasm-user/Desktop">GET /api/linux/files (Bureau)</option>
          </select>
        </div>

        <div class="form-group" id="bodyGroup" style="display: none;">
          <label for="cfgBody">Corps JSON (POST) :</label>
          <textarea id="cfgBody" rows="4" style="font-family: monospace; font-size: 13px;">{
  "command": "uname -a && free -h"
}</textarea>
        </div>

        <button type="button" class="btn" onclick="executeTest()">Envoyer la requête</button>

        <div class="response-box" id="responseBox">
          <label>Réponse du serveur (<span id="resStatus"></span>) :</label>
          <pre id="resContent" style="max-height: 380px; overflow-y: auto;"></pre>
          <img id="imgPreview" class="screenshot-preview" style="display: none;" alt="Aperçu écran">
        </div>
      </div>
    </section>

    <!-- 5. Guide Agent IA -->
    <section id="ai-guide">
      <h2>5. Guide d'Intégration pour Agents IA</h2>
      <p>Vous pouvez copier ce bloc de contexte directement dans les instructions de votre agent IA (Gemini, Claude, GPT) :</p>
      <pre>Tu disposes d'un accès complet à une machine Linux Ubuntu via l'API REST suivante :
Base URL : ${origin}
Mot de passe : &lt;VOTRE_MOT_DE_PASSE&gt; (en-tête Authorization: Bearer &lt;VOTRE_MOT_DE_PASSE&gt;)

- Pour exécuter des commandes : POST /api/linux/exec { "command": "votre commande" }
- Pour voir l'écran : GET /api/linux/screenshot?format=base64
- Pour interagir avec la souris : POST /api/linux/mouse { "action": "click", "x": 100, "y": 200 }
- Pour taper au clavier : POST /api/linux/keyboard { "text": "texte" } ou { "key": "Return" }
- Pour gérer les fichiers : GET/POST /api/linux/files</pre>
    </section>
  </div>

  <script>
    function handleEndpointChange() {
      const val = document.getElementById('cfgEndpoint').value;
      const [method] = val.split('|');
      document.getElementById('bodyGroup').style.display = method === 'POST' ? 'block' : 'none';
    }

    async function executeTest() {
      const pwd = document.getElementById('cfgPassword').value.trim();
      const val = document.getElementById('cfgEndpoint').value;
      const [method, endpoint] = val.split('|');
      const bodyText = document.getElementById('cfgBody').value;
      const resBox = document.getElementById('responseBox');
      const resStatus = document.getElementById('resStatus');
      const resContent = document.getElementById('resContent');
      const imgPreview = document.getElementById('imgPreview');

      resBox.style.display = 'block';
      imgPreview.style.display = 'none';

      if (!pwd && endpoint !== '/api/health') {
        resStatus.textContent = 'Mot de passe requis';
        resContent.textContent = 'Veuillez renseigner votre mot de passe secret dans le champ ci-dessus pour pouvoir envoyer cette requête.';
        return;
      }

      resStatus.textContent = 'En cours...';
      resContent.textContent = 'Requête transmise à la passerelle Cloudflare Worker...';

      try {
        const fullUrl = '${origin}' + endpoint;
        const options = {
          method: method,
          headers: {
            'Authorization': 'Bearer ' + pwd
          }
        };

        if (method === 'POST') {
          options.headers['Content-Type'] = 'application/json';
          options.body = bodyText;
        }

        const res = await fetch(fullUrl, options);
        resStatus.textContent = 'HTTP ' + res.status + ' ' + res.statusText;

        const contentType = res.headers.get('content-type') || '';
        if (contentType.includes('image/')) {
          const blob = await res.blob();
          const imgUrl = URL.createObjectURL(blob);
          imgPreview.src = imgUrl;
          imgPreview.style.display = 'block';
          resContent.textContent = '[Image PNG reçue avec succès : ' + blob.size + ' octets]';
        } else {
          const data = await res.json().catch(() => res.text());
          resContent.textContent = typeof data === 'string' ? data : JSON.stringify(data, null, 2);
        }
      } catch (err) {
        resStatus.textContent = 'Erreur';
        resContent.textContent = err.message;
      }
    }
  </script>
</body>
</html>`;
}
