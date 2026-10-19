// Proxy CORS minimal vers l'API d'inférence Roboflow.
// Secrets attendus : ROBOFLOW_API_KEY (obligatoire, jeton historique),
// RF_TOKEN_1/2/3 (jetons supplémentaires de la même organisation), ACCESS_CODE.
// Les jetons servent au routage : dès qu'une tentative échoue, la suivante
// utilise le jeton suivant (voir roboflowTokens / acrossTokens).
// Aucune clé n'est exposée au navigateur.
//
// Chaque modèle est décrit par : block (type de step Workflow), family
// (famille de réglages), vf/vv (champ+valeur de version), task (task_type,
// null = absent), backend (Qwen), out (champ de sortie du step).
// Tous les blocs tiers utilisent "rf_key:account" -> facturé en crédits
// Roboflow (le bloc glm_ocr est auto-hébergé, sans champ api_key).

const ROBOFLOW_URL = "https://serverless.roboflow.com/workflows/run";

/* Génération d'image via Workers AI (compte Cloudflare du projet) :
   gratuit dans la limite du quota journalier de neurons du plan Free. */
const CF_ACCOUNT_ID = "aeb92cddbb638d207038daf831a5aab2";
const IMG_MODELS = {
  "flux-2-klein-9b": "black-forest-labs/flux-2-klein-9b",
  "flux-2-klein-4b": "black-forest-labs/flux-2-klein-4b",
};

/* Génération d'image via Hugging Face ZeroGPU (Spaces officiels, appels
   anonymes — aucun token n'est stocké ni transmis). Quota HF : ~180 s de
   GPU/j par IP. Le Worker sert de relais : le HTML reste une simple UI. */
const HF_IMG_MODELS = {
  "hf-flux-2-klein-9b": {
    label: "FLUX.2 klein 9B (Space BFL, variante KV)",
    host: "https://black-forest-labs-flux-klein-9b-kv.hf.space",
    api: "generate",
    bounds: { w: [256, 1024, 8], h: [256, 1024, 8], s: [1, 8, 4] },
    data: (p, w, h, s, g, mode, params, refs) => [p, refs || [], 0, true, w, h, s, false],
  },
  "hf-flux-2-klein-4b": {
    label: "FLUX.2 klein 4B (Space officiel BFL)",
    host: "https://black-forest-labs-flux-2-klein-4b.hf.space",
    api: "infer",
    bounds: { w: [256, 1024, 8], h: [256, 1024, 8], s: [1, 50, 4] },
    data: (p, w, h, s, g, mode, params, refs) => [p, refs || [], "Distilled (4 steps)", 0, true, w, h,
      s, g === undefined ? 1.0 : g, false],
  },
  /* FLUX.3 Klein : Space officiel Black Forest Labs, appelable SANS jeton.
     Signature du Space : prompt, images de référence, mode, graine, tirage
     aléatoire, largeur, hauteur, étapes, guidance, upsampling du prompt.
     Le 4K demanderait un compte PRO (ZeroGPU anonyme plafonne la durée GPU) :
     on s'arrête donc à 2048 px de côté. */
  "hf-flux-3-klein": {
    label: "FLUX.3 Klein (Space officiel BFL)",
    host: "https://black-forest-labs-flux-3-image-klein.hf.space",
    api: "generate",
    bounds: { w: [512, 2048, 64], h: [512, 2048, 64], s: [1, 50, 16] },
    modes: ["Distilled (16 steps)", "Base (50 steps)"],
    /* `upsample` n'est activé que si le réglage « Amélioration du prompt »
       vaut oui : par défaut le prompt part TEL QUEL. */
    data: (p, w, h, s, g, mode, params, refs) => [p, refs || [], mode || "Distilled (16 steps)", 0, true, w, h,
      s === undefined ? 16 : s, g === undefined ? 1.0 : g,
      !!(params && (params.upsample === true || params.upsample === "oui"))],
  },
  "ideogram-v4": {
    label: "Ideogram v4 (Space officiel)",
    host: "https://ideogram-ai-ideogram4.hf.space",
    api: "generate",
    bounds: { w: [512, 2048, 64], h: [512, 2048, 64], s: null },
    modes: ["Turbo · 12 steps", "Default · 20 steps", "Quality · 48 steps"],
    data: (p, w, h, s, g, mode) => [p, mode, "Ideogram (remote)", w, h, 0, true],
  },
};

/* --- Vidéo : Wan 2.2 -------------------------------------------------------
   Deux voies pour un même modèle.
   1. Pollinations (`alibaba/wan-2.2-fast`) : vraie génération texte→vidéo,
      facturée en Pollen — donc inutilisable quand le solde est à zéro.
   2. Space public Wan 2.2 (ZeroGPU, gratuit) : il ne sait qu'animer une image,
      alors le Worker fabrique d'abord l'image de départ à partir du prompt
      (FLUX.2 klein 4B) puis la lui envoie. Deux appels, aucun crédit.
   Le champ `via` de la réponse dit laquelle des deux a produit la vidéo. */
const HF_VID_MODELS = {
  "wan-2-2-t2v": {
    label: "Wan 2.2 (Space public, anime une image de départ)",
    host: "https://prithivmlmods-wan2-2-fast.hf.space",
    api: "generate_video",
  },
};
const POLL_VID_MAP = { "wan-2-2-t2v": "alibaba/wan-2.2-fast" };

function b64encode(buf) {
  const u = new Uint8Array(buf);
  let bin = "";
  for (let i = 0; i < u.length; i += 32768) {
    bin += String.fromCharCode.apply(null, u.subarray(i, i + 32768));
  }
  return btoa(bin);
}

function findGradioFile(o) {
  if (Array.isArray(o)) {
    for (const v of o) { const r = findGradioFile(v); if (r) return r; }
  } else if (o && typeof o === "object") {
    if (typeof o.url === "string" && o.url.startsWith("http")) return o;
    for (const v of Object.values(o)) { const r = findGradioFile(v); if (r) return r; }
  }
  return null;
}

function stepInt(v, bounds, dflt) {
  const [mn, mx, st] = bounds;
  let n = int(v, mn, mx);
  if (n === undefined) n = dflt;
  return Math.round(n / st) * st;
}

/* Dépose une image dans le Space (protocole Gradio) et rend son chemin.
   Le Space attend ensuite {"path": …, "meta": {"_type": "gradio.FileData"}}
   dans `input_images`. Sans ce dépôt, l'image de référence n'existe pas pour
   lui : il ne peut donc pas l'éditer. */
async function hfUpload(cfg, b64, mime, env) {
  const bytes = b64ToBytes(String(b64 || ""));
  if (!bytes || !bytes.length) return null;
  const fd = new FormData();
  fd.append("files", new Blob([bytes], { type: mime || "image/png" }), "image.png");
  const headers = {};
  if (env && env.HF_TOKEN) headers.Authorization = "Bearer " + String(env.HF_TOKEN).trim();
  let r;
  try { r = await fetch(cfg.host + "/gradio_api/upload", { method: "POST", headers, body: fd }); }
  catch (e) { return null; }
  if (!r.ok) return null;
  let j = null;
  try { j = await r.json(); } catch (e) { return null; }
  const path = Array.isArray(j) ? j[0] : (j && (j.files && j.files[0] && j.files[0].path));
  return path ? String(path) : null;
}
async function hfImage(cfg, ib, env) {
  const iprompt = String(ib.prompt ?? "").trim();
  if (!iprompt) return json({ error: "prompt vide" }, 400);
  if (iprompt.length > 4000) return json({ error: "prompt trop long (max 4000 caractères)" }, 400);
  const ip = ib.params && typeof ib.params === "object" ? ib.params : {};
  const w = stepInt(ip.width, cfg.bounds.w, cfg.bounds.w[1]);
  const h = stepInt(ip.height, cfg.bounds.h, cfg.bounds.h[1]);
  const s = cfg.bounds.s ? int(ip.num_steps, cfg.bounds.s[0], cfg.bounds.s[1]) : undefined;
  const g = num(ip.guidance, 0, 10);
  let mode;
  if (cfg.modes) {
    mode = cfg.modes.includes(ip.mode) ? ip.mode : cfg.modes[0];
  }
  /* Image de référence : on la dépose d'abord, puis on la passe au Space.
     Une seule image est transmise — celle que l'utilisateur a jointe. */
  let refs = [];
  if (ib && ib.image && cfg.refs !== false) {
    const rp = await hfUpload(cfg, ib.image, detectMime(String(ib.image)), env);
    if (rp) refs = [{ image: { path: rp, meta: { _type: "gradio.FileData" } }, caption: null }];
  }
  const payload = cfg.data(iprompt, w, h, s, g, mode, ip, refs);

  /* Sans jeton, ces Spaces tournent sur le quota ZeroGPU ANONYME (d'où les
     refus « quota » ou « durée GPU demandée trop longue » sur Ideogram).
     Si le secret HF_TOKEN existe, il est transmis : le quota est alors celui
     du compte. Absent, rien ne change. */
  const hfHeaders = { "Content-Type": "application/json" };
  if (env && env.HF_TOKEN) hfHeaders.Authorization = "Bearer " + String(env.HF_TOKEN).trim();
  let post;
  try {
    post = await fetch(`${cfg.host}/gradio_api/call/${cfg.api}`, {
      method: "POST",
      headers: hfHeaders,
      body: JSON.stringify({ data: payload }),
    });
  } catch (err) {
    return json({ error: `Space HF injoignable : ${err.message}` }, 502);
  }
  if (!post.ok) return json({ error: `Space HF HTTP ${post.status}` }, 502);
  let evId;
  try { evId = (await post.json()).event_id; } catch { return json({ error: "réponse du Space HF invalide" }, 502); }
  if (!evId) return json({ error: "pas d'event_id renvoyé par le Space HF" }, 502);

  let sse;
  HFV_STAGE = "ouverture du flux";
  try {
    sse = await fetch(`${cfg.host}/gradio_api/call/${cfg.api}/${evId}`, { headers: env && env.HF_TOKEN ? { Authorization: "Bearer " + String(env.HF_TOKEN).trim() } : {} });
  } catch (err) {
    return json({ error: `flux SSE HF injoignable : ${err.message}` }, 502);
  }
  const txt = await sse.text();
  if (/event:\s*error/.test(txt)) {
    let msg = `échec de génération sur le Space HF (${cfg.label}) — quota ZeroGPU anonyme peut-être épuisé pour cette IP`;
    const dm = txt.match(/data:\s*(\{.*\})/);
    if (dm) {
      try { const j = JSON.parse(dm[1]); if (j && j.error) msg = String(j.error).slice(0, 300); } catch { /* brut */ }
    } else {
      msg += " · " + txt.replace(/\s+/g, " ").slice(0, 140);
    }
    return json({ error: msg }, 502);
  }
  const dm2 = txt.match(/^data:\s*(\[.*\])\s*$/m);
  let file = null;
  if (dm2) { try { file = findGradioFile(JSON.parse(dm2[1])); } catch { /* null */ } }
  if (!file) return json({ error: "aucun fichier image dans la réponse du Space HF" }, 502);

  let fr;
  try { fr = await fetch(file.url); } catch (err) { return json({ error: `téléchargement de l'image HF : ${err.message}` }, 502); }
  if (!fr.ok) return json({ error: `téléchargement de l'image HF HTTP ${fr.status}` }, 502);
  const buf = await fr.arrayBuffer();
  let mime = fr.headers.get("content-type") || "";
  if (!mime.startsWith("image/")) {
    mime = /\.webp(\?|$)/.test(file.url) ? "image/webp" : "image/png";
  }
  return json({ image: b64encode(buf), mime });
}

/* Fabrique l'image de départ d'une vidéo avec le modèle d'image choisi dans
   les réglages de l'application. Chaque canal répond dans son propre format :
   on renvoie toujours { image, mime }. */
/* Les modèles d'image Cloudflare (klein 9B/4B) existent aussi en Space HF :
   pour l'image de départ d'une vidéo, on passe par le Space, plus rapide. */
const POSTER_ALIAS = { "flux-2-klein-9b": "hf-flux-2-klein-9b", "flux-2-klein-4b": "hf-flux-2-klein-4b" };
async function posterImage(env, model, prompt) {
  model = POSTER_ALIAS[model] || model;
  const ib = { prompt, params: { width: 768, height: 768 } };
  if (HF_IMG_MODELS[model]) {
    const r = await hfImage(HF_IMG_MODELS[model], ib, env);
    const j = await r.json();
    if (j && j.image) return { image: j.image, mime: j.mime || "image/png" };
    throw new Error(String((j && j.error) || "image vide"));
  }
  if (MS_IMG[model]) {
    const r = await msImage(env, MS_IMG[model], ib);
    const j = await r.json();
    if (j && j.image) return { image: j.image, mime: j.mime || "image/png" };
    throw new Error(String((j && j.error) || "image vide"));
  }
  if (POLL_IMG_MAP[model]) {
    if (!env.POLLINATIONS_API_KEY) throw new Error("secret POLLINATIONS_API_KEY manquant");
    const up = await fetch(POLL_URL + "/v1/images/generations", {
      method: "POST",
      headers: { "Content-Type": "application/json", Authorization: "Bearer " + env.POLLINATIONS_API_KEY },
      body: JSON.stringify({ model: POLL_IMG_MAP[model], prompt, size: "1024x1024", response_format: "b64_json" }),
    });
    const t = await up.text();
    let j = null; try { j = JSON.parse(t); } catch { /* brut */ }
    if (!up.ok) throw new Error(pollErrorMessage(up.status, t));
    const b64 = j && j.data && j.data[0] && j.data[0].b64_json;
    if (!b64) throw new Error("aucune image dans la réponse Pollinations");
    return { image: b64, mime: detectMime(b64) };
  }
  throw new Error("modèle d'image de départ inconnu : " + String(model));
}

/* Fabrique une vidéo Wan 2.2 : image de départ puis animation.
   `opts.image` = image jointe par l'utilisateur (elle sert telle quelle, pas
   de génération) ; `opts.posterModel` = modèle qui fabrique l'image de départ. */
let HFV_STAGE = "départ";
async function hfVideo(env, prompt, dur, opts) {
  const cfg = HF_VID_MODELS["wan-2-2-t2v"];
  opts = opts || {};
  let poster = "", posterMime = "image/png", posterErr = "", starter = "worker";
  HFV_STAGE = "image de départ";
  if (opts.image) {
    /* L'image vient de l'utilisateur : on l'anime directement. */
    poster = String(opts.image);
    posterMime = String(opts.imageMime || "image/png");
    starter = "user";
  }
  const wanted = String(opts.posterModel || "flux-2-klein-4b");
  if (!poster) {
    try {
      const a = await posterImage(env, wanted, prompt);
      poster = a.image; posterMime = a.mime;
    } catch (err) { posterErr = wanted + " : " + String((err && err.message) || err); }
  }
  if (!poster && wanted !== "flux-2-klein-4b") {
    /* le choix des réglages a échoué : on retombe sur le modèle par défaut */
    try {
      const a = await posterImage(env, "flux-2-klein-4b", prompt);
      poster = a.image; posterMime = a.mime; posterErr = "";
    } catch (err) { posterErr = (posterErr ? posterErr + " · " : "") + "flux-2-klein-4b : " + String((err && err.message) || err); }
  }
  if (!poster) {
    /* dernier recours pour l'image de départ : Qwen Image 2.1 Turbo */
    try {
      const a = await posterImage(env, "qwen-image-2-1-turbo", prompt);
      poster = a.image; posterMime = a.mime; posterErr = "";
    } catch (err2) { posterErr = (posterErr ? posterErr + " · " : "") + String((err2 && err2.message) || err2); }
  }
  if (!poster) return { error: "aucune image de départ n'a pu être produite (" + posterErr + ")" };

  HFV_STAGE = "envoi au Space";
  /* `dur` part en NOMBRE : le Space lève un TypeError si la durée arrive en
     chaîne (son code fait de l'arithmétique dessus). Le nombre d'étapes, lui,
     reste une chaîne — c'est ce que son interface envoie. */
  const payload = ["data:" + posterMime + ";base64," + poster, prompt, "4", "", Number(dur), 4.0, 3.0, 0, true];
  /* Le jeton peut venir de trois endroits, dans cet ordre : celui que
     l'utilisateur a collé dans ses réglages (il reste sur son appareil), le
     secret du Worker, puis rien — c'est-à-dire le quota anonyme, très court. */
  const hfTok = String((opts && opts.hfToken) || (env && (env.HUGGINGFACE_TOKEN || env.HF_TOKEN)) || "").trim();
  const hfHeaders = { "Content-Type": "application/json" };
  if (hfTok) hfHeaders.Authorization = "Bearer " + hfTok;
  let post;
  try {
    post = await fetch(`${cfg.host}/gradio_api/call/${cfg.api}`, {
      method: "POST", headers: hfHeaders, body: JSON.stringify({ data: payload }),
    });
  } catch (err) { return { error: `Space vidéo HF injoignable : ${err.message}`, poster, posterMime }; }
  if (!post.ok) return { error: `Space vidéo HF HTTP ${post.status}`, poster, posterMime };
  let evId;
  try { evId = (await post.json()).event_id; } catch { /* null */ }
  if (!evId) return { error: "pas d'event_id renvoyé par le Space vidéo", poster, posterMime };
  let sse;
  try {
    sse = await fetch(`${cfg.host}/gradio_api/call/${cfg.api}/${evId}`, {
      headers: hfTok ? { Authorization: "Bearer " + hfTok } : {},
    });
  } catch (err) { return { error: `flux SSE vidéo injoignable : ${String((err && err.message) || err)}`, poster, posterMime }; }
  let txt = "";
  HFV_STAGE = "lecture du flux";
  try {
    txt = await sse.text();
  } catch (err) {
    /* La connexion SSE peut être coupée en plein milieu (quota, réseau) :
       on le dit clairement au lieu de laisser remonter un « TypeError ». */
    return { error: "la réponse du Space vidéo a été coupée en cours de route (quota gratuit ou réseau) — réessaie dans un instant", poster, posterMime };
  }
  if (/event:\s*error/.test(txt)) {
    /* Message brut d'abord : c'est lui qui dit la vraie raison. */
    let msg = "Space vidéo : " + txt.replace(/\s+/g, " ").slice(0, 240);
    const dm = txt.match(/data:\s*(\{.*\})/);
    if (dm) {
      try {
        const j = JSON.parse(dm[1]);
        const e = j && (j.error || j.detail || j.message);
        if (e) msg = typeof e === "string" ? e : JSON.stringify(e).slice(0, 300);
      } catch { /* on garde le texte brut */ }
    }
    if (/ZeroGPU quota|exceeded your/i.test(msg)) {
      const tm = msg.match(/Try again in ([0-9:]+)/i);
      msg = "Quota gratuit Hugging Face épuisé pour cette adresse IP"
        + (tm ? " — il se rouvre dans " + tm[1] : "")
        + ". Un jeton Hugging Face (secret HF_TOKEN côté Worker) augmente nettement ce quota.";
    }
    return { error: msg.slice(0, 320), poster, posterMime, starter };
  }
  HFV_STAGE = "analyse de la réponse";
  /* la vidéo arrive soit en data-URI base64, soit en fichier téléchargeable */
  const du = txt.match(/data:video\/[a-z0-9.+-]+;base64,([A-Za-z0-9+/=]+)/i);
  if (du) return { video: du[1], mime: "video/mp4", poster, posterMime, starter, duration: dur };
  const dm2 = txt.match(/^data:\s*(\[.*\])\s*$/m);
  if (dm2) {
    let file = null;
    try { file = findGradioFile(JSON.parse(dm2[1])); } catch { /* null */ }
    if (file) {
      try {
        const fr = await fetch(file.url);
        if (fr.ok) {
          const buf = await fr.arrayBuffer();
          if (buf.byteLength > 0) return { video: b64encode(buf), mime: "video/mp4", poster, posterMime, starter, duration: dur };
        }
      } catch { /* échec silencieux, message plus bas */ }
    }
  }
  return { error: "aucune vidéo dans la réponse du Space HF", poster, posterMime, starter };
}

/* Aucune exception ne doit sortir de la fabrication vidéo : l'application
   doit recevoir un message lisible, avec l'image de départ déjà produite. */
async function hfVideoSafe(env, prompt, dur, opts) {
  try {
    return await hfVideo(env, prompt, dur, opts);
  } catch (e) {
    return { error: "Space vidéo [" + HFV_STAGE + "] : " + String((e && e.message) || e) };
  }
}

const MODELS = {
  /* Les trois familles ci-dessous passent par un bloc dédié Roboflow ; `or`
     donne l'identifiant du MÊME modèle chez OpenRouter : si le bloc dédié
     échoue (500 « stopped generation », saturation, outil natif défaillant…),
     la requête est rejouée à l'identique via le bloc OpenRouter managé. */
  /* ---- Anthropic -------------------------------------------------- */
  "claude-opus-5-5":   { block: "roboflow_core/anthropic_claude@v5", family: "claude", or: "anthropic/claude-opus-5.5" },
  "claude-haiku-5-5": { block: "roboflow_core/anthropic_claude@v5", family: "claude", or: "anthropic/claude-haiku-5.5" },
  "claude-sonnet-5-5": { block: "roboflow_core/anthropic_claude@v5", family: "claude", or: "anthropic/claude-sonnet-5.5" },
  "claude-fable-5-1":  { block: "roboflow_core/anthropic_claude@v5", family: "claude", or: "anthropic/claude-fable-5.1" },
  /* ---- OpenAI ------------------------------------------------------ */
  "gpt-6-astra": { block: "roboflow_core/open_ai@v7", family: "openai", or: "openai/gpt-6-astra" },
  "gpt-6.1-sol": { block: "roboflow_core/open_ai@v7", family: "openai", or: "openai/gpt-6.1-sol" },
  "gpt-6-luna":  { block: "roboflow_core/open_ai@v7", family: "openai", or: "openai/gpt-6-luna" },
  /* ---- Google Gemini (v3 : rf_key:account ; pas de task_type) ------ */
  "gemini-3-8-flash":      { block: "roboflow_core/openrouter@v1", family: "router", vf: "model_id", vv: "google/gemini-3.8-flash" },
  "gemini-3-7-flash":      { block: "roboflow_core/google_gemini@v3", family: "gemini", task: null, vv: "gemini-3.7-flash", or: "google/gemini-3.7-flash" },
  "gemini-3-5-flash-lite": { block: "roboflow_core/google_gemini@v3", family: "gemini", task: null, vv: "gemini-3.5-flash-lite", or: "google/gemini-3.5-flash-lite" },
  "gemini-3-1-pro":        { block: "roboflow_core/google_gemini@v3", family: "gemini", task: null, vv: "gemini-3.1-pro-preview", or: "google/gemini-3.1-pro-preview" },
  /* Gemini 4 Argon (13/10/2026) — frontier 1 M entrée / 1 M sortie.
     Bloc dédié google_gemini v6 (v3/v4 rejettent le modèle : leur liste de
     model_version s'arrête à gemini-3.7-flash, v5 n'existe pas). */
  "gemini-4-argon":        { block: "roboflow_core/google_gemini@v6", family: "gemini", task: null, vv: "gemini-4-argon", or: "google/gemini-4-argon" },
  "gemini-2-5-pro":          { block: "roboflow_core/openrouter@v1", family: "router", vf: "model_id", vv: "google/gemini-2.5-pro" },
  /* ---- Meta (bloc générique managé Roboflow) ----------------------- */
  "muse-spark-1-3": { block: "roboflow_core/openrouter@v1", family: "router", vf: "model_id", vv: "meta/muse-spark-1.3" },
  "muse-wonder": { block: "roboflow_core/openrouter@v1", family: "router", vf: "model_id", vv: "meta/muse-wonder" },
  /* Mistral (bloc Roboflow -> OpenRouter). Les identifiants sont ceux
     d'OpenRouter : « mistral/mistral-large-4 » (forme de la passerelle Vercel)
     est refusé par le proxy, « mistralai/mistral-large-4-0 » répond. */
  "mistral-large-4": { block: "roboflow_core/openrouter@v1", family: "mistral", vf: "model_id", vv: "mistralai/mistral-large-4-0" },
  "mistral-medium-3-5": { block: "roboflow_core/openrouter@v1", family: "mistral", vf: "model_id", vv: "mistralai/mistral-medium-3-5" },
  "mistral-small-4": { block: "roboflow_core/openrouter@v1", family: "mistral", vf: "model_id", vv: "mistralai/mistral-small-2603" },
  /* ---- Alibaba / Qwen (bloc dédié, backend managé Roboflow) -------- */
  "qwen-3-8-max":        { block: "roboflow_core/openrouter@v1", family: "router", vf: "model_id", vv: "qwen/qwen3.8-max-0902" },
  "qwen-3-8-flash-next": { block: "roboflow_core/openrouter@v1", family: "router", vf: "model_id", vv: "qwen/qwen3.8-omni-flash" },
  "qwen-3-8-27b":        { block: "roboflow_core/openrouter@v1", family: "router", vf: "model_id", vv: "qwen/qwen3.8-27b" },
  /* ---- Moonshot Kimi (bloc dédié Roboflow) ------------------------- */
  "kimi-k3": { block: "roboflow_core/openrouter@v1", family: "router", vf: "model_id", vv: "moonshotai/kimi-k3" },
  /* Servis aussi par ModelScope (voie de secours quand Roboflow échoue). */
  "glm-5-3": { block: "roboflow_core/openrouter@v1", family: "router", vf: "model_id", vv: "z-ai/glm-5.3", or: "z-ai/glm-5.3" },
  "qwen-3-8-2-4t": { block: "roboflow_core/openrouter@v1", family: "router", vf: "model_id", vv: "qwen/qwen3.8-2.4t-a95b", or: "qwen/qwen3.8-2.4t-a95b" },
  /* ---- Zhipu / GLM (bloc générique managé Roboflow) ---------------- */
  "glm-5-3-flash": { block: "roboflow_core/openrouter@v1", family: "router", vf: "model_id", vv: "z-ai/glm-5.3-flash" },
  /* ---- DeepSeek (bloc générique managé Roboflow) ------------------- */
  "deepseek-v4-1-flash": { block: "roboflow_core/openrouter@v1", family: "router", vf: "model_id", vv: "deepseek/deepseek-v4.1-flash-1017" },
  /* ---- Génération d'image (sortie image via OpenRouter managé) ----- */
  "gemini-2-5-flash-image": { block: "roboflow_core/openrouter@v1", family: "router", vf: "model_id", vv: "google/gemini-2.5-flash-image" },
  "nano-banana-2-1":        { block: "roboflow_core/openrouter@v1", family: "router", vf: "model_id", vv: "google/gemini-nano-banana-2.1" },
  "gpt-5-image":            { block: "roboflow_core/openrouter@v1", family: "router", vf: "model_id", vv: "openai/gpt-5-image" },
  /* ---- Xiaomi / Mimo (bloc générique managé Roboflow) -------------- */
  "mimo-v2-6-pro":   { block: "roboflow_core/openrouter@v1", family: "router", vf: "model_id", vv: "xiaomi/mimo-v2.6-pro" },
  "mimo-v2-6-flash": { block: "roboflow_core/openrouter@v1", family: "router", vf: "model_id", vv: "xiaomi/mimo-v2.6-flash" },
};

const REASONING_EFFORT = ["low", "medium", "high", "xhigh", "max"];
const IMAGE_DETAIL = ["auto", "high", "low"];
const THINKING_LEVEL = ["low", "high"];

// Garde-fou : le plan Workers Free plafonne à 10 ms de CPU par requête.
const MAX_BODY_BYTES = 6_000_000;

/* --- Canal Groq (API gratuite, compatible OpenAI) --- */
const GROQ_MAP = {
  "groq-gpt-oss-120b": "openai/gpt-oss-120b",
  "groq-gpt-oss-20b": "openai/gpt-oss-20b",
  "groq-qwen-3-8-27b": "qwen/qwen3.8-27b",
};
/* Sortie max par modèle (contexte 131 k en entrée pour les trois) : on
   l'envoie systématiquement pour que rien ne soit coupé par défaut. */
const GROQ_MAXOUT = {
  "groq-gpt-oss-120b": 65536,
  "groq-gpt-oss-20b": 65536,
  /* qwen3.8-27b : le palier Groq du compte limite la SORTIE à 1 000 tokens
     par minute (OTPM). Demander plus fait rejeter la requête en 429
     « Request too large … reduce max_tokens » — d'où ce plafond à 1 000. */
  "groq-qwen-3-8-27b": 1000,
};
/* Quota Groq (OTPM = tokens de sortie par minute, ~1 000 sur qwen3.8-27b au
   palier on_demand) : quand la sortie attendue dépasse le budget restant de
   la minute, Groq répond 429 « Request too large … reduce max_tokens ».
   On réessaie alors avec des plafonds décroissants avant d'abandonner. */
const GROQ_RETRY_MAX = [1000, 500, 256, 128];
const GROQ_TOO_LARGE_RE = /Request too large|reduce max_tokens/i;
/* Un échec Groq du type « quota / requête trop grosse » se soigne ailleurs :
   le modèle texte correspondant existe chez ModelScope. Les autres erreurs
   (clé absente, panne réseau) ne relèvent pas d'un relais. */
const GROQ_RELAY_RE = /Request too large|reduce max_tokens|tokens per minute|rate limit/i;
function groqCanRelay(status, raw) {
  const txt = String(raw || "");
  return status === 429 || status === 413 || GROQ_RELAY_RE.test(txt);
}
function groqApiError(status, raw) {
  let msg = "";
  try { msg = (JSON.parse(raw) || {}).error?.message || ""; } catch {}
  const txt = msg || String(raw);
  if (status === 429 && GROQ_TOO_LARGE_RE.test(txt) && /output tokens/i.test(txt)) {
    return "Quota Groq atteint : votre compte est limité à environ 1 000 tokens de SORTIE par minute sur ce modèle (palier Groq). Attendez ~1 minute puis réessayez, ou choisissez un autre modèle.";
  }
  if (status === 429 && /tokens per minute/i.test(txt)) {
    return "Quota Groq atteint : la limite de tokens par minute de votre compte est épuisée sur ce modèle. Attendez ~1 minute puis réessayez, ou choisissez un autre modèle.";
  }
  if (status === 413 && /Request too large/i.test(txt)) {
    const m = txt.match(/tokens per minute \(([A-Z]+)\): Limit (\d+), Requested (\d+)/i);
    if (m) {
      const lim = Number(m[2]).toLocaleString("fr-FR");
      const req = Number(m[3]).toLocaleString("fr-FR");
      return "Message trop long pour le quota Groq : votre compte (palier gratuit) ne peut traiter que "
        + lim + " tokens par minute sur ce modèle — demande actuelle : " + req
        + " tokens. Raccourcissez le message ou démarrez une nouvelle conversation (l’historique complet est renvoyé à chaque tour), attendez ~1 minute, ou utilisez un modèle au quota supérieur.";
    }
    return "Message trop long pour le quota Groq de ce modèle (limite de tokens par minute). Raccourcissez le message, ou attendez ~1 minute.";
  }
  return "Groq HTTP " + status + " — " + (msg || String(raw).slice(0, 200));
}

/* Relais ModelScope d'un modèle Groq : la demande est rejouée chez
   ModelScope avec le prompt DÉJÀ construit par la voie Groq (intro, consignes,
   historique d'outils…) — le modèle voit donc exactement le même contexte.
   Renvoie null si ModelScope n'est pas configuré pour ce modèle. */
let MS_TEXT_RELAY = 0;
let MS_TEXT_RELAY_LAST = null;
async function msTextRelay(env, model, prompt, params, image) {
  const msModel = MS_TEXT[model];
  if (!msModel || !msKey(env)) return null;
  const r = await msChat(env, msModel, prompt, params, image);
  if (r.ok) {
    MS_TEXT_RELAY++;
    MS_TEXT_RELAY_LAST = { at: Date.now(), model: model, relais: msModel, etat: "ok" };
    return json({ output: r.output, applied: [], via: "modelscope" });
  }
  MS_TEXT_RELAY_LAST = { at: Date.now(), model: model, relais: msModel, etat: "échec", erreur: String(r.error || "").slice(0, 200) };
  return null;
}

/* --- Canal Pollinations (gen.pollinations.ai, compatible OpenAI) --- */
const POLL_TEXT_MAP = {
  "po-gpt-6-astra": "openai/gpt-6-astra",
  "po-gpt-6-1-sol": "openai/gpt-6.1-sol",
  "po-gpt-6-luna": "openai/gpt-6-luna",
};
/* Modèle Pollinations -> son jumeau côté Roboflow. Quand Pollinations refuse
   (crédits épuisés, quota, panne), la même demande part par la voie Roboflow
   sans que l'utilisateur ait à changer de modèle. */
const POLL_FB = {
  "po-gpt-6-astra": "gpt-6-astra",
  "po-gpt-6-1-sol": "gpt-6.1-sol",
  "po-gpt-6-luna": "gpt-6-luna",
};
const POLL_IMG_MAP = {
  "po-gpt-image-2": "openai/gpt-image-2",
  "po-mai-image-2-6": "microsoft/mai-image-2.6",
  "po-mai-image-2-6-flash": "microsoft/mai-image-2.6-flash",
  "po-flux-2-dev-free": "community/scriptsnsenses-sys/flux-2-dev-free",
};
const POLL_URL = "https://gen.pollinations.ai";

/* --- IFM : passerelle OpenAI-compatible (K2 Horizon 375B A23B) ---------
   Même forme que Groq : POST /v1/chat/completions, flux SSE `choices[].delta`.
   La clé vit uniquement en secret du worker (`IFM_API_KEY`). */
const IFM_URL = "https://api.ifm.ai/v1";
const IFM_MAP = {
  "ifm-k2-horizon": "IFM/K2-Horizon-375B-A23B",
};
/* Pollinations, à court de crédits, renvoie parfois HTTP 200 avec ce texte
   comme « réponse » : on le détecte pour le traiter en erreur 402. */
function pollCreditText(s) {
  return /enough credits|enter\.pollinations\.ai\/top-up|low_balance_topup|complete a quest/i.test(s || "");
}
function pollErrorMessage(status, raw) {
  let msg = "";
  try { msg = (JSON.parse(raw) || {}).error?.message || ""; } catch {}
  if (status === 402) return "Solde Pollinations épuisé : ce modèle est facturé à l'usage. Recharge ton compte sur enter.pollinations.ai/top-up.";
  if (status === 429) return "Quota Pollinations atteint : trop de requêtes sur ce modèle. Attends quelques secondes puis réessaie.";
  if (status === 503) return "Le serveur communautaire qui héberge ce modèle est momentanément indisponible (503). Réessaie dans quelques instants.";
  return "Pollinations HTTP " + status + " — " + (msg || String(raw).slice(0, 200));
}

/* --- Canal Google (Gemini API gratuite) --- */
const GOOGLE_MAP = {
  "gg-gemini-3-8-flash": { m: "gemini-3.8-flash" },
  "gg-gemini-3-7-flash": { m: "gemini-3.7-flash" },
  "gg-gemini-3-6-flash": { m: "gemini-3.6-flash" },
  "gg-gemini-3-5-flash": { m: "gemini-3.5-flash" },
  "gg-gemini-3-flash": { m: "gemini-3-flash-preview" },
  "gg-gemini-2-5-flash": { m: "gemini-2.5-flash" },
  "gg-gemini-2-5-pro": { m: "gemini-2.5-pro" },
  "gg-gemma-4-31b": { m: "gemma-4-31b-it" },
  "gg-gemma-4-26b-a4b": { m: "gemma-4-26b-a4b-it" },
  "gg-gemini-3-5-flash-lite": { m: "gemini-3.5-flash-lite" },
  "gg-gemini-3-1-flash-lite": { m: "gemini-3.1-flash-lite" },
  "gg-gemini-2-5-flash-lite": { m: "gemini-2.5-flash-lite" },
  "gg-gemini-3-8-flash-tts": { m: "gemini-3.8-flash-tts", kind: "tts" },
  "gg-gemini-3-8-flash-lite-tts": { m: "gemini-3.8-flash-lite-tts", kind: "tts" },
  "gg-gemini-embedding-2": { m: "gemini-embedding-2", kind: "embed" },
  "gg-gemini-3-5-transcribe": { m: "gemini-3.5-transcribe", kind: "transcribe" },
};

/* Construit la charge utile d'entrée d'un modèle multimodal.
   `image` (compat, 1 seule image) et `images` (plusieurs images nommées,
   envoyées dans l'ordre) sont acceptés ensemble : les images nommées sont
   placées en tête, chacune précédée de son nom pour que le modèle sache
   de laquelle on parle. */
const MAX_INLINE_IMAGES = 4;
function normImageList(body) {
  const out = [];
  if (Array.isArray(body.images)) {
    for (const im of body.images) {
      if (!im || typeof im !== "object") continue;
      const b64 = String(im.b64 || "");
      if (!b64 || b64.length > 3000000) continue;
      out.push({ name: String(im.name || "image"), b64, mime: String(im.mime || detectMime(b64)) });
      if (out.length >= MAX_INLINE_IMAGES) break;
    }
  }
  if (!out.length) {
    const b64 = String(body.image || "");
    if (body.image_kind === "upload" && b64 && b64.length <= 3000000) {
      out.push({ name: "image", b64, mime: detectMime(b64) });
    }
  }
  return out;
}
function dataUri(im) { return "data:" + im.mime + ";base64," + im.b64; }

function b64ToBytes(b64) {
  const bin = atob(b64);
  const u8 = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
  return u8;
}

function detectMime(b64) {
  if (b64.startsWith("/9j/")) return "image/jpeg";
  if (b64.startsWith("iVBOR")) return "image/png";
  if (b64.startsWith("UklGR")) return "image/webp";
  if (b64.startsWith("R0lGOD")) return "image/gif";
  return "image/jpeg";
}

const BROWSER_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36";

function json(obj, status = 200) {
  return new Response(JSON.stringify(obj), {
    status,
    headers: {
      "Content-Type": "application/json; charset=utf-8",
      "Access-Control-Allow-Origin": "*",
      "Access-Control-Allow-Methods": "POST, GET, OPTIONS",
      "Access-Control-Allow-Headers": "Content-Type, X-Access-Code",
      "Access-Control-Max-Age": "86400",
      "Cache-Control": "no-store",
    },
  });
}

function int(v, min, max) {
  const n = Number(v);
  if (!Number.isFinite(n)) return undefined;
  const r = Math.trunc(n);
  if (r < min || (max !== undefined && r > max)) return undefined;
  return r;
}
function num(v, min, max) {
  const n = Number(v);
  if (!Number.isFinite(n) || n < min || n > max) return undefined;
  return n;
}
function bool(v) {
  if (typeof v === "boolean") return v;
  if (v === "true") return true;
  if (v === "false") return false;
  return undefined;
}
function oneOf(v, list) {
  return typeof v === "string" && list.includes(v) ? v : undefined;
}

/* Le point d'entrée Roboflow est protégé par un pare-feu applicatif (WAF) qui
   bloque certaines charges : une commande de téléchargement (curl, wget…)
   écrite entre accents graves et contenant une URL est prise pour une
   injection de commande, et le fournisseur répond 403 « Attention Required! »
   au lieu de traiter le message. On rejoue alors la requête une fois avec ces
   accents graves remplacés par des apostrophes (le texte reste lisible). */
/* Roboflow renvoie tantôt du texte, tantôt une image (objet base64, URI de
   données, URL d'image). On sait le reconnaître pour rejouer la demande par
   la voie OpenRouter quand le modèle choisi doit rendre du texte. */
function looksLikeImage(v, certainOnly) {
  if (v == null) return false;
  if (typeof v === "object") {
    if (Array.isArray(v)) return v.length > 0 && v.every((x) => looksLikeImage(x, certainOnly));
    if (v.type === "base64" && typeof v.value === "string") return true;
    return looksLikeImage(v.value != null ? v.value : (v.url != null ? v.url : v.image), certainOnly);
  }
  const s = String(v).trim();
  if (!s) return false;
  if (/^data:image\//i.test(s)) return true;                       /* data URI */
  /* Un simple lien vers une image peut être une réponse légitime : ce cas
     n'est « certain » que si l'autre voie ne répond pas la même chose. */
  if (/^https?:\/\/\S+\.(png|jpe?g|gif|webp|bmp|avif|tiff?|svg)(\?\S*)?$/i.test(s)) return !certainOnly;
  /* Base64 brut : très long, sans espaces, et l'en-tête est celui d'une image. */
  if (s.length > 240 && !/\s/.test(s) && /^[A-Za-z0-9+/=]+$/.test(s)) {
    return /^(iVBORw0KGgo|\/9j\/|R0lGOD|UklGR|Qk|SUkq|AAAAIGZ0eXB)/.test(s.slice(0, 20));
  }
  return false;
}
function defuseWaf(s, level) {
  /* Mesure du 13/10 : le pare-feu de Roboflow refuse une commande de téléchargement
     placée entre apostrophes ou accents graves ('curl -s http://…' / `wget http://…`).
     Les guillemets doubles passent. On ne réécrit que ces passages-là, et
     uniquement dans la copie envoyée au modèle (jamais dans l'affichage).
     Trois niveaux, du plus discret au plus radical — c'est la montée en
     puissance utilisée quand le fournisseur refuse quand même :
       1. commandes entre ' ou ` → guillemets doubles (aucun autre changement) ;
       2. toute apostrophe / accent grave → espace, apostrophes typographiques
          normalisées (« l’appareil » → « l appareil ») ;
       3. en plus, les accents sont retirés (« é » → « e »).
     Le sens du message reste lisible aux trois niveaux. */
  const lvl = Math.max(1, Math.min(3, Number(level) || 1));
  const span = /'([^'\n]{0,600})'|`([^`\n]{0,600})`/g;
  let out = String(s).replace(span, (m, simple, grave) => {
    const inner = (simple !== undefined) ? simple : grave;
    const shell = /(^|[\s"'(])(curl|wget|bash|sh|zsh|python3?|node|nc|ncat|powershell)\b/i.test(inner);
    if (shell && /https?:\/\//i.test(inner)) return '"' + inner.replace(/"/g, "'") + '"';
    return m;
  });
  if (lvl >= 2) {
    out = out.replace(/[`´‘’‚‛′]/g, "'").replace(/'/g, " ");
  }
  if (lvl >= 3) {
    try { out = out.normalize("NFD").replace(/[\u0300-\u036f]/g, ""); }
    catch (e) { /* moteur sans normalize : on garde le niveau 2 */ }
  }
  return out;
}
/* Une réponse « pare-feu » ne ressemble jamais à du JSON : 403 sec, page HTML,
   code 1010/1020, ou texte « Attention Required ». On la reconnaît pour
   réessayer — jamais pour l'afficher. */
function wafLike(status, text) {
  const t = String(text || "");
  if (/Attention Required|Cloudflare|cf-error-details|error code:?\s*10(10|20|40)/i.test(t)) return true;
  if (/^\s*<(!doctype|html)/i.test(t)) return true;
  return Number(status) === 403;
}
// Ne conserve que les réglages valides pour la famille du modèle.
function sanitize(params, family) {
  const p = params && typeof params === "object" ? params : {};
  const out = {};
  let v;

  if (family === "claude") {
    if ((v = bool(p.extended_thinking)) !== undefined) out.extended_thinking = v;
    if (out.extended_thinking) {
      if ((v = int(p.thinking_budget_tokens, 1024, 128000)) !== undefined) {
        out.thinking_budget_tokens = v;
      }
    }
    if ((v = int(p.max_tokens, 1, 128000)) !== undefined) out.max_tokens = v;
    if ((v = num(p.temperature, 0, 1)) !== undefined) out.temperature = v;
    if ((v = int(p.max_image_size, 64, 8192)) !== undefined) out.max_image_size = v;
    if ((v = int(p.max_concurrent_requests, 1, 64)) !== undefined) out.max_concurrent_requests = v;
  } else if (family === "openai") {
    if ((v = oneOf(p.reasoning_effort, REASONING_EFFORT)) !== undefined) out.reasoning_effort = v;
    if ((v = oneOf(p.image_detail, IMAGE_DETAIL)) !== undefined) out.image_detail = v;
    if ((v = int(p.max_tokens, 16, 128000)) !== undefined) out.max_tokens = v;
    if ((v = num(p.temperature, 0, 2)) !== undefined) out.temperature = v;
    if ((v = int(p.max_concurrent_requests, 1, 64)) !== undefined) out.max_concurrent_requests = v;
  } else if (family === "gemini") {
    if ((v = oneOf(p.thinking_level, THINKING_LEVEL)) !== undefined) out.thinking_level = v;
    if ((v = int(p.max_tokens, 1, 65536)) !== undefined) out.max_tokens = v;
    /* Toujours transmis (false par défaut) : quand le bloc laisse l'outil
       natif actif, Gemini tente des appels d'outil et l'API peut répondre
       500 « stopped generation with finish reason: MALFORMED_FUNCTION_CALL ».
       L'utilisateur l'active explicitement quand il en a besoin. */
    out.google_code_execution = bool(p.google_code_execution) === true;
  } else if (family === "router") {
    if ((v = num(p.temperature, 0, 2)) !== undefined) out.temperature = v;
    if ((v = int(p.max_tokens, 1, 128000)) !== undefined) out.max_tokens = v;
    if ((v = int(p.max_concurrent_requests, 1, 64)) !== undefined) out.max_concurrent_requests = v;
  } else if (family === "qwen") {
    if ((v = bool(p.enable_thinking)) !== undefined) out.enable_thinking = v;
    if ((v = int(p.max_tokens, 1, 65536)) !== undefined) out.max_tokens = v;
    if ((v = num(p.temperature, 0, 2)) !== undefined) out.temperature = v;
  } else if (family === "mistral") {
    /* Paramètres des Mistral relayés par le bloc OpenRouter. Liste calée sur
       « supported_parameters » des trois modèles : température, top-p, tokens,
       pénalités, graine. Pas de safe_prompt (propre à l'API Mistral directe). */
    if ((v = num(p.temperature, 0, 1.5)) !== undefined) out.temperature = v;
    if ((v = num(p.top_p, 0, 1)) !== undefined) out.top_p = v;
    if ((v = int(p.max_tokens, 1, 209715)) !== undefined) out.max_tokens = v;
    if ((v = num(p.presence_penalty, -2, 2)) !== undefined) out.presence_penalty = v;
    if ((v = num(p.frequency_penalty, -2, 2)) !== undefined) out.frequency_penalty = v;
    if ((v = int(p.seed, 0, 2147483647)) !== undefined) out.seed = v;
  } else if (family === "glm") {
    if ((v = int(p.max_new_tokens, 1, 8192)) !== undefined) out.max_new_tokens = v;
  }
  return out;
}

/* ---- Missions d'agent en arrière-plan -------------------------------
   La mission n'est plus pilotée par le navigateur : l'app la confie au
   worker (POST /api/agent/job/start), qui exécute lui-même la boucle
   modèle → commande → modèle. L'app peut donc être fermée : la mission
   continue, et au rechargement elle s'y raccroche (GET /api/agent/job?id=…)
   pour réafficher les étapes et le résultat. L'état vit dans la mémoire de
   l'isolate, gardé 30 min après la dernière écriture. ---- */
const JOBS = new Map();
/* Plus AUCUNE limite de durée ni de nombre d'étapes : une mission s'arrête
   quand le modèle rend sa réponse finale, ou quand on l'arrête depuis l'app.
   Seule contrainte conservée : la taille de la trace envoyée au modèle à
   chaque étape (sinon le prompt grossit sans fin et le modèle décroche). */
const JOB_KEEP_MS = 30 * 60 * 1000;    /* conservation après la fin */
const JOB_TR_KEEP = 48000;             /* trace gardée dans le prompt (caractères) */
const JOB_HIST_KEEP = 32000;           /* contexte de conversation gardé */
const JOB_SAVE_STEPS = 400;            /* étapes conservées sur le stockage durable */
const JOB_SAVE_OUT = 2000;             /* sortie conservée pour les étapes anciennes */
function tailKeep(s, n) {
  const t = String(s == null ? "" : s);
  return t.length > n ? t.slice(t.length - n) : t;
}
const VM_API = "https://vmai.hugdu77777.workers.dev/api/linux";

function jobSweep() {
  const now = Date.now();
  for (const [id, j] of JOBS) if (j.updatedAt && now - j.updatedAt > JOB_KEEP_MS) JOBS.delete(id);
}

/* Lecture d'un appel d'outil. Les modèles renvoient presque toujours du JSON,
   mais pas toujours PROPRE : retours à la ligne bruts dans une chaîne (ce qui
   est interdit en JSON), guillemets internes non échappés, ou réponse coupée
   en pleine commande. Sans réparation, ce JSON s'affichait comme un message
   au lieu d'être exécuté. On essaie donc, dans l'ordre : le texte tel quel,
   sa version réparée, puis une lecture directe de la commande. */
function jsonRepair(t) {
  let out = "", str = false, esc = false;
  for (let i = 0; i < t.length; i++) {
    const c = t[i];
    if (esc) { out += c; esc = false; continue; }
    if (c === "\\") { out += c; esc = true; continue; }
    if (c === '"') { str = !str; out += c; continue; }
    if (str && (c === "\n" || c === "\r" || c === "\t")) {
      out += c === "\n" ? "\\n" : (c === "\r" ? "\\r" : "\\t");
      continue;
    }
    out += c;
  }
  return out;
}
/* Corps d'une chaîne JSON (entre guillemets) : on rejoue les échappements. */
function jsonStringBody(body) {
  let out = "";
  for (let i = 0; i < body.length; i++) {
    const c = body[i];
    if (c !== "\\") { out += c; continue; }
    const n = body[++i];
    if (n === "n") out += "\n";
    else if (n === "r") out += "\r";
    else if (n === "t") out += "\t";
    else if (n === "u") { out += String.fromCharCode(parseInt(body.slice(i + 1, i + 5), 16) || 32); i += 4; }
    else out += n === undefined ? "\\" : n;
  }
  return out;
}
/* Dernier recours : on lit la commande directement dans le texte, guillemets
   internes, retours à la ligne et fin de réponse coupée compris. */
function looseAgentCall(s) {
  const am = /"action"\s*:\s*"?([a-zA-Z]+)"?/.exec(s);
  if (!am) return null;
  const action = am[1].toLowerCase();
  const key = action === "fetch" ? "url" : (action === "final" ? "message" : "command");
  const km = new RegExp('"' + key + '"\\s*:\\s*"').exec(s);
  const o = { action: action };
  if (km) {
    let i = km.index + km[0].length, body = "", esc = false;
    for (; i < s.length; i++) {
      const c = s[i];
      if (esc) { body += "\\" + c; esc = false; continue; }
      if (c === "\\") { esc = true; continue; }
      if (c === '"') {
        /* fermant réel seulement si la suite du JSON (virgule, accolade, fin)
           vient juste après ; sinon c'est un guillemet interne non échappé. */
        const rest = s.slice(i + 1).replace(/^\s+/, "");
        if (rest === "" || rest[0] === "," || rest[0] === "}") break;
        body += '"';
        continue;
      }
      body += c;
    }
    o[key] = jsonStringBody(body);
    if (!String(o[key]).trim()) return null;
  } else if (action !== "final") {
    return null;
  }
  const mm = /"mode"\s*:\s*"([a-zA-Z]+)"/.exec(s);
  if (mm) o.mode = mm[1];
  return o;
}
function parseAgentCall(s) {
  s = String(s || "");
  const cand = [];
  const fences = s.match(/```[a-zA-Z]*\s*([\s\S]*?)```/g) || [];
  for (const f of fences) cand.push(f.replace(/```[a-zA-Z]*/, "").replace(/```$/, "").trim());
  cand.push(s.trim());
  const i = s.indexOf("{");
  if (i >= 0) { const k = s.lastIndexOf("}"); if (k > i) cand.push(s.slice(i, k + 1)); }
  for (const c of cand) {
    for (const v of [c, jsonRepair(c)]) {
      try {
        const o = JSON.parse(v);
        if (o && typeof o === "object" && (o.action || o.command)) {
          if (!o.action && o.command) o.action = "bash";
          return o;
        }
      } catch { /* essai suivant */ }
    }
  }
  return looseAgentCall(s);
}

async function vmCall(env, path, body) {
  const rq = new Request(VM_API + path, {
    method: "POST",
    headers: { "Authorization": "Bearer " + (env.VMAI_PASS || ""), "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  });
  const r = env.VMAI && typeof env.VMAI.fetch === "function" ? await env.VMAI.fetch(rq) : await fetch(rq);
  try { return await r.json(); } catch { return { ok: false, error: "réponse illisible de la VM" }; }
}

/* Vue publique d'une mission. `since` (facultatif) permet à l'app de ne
   recevoir que les étapes qu'elle n'a pas encore affichées : une mission très
   longue ne retransfère donc jamais tout son déroulé à chaque consultation. */
function jobPublic(j, since) {
  const all = j.steps || [];
  const from = Math.min(Math.max(0, Number(since) || 0), all.length);
  return { id: j.id, chat: j.chat || "", mission: j.mission, model: j.model, steps: all.slice(from),
    stepsFrom: from, stepsTotal: all.length, done: !!j.done,
    error: j.error || "", final: j.finalMd || "", updatedAt: j.updatedAt, step: j.step };
}

/* ---- Disponibilité des secrets ---------------------------------------
   Ce que l'appelant peut voir : le NOM du secret, l'API qu'il sert, et s'il
   est présent. Jamais sa valeur. Chaque test est un vrai appel à l'API, avec
   la clé du worker : ce qui est affiché est une preuve, pas une supposition. */
const SECRET_CATALOG = (function () {
  const out = [];
  for (let i = 1; i <= 8; i++) {
    out.push({ name: "ROBOFLOW_API_KEY_" + i, api: "Roboflow", label: "Jeton " + i });
  }
  out.push({ name: "MODELSCOPE_API_KEY", api: "ModelScope", label: "Jeton d’inférence" });
  out.push({ name: "POLLINATIONS_API_KEY", api: "Pollinations", label: "Clé" });
  out.push({ name: "GROQ_API_KEY", api: "Groq", label: "Clé" });
  out.push({ name: "HUGGINGFACE_TOKEN", api: "Hugging Face", label: "Jeton" });
  out.push({ name: "NVIDIA_NIM_API_KEY", api: "NVIDIA NIM", label: "Clé" });
  return out;
})();
function secretValue(env, name) {
  let v = String((env && env[name]) || "").trim();
  /* Ancien nom, encore lu : un jeton posé sous HF_TOKEN n'est pas perdu. */
  if (!v && name === "HUGGINGFACE_TOKEN") v = String((env && env.HF_TOKEN) || "").trim();
  /* Le jeton historique — celui qui n'a pas de numéro — EST le numéro 4 : on
     l'affiche à sa place au lieu de laisser un trou dans la liste. */
  if (!v && name === "ROBOFLOW_API_KEY_4") v = String((env && env.ROBOFLOW_API_KEY) || "").trim();
  return v;
}
function secretFin(v) { return v ? v.slice(-4) : ""; }
async function secretTest(env, name) {
  const v = secretValue(env, name);
  const t0 = Date.now();
  if (!v) return { name: name, ok: false, status: 0, ms: 0, message: "Secret absent du worker : rien à tester." };
  const fin = (r, ok, message) => ({ name: name, ok: ok, status: r && r.status != null ? r.status : 0, ms: Date.now() - t0, message: message });
  try {
    if (/^ROBOFLOW_API_KEY(_\d+)?$/.test(name)) {
      const r = await fetch("https://api.roboflow.com/?api_key=" + encodeURIComponent(v),
        { headers: { "Accept": "application/json", "User-Agent": BROWSER_UA } });
      const txt = await r.text();
      let j = {}; try { j = JSON.parse(txt); } catch (e) { j = {}; }
      if (r.ok && j && j.workspace) return fin(r, true, "Jeton valide — espace de travail « " + j.workspace + " ».");
      const why = (j && j.error && (j.error.message || j.error.type)) || String(txt).slice(0, 200) || ("HTTP " + r.status);
      return fin(r, false, "Jeton refusé (HTTP " + r.status + ") : " + why);
    }
    if (name === "MODELSCOPE_API_KEY") {
      /* page_size=1 : la liste complète met une trentaine de secondes, un
         seul élément répond en une seconde — on ne teste que l'authentification. */
      const r = await fetch("https://api-inference.modelscope.ai/v1/models?page_size=1", {
        headers: { "Authorization": "Bearer " + v, "User-Agent": BROWSER_UA } });
      const txt = await r.text();
      if (r.ok) return fin(r, true, "Jeton accepté — l’API d’inférence répond.");
      return fin(r, false, "Refusé (HTTP " + r.status + ") : " + String(txt).slice(0, 200));
    }
    if (name === "POLLINATIONS_API_KEY") {
      const r = await fetch("https://gen.pollinations.ai/account/balance", {
        headers: { "Authorization": "Bearer " + v, "User-Agent": BROWSER_UA } });
      const txt = await r.text();
      let j = {}; try { j = JSON.parse(txt); } catch (e) { j = {}; }
      if (!r.ok) return fin(r, false, "Clé refusée (HTTP " + r.status + ") : " + String(txt).slice(0, 200));
      const bal = (j && (j.balance != null ? j.balance : (j.account && j.account.total))) ;
      return fin(r, true, "Clé valide — solde : " + (bal == null ? "inconnu" : String(bal) + " Pollen") + ".");
    }
    if (name === "GROQ_API_KEY") {
      const r = await fetch("https://api.groq.com/openai/v1/models", {
        headers: { "Authorization": "Bearer " + v, "User-Agent": BROWSER_UA } });
      const txt = await r.text();
      if (r.ok) {
        let n = 0; try { n = ((JSON.parse(txt) || {}).data || []).length; } catch (e) {}
        return fin(r, true, "Clé valide — " + n + " modèles visibles.");
      }
      return fin(r, false, "Clé refusée (HTTP " + r.status + ") : " + String(txt).slice(0, 200));
    }
    if (name === "HUGGINGFACE_TOKEN") {
      const r = await fetch("https://huggingface.co/api/whoami-v2", {
        headers: { "Authorization": "Bearer " + v, "User-Agent": BROWSER_UA } });
      const txt = await r.text();
      let j = {}; try { j = JSON.parse(txt); } catch (e) { j = {}; }
      if (r.ok) return fin(r, true, "Jeton valide — compte « " + (j.name || j.fullname || "?") + " ».");
      return fin(r, false, "Jeton refusé (HTTP " + r.status + ") : " + String(txt).slice(0, 200));
    }
    if (name === "NVIDIA_NIM_API_KEY") {
      const r = await fetch("https://integrate.api.nvidia.com/v1/models", {
        headers: { "Authorization": "Bearer " + v, "User-Agent": BROWSER_UA } });
      const txt = await r.text();
      if (r.ok) return fin(r, true, "Clé valide — le catalogue NVIDIA répond.");
      return fin(r, false, "Clé refusée (HTTP " + r.status + ") : " + String(txt).slice(0, 200));
    }
    return fin(null, false, "Aucun test connu pour ce secret.");
  } catch (e) {
    return fin(null, false, "Injoignable : " + String((e && e.message) || e));
  }
}

/* Registre des missions : un objet durable unique (« index ») garde la liste
   de ce qui a été lancé, avec sa discussion et son état. C'est ce qui permet
   de LISTER les tâches actives — un Durable Object, lui, ne s'énumère pas. */
async function jobIndexPut(env, entry) {
  if (!env || !env.MISSION || !entry || !entry.id) return false;
  try {
    const stub = env.MISSION.get(env.MISSION.idFromName("index"));
    const r = await stub.fetch("https://mission/reg", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(entry),
    });
    return !!(r && r.ok);
  } catch (e) { return false; }
}
async function jobIndexList(env) {
  if (!env || !env.MISSION) return [];
  try {
    const stub = env.MISSION.get(env.MISSION.idFromName("index"));
    const r = await stub.fetch("https://mission/list");
    const jr = await r.json().catch(() => ({}));
    return (jr && jr.jobs) || [];
  } catch (e) { return []; }
}

/* Une étape : construire le prompt, appeler le modèle, exécuter ce qu'il
   demande, mémoriser le tout. */
async function jobStep(env, j) {
  const ctxHist = j.hist ? "\n\n## Conversation précédente (contexte)\n" + j.hist : "";
  const prompt = j.step === 0
    ? j.intro + ctxHist + "\n\nVoici le premier message de l'utilisateur :\n" + j.mission
    : j.rappel + ctxHist + "\n\n## Mission\n" + j.mission
      + (j.tr ? "\n\n## Historique des commandes\n" + j.tr : "")
      + "\n\nRéponds par un seul objet JSON.";

  let out = "";
  let lastErr = "";
  /* Un incident de fournisseur ne doit pas tuer une mission : on rejoue
     l'étape (les jetons tournent, les voies changent) avant de renoncer. */
  let wafRefus = false;
  for (let essai = 0; essai < 4 && !out; essai++) {
  /* `_waf` : à chaque nouvel essai, le message part réécrit un peu plus
     (apostrophes remplacées, puis accents retirés). Un refus du filtre du
     fournisseur ne doit jamais arrêter une mission ni s'afficher : on
     réessaie simplement avec la copie suivante. */
  const wafLvl = Math.min(3, essai + 1);
  try {
    /* Appel interne : on traverse exactement la même route que l'app,
       sans passer par le réseau (un worker ne peut pas s'appeler lui-même). */
    const req = new Request(j.origin + "/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-Access-Code": String(env.ACCESS_CODE || j.code || ""), "User-Agent": "Mozilla/5.0" },
      body: JSON.stringify({ model: j.model, prompt: prompt, image: j.image, image_kind: j.image_kind, params: j.params, _waf: wafLvl }),
    });
    const r = await fetchHandler(req, env, { waitUntil: () => {}, passThroughOnException: () => {} });
    const jr = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(String(jr.error || ("HTTP " + r.status)));
    out = String(jr.output == null ? "" : jr.output);
  } catch (e) {
    const msg = String(e && e.message ? e.message : e);
    wafRefus = /filtr|pare-feu|WAF|Attention Required|Cloudflare|HTTP 403|1010/i.test(msg);
    lastErr = "appel du modèle impossible : " + msg;
    /* Rien ne s'affiche tant qu'il reste un espoir : on attend un peu et on
       rejoue — la rotation change de jeton, et parfois de fournisseur. */
    if (essai < 3) await new Promise(function (res) { setTimeout(res, 500 + essai * 900); });
  }
  }
  if (!out) {
    j.error = wafRefus ? "mission interrompue — relance-la" : (lastErr || "appel du modèle impossible");
    j.done = true;
    return;
  }

  const call = parseAgentCall(out);
  if (!call || call.action === "final") {
    j.finalMd = call ? String(call.message || "") : out;
    j.done = true; j.updatedAt = Date.now();
    return;
  }

  if (call.action === "bash") {
    const cmd = String(call.command || "");
    let res = {};
    try { res = await vmCall(env, "/exec", { command: cmd, timeout: 300 }); }
    catch (e) { res = { ok: false, stderr: String(e && e.message ? e.message : e) }; }
    const so = String(res.stdout || ""), se = String(res.stderr || "");
    const ex = res.exit_code == null ? 0 : res.exit_code;
    j.steps.push({ kind: "bash", cmd: cmd, out: so, err: se, exit: ex });
    j.tr = tailKeep(j.tr + "\n[commande] " + cmd + "\n[exit] " + (res.exit_code == null ? "?" : res.exit_code)
      + "\n[stdout]\n" + so + (se ? "\n[stderr]\n" + se : "") + "\n", JOB_TR_KEEP);
  } else if (call.action === "fetch") {
    const url = String(call.url || "").trim();
    const fmode = String(call.mode || "text").toLowerCase() === "html" ? "html" : "text";
    const label = "fetch [" + fmode + "] " + url;
    let fj = {};
    try {
      const body = { url: url, mode: fmode, timeout: 120 };
      if (call.wait_ms != null) body.wait_ms = Math.min(Math.max(0, Number(call.wait_ms) || 0), 15000);
      if (call.selector) body.selector = String(call.selector).slice(0, 200);
      fj = await vmCall(env, "/fetch", body);
    } catch (e) { fj = { ok: false, error: String(e && e.message ? e.message : e) }; }
    const head = (fj.ok ? "HTTP " + (fj.status == null ? "?" : fj.status) : "ÉCHEC")
      + (fj.title ? " · " + fj.title : "") + (fj.chars ? " · " + fj.chars + " caractères" : "")
      + (fj.ms ? " · " + fj.ms + " ms" : "") + (fj.path ? "\n(copie enregistrée : " + fj.path + ")" : "");
    const full = String(fj.content == null ? "" : fj.content);
    const shown = fj.ok ? (head + (full ? "\n\n" + full : "")) : String(fj.error || head);
    j.steps.push({ kind: "fetch", cmd: label, out: shown, err: fj.ok ? "" : String(fj.error || ""), exit: fj.ok ? 0 : 1 });
    j.tr = tailKeep(j.tr + "\n[fetch " + fmode + "] " + url + "\n[statut] " + (fj.status == null ? "?" : fj.status)
      + " · " + (fj.title || "") + "\n[contenu complet]\n" + full + "\n", JOB_TR_KEEP);
  }

  j.step++;
  j.updatedAt = Date.now();
}

async function jobRun(env, j) {
  if (j.running) return;          /* une seule boucle par mission */
  j.running = true;
  try {
    /* Aucune limite : on avance tant que la mission n'est pas finie. */
    while (!j.done && !j.stop) {
      await jobStep(env, j);
      j.updatedAt = Date.now();
    }
    if (!j.done && !j.error && j.stop) j.error = "mission arrêtée à la demande";
  } catch (e) {
    j.error = String(e && e.message ? e.message : e);
  }
  j.done = true;
  j.running = false;
  j.updatedAt = Date.now();
  /* Le résultat part dans sa discussion, comme pour une mission durable. */
  try { await jobDeliver(env, j); } catch (e) {}
}

/* Bascule Pollinations -> Roboflow : on rejoue la demande, en interne, sur le
   modèle jumeau. Renvoie le texte, ou null si la bascule n'était pas possible. */
async function pollToRoboflow(env, body, model) {
  const twin = POLL_FB[model];
  if (!twin || !MODELS[twin]) return null;
  const p = Object.assign({}, body.params || {});
  delete p._fallback; delete p._raw;          /* réglages internes, pas au modèle */
  try {
    const req = new Request("https://internal/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-Access-Code": String(env.ACCESS_CODE || "") },
      body: JSON.stringify({ model: twin, prompt: String(body.prompt || ""), image: body.image,
        image_kind: body.image_kind, images: body.images, params: p }),
    });
    const r = await fetchHandler(req, env, { waitUntil() {}, passThroughOnException() {} });
    const jr = await r.json().catch(() => ({}));
    /* Une réponse vide ne vaut pas une bascule réussie : mieux vaut alors
       afficher la raison d'origine (crédits, quota, panne). */
    const txt = String(jr && jr.output != null ? jr.output : "").trim();
    if (r.ok && txt) {
      return { output: String(jr.output), applied: jr.applied || [], via: "roboflow", from: model };
    }
  } catch (e) { /* on garde l'erreur d'origine */ }
  return null;
}

/* Bascule en mode flux : le texte sauvé part en UN seul bloc SSE, précédé du
   marqueur « roboflow » — l'app n'a donc rien à faire de particulier. */
function pollFallbackStream(fb) {
  const enc = new TextEncoder();
  const one = new ReadableStream({
    start(c) {
      c.enqueue(enc.encode("data: " + JSON.stringify({ fb: "roboflow" }) + "\n\n"));
      c.enqueue(enc.encode("data: " + JSON.stringify({ t: fb.output }) + "\n\n"));
      c.enqueue(enc.encode("data: [DONE]\n\n"));
      c.close();
    },
  });
  return new Response(one, {
    headers: {
      "Content-Type": "text/event-stream; charset=utf-8",
      "Access-Control-Allow-Origin": "*",
      "Access-Control-Allow-Methods": "POST, GET, OPTIONS",
      "Access-Control-Allow-Headers": "Content-Type, X-Access-Code",
      "Cache-Control": "no-store",
    },
  });
}

/* ---- ModelScope (jetons « ms-… ») -------------------------------------
   Le jeton ModelScope (secret MODELSCOPE_API_KEY) ouvre l'API d'inférence,
   compatible OpenAI, et donne droit aux crédits gratuits de la plateforme.
   Deux usages ici :
     • voie de secours des modèles déjà servis par Roboflow : dès que Roboflow
       échoue (panne, jeton refusé, plafond de crédits atteint), la même
       demande part chez ModelScope ;
     • voie principale des modèles qui n'existent que là.
   Textes : POST {MS_BASE}/chat/completions.
   Images  : POST {MS_BASE}/images/generations en mode ASYNCHRONE
             (en-tête X-ModelScope-Async-Mode) qui rend un identifiant de
             tâche, puis sondage de {MS_BASE}/tasks/{id} avec l'en-tête
             X-ModelScope-Task-Type: image_generation jusqu'à « SUCCEED ».
   Rien n'est facturé pour /v1/models (lecture libre). */
/* Deux hôtes, et l'ordre compte : le .cn refuse des jetons que le .ai
   accepte (même compte, même jeton, deux serveurs différents). On essaie
   donc .ai d'abord, .cn seulement s'il ne répond pas du tout. */
const MS_HOSTS = ["https://api-inference.modelscope.ai/v1", "https://api-inference.modelscope.cn/v1"];
const MS_BASE = MS_HOSTS[0];            /* hôte principal */
const MS_SITES = ["https://modelscope.ai", "https://modelscope.cn"];
const MS_SITE = MS_SITES[0];
/* Modèles servis AUSSI par ModelScope, sous leur nom ModelScope. */
/* Les identifiants suivent le dépôt réel : l'organisation de GLM s'appelle
   désormais zai-org (ZhipuAI est refusé), et les MiMo s'appellent
   MiMo-V2.6-… sur le dépôt. Un identifiant juste vaut mieux même quand le
   fournisseur n'est pas encore ouvert : le jour où il l'est, c'est branché. */
const MS_TEXT = {
  "kimi-k3": "moonshotai/Kimi-K3",
  "glm-5-3": "zai-org/GLM-5.3",
  "glm-5-3-flash": "zai-org/GLM-5.3-Flash",
  "deepseek-v4-1-flash": "deepseek-ai/DeepSeek-V4.1-Flash",
  "mimo-v2-6-pro": "XiaomiMiMo/MiMo-V2.6-Pro-RL",
  "mimo-v2-6-flash": "XiaomiMiMo/MiMo-V2.6-Flash-RL",
  "qwen-3-8-flash-next": "Qwen/Qwen3.8-Flash-Next",
  "qwen-3-8-27b": "Qwen/Qwen3.8-27B",
  "groq-qwen-3-8-27b": "Qwen/Qwen3.8-27B",
  "qwen-3-8-max": "Qwen/Qwen3.8-2.4T-A95B",
  "qwen-3-8-2-4t": "Qwen/Qwen3.8-2.4T-A95B",
};
/* Modèles d'image servis par ModelScope. */
/* Modèles servis par ModelScope EN PREMIER : Roboflow ne prend le relais que
   si ModelScope refuse (Qwen 3.8 Max est dans ce cas — ses crédits gratuits
   passent avant). Aucune bascule vers un autre modèle n'est jamais faite :
   si les deux voies échouent, l'erreur remonte telle quelle. */
/* ModelScope d'abord pour les modèles open source : c'est la plateforme qui
   porte leur nom, et ses crédits sont gratuits. Roboflow ne prend le relais
   que si ModelScope refuse (modèle absent, quota, panne) — jamais d'un autre
   modèle : toujours le même, par la voie de secours. */
const MS_FIRST = {
  "qwen-3-8-max": true,
  "qwen-3-8-2-4t": true,
  "qwen-3-8-flash-next": true,
  "qwen-3-8-27b": true,
  "groq-qwen-3-8-27b": true,
  "deepseek-v4-1-flash": true,
  "kimi-k3": true,
  "glm-5-3": true,
  "glm-5-3-flash": true,
  "mimo-v2-6-pro": true,
};
const MS_IMG = {
  "qwen-image-2-1": "Qwen/Qwen-Image-2.1",
  "qwen-image-2-1-turbo": "Qwen/Qwen-Image-2.1-Turbo",
  /* Wan 2.2 n'est plus ici : la famille Wan 2.2 de ModelScope est vidéo
     (texte→image n'existe pas sur leur inférence), elle passe par /api/video. */
};
function msKey(env) { return String(env.MODELSCOPE_API_KEY || "").trim(); }
/* Un appel de texte chez ModelScope. Renvoie { ok, output } ou { ok:false, error }. */
async function msChat(env, msModel, prompt, params, image) {
  const key = msKey(env);
  if (!key) return { ok: false, error: "secret MODELSCOPE_API_KEY manquant côté Worker" };
  const p = params && typeof params === "object" ? params : {};
  /* L'application joint toujours une image (un pixel blanc 1×1) quand la
     demande est purement textuelle. ModelScope refuse ce faux format :
     « invalid image format ». On ne lui envoie donc une image que si c'en
     est vraiment une — c'est-à-dire nettement plus qu'un pixel. */
  const vraieImage = !!image && String(image).length > 200;
  const content = [];
  if (vraieImage) content.push({ type: "image_url", image_url: { url: dataUri({ b64: image, mime: detectMime(image) }) } });
  content.push({ type: "text", text: String(prompt || "") });
  const body = {
    model: msModel,
    messages: [{ role: "user", content: vraieImage ? content : String(prompt || "") }],
    max_tokens: Math.max(64, Math.min(32000, Number(p.max_tokens) || 4096)),
  };
  if (p.temperature !== undefined && p.temperature !== null && p.temperature !== "") body.temperature = Number(p.temperature);
  if (p.top_p) body.top_p = Number(p.top_p);
  /* Un hôte peut refuser le jeton, l'autre l'accepter : on les essaie dans
     l'ordre, et on ne réessaie que si le refus vient de l'authentification. */
  let r = null, t = "", j = null, dernier = "";
  for (const host of MS_HOSTS) {
    try {
      r = await fetch(host + "/chat/completions", {
        method: "POST",
        headers: { "Content-Type": "application/json", Authorization: "Bearer " + key, "User-Agent": BROWSER_UA },
        body: JSON.stringify(body),
      });
    } catch (e) {
      dernier = "ModelScope injoignable : " + (e && e.message ? e.message : e);
      r = null;
      continue;
    }
    t = await r.text();
    j = null;
    try { j = JSON.parse(t); } catch { j = null; }
    if (r.ok && j) break;
    const mm = (j && j.error && (j.error.message || j.error.code)) || ("HTTP " + r.status);
    dernier = String(mm).slice(0, 300);
    /* « invalid model id » ou « no provider » ne changeront pas d'hôte :
       seule l'authentification justifie d'essayer le second. */
    if (!/authenticat|authorization|unauthor/i.test(dernier)) break;
  }
  if (!r || !r.ok || !j) {
    return { ok: false, error: "ModelScope : " + (dernier || "réponse vide") };
  }
  let out = "";
  try { out = String(j.choices[0].message.content ?? ""); } catch (e) { out = ""; }
  if (!out.trim()) return { ok: false, error: "ModelScope : réponse vide" };
  return { ok: true, output: out, msModel: msModel };
}
/* ---- Canal Arena (Firebase) ------------------------------------------
   Le canal « chan/main » fait le lien entre l'application et Arena. Ces deux
   routes parlent à Firestore au nom du compte ADMIN (identifiants Firebase
   stockés en secrets du worker : FIREBASE_EMAIL / FIREBASE_PASSWORD) — le
   compte de l'utilisateur, lui, n'est jamais touché.
     POST /api/arena/send  { text, kind, req, note }  → écrit un message
     GET  /api/arena/poll  ?since=<id>                → relit le canal */
const ARENA_PROJECT = "ai-agent-1234a";
const ARENA_FS = "https://firestore.googleapis.com/v1/projects/" + ARENA_PROJECT + "/databases/(default)/documents";
let ARENA_TOKEN = { id: "", uid: "", expires: 0 };
let ARENA_FS_ERR = null;               /* dernière erreur Firestore (diagnostic) */
let ARENA_LAST = null;                 /* dernier aller-retour (diagnostic) */

function fsStr(v) { return { stringValue: String(v == null ? "" : v) }; }
function fsInt(v) { return { integerValue: String(Math.round(Number(v) || 0)) }; }
function fsVal(f) {
  if (!f || typeof f !== "object") return null;
  if (f.stringValue !== undefined) return f.stringValue;
  if (f.integerValue !== undefined) return Number(f.integerValue);
  if (f.doubleValue !== undefined) return Number(f.doubleValue);
  if (f.booleanValue !== undefined) return !!f.booleanValue;
  return null;
}
function fsDoc(d) {
  const out = {};
  for (const k of Object.keys(d.fields || {})) out[k] = fsVal(d.fields[k]);
  out.id = String(d.name || "").split("/").pop();
  return out;
}
/* Jeton du compte de service : connexion par courriel + mot de passe, renouvelé
   à l'expiration. Aucune valeur n'est jamais journalisée ni rendue au client. */
async function arenaToken(env) {
  const email = String(env.FIREBASE_EMAIL || "").trim();
  const pass = String(env.FIREBASE_PASSWORD || "");
  if (!email || !pass) return "";
  if (ARENA_TOKEN.id && ARENA_TOKEN.expires > Date.now() + 60000) return ARENA_TOKEN.id;
  const r = await fetch("https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key=AIzaSyC58J1paO0ULhTW0RuDfUePVeuOUmG5fZY", {
    method: "POST",
    headers: { "Content-Type": "application/json", "User-Agent": BROWSER_UA },
    body: JSON.stringify({ email: email, password: pass, returnSecureToken: true }),
  });
  let j = await r.json().catch(() => null);
  if (r.ok && j && j.idToken) {
    ARENA_TOKEN = { id: j.idToken, uid: String(j.localId || ""), expires: Date.now() + (Number(j.expiresIn || 3600) * 1000) };
    return ARENA_TOKEN.id;
  }
  ARENA_LAST = { at: Date.now(), action: "login", ok: false, statut: r.status };
  /* Repli : une session ANONYME. Les règles du projet autorisent cette voie
     (l'application elle-même s'en sert), et elle évite de rester bloqué si le
     compte du service est momentanément refusé. */
  try {
    const r2 = await fetch("https://identitytoolkit.googleapis.com/v1/accounts:signUp?key=AIzaSyC58J1paO0ULhTW0RuDfUePVeuOUmG5fZY", {
      method: "POST",
      headers: { "Content-Type": "application/json", "User-Agent": BROWSER_UA },
      body: JSON.stringify({ returnSecureToken: true }),
    });
    const j2 = await r2.json().catch(() => null);
    if (r2.ok && j2 && j2.idToken) {
      ARENA_TOKEN = { id: j2.idToken, uid: String(j2.localId || ""), expires: Date.now() + (Number(j2.expiresIn || 3600) * 1000) };
      ARENA_LAST = { at: Date.now(), action: "login", ok: true, mode: "anonyme" };
      return ARENA_TOKEN.id;
    }
    ARENA_LAST = { at: Date.now(), action: "login", ok: false, statut: r2.status };
  } catch (e) { /* réseau : on réessaiera */ }
  return "";
}
async function arenaWrite(env, doc) {
  const tok = await arenaToken(env);
  if (!tok) return { ok: false, error: "compte du canal non configuré (FIREBASE_EMAIL / FIREBASE_PASSWORD)" };
  const r = await fetch(ARENA_FS + "/chan/main/box", {
    method: "POST",
    headers: { "Content-Type": "application/json", Authorization: "Bearer " + tok, "User-Agent": BROWSER_UA },
    body: JSON.stringify({ fields: doc }),
  });
  const j = await r.json().catch(() => null);
  if (!r.ok || !j) return { ok: false, error: "écriture refusée (" + r.status + ")" };
  return { ok: true, id: String(j.name || "").split("/").pop() };
}
async function arenaRead(env, n) {
  const tok = await arenaToken(env);
  if (!tok) return { ok: false, error: "compte du canal non configuré" };
  const url = ARENA_FS + "/chan/main/box?orderBy=ts%20desc&pageSize=" + Math.min(200, Math.max(1, n || 20));
  const r = await fetch(url, { headers: { Authorization: "Bearer " + tok, "User-Agent": BROWSER_UA } });
  const j = await r.json().catch(() => null);
  if (!r.ok || !j) return { ok: false, error: "lecture refusée (" + r.status + ")" };
  const docs = (j.documents || []).map(fsDoc);
  docs.sort((a, b) => (a.ts || 0) - (b.ts || 0));
  return { ok: true, messages: docs };
}

/* ---- le résultat d'une mission est écrit DANS SA DISCUSSION -------------
   C'est la passerelle qui enregistre, pas le navigateur : une mission qui se
   termine pendant que l'application est fermée (ou depuis un autre appareil)
   est donc rangée quand même dans la discussion qui l'a lancée, avec ses
   étapes et son résultat final. Le compte du propriétaire sert d'identité ;
   le jeton reste ici, jamais dans la page. */
function fsNewVal(o) {
  if (o === null || o === undefined) return { nullValue: null };
  if (typeof o === "boolean") return { booleanValue: o };
  if (typeof o === "number") return Number.isInteger(o) ? { integerValue: String(o) } : { doubleValue: o };
  if (typeof o === "string") return { stringValue: o };
  if (Array.isArray(o)) return { arrayValue: { values: o.map(fsNewVal) } };
  if (typeof o === "object") {
    const f = {};
    for (const k of Object.keys(o)) f[k] = fsNewVal(o[k]);
    return { mapValue: { fields: f } };
  }
  return { stringValue: String(o) };
}
/* PATCH avec masque : sans masque, Firestore REMPLACE tout le document et les
   autres champs (le titre, les réglages) disparaîtraient. */
async function fbWrite(env, path, fields) {
  const tok = await arenaToken(env);
  if (!tok) return { ok: false, error: "compte Firebase non configuré" };
  const mask = Object.keys(fields).map((k) => "updateMask.fieldPaths=" + encodeURIComponent(k)).join("&");
  const body = {};
  for (const k of Object.keys(fields)) body[k] = fsNewVal(fields[k]);
  const r = await fetch(ARENA_FS + "/" + path + "?" + mask, {
    method: "PATCH",
    headers: { "Content-Type": "application/json", Authorization: "Bearer " + tok, "User-Agent": BROWSER_UA },
    body: JSON.stringify({ fields: body }),
  });
  if (!r.ok) return { ok: false, error: "écriture refusée (" + r.status + ")" };
  return { ok: true };
}
/* Un message trop gros pour un document (une mission peut produire des
   mégaoctets de sorties) part en morceaux, exactement comme le fait
   l'application : en le rouvrant, il est recollé sans rien perdre. */
async function fbWriteMessage(env, uid, cid, id, doc) {
  const small = {}, big = {};
  for (const k of Object.keys(doc)) {
    if (k === "role" || k === "order" || k === "createdAt" || k === "modelName" || k === "kind" || k === "agent") small[k] = doc[k];
    else big[k] = doc[k];
  }
  const base = "users/" + uid + "/chats/" + cid + "/messages/" + id;
  const jsonJSON = JSON.stringify(big);
  if (jsonJSON.length <= 450000) return fbWrite(env, base, doc);
  const parts = [];
  for (let i = 0; i < jsonJSON.length; i += 400000) parts.push(jsonJSON.slice(i, i + 400000));
  const head = await fbWrite(env, base, Object.assign({}, small, { parts: parts.length }));
  if (!head.ok) return head;
  for (let i = 0; i < parts.length; i++) {
    const r = await fbWrite(env, base + "/parts/p" + i, { d: parts[i] });
    if (!r.ok) return r;
  }
  return { ok: true };
}
/* Numéro du dernier message de la discussion : le nouveau passe après, sans
   jamais écraser ce qui est déjà là. */
async function fbLastOrder(env, uid, cid) {
  const tok = await arenaToken(env);
  if (!tok) return 0;
  try {
    const r = await fetch(ARENA_FS + "/users/" + uid + "/chats/" + cid + "/messages?orderBy=order%20desc&pageSize=1",
      { headers: { Authorization: "Bearer " + tok, "User-Agent": BROWSER_UA } });
    const j = await r.json().catch(() => null);
    const d = j && j.documents && j.documents[0];
    if (!d || !d.fields || !d.fields.order) return 0;
    return Number(fsVal(d.fields.order)) || 0;
  } catch (e) { return 0; }
}
/* Livre une mission terminée dans sa discussion : la question, la réponse, les
   étapes — puis on note que c'est fait (jamais deux fois). */
async function jobDeliver(env, j) {
  if (!j || j.delivered || !j.chat || !j.uid) return false;
  if (j.noDeliver) { j.delivered = true; return false; }
  const uid = String(j.uid), cid = String(j.chat);
  const base = await fbLastOrder(env, uid, cid);
  const o1 = base + 1, o2 = base + 2;
  const finalText = String(j.finalMd || "") || (j.error ? "*(" + j.error + ")*" : "");
  const aText = (j.finalMd && j.error) ? (String(j.finalMd) + "\n\n*(" + j.error + ")*") : finalText;
  const w1 = await fbWriteMessage(env, uid, cid, "m" + o1, {
    role: "user", text: String(j.mission || ""), kind: "none", modelName: String(j.model || ""),
    order: o1, createdAt: Number(j.startedAt) || Date.now(), images: [],
  });
  const w2 = await fbWriteMessage(env, uid, cid, "m" + o2, {
    role: "assistant", text: aText, modelName: String(j.model || ""),
    order: o2, createdAt: Date.now(), steps: (j.steps || []).slice(0, 400), agent: true,
  });
  if (!w1.ok || !w2.ok) {
    ARENA_LAST = { at: Date.now(), action: "mission", ok: false, error: (w1.error || w2.error || "") };
    return false;
  }
  const state = (j.error && !j.finalMd) ? "error" : "done";
  const w3 = await fbWrite(env, "users/" + uid + "/chats/" + cid, {
    updatedAt: Date.now(),
    job: { id: String(j.id || ""), state: state, at: Date.now(), mission: String(j.mission || ""), model: String(j.model || ""), saved: "passerelle" },
  });
  ARENA_LAST = { at: Date.now(), action: "mission", ok: true, chat: cid, messages: [o1, o2] };
  j.delivered = true;
  return !!(w3 && w3.ok);
}

/* L'application est-elle ouverte ? Le marqueur de présence est écrit par
   l'app elle-même (chan/main.arenaApp) et rafraîchi toutes les minutes : il est
   considéré comme frais s'il a moins de trois minutes. C'est ce qui permet à
   Arena de savoir s'il parle à quelqu'un ou dans le vide. */
async function arenaAppOnline(env) {
  try {
    const tok = await arenaToken(env);
    if (!tok) return false;
    const r = await fetch(ARENA_FS + "/chan/main", { headers: { Authorization: "Bearer " + tok, "User-Agent": BROWSER_UA } });
    const j = await r.json().catch(() => null);
    const m = j && j.fields && j.fields.arenaApp && j.fields.arenaApp.mapValue;
    if (!m || !m.fields) return false;
    const online = fsVal(m.fields.online) === true;
    const at = fsVal(m.fields.at) || 0;
    return online && (Date.now() - at) < 180000;
  } catch (e) { return false; }
}

/* Dernier échange avec le service d'images de ModelScope (diagnostic) :
   aucun secret n'y figure, seulement ce qui est envoyé et ce qui revient. */
let MS_IMG_LAST = null;

/* Une image chez ModelScope : soumission, sondage, téléchargement. */
async function msImage(env, msModel, ib) {
  const key = msKey(env);
  if (!key) return json({ error: "secret MODELSCOPE_API_KEY manquant côté Worker" }, 500);
  const p = (ib.params && typeof ib.params === "object") ? ib.params : {};
  const body = { model: msModel, prompt: String(ib.prompt || "") };
  /* Résolution : le champ `size` de l'API /images/generations accepte une
     taille précise (« largeurxhauteur »). L'app l'envoie sous le nom
     `resolution` (pastilles), les anciens réglages sous width/height.
     Sans réglage, on demande la taille native de Qwen Image (1328 × 1328) :
     la valeur 1024 par défaut donnait une image visiblement moins fine. */
  const rres = /^(\d{2,4})\s*[x×*]\s*(\d{2,4})$/i.exec(String(p.resolution || "").trim());
  const snap16 = (n) => Math.max(64, Math.min(2048, Math.round(n / 16) * 16));
  const rw = rres ? Number(rres[1]) : int(p.width, 64, 2048);
  const rh = rres ? Number(rres[2]) : int(p.height, 64, 2048);
  body.size = snap16(rw || 1328) + "x" + snap16(rh || 1328);
  if ((p.num_steps = int(p.num_steps, 1, 100)) !== undefined) body.num_inference_steps = p.num_steps;
  if (ib.image) body.image_url = dataUri({ b64: String(ib.image), mime: detectMime(String(ib.image)) });
  const H = { "Content-Type": "application/json", Authorization: "Bearer " + key,
    "User-Agent": BROWSER_UA };
  let sub;
  try {
    sub = await fetch(MS_BASE + "/images/generations", {
      method: "POST",
      headers: Object.assign({ "X-ModelScope-Async-Mode": "true" }, H),
      body: JSON.stringify(body),
    });
  } catch (e) { return json({ error: "ModelScope injoignable : " + (e && e.message ? e.message : e) }, 502); }
  const st = await sub.text();
  let sj = null;
  try { sj = JSON.parse(st); } catch { sj = null; }
  if (!sub.ok || !sj) {
    const m = (sj && sj.error && (sj.error.message || sj.error.code)) || ("HTTP " + sub.status);
    MS_IMG_LAST = { at: Date.now(), model: msModel, taille: body.size || "(défaut)", etat: "refus",
      soumission: sj, statut: sub.status, brut: String(st || "").slice(0, 600) };
    return json({ error: "ModelScope « " + msModel + " » : " + String(m).slice(0, 300) }, sub.status === 401 ? 401 : 502);
  }
  /* La soumission peut aussi répondre 200 avec un code d'erreur dans le corps. */
  if (sj.code && sj.message && !sj.task_id && !sj.output_images) {
    MS_IMG_LAST = { at: Date.now(), model: msModel, taille: body.size || "(défaut)", etat: "refus",
      soumission: sj, statut: sub.status };
    return json({ error: "ModelScope « " + msModel + " » : " + String(sj.message).slice(0, 300) }, 502);
  }
  /* Toute la réponse est conservée : c'est elle qui dit pourquoi ça ne marche pas. */
  MS_IMG_LAST = { at: Date.now(), model: msModel, taille: body.size || "(défaut)", soumission: sj,
    tailleReponse: String(st || "").length };
  /* Certaines versions rendent l'image tout de suite ; sinon on sonde la tâche. */
  let urls = (sj.output_images || (sj.data && sj.data[0] && (sj.data[0].url || sj.data[0].b64_json)) || []).length ? (sj.output_images || []) : [];
  if (!urls.length && sj.task_id) {
    const t0 = Date.now();
    for (let i = 0; i < 40; i++) {
      if (Date.now() - t0 > 180000) break;
      await new Promise((res) => setTimeout(res, i < 4 ? 1200 : 2500));
      let pr;
      try {
        pr = await fetch(MS_BASE + "/tasks/" + encodeURIComponent(sj.task_id), {
          headers: { Authorization: "Bearer " + key, "X-ModelScope-Task-Type": "image_generation", "User-Agent": BROWSER_UA },
        });
      } catch (e) { continue; }
      let pj = null;
      const pt = await pr.text();
      try { pj = JSON.parse(pt); } catch { pj = null; }
      const status = String((pj && pj.task_status) || "").toUpperCase();
      if (status === "SUCCEED") { urls = (pj && pj.output_images) || []; break; }
      if (status === "FAILED") {
        MS_IMG_LAST = Object.assign({}, MS_IMG_LAST, { at: Date.now(), etat: "échec", tache: pj });
        return json({ error: "ModelScope « " + msModel + " » : génération échouée — " +
          String(JSON.stringify((pj && (pj.error || pj.message)) || pj || {})).slice(0, 250) }, 502);
      }
    }
  }
  const url = urls && urls[0];
  if (!url) {
    MS_IMG_LAST = Object.assign({}, MS_IMG_LAST, { at: Date.now(), etat: "sans image" });
    return json({ error: "ModelScope « " + msModel + " » : aucune image rendue (tâche trop longue ou refusée)" }, 502);
  }
  MS_IMG_LAST = Object.assign({}, MS_IMG_LAST, { at: Date.now(), etat: "ok", image: String(url).slice(0, 120) });
  /* b64_json éventuel : déjà prêt. */
  if (/^[A-Za-z0-9+/=]{200,}$/.test(String(url))) return json({ image: String(url), mime: detectMime(String(url)), via: "modelscope" });
  let im;
  try { im = await fetch(url, { headers: { "User-Agent": BROWSER_UA } }); }
  catch (e) {
    MS_IMG_LAST = Object.assign({}, MS_IMG_LAST, { at: Date.now(), etat: "téléchargement impossible", url: String(url).slice(0, 160), motif: String(e && e.message ? e.message : e) });
    return json({ error: "image ModelScope non téléchargeable : " + (e && e.message ? e.message : e) }, 502);
  }
  if (!im.ok) {
    MS_IMG_LAST = Object.assign({}, MS_IMG_LAST, { at: Date.now(), etat: "téléchargement refusé", url: String(url).slice(0, 160), statut: im.status });
    return json({ error: "image ModelScope non téléchargeable (HTTP " + im.status + ")" }, 502);
  }
  const buf = new Uint8Array(await im.arrayBuffer());
  const typeRecu = (im.headers.get("Content-Type") || "").split(";")[0];
  MS_IMG_LAST = Object.assign({}, MS_IMG_LAST, { at: Date.now(), url: String(url).slice(0, 160),
    octets: buf.length, type: typeRecu });
  /* Un corps vide n'est pas une image : on le dit, au lieu de rendre une
     image vide que l'app afficherait comme un texte. */
  if (!buf.length) {
    MS_IMG_LAST = Object.assign({}, MS_IMG_LAST, { etat: "corps vide" });
    return json({ error: "ModelScope a rendu une image vide (0 octet, type « " + (typeRecu || "?") + " »)" }, 502);
  }
  /* b64encode attend les OCTETS : lui passer une chaîne donnait une chaîne
     vide (d'où les images « générées » mais sans contenu). */
  const b64 = b64encode(buf);
  const mime = typeRecu || detectMime(b64);
  if (!b64) {
    MS_IMG_LAST = Object.assign({}, MS_IMG_LAST, { etat: "encodage vide" });
    return json({ error: "image ModelScope illisible après téléchargement (" + buf.length + " octets reçus)" }, 502);
  }
  return json({ image: b64, mime: mime, via: "modelscope" });
}
/* Appel quotidien : le jeton s'authentifie auprès de ModelScope (identité +
   solde de crédits). Renvoie { ok, token, identity, balance }. */
async function msDaily(env) {
  const key = msKey(env);
  if (!key) return { ok: false, error: "secret MODELSCOPE_API_KEY manquant côté Worker", token: "absent" };
  const H = { Authorization: "Bearer " + key, "User-Agent": BROWSER_UA, Accept: "application/json" };
  const out = { ok: false, token: "inconnu", identity: "", balance: null, error: "" };
  try {
    let r = null, t = "";
    for (const site of MS_SITES) {
      try {
        r = await fetch(site + "/openapi/v1/magicubes/balance", { headers: H });
        t = await r.text();
      } catch (e) { r = null; continue; }
      if (r.ok) break;
    }
    if (!r) return { ok: false, token: "inconnu", error: "ModelScope injoignable", identity: "", balance: null };
    let j = null;
    try { j = JSON.parse(t); } catch { j = null; }
    if (r.status === 401 || r.status === 403 || (j && j.code === "InvalidAuthentication")) {
      out.token = "refusé";
      out.error = "jeton refusé par ModelScope (" + String((j && j.message) || r.status).slice(0, 200) + ")";
      return out;
    }
    if (r.ok && j && j.success && j.data) {
      out.balance = { total: j.data.total_balance, available: j.data.available_balance, frozen: j.data.frozen_amount };
      out.token = "valide";
    }
  } catch (e) {
    out.error = "ModelScope injoignable : " + (e && e.message ? e.message : e);
    return out;
  }
  try {
    let r2 = { ok: false };
    for (const site of MS_SITES) {
      try { r2 = await fetch(site + "/openapi/v1/users/me", { headers: H }); } catch (e) { r2 = { ok: false }; }
      if (r2.ok) break;
    }
    if (r2.ok) {
      const j2 = await r2.json().catch(() => null);
      const d = (j2 && j2.data) || {};
      out.identity = String(d.name || d.nickname || d.username || "");
      if (out.token === "inconnu") out.token = "valide";
    }
  } catch (e) { /* l'identité est un bonus, pas une condition */ }
  out.ok = out.token === "valide";
  /* Le catalogue est lu au passage : c'est une simple lecture (aucun crédit
     consommé) et il dit quels noms de modèles configurés existent vraiment
     chez ModelScope — donc lesquels sont à corriger. */
  if (out.ok) {
    try {
      const r3 = await fetch(MS_HOSTS[0] + "/models", { headers: H });
      const j3 = await r3.json();
      const ids = (j3 && j3.data ? j3.data : []).map((m) => m.id);
      if (ids.length) {
        out.catalog = ids.length;
        out.models = {};
        for (const [k, v] of Object.entries(MS_TEXT)) out.models[k] = ids.indexOf(v) >= 0 ? "ok" : "inconnu";
        for (const [k, v] of Object.entries(MS_IMG)) out.models[k] = ids.indexOf(v) >= 0 ? "ok" : "inconnu";
      }
    } catch (e) { /* le catalogue est un bonus, jamais une condition */ }
  }
  return out;
}

/* ---- Jetons Roboflow, en rotation ------------------------------------
   Certaines erreurs rares viennent des limites propres à un jeton d'API (et
   non du modèle) : un jeton peut être refusé là où un autre passe, à modèle et
   organisation identiques. On dispose donc de plusieurs jetons de la MÊME
   organisation (crédits partagés) et l'on passe au suivant dès qu'une
   tentative échoue, en alternant les voies :
     jeton 1 voie principale -> jeton 1 OpenRouter -> jeton 2 voie principale
     -> jeton 2 OpenRouter -> …
   Le dernier jeton est le jeton historique (ROBOFLOW_API_KEY). */
function roboflowTokens(env) {
  /* Les jetons portent tous le même nom, numéroté : ROBOFLOW_API_KEY_1 … _8.
     L'ancien nom (ROBOFLOW_API_KEY) et les RF_TOKEN_* restent lus en secours,
     pour qu'aucun déploiement ne perde un jeton au passage. */
  const order = [];
  for (let i = 1; i <= 8; i++) {
    /* _4 : numéroté s'il existe, sinon le jeton historique (même clé). */
    order.push(env["ROBOFLOW_API_KEY_" + i] || (i === 4 ? env.ROBOFLOW_API_KEY : ""));
  }
  order.push(env.RF_TOKEN_1, env.RF_TOKEN_2, env.RF_TOKEN_3);
  const out = [];
  for (const t of order) {
    const k = String(t || "").trim();
    if (k && out.indexOf(k) < 0) out.push(k);      /* jamais deux fois le même */
  }
  return out;
}
/* Développe une liste de blocs en (jeton, bloc) : le jeton tourne le plus vite. */
function acrossTokens(env, cands) {
  const toks = roboflowTokens(env);
  if (!toks.length) return cands.slice();
  const out = [];
  for (const t of toks) for (const c of cands) out.push(Object.assign({}, c, { _token: t }));
  return out;
}

/* ---- Missions durables ------------------------------------------------
   Constat du 13/10 : un moteur qui vit seulement dans la mémoire d'un isolate
   disparaît dès que plus personne n'interroge la passerelle (l'isolate est
   recyclé) — la mission est alors « inconnue ou expirée », donc perdue.
   Chaque mission vit donc maintenant dans son propre Durable Object, rattaché
   à la plateforme Cloudflare : il poursuit le travail sur alarmes, survit au
   recyclage de l'isolate et au redéploiement du worker, et l'app peut s'y
   raccrocher à tout moment — même après avoir été fermée. ---- */
export class MissionJob {
  constructor(ctx, env) {
    this.ctx = ctx;
    this.env = env;
    this.state = null;
    /* À la (re)naissance de l'objet, on relit l'état et on relance la chaîne
       d'alarmes si la mission n'était pas terminée. */
    ctx.blockConcurrencyWhile(async () => {
      this.state = (await ctx.storage.get("job")) || null;
      if (this.state && !this.state.done && !this.state.stop) {
        this.state.updatedAt = Date.now();
        this.state.running = false;
        await ctx.storage.put("job", this.state);
        await ctx.storage.setAlarm(Date.now() + 50);
      } else if (this.state && this.state.done && !this.state.delivered) {
        /* Terminée mais pas encore rangée dans sa discussion : on repasse. */
        await ctx.storage.setAlarm(Date.now() + 50);
      }
    });
  }
  async load() {
    if (!this.state) this.state = (await this.ctx.storage.get("job")) || null;
    return this.state;
  }
  /* Écriture durable de l'état. La valeur rangée est ALLÉGÉE : la trace et les
     très longues sorties d'étapes ne sont pas nécessaires au stockage (elles
     sont déjà affichées dans l'app), alors qu'elles peuvent peser des mégaoctets
     — au-delà, l'écriture échouerait et la mission ne survivrait pas au
     redémarrage. Rien n'est perdu de ce qui compte : le déroulé, le résultat. */
  async save() {
    if (!this.state) return;
    const st = this.state;
    const steps = (st.steps || []);
    const light = steps.length > JOB_SAVE_STEPS ? steps.slice(steps.length - JOB_SAVE_STEPS) : steps;
    const out = light.map((x, i) => (i < light.length - 60) ? Object.assign({}, x, {
      out: tailKeep(x.out, JOB_SAVE_OUT), err: tailKeep(x.err, 200),
    }) : x);
    await this.ctx.storage.put("job", Object.assign({}, st, {
      steps: out, tr: tailKeep(st.tr, JOB_TR_KEEP), hist: tailKeep(st.hist, JOB_HIST_KEEP),
    }));
  }

  /* Le registre ne grossit pas sans fin : on ne garde que les 300 missions
     les plus récentes. */
  async pruneIndex() {
    if (!this.ctx.storage.list || !this.ctx.storage.delete) return;
    const m = await this.ctx.storage.list({ prefix: "job:" });
    const arr = [];
    if (m && typeof m.forEach === "function") m.forEach((v, k) => arr.push([k, v]));
    if (arr.length <= 300) return;
    arr.sort((a, b) => ((b[1].at || 0) - (a[1].at || 0)));
    for (const [k] of arr.slice(300)) { try { await this.ctx.storage.delete(k); } catch (e) {} }
  }

  async fetch(request) {
    const path = new URL(request.url).pathname;
    const st = await this.load();
    if (path === "/start") {
      let b = {};
      try { b = await request.json(); } catch { /* corps illisible */ }
      this.state = {
        id: String(b.id || ""), chat: String(b.chat || ""), uid: String(b.uid || ""),
        model: String(b.model || ""), mission: String(b.mission || ""),
        intro: String(b.intro || ""), rappel: String(b.rappel || ""), hist: String(b.hist || ""),
        image: String(b.image || ""), image_kind: String(b.image_kind || "none"),
        params: (b.params && typeof b.params === "object") ? b.params : {},
        origin: String(b.origin || ""), code: String(b.code || ""),
        steps: [], tr: "", finalMd: "", error: "", stop: false,
        step: 0, done: false, running: false, startedAt: Date.now(), updatedAt: Date.now(),
      };
      await this.save();
      await this.ctx.storage.setAlarm(Date.now() + 50);
      return json({ ok: true, job: this.state.id, durable: true });
    }
    if (path === "/state") {
      if (!st) return json({ ok: false, error: "mission inconnue ou expirée" }, 404);
      const since = new URL(request.url).searchParams.get("since");
      return json({ ok: true, job: jobPublic(st, since) });
    }
    /* Raccourcis d'identifiant : « hugdu77777 » -> adresse email du compte.
       Rangé dans le même objet durable que les missions, sous une instance
       dédiée (« aliases ») : rien à migrer, et le raccourci survit aux
       redéploiements. */
    /* Journal des neurones Cloudflare : un compteur par jour, tenu dans
       l'instance « credits ». Cloudflare ne publie pas de solde restant par
       API ; en revanche chaque réponse porte le coût réel de l'appel. On
       additionne donc ce que l'application a consommé aujourd'hui. */
    if (path === "/ledger") {
      const day = new Date().toISOString().slice(0, 10);
      let L = (await this.ctx.storage.get("ledger")) || null;
      if (!L || L.day !== day) L = { day: day, neurons: 0 };
      if (request.method === "POST") {
        let b = {};
        try { b = await request.json(); } catch { /* corps illisible */ }
        const n = Math.max(0, Math.min(1e9, Number(b.neurons) || 0));
        L.neurons = Math.round((L.neurons + n) * 100) / 100;
        await this.ctx.storage.put("ledger", L);
      }
      return json({ ok: true, day: L.day, neurons: L.neurons });
    }
    if (path === "/alias") {
      if (request.method === "POST") {
        let b = {};
        try { b = await request.json(); } catch { /* corps illisible */ }
        const name = String(b.name || "").trim().toLowerCase();
        const email = String(b.email || "").trim().toLowerCase();
        if (!name || !email || name.indexOf("@") >= 0) return json({ ok: false, error: "raccourci invalide" }, 400);
        await this.ctx.storage.put("alias:" + name, email);
        return json({ ok: true, name: name });
      }
      const q = new URL(request.url).searchParams.get("name") || "";
      const key = q.trim().toLowerCase();
      if (!key) return json({ ok: false, error: "nom manquant" }, 400);
      const em = await this.ctx.storage.get("alias:" + key);
      return json({ ok: !!em, email: em || "" });
    }
    /* Tenue du registre : une ligne par mission, pour pouvoir les LISTER. */
    if (path === "/reg") {
      let b = {};
      try { b = await request.json(); } catch { /* corps illisible */ }
      const jid = String(b.id || "");
      if (!jid) return json({ ok: false, error: "identifiant manquant" }, 400);
      const prev = (await this.ctx.storage.get("job:" + jid)) || {};
      const entry = {
        id: jid,
        chat: String(b.chat || prev.chat || ""),
        model: String(b.model || prev.model || ""),
        mission: String(b.mission || prev.mission || "").slice(0, 180),
        at: Number(b.at || prev.at || Date.now()),
        state: String(b.state || prev.state || "run"),
        updatedAt: Date.now(),
      };
      await this.ctx.storage.put("job:" + jid, entry);
      await this.pruneIndex();
      return json({ ok: true, id: jid });
    }
    if (path === "/list") {
      const jobs = [];
      if (this.ctx.storage.list) {
        const m = await this.ctx.storage.list({ prefix: "job:" });
        if (m && typeof m.forEach === "function") m.forEach((v) => jobs.push(v));
      }
      jobs.sort((a, b) => (b.at || 0) - (a.at || 0));
      return json({ ok: true, jobs: jobs.slice(0, 120) });
    }
    if (path === "/stop") {
      const force = new URL(request.url).searchParams.get("force") === "1";
      if (!st) return json({ ok: false, error: "mission inconnue" }, 404);
      if (!st.done) {
        st.stop = true;
        st.updatedAt = Date.now();
        /* Une étape est peut-être en cours : sans « force », c'est elle qui
           posera le motif en finissant. Avec « force », on tranche tout de
           suite : plus aucune alarme, plus aucune étape, la mission est dite
           arrêtée même si un appel réseau traîne encore. */
        if (force) {
          st.done = true; st.running = false;
          if (!st.error) st.error = "mission arrêtée à la demande";
        } else if (!st.running && !st.error) {
          st.error = "mission arrêtée à la demande"; st.done = true;
        }
        await this.save();
      }
      if (force) {
        try { if (this.ctx.storage.deleteAlarm) await this.ctx.storage.deleteAlarm(); } catch (e) {}
      }
      return json({ ok: true, forced: force, done: !!st.done });
    }
    return json({ ok: false, error: "route inconnue" }, 404);
  }

  /* Une alarme = une étape de mission. On garde chaque exécution courte et on
     reprogramme la suivante : la mission avance donc même si l'app est fermée
     et même si aucune requête n'arrive. */
  async alarm() {
    const st = await this.load();
    if (!st) return;
    /* Mission déjà finie : on garde l'état le temps que l'app vienne le
       chercher, puis on efface — rien ne s'accumule. */
    if (st.done || st.stop) {
      /* Le résultat part dans la discussion (une fois), puis l'état est gardé
         le temps que l'application vienne le chercher. Tant que la livraison
         n'a pas réussi, on réessaie au lieu d'effacer la mission. */
      if (!st.delivered && st.chat && st.uid) {
        try { await jobDeliver(this.env, st); } catch (e) {}
        if (!st.delivered) st.tries = (st.tries || 0) + 1;   /* essais RATÉS */
        await this.save();
      }
      const fini = st.delivered || !st.chat || !st.uid || (st.tries || 0) > 60;
      if (fini && Date.now() - (st.updatedAt || 0) > JOB_KEEP_MS) {
        await this.ctx.storage.deleteAll();
        this.state = null;
      } else {
        await this.ctx.storage.setAlarm(Date.now() + 5 * 60 * 1000);
      }
      return;
    }
    if (st.running) { await this.ctx.storage.setAlarm(Date.now() + 500); return; }
    st.running = true; st.updatedAt = Date.now();
    await this.save();          /* une étape en cours se voit depuis /stop */
    try {
      await jobStep(this.env, st);
    } catch (e) {
      st.error = String(e && e.message ? e.message : e);
      st.done = true;
    }
    /* Un arrêt forcé a pu être écrit sur le disque pendant l'étape (l'appel
       au modèle, lui, ne s'interrompt pas). S'il dit « terminée », il gagne :
       on jette le résultat de cette étape et on n'écrit rien par-dessus. */
    const disk = await this.ctx.storage.get("job");
    if (disk && disk.done && disk.stop) {
      this.state = disk;
      try { if (this.ctx.storage.deleteAlarm) await this.ctx.storage.deleteAlarm(); } catch (e) {}
      return;
    }
    st.running = false;
    if (!st.done) {
      if (st.stop) {
        st.error = "mission arrêtée à la demande";
        st.done = true;
      }
    }
    st.updatedAt = Date.now();
    /* Mission finie : le résultat est écrit dans sa discussion TOUT DE SUITE,
       sans attendre que l'application soit là. */
    if (st.done && !st.delivered && st.chat && st.uid) {
      try { await jobDeliver(this.env, st); } catch (e) {}
    }
    await this.save();
    await this.ctx.storage.setAlarm(st.done ? (Date.now() + JOB_KEEP_MS + 1000) : (Date.now() + 250));
  }
}

/* ---- Application : index.html servi en HTTPS sur /app (canal de livraison
   fiable, sans transit corrupteur ; no-store pour éviter toute version périmée) */
/*APP_HTML_START*/
const APP_HTML = "<!DOCTYPE html>\n<html lang=\"fr\">\n<head>\n<meta charset=\"utf-8\">\n<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n<meta http-equiv=\"cache-control\" content=\"no-store\">\n<title>AI Agent</title>\n<style>\n:root{\n  --bg:#ffffff; --ink:#111113; --ink-2:#55565c; --ink-3:#8b8c93;\n  --line:#e8e8ec; --line-2:#f1f1f4; --accent:#111113;\n  --code-bg:#0d0d0f; --code-ink:#e6e6e9;\n  --mono:ui-monospace,SFMono-Regular,\"SF Mono\",Menlo,Consolas,\"Liberation Mono\",monospace;\n  --sans:-apple-system,BlinkMacSystemFont,\"Segoe UI\",Roboto,Helvetica,Arial,sans-serif;\n}\n*{box-sizing:border-box}\n/* « hidden » doit gagner contre les display:inline-flex des boutons. */\n[hidden]{display:none !important}\nhtml,body{height:100%}\nbody{\n  margin:0;background:var(--bg);color:var(--ink);\n  font-family:var(--sans);font-size:15px;line-height:1.6;\n  -webkit-font-smoothing:antialiased;\n  display:flex;flex-direction:column;\n}\n\n/* ---------- header ---------- */\nheader{\n  flex:0 0 auto;display:flex;align-items:center;gap:14px;\n  padding:10px 20px;border-bottom:1px solid var(--line);\n  position:sticky;top:0;background:rgba(255,255,255,.92);backdrop-filter:blur(8px);z-index:20;\n}\n.brand{display:flex;align-items:center;gap:9px;font-weight:600;letter-spacing:-.01em;font-size:14px;white-space:nowrap}\n.brand svg{display:block}\n.pills{display:flex;gap:5px;flex-wrap:wrap;flex:1 1 auto;min-width:0}\n.pill{\n  display:inline-flex;align-items:center;gap:6px;\n  padding:5px 10px 5px 8px;border:1px solid var(--line);border-radius:999px;\n  background:#fff;color:var(--ink-2);font-size:12.5px;font-family:var(--sans);\n  cursor:pointer;transition:border-color .12s,color .12s,background .12s;white-space:nowrap;\n}\n.pill:hover{border-color:#cfcfd6;color:var(--ink)}\n.pill[aria-pressed=\"true\"]{background:var(--ink);border-color:var(--ink);color:#fff}\n.pill svg{display:block;flex:0 0 auto}\n.pill[aria-pressed=\"true\"] svg .mk{fill:#fff;stroke:#fff}\n.iconbtn{\n  width:30px;height:30px;display:inline-flex;align-items:center;justify-content:center;\n  border:1px solid var(--line);border-radius:8px;background:#fff;cursor:pointer;color:var(--ink-2);\n}\n.iconbtn:hover{border-color:#cfcfd6;color:var(--ink)}\n.iconbtn.on{background:var(--ink);border-color:var(--ink);color:#fff}\n\n#proxywarn{\n  flex:0 0 auto;padding:9px 20px;border-bottom:1px solid #f0dcc0;background:#fffbf3;\n  color:#6b4e16;font-size:12.5px;line-height:1.55;\n}\n#proxywarn[hidden]{display:none}\n#proxywarn code{\n  font-family:var(--mono);font-size:11.5px;background:#f6ecd8;\n  padding:.1em .35em;border-radius:4px;\n}\n\n/* ---------- conversation ---------- */\nmain{flex:1 1 auto;overflow-y:auto;padding:26px 20px 34px}\n.wrap{max-width:780px;margin:0 auto}\n.msg{margin:0 0 30px}\n.msg .who{\n  display:flex;align-items:center;gap:7px;\n  font-size:11px;font-weight:600;letter-spacing:.07em;text-transform:uppercase;\n  color:var(--ink-3);margin-bottom:9px;\n}\n.msg.user .who{color:var(--ink-2)}\n.bubble{min-width:0}\n.msg.user .bubble{\n  background:#fafafb;border:1px solid var(--line);border-radius:12px;padding:12px 15px;\n}\n.empty{color:var(--ink-3);font-size:14px;text-align:center;padding:70px 0 40px}\n.empty p{margin:0 0 6px}\n.err{\n  border:1px solid #f0c9c9;background:#fdf6f6;color:#8a2c2c;\n  border-radius:10px;padding:11px 14px;font-family:var(--mono);font-size:12px;white-space:pre-wrap;\n}\n.thinking{display:inline-flex;gap:4px;align-items:center;height:18px}\n.thinking i{width:5px;height:5px;border-radius:50%;background:var(--ink-3);animation:b 1.1s infinite ease-in-out}\n.thinking i:nth-child(2){animation-delay:.16s}.thinking i:nth-child(3){animation-delay:.32s}\n/* « Generating… » — indicateur d'attente texte, points allumés en séquence. */\n.gentxt{display:inline-flex;align-items:baseline;font:12px/18px var(--mono);color:var(--ink-2);letter-spacing:.03em}\n.gentxt i{font-style:normal;animation:gdot 1.15s infinite linear}\n.gentxt i:nth-of-type(2){animation-delay:.16s}\n.gentxt i:nth-of-type(3){animation-delay:.32s}\n/* Les points apparaissent un à un (vitesse normale) puis s'effacent\n   de gauche à droite, plus vite qu'ils ne sont apparus. */\n@keyframes gdot{0%{opacity:0}28%{opacity:1}56%{opacity:1}70%{opacity:0}100%{opacity:0}}\n/* Attente d'image : zone grise arrondie, chronomètre, reflets animés. */\n.vidnote{margin:8px 0 0;font-size:12.5px;color:var(--ink-3);max-width:420px;line-height:1.45}\n.imgwait{position:relative;overflow:hidden;width:min(100%,420px);aspect-ratio:1/1;\n  border-radius:14px;background:#e9e9e6;display:flex;align-items:center;justify-content:center}\n.imgwait::before{content:\"\";position:absolute;inset:-45%;\n  background:linear-gradient(105deg,transparent 38%,rgba(255,255,255,.85) 50%,transparent 62%);\n  animation:sheen 2.1s ease-in-out infinite}\n@keyframes sheen{0%{transform:translateX(-58%)}100%{transform:translateX(58%)}}\n.imgwait .clock{position:relative;z-index:1;display:flex;align-items:center;gap:8px;\n  font:500 14px/1 var(--mono);color:var(--ink-2);background:rgba(255,255,255,.72);\n  padding:8px 13px;border-radius:999px;animation:pulse 1.5s ease-in-out infinite}\n.imgwait .clock svg{width:15px;height:15px}\n@keyframes pulse{0%,100%{background:rgba(255,255,255,.6)}50%{background:rgba(255,255,255,.92)}}\n/* Visionneuse : image en grand sur fond grisé, croix en haut à droite. */\n.lightbox{position:fixed;inset:0;z-index:70;display:flex;align-items:center;justify-content:center;\n  padding:26px;background:rgba(17,17,16,.62);animation:lbin .18s ease}\n@keyframes lbin{from{opacity:0}to{opacity:1}}\n.lightbox img{max-width:100%;max-height:100%;object-fit:contain;border-radius:10px;background:#fff}\n.lbclose{position:fixed;top:16px;right:16px;width:38px;height:38px;border:0;cursor:pointer;\n  border-radius:11px;background:rgba(224,224,220,.72);color:#111;\n  display:flex;align-items:center;justify-content:center;padding:0}\n.lbclose:hover{background:rgba(224,224,220,.92)}\n.lbclose svg{width:17px;height:17px}\n.thinking i{width:5px;height:5px;border-radius:50%;background:var(--ink-3);animation:b 1.1s infinite ease-in-out}\n.thinking i:nth-child(2){animation-delay:.16s}.thinking i:nth-child(3){animation-delay:.32s}\n@keyframes b{0%,80%,100%{opacity:.22;transform:translateY(0)}40%{opacity:1;transform:translateY(-3px)}}\n\n/* ---------- markdown ---------- */\n.md>*:first-child{margin-top:0}\n.md>*:last-child{margin-bottom:0}\n.md h1,.md h2,.md h3,.md h4{margin:1.5em 0 .55em;line-height:1.28;font-weight:650;letter-spacing:-.015em}\n.md h1{font-size:1.45em}.md h2{font-size:1.24em}.md h3{font-size:1.09em}.md h4{font-size:1em}\n.md h1{border-bottom:1px solid var(--line);padding-bottom:.3em}\n.md h2{border-bottom:1px solid var(--line-2);padding-bottom:.25em}\n.md p{margin:0 0 1em}\n.md a{color:var(--ink);text-decoration:underline;text-underline-offset:2px;text-decoration-color:#c9c9d1}\n.md ul,.md ol{margin:0 0 1em;padding-left:1.7em}\n.md li::marker{color:var(--ink-3)}\n.md li{margin:.25em 0}\n.md li>ul,.md li>ol{margin:.25em 0}\n.md blockquote{\n  margin:0 0 1em;padding:.1em 0 .1em 14px;border-left:2px solid var(--line);color:var(--ink-2);\n}\n.md hr{border:0;border-top:1px solid var(--line);margin:1.6em 0}\n.md table{border-collapse:collapse;margin:0 0 1em;width:100%;font-size:13.5px;display:block;overflow-x:auto}\n.md th,.md td{border:1px solid var(--line);padding:6px 10px;text-align:left}\n.md th{background:#fafafb;font-weight:600}\n.md code{\n  font-family:var(--mono);font-size:.86em;background:#f4f4f6;\n  padding:.13em .38em;border-radius:4px;\n}\n.md strong{font-weight:650}\n.md img{max-width:100%;height:auto}\n/* Les très longues suites sans espace (URL, base64, sorties d'outils) ne\n   doivent jamais élargir la page : elles reviennent à la ligne au milieu. */\n.md,.bubble,.err,.zone,.agent-log{overflow-wrap:anywhere}\n.md code,.md a{word-break:break-word}\n\n/* ---------- math ---------- */\n.math{overflow-x:auto;overflow-y:hidden;padding:.15em 0}\n.math-block{display:block;margin:0 0 1em;text-align:center;font-size:1.06em}\n.math-inline{display:inline;vertical-align:middle}\n.math-fallback{font-family:var(--mono);font-size:.86em;color:var(--ink-2)}\n\n/* ---------- code zones ---------- */\n.zone{\n  margin:0 0 1em;border:1px solid #26262b;border-radius:10px;\n  background:var(--code-bg);overflow:hidden;\n}\n.zone-bar{\n  display:flex;align-items:center;gap:8px;padding:7px 9px 7px 13px;\n  border-bottom:1px solid #232329;background:#141417;\n}\n.zone-lang{\n  font-family:var(--mono);font-size:10.5px;letter-spacing:.08em;text-transform:uppercase;\n  color:#7d7d88;flex:1 1 auto;\n}\n.zbtn{\n  font-family:var(--sans);font-size:11.5px;padding:4px 10px;border-radius:6px;cursor:pointer;\n  border:1px solid #33333b;background:#1c1c21;color:#c9c9d1;transition:.12s;white-space:nowrap;\n}\n.zbtn:hover{background:#26262d;color:#fff;border-color:#45454f}\n.zbtn.run{background:#f5f5f7;color:#111113;border-color:#f5f5f7;font-weight:600}\n.zbtn.run:hover{background:#fff;color:#000}\n.zone pre{\n  margin:0;padding:13px 15px;overflow-x:auto;\n  font-family:var(--mono);font-size:12.5px;line-height:1.62;color:var(--code-ink);\n  tab-size:2;\n}\n.zone pre code{background:none;padding:0;font-size:inherit}\n.zone-render{padding:0;background:#fff;min-height:90px}\n.zone-frame{display:block;width:100%;height:340px;border:0;background:#fff}\n/* --- boutons d'action sous les messages ------------------------------ */\n/* ---------- crédits (écran Disponibilité) ---------- */\n.crgrid{display:grid;grid-template-columns:1fr 1fr;gap:10px}\n.crgrid .cred{border:1px solid var(--line);border-radius:12px;padding:12px 13px;background:#fff;min-width:0}\n.cred span{display:block;font-size:10.5px;color:var(--ink-3);text-transform:uppercase;letter-spacing:.07em;margin-bottom:6px}\n.cred b{display:block;font-size:20px;line-height:1.15;letter-spacing:-.02em;font-weight:650}\n.cred em{display:block;font-style:normal;font-size:11.5px;line-height:1.45;color:var(--ink-2);margin-top:5px}\n.cred.ko b{color:var(--ink-2);font-size:15px}\n/* ---------- feuille « Téléchargement » (secours universel) ---------- */\n.dlmask{position:fixed;inset:0;z-index:80;display:flex;align-items:flex-end;justify-content:center;\n  padding:18px;background:rgba(17,17,16,.42);animation:lbin .16s ease}\n.dlcard{width:100%;max-width:420px;background:#fff;border-radius:16px;padding:18px 18px 14px;\n  box-shadow:0 18px 60px rgba(0,0,0,.22);animation:dlup .2s cubic-bezier(.2,.8,.3,1) both}\n@keyframes dlup{from{transform:translateY(14px);opacity:0}to{transform:none;opacity:1}}\n.dlcard h4{margin:0 0 3px;font-size:14.5px;font-weight:650;line-height:1.35;overflow-wrap:anywhere}\n.dlmeta{margin:0 0 13px;font:12px var(--mono);color:var(--ink-3)}\n.dlbig{display:flex;align-items:center;justify-content:center;gap:8px;height:44px;border-radius:11px;\n  background:var(--ink);color:#fff;text-decoration:none;font-size:14px;font-weight:600}\n.dlbig svg{width:16px;height:16px}\n.dlrow{display:flex;gap:8px;margin-top:9px}\n.dlrow a,.dlrow button{flex:1 1 0;height:38px;display:flex;align-items:center;justify-content:center;\n  border:1px solid var(--line);border-radius:10px;background:#fff;color:var(--ink);font:inherit;\n  font-size:12.5px;text-decoration:none;cursor:pointer}\n.dlnote{margin:11px 0 0;font-size:12px;line-height:1.5;color:var(--ink-3)}\n.dlclose{width:100%;margin-top:11px;height:36px;border:0;border-radius:10px;background:#f4f4f6;\n  color:var(--ink-2);font:inherit;font-size:12.5px;cursor:pointer}\n/* Rappel discret après un téléchargement : un lien reste à portée de main. */\n.dlhint{position:fixed;left:50%;bottom:92px;transform:translateX(-50%);z-index:78;\n  display:flex;align-items:center;gap:10px;max-width:min(92vw,440px);\n  padding:9px 10px 9px 14px;border-radius:12px;background:#111113;color:#fff;\n  box-shadow:0 10px 30px rgba(0,0,0,.28);font-size:12.5px;animation:toastIn .25s both}\n.dlhint button{border:0;border-radius:8px;background:#fff;color:#111;font:inherit;font-size:12px;\n  font-weight:600;padding:6px 11px;cursor:pointer;white-space:nowrap}\n.dlhint.out{animation:toastOut .3s ease both}\n.macts{display:flex;gap:6px;margin-top:10px}\n.mact,.iact{display:inline-flex;align-items:center;justify-content:center;width:30px;height:30px;\n  border:1px solid var(--line);border-radius:8px;background:#fff;color:var(--ink-2);cursor:pointer;padding:0}\n.mact:hover,.iact:hover{border-color:var(--ink-3);color:var(--ink)}\n.mact svg,.iact svg{width:15px;height:15px;animation:actIn .22s ease}\n@keyframes actIn{0%{transform:scale(.4);opacity:0}70%{transform:scale(1.14);opacity:1}100%{transform:scale(1);opacity:1}}\n.imgacts{display:flex;gap:6px;margin-top:8px}\n.attach{display:flex;gap:8px;overflow-x:auto;overscroll-behavior-x:contain;padding-bottom:2px}\n.attach-chip{display:inline-flex;align-items:center;gap:8px;margin-top:8px;padding:5px 10px 5px 5px;\n  border:1px solid var(--line);border-radius:9px;background:#fff}\n.attach-chip img{width:38px;height:38px;object-fit:cover;border-radius:6px;cursor:zoom-in;display:block}\n.attach-chip .nm{font-family:var(--mono);font-size:11px;color:var(--ink-2)}\n/* --- bouton hamburger (animation vers croix) ------------------------- */\n.burger{display:inline-flex;flex-direction:column;justify-content:center;gap:4.5px;\n  width:34px;height:34px;padding:7px 7px;border:1px solid var(--line);border-radius:9px;\n  background:#fff;cursor:pointer;margin-right:4px}\n.burger span{display:block;height:1.8px;width:100%;background:var(--ink);border-radius:2px;\n  transition:transform .28s cubic-bezier(.4,0,.2,1),opacity .2s}\n.burger.open span:nth-child(1){transform:translateY(6.3px) rotate(45deg)}\n.burger.open span:nth-child(2){opacity:0;transform:scaleX(.2)}\n.burger.open span:nth-child(3){transform:translateY(-6.3px) rotate(-45deg)}\n/* --- barre latérale --------------------------------------------------- */\n#sbackdrop{position:fixed;inset:0;background:rgba(17,17,16,.34);z-index:58;opacity:0;\n  pointer-events:none;transition:opacity .25s}\n#sbackdrop.on{opacity:1;pointer-events:auto}\n#sidebar{position:fixed;top:0;left:0;bottom:0;width:286px;max-width:86vw;z-index:59;\n  background:#fbfbfa;border-right:1px solid var(--line);display:flex;flex-direction:column;\n  transform:translateX(-102%);transition:transform .28s cubic-bezier(.4,0,.2,1)}\n#sidebar.open{transform:translateX(0)}\n.sidespace{height:14px;flex:none}\n#newchat{display:flex;align-items:center;gap:8px;margin:0 12px 10px;padding:9px 12px;\n  border:1px solid var(--line);border-radius:10px;background:#fff;color:var(--ink);\n  font:inherit;font-size:13.5px;cursor:pointer}\n#newchat:hover{border-color:var(--ink-3)}\n.chatlist{flex:1;overflow-y:auto;padding:0 8px 10px}\n.chitem{display:flex;align-items:center;gap:6px;width:100%;border:0;background:none;\n  padding:8px 8px;border-radius:9px;cursor:pointer;text-align:left;font:inherit}\n.chitem:hover{background:var(--line-2)}\n.chitem.sel{background:#eeeef0}\n/* La discussion « Arena » est épinglée en tête : sa marque à gauche du nom. */\n.chitem .chmark{flex:none;display:inline-flex;align-items:center;opacity:.92}\n.chitem.pinned{background:#f7f7f9}\n.chitem.pinned .cht{font-weight:600}\n.chitem.pinned.sel{background:#eeeef0}\n/* Une mission tourne encore dans cette discussion : elle avance toute seule,\n   même si l'application est fermée. */\n.chjob{flex:none;display:inline-flex;align-items:center;gap:5px;font-size:9.5px;font-weight:700;\n  letter-spacing:.4px;text-transform:uppercase;color:#0b6b3a;background:#e8f6ee;\n  border:1px solid #cbe9d8;border-radius:999px;padding:2px 7px}\n.chjob i{width:5px;height:5px;border-radius:50%;background:#12a35c;animation:jobpulse 1.3s ease-in-out infinite}\n@keyframes jobpulse{0%,100%{opacity:.3}50%{opacity:1}}\n.chdot{flex:none;width:7px;height:7px;border-radius:50%;background:#111113;animation:fadein .25s ease}\n.chitem .cht{flex:1;min-width:0;font-size:13.5px;color:var(--ink);white-space:nowrap;\n  overflow:hidden;text-overflow:ellipsis;border:0;background:none;padding:0;text-align:left}\n.chitem .cht:focus{outline:none}\n.chmore{flex:none;border:0;background:none;cursor:pointer;color:var(--ink-3);\n  font:700 13px/1 var(--sans);letter-spacing:1px;padding:4px 6px;border-radius:7px}\n.chmore:hover{background:#e4e4e8;color:var(--ink)}\n.chpop{position:absolute;z-index:61;background:#fff;border:1px solid var(--line);\n  border-radius:9px;box-shadow:0 8px 24px rgba(0,0,0,.12);padding:4px;min-width:118px}\n.chpop button{display:block;width:100%;border:0;background:none;text-align:left;\n  font:inherit;font-size:13px;padding:6px 9px;border-radius:6px;cursor:pointer;color:var(--ink)}\n.chpop button:hover{background:var(--line-2)}\n.chpop .del{color:#d33}\n.sideuser{flex:none;display:flex;align-items:center;gap:10px;margin:8px;padding:9px 10px;\n  border:1px solid var(--line);border-radius:11px;background:#fff;cursor:pointer;text-align:left}\n.sideuser:hover{border-color:var(--ink-3)}\n.sideuser .un{flex:1;min-width:0;font-size:13.5px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}\n.avat{width:30px;height:30px;flex:none;border-radius:50%;object-fit:cover;display:flex;\n  align-items:center;justify-content:center;color:#fff;font:600 14px/1 var(--sans)}\n.sideauth{display:flex;flex-direction:column;gap:6px;margin:8px}\n.sideauth button{padding:8px 10px;border:1px solid var(--line);border-radius:9px;\n  background:#fff;font:inherit;font-size:13px;cursor:pointer}\n.sideauth button.pri{background:var(--ink);color:#fff;border-color:var(--ink)}\n/* --- pièces jointes multiples : défilement horizontal de la zone seule - */\n#pending{max-width:100%;overflow-x:auto;overflow-y:hidden;display:none;gap:6px;\n  scrollbar-width:thin;padding:2px 0}\n#pending.on{display:flex}\n.pchip{flex:none;display:inline-flex;align-items:center;gap:7px;padding:3px 8px 3px 3px;\n  border:1px solid var(--line);border-radius:8px;background:#fff}\n.pchip img{width:26px;height:26px;border-radius:5px;object-fit:cover}\n.pchip .pn{font-family:var(--mono);font-size:11px;color:var(--ink-2);max-width:130px;\n  white-space:nowrap;overflow:hidden;text-overflow:ellipsis}\n.pchip button{border:0;background:none;cursor:pointer;color:var(--ink-3);font-size:13px;padding:0 2px}\n.pchip button:hover{color:var(--ink)}\n/* --- modales ----------------------------------------------------------- */\n.modal{position:fixed;inset:0;z-index:80;background:rgba(17,17,16,.45);display:flex;\n  align-items:center;justify-content:center;padding:20px}\n.modal[hidden]{display:none}\n.modalcard{position:relative;background:#fff;border-radius:16px;padding:26px 26px 22px;\n  width:min(380px,100%);max-height:92vh;overflow-y:auto}\n/* La croix est un vrai bouton : même contour, même forme et même taille que\n   le bouton « revenir » des paramètres — les deux se font face au même niveau. */\n.modalx{position:absolute;top:10px;right:12px;width:34px;height:34px;padding:0;border:1px solid var(--line);\n  border-radius:10px;background:#fff;color:var(--ink-2);cursor:pointer;display:inline-flex;\n  align-items:center;justify-content:center;\n  transition:border-color .15s ease,color .15s ease,transform .12s ease}\n.modalx svg{display:block}\n.modalx:hover{border-color:var(--ink-3);color:var(--ink)}\n.modalx:active{transform:scale(.94)}\n.modalcard h2{margin:0 0 16px;font-size:18px}\n.gbtn{display:flex;align-items:center;justify-content:center;gap:9px;width:100%;\n  padding:10px;border:1px solid var(--line);border-radius:10px;background:#fff;\n  font:inherit;font-size:14px;cursor:pointer}\n.gbtn:hover{background:var(--line-2)}\n.authsep{display:flex;align-items:center;gap:10px;margin:14px 0;color:var(--ink-3);font-size:12px}\n.authsep::before,.authsep::after{content:\"\";flex:1;height:1px;background:var(--line)}\n.flab{display:block;font-size:12.5px;color:var(--ink-2);margin:0 0 10px}\n.flab input{display:block;width:100%;margin-top:4px;padding:9px 11px;border:1px solid var(--line);\n  border-radius:9px;font:inherit;font-size:14px;background:#fff;color:var(--ink)}\n.flab input:focus{outline:none;border-color:var(--ink-3)}\n.pbtn{width:100%;padding:10px;border:0;border-radius:10px;background:var(--ink);color:#fff;\n  font:inherit;font-size:14px;cursor:pointer}\n.pbtn:hover{opacity:.9}\n.authalt{font-size:12.5px;color:var(--ink-2);text-align:center;margin:12px 0 0}\n.authalt button{border:0;background:none;color:var(--ink);text-decoration:underline;\n  font:inherit;cursor:pointer;padding:0}\n.autherr{color:#d33;font-size:12.5px;margin:10px 0 0}\n/* --- réglages plein écran --------------------------------------------- */\n#settings{z-index:82}\n#settings{padding:0;align-items:stretch;justify-content:stretch;background:#fff}\n#settings .setcard{position:relative;background:#fff;width:100%;height:100%;max-height:100%;\n  border-radius:0;padding:62px 22px 52px;overflow-y:auto}\n.setavatar{width:104px;height:104px;border-radius:50%;margin:0 auto 10px;overflow:hidden;\n  display:flex;align-items:center;justify-content:center;color:#fff;font:600 44px/1 var(--sans)}\n.setavatar img{width:100%;height:100%;object-fit:cover}\n.linkbtn{display:block;margin:0 auto 22px;border:0;background:none;color:var(--ink-2);\n  text-decoration:underline;font:inherit;font-size:13px;cursor:pointer}\n.setwrap{max-width:560px;margin:0 auto}\n.settitle{font-size:21px;margin:10px 0 6px;text-align:center;letter-spacing:-.02em}\n.setsub{font-size:12.5px;color:var(--ink-2);text-align:center;margin:0 0 28px;\n  line-height:1.55;overflow-wrap:anywhere}\n.setsec{border-top:1px solid var(--line);padding:16px 0}\n.setsec h3{margin:0 0 8px;font-size:11.5px;letter-spacing:.08em;text-transform:uppercase;color:var(--ink-3)}\n.setrowline{display:flex;align-items:center;gap:10px;padding:9px 0}\n.setlab{flex:none;width:112px;font-size:13px;color:var(--ink-2)}\n.setval{flex:1;min-width:0;font-size:14px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}\n.setact{flex:none;border:1px solid var(--line);background:#fff;border-radius:8px;\n  padding:6px 12px;font:inherit;font-size:12.5px;cursor:pointer}\n.setact:hover{border-color:var(--ink-3)}\n.wide{width:100%}\n.setnote{font-size:12.5px;color:var(--ink-3);margin:0;line-height:1.62}\n.setpop{position:absolute;inset:0;background:rgba(17,17,16,.45);display:flex;align-items:center;\n  justify-content:center;padding:20px;z-index:5}\n.setpop[hidden]{display:none}\n.setpopcard{background:#fff;border-radius:14px;padding:22px;width:min(360px,100%);\n  box-shadow:0 12px 40px rgba(0,0,0,.18)}\n.setpopcard h3{margin:0 0 14px;font-size:15px}\n.setpoprow{display:flex;gap:8px;justify-content:flex-end;margin-top:14px}\n.setpoprow .pbtn{width:auto;padding:9px 16px}\n.setgrid .flab:first-child{grid-column:1/-1}\n.setrow{display:flex;gap:8px;flex-wrap:wrap}\n.setrow .pbtn{width:auto;flex:1;padding:10px 18px}\n.dbtn{padding:10px 14px;border:1px solid var(--line);border-radius:10px;background:#fff;\n  font:inherit;font-size:13.5px;cursor:pointer}\n.dbtn:hover{border-color:var(--ink-3)}\n.setdanger{margin-top:30px;border-top:1px solid var(--line);padding-top:16px}\n.dangerbtn{border:1px solid #e7bcbc;border-radius:10px;background:#fff;color:#d33;\n  font:inherit;font-size:13.5px;padding:10px 14px;cursor:pointer}\n.dangerbtn:hover{background:#fdf3f3}\n.viafb{font-size:12.5px;color:var(--ink-3);margin:0 0 8px}\n.zone-warn{margin:0;padding:12px 14px;font-size:12.5px;color:var(--ink-2)}\n\n/* tokens */\n.t-com{color:#6a6a76;font-style:italic}\n.t-str{color:#a5d6a7}\n.t-tag{color:#79b8ff}\n.t-attr{color:#ffcc80}\n.t-kw{color:#c792ea}\n.t-num{color:#f78c6c}\n.t-punc{color:#8e8e99}\n.t-fn{color:#82aaff}\n.t-prop{color:#82aaff}\n.t-sel{color:#ffcb6b}\n.t-txt{color:#d4d4dc}\n\n/* ---------- composer ---------- */\nfooter{flex:0 0 auto;border-top:1px solid var(--line);background:#fff;padding:12px 20px 14px}\n.composer{max-width:780px;margin:0 auto}\n.tools{display:flex;align-items:center;gap:7px;margin-bottom:8px;flex-wrap:wrap}\n.chip{\n  display:inline-flex;align-items:center;gap:6px;padding:4px 10px;border-radius:999px;\n  border:1px dashed #d3d3da;background:#fff;color:var(--ink-3);font-size:11.5px;cursor:pointer;\n  font-family:var(--sans);transition:.12s;\n}\n.chip:hover{border-color:#b6b6c0;color:var(--ink-2)}\n.chip[aria-pressed=\"true\"]{border-style:solid;background:var(--ink);border-color:var(--ink);color:#fff}\n.chip[aria-pressed=\"true\"] svg .mk{fill:#fff}\n.hint{font-size:11px;color:var(--ink-3);margin-left:auto}\n.box{\n  position:relative;\n  display:flex;align-items:flex-end;gap:8px;border:1px solid var(--line);\n  border-radius:12px;padding:8px 8px 8px 13px;background:#fff;transition:border-color .12s;\n}\n.box:focus-within{border-color:#b9b9c2}\n#prompt{\n  flex:1 1 auto;border:0;outline:none;resize:none;background:transparent;\n  font-family:var(--sans);font-size:14.5px;line-height:1.55;color:var(--ink);\n  max-height:200px;min-height:24px;padding:3px 0;\n}\n#prompt{white-space:pre-wrap;overflow-wrap:anywhere;overflow-y:auto;cursor:text}\n#prompt.empty::before{content:attr(data-ph);color:#b3b4bb;pointer-events:none}\n/* Marqueurs markdown : visibles (on peut les effacer) mais discrets. */\n#prompt .mdm{color:#c2c2ca}\n#prompt code{font-family:var(--mono);font-size:.9em;background:#f4f4f6;border-radius:4px;padding:.05em .25em}\n#prompt strong{font-weight:700}\n#prompt strong.mdh{font-size:1.06em}\n#prompt em{font-style:italic}\n#prompt s{text-decoration:line-through;color:var(--ink-2)}\n#prompt .mdq{color:var(--ink-2);font-style:italic}\n/* Listes : signe dans la marge, texte décalé, lignes suivantes alignées. */\n#prompt .mdli{display:inline-block;padding-left:1.6em;text-indent:-1.1em}\n#prompt .mdz{font-size:0}\n#send{\n  flex:0 0 auto;width:32px;height:32px;border-radius:9px;border:0;background:var(--ink);\n  color:#fff;cursor:pointer;display:inline-flex;align-items:center;justify-content:center;\n}\n#send:disabled{background:#d7d7dd;cursor:not-allowed}\n#send:not(:disabled):hover{background:#000}\n\n/* ---------- pièce jointe ---------- */\n.attach{\n  flex:0 0 auto;width:32px;height:32px;border-radius:9px;cursor:pointer;\n  border:1px solid var(--line);background:#fff;color:var(--ink-2);\n  display:inline-flex;align-items:center;justify-content:center;\n}\n.attach:hover{border-color:#b9b9c2;color:var(--ink)}\n.attach.on{background:var(--ink);border-color:var(--ink);color:#fff}\n#imgmenu{\n  position:absolute;bottom:calc(100% + 8px);left:0;z-index:70;\n  min-width:196px;background:#fff;border:1px solid var(--line);border-radius:10px;\n  box-shadow:0 10px 28px rgba(20,20,26,.10);padding:5px;\n  display:flex;flex-direction:column;gap:2px;\n}\n#imgmenu[hidden]{display:none}\n#imgmenu button{\n  display:flex;align-items:center;gap:8px;width:100%;text-align:left;\n  border:0;background:none;border-radius:7px;padding:7px 9px;\n  font-size:12.5px;color:var(--ink-2);cursor:pointer;font-family:var(--sans);\n}\n#imgmenu button:hover{background:var(--line-2);color:var(--ink)}\n#imgmenu button[aria-pressed=\"true\"]{background:var(--ink);color:#fff}\n.pending{\n  display:none;align-items:center;gap:6px;max-width:100%;\n  overflow-x:auto;overflow-y:hidden;overscroll-behavior-x:contain;\n  padding:0;border:0;background:none;scrollbar-width:thin;\n  font-size:11.5px;color:var(--ink-2);\n}\n.pending.on{display:flex}\n.pending img{width:26px;height:26px;object-fit:cover;border-radius:5px;border:1px solid var(--line);background:#fff}\n.pending b{font-weight:500;max-width:200px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}\n.pending button{\n  border:0;background:none;cursor:pointer;color:var(--ink-3);font-size:14px;\n  line-height:1;padding:2px 3px;border-radius:5px;\n}\n.pending button:hover{color:var(--ink);background:#ececf0}\n\n/* ---------- réglages plein écran ---------- */\n#cfg{position:fixed;inset:0;background:#fff;z-index:9998;display:none;overflow-y:auto}\n#cfg.open{display:block}\n#cfgclose{\n  position:absolute;top:14px;right:16px;width:38px;height:38px;border-radius:50%;\n  border:1px solid var(--line);background:#fff;color:var(--ink);cursor:pointer;\n  display:flex;align-items:center;justify-content:center;z-index:2;\n}\n#cfgclose:hover{background:var(--ink);color:#fff;border-color:var(--ink)}\n.cfgwrap{max-width:520px;margin:0 auto;padding:30px 20px 60px}\n.cfgwrap h2{\n  margin:0 0 4px;font-size:19px;font-weight:650;letter-spacing:-.02em;\n  display:flex;align-items:center;gap:9px;\n}\n.cfgsub{margin:0 0 26px;font-size:12.5px;color:var(--ink-3)}\n.field{padding:14px 0;border-top:1px solid var(--line)}\n.field:first-of-type{border-top:0}\n.field label{display:flex;align-items:center;justify-content:space-between;gap:12px;font-size:13.5px;font-weight:500}\n.field input[type=number]{\n  width:130px;font-family:var(--mono);font-size:12.5px;text-align:right;\n  padding:6px 9px;border:1px solid var(--line);border-radius:7px;background:#fff;color:var(--ink);\n}\n.field input[type=number]:focus{outline:none;border-color:#b9b9c2}\n.field select{\n  font-family:var(--sans);font-size:12.5px;padding:6px 9px;\n  border:1px solid var(--line);border-radius:7px;background:#fff;color:var(--ink);\n}\n.field select:focus{outline:none;border-color:#b9b9c2}\n/* La flèche est dessinée par nous : sans ça, chaque navigateur met la sienne. */\n.field select{appearance:none;-webkit-appearance:none;-moz-appearance:none;cursor:pointer;\n  padding-right:26px;\n  background-image:url(\"data:image/svg+xml;charset=utf-8,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12' viewBox='0 0 24 24' fill='none' stroke='%238a8a92' stroke-width='2.2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='m6 9 6 6 6-6'/%3E%3C/svg%3E\");\n  background-repeat:no-repeat;background-position:right 8px center}\n.field .note{margin:6px 0 0;font-size:11.5px;color:var(--ink-3);line-height:1.5}\n/* Résolution : pastilles dessinées par l'app (jamais de menu natif). */\n.resrow{display:flex;flex-wrap:wrap;gap:6px;margin-top:8px}\n.reschip{font-family:var(--mono);font-size:11.5px;padding:6px 10px;border-radius:999px;\n  border:1px solid var(--line);background:#fff;color:var(--ink-2);cursor:pointer;transition:.12s}\n.reschip:hover{border-color:#c7c7cf;color:var(--ink)}\n.reschip[aria-pressed=\"true\"]{background:var(--ink);border-color:var(--ink);color:#fff}\n/* Format : un petit cadre à l'échelle du ratio, puis le ratio lui-même. */\n.fmtchip{display:inline-flex;align-items:center;gap:7px}\n.fmtbox{display:block;border:1.5px solid currentColor;border-radius:2.5px;flex:0 0 auto;opacity:.9;background:currentColor;background-clip:content-box;padding:1px;box-sizing:border-box}\n.imgsize{margin:8px 0 0;font-family:var(--mono);font-size:11.5px;color:var(--ink-2)}\n.sw{\n  position:relative;width:38px;height:22px;flex:0 0 auto;border-radius:999px;\n  border:1px solid var(--line);background:#ececf0;cursor:pointer;transition:.15s;padding:0;\n}\n.sw::after{\n  content:\"\";position:absolute;top:2px;left:2px;width:16px;height:16px;border-radius:50%;\n  background:#fff;transition:.15s;box-shadow:0 1px 2px rgba(0,0,0,.18);\n}\n.sw[aria-checked=\"true\"]{background:var(--ink);border-color:var(--ink)}\n.sw[aria-checked=\"true\"]::after{left:18px}\n#cfgreset{\n  margin-top:26px;font-family:var(--sans);font-size:12.5px;padding:7px 14px;\n  border:1px solid var(--line);border-radius:8px;background:#fff;color:var(--ink-2);cursor:pointer;\n}\n#cfgreset:hover{border-color:#b9b9c2;color:var(--ink)}\n.pillhint{font-size:11px;color:var(--ink-3);margin-left:2px}\n\n.cfgcard{border:1px solid var(--line);border-radius:12px;padding:10px 12px;margin-bottom:14px;background:var(--line-2)}\n.cfgcard .desc{font-size:12.5px;color:var(--ink-2);margin:0 0 8px}\n.cfgcard .stats{display:flex;flex-wrap:wrap;gap:6px}\n.cfgcard .stats span{font-size:11px;color:var(--ink-3);background:#fff;border:1px solid var(--line);border-radius:999px;padding:3px 9px}\n.cfgcard .stats b{color:var(--ink);font-weight:600}\n\n/* ---------- sélecteur de modèle (menu déroulant) ---------- */\n.modelwrap{position:relative;flex:1 1 auto;min-width:0;display:flex}\n#modelbtn{display:flex;align-items:center;gap:8px;min-width:0;max-width:100%;\n  border:1px solid var(--line);border-radius:999px;background:#fff;\n  padding:6px 14px 6px 10px;color:var(--ink-2);font-size:13px;font-weight:600;\n  font-family:var(--sans);cursor:pointer;user-select:none;-webkit-user-select:none;touch-action:manipulation}\n#modelbtn:hover{border-color:#cfcfd6;color:var(--ink)}\n#modelbtn .lbl{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}\n#modelbtn svg{flex:0 0 auto;display:block}\n#dd{position:absolute;top:calc(100% + 8px);left:0;z-index:70;\n  width:min(340px,calc(100vw - 24px));max-height:min(430px,70vh);overflow:auto;\n  background:#fff;border:1px solid var(--line);border-radius:14px;\n  box-shadow:0 14px 44px rgba(10,10,14,.14);padding:6px}\n.ddhead{font-size:10.5px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;\n  color:var(--ink-3);padding:8px 10px 3px;display:flex;align-items:center;gap:6px}\n.dditem{display:flex;align-items:center;gap:8px;width:100%;text-align:left;\n  padding:7px 10px;border:0;border-radius:9px;background:none;cursor:pointer;\n  font-family:var(--sans);font-size:13px;color:var(--ink)}\n.dditem:hover{background:var(--line-2)}\n.dditem[aria-selected=\"true\"]{background:var(--ink);color:#fff}\n.dditem[aria-selected=\"true\"] svg .mk{fill:#fff;stroke:#fff}\n.dditem .mid{margin-left:auto;font-family:var(--mono);font-size:10px;color:var(--ink-3)}\n.dditem[aria-selected=\"true\"] .mid{color:#bbbbc2}\n\n/* ---------- fullscreen render ---------- */\n#fs{position:fixed;inset:0;background:#fff;z-index:9999;display:none}\n#fs.open{display:block}\n#fs iframe{width:100%;height:100%;border:0;display:block}\n#fsclose{\n  position:absolute;top:14px;right:16px;width:38px;height:38px;border-radius:50%;\n  border:1px solid var(--line);background:rgba(255,255,255,.94);color:var(--ink);\n  cursor:pointer;display:flex;align-items:center;justify-content:center;z-index:2;\n  box-shadow:0 1px 6px rgba(0,0,0,.08);\n}\n#fsclose:hover{background:#111113;color:#fff;border-color:#111113}\n\n@media (max-width:680px){\n  .brand span{display:none}\n  header{padding:9px 12px;gap:9px}\n  main,footer{padding-left:12px;padding-right:12px}\n  .pill span{display:none}\n  .pill{padding:5px 8px}\n}\n\n/* --- animations d'interface (déploiements, entrées, popups) ------------- */\n@keyframes rise{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:none}}\n@keyframes pop{from{opacity:0;transform:scale(.86)}to{opacity:1;transform:scale(1)}}\n@keyframes fadein{from{opacity:0}to{opacity:1}}\n@keyframes ddin{from{opacity:0;transform:translateY(-7px)}to{opacity:1;transform:none}}\n.msg{animation:rise .3s cubic-bezier(.2,.7,.3,1) both}\n.chitem{animation:rise .22s ease-out both}\n.chpop{animation:pop .16s ease-out both}\n#dd:not([hidden]){animation:ddin .18s ease-out both;transform-origin:top center}\n.pchip{animation:pop .18s ease-out both}\n.attach-chip{animation:pop .2s ease-out both}\n#authmodal:not([hidden]){animation:fadein .2s both}\n#authmodal .modalcard{animation:rise .24s ease-out both}\n#settings:not([hidden]){animation:fadein .2s both}\n#settings .setcard{animation:fadein .26s both}\n.setpop:not([hidden]){animation:fadein .18s both}\n.setpopcard{animation:pop .2s ease-out both}\n.lightbox{animation:fadein .2s both}\n.lightbox img{animation:pop .22s ease-out both}\n.pbtn:active,.dbtn:active,.setact:active,.dditem:active,.chitem:active,.iact:active,\n.mact:active,.chmore:active,#newchat:active{transform:scale(.96)}\n.pbtn,.dbtn,.setact,.dditem,.iact,.mact,.chmore,#newchat{transition:transform .12s ease,border-color .15s,background .15s}\n/* --- Mode agent (phases 37-38) ---------------------------------------- */\n.agentbtn{border:1px solid #d9d9d6;background:#fff;border-radius:999px;padding:5px 13px;\n  font-size:12px;font-weight:600;cursor:pointer;color:#111113;white-space:nowrap;\n  transition:transform .12s ease,background .18s ease,color .18s ease,box-shadow .18s ease}\n.agentbtn:hover{background:#f4f4f2}\n.agentbtn:active{transform:scale(.94)}\n.agentbtn.on{background:#111113;color:#fff;border-color:#111113;\n  box-shadow:0 2px 10px rgba(17,17,19,.25);animation:agtPop .25s ease}\n/* --- Phase 69 : le mode de la discussion dans la zone noire ------------ */\n.agentbtn.mode{display:inline-flex;flex-direction:row;align-items:center;gap:6px;cursor:default;\n  background:#111113;color:#fff;border-color:#111113;padding:5px 12px;line-height:1.25;\n  max-width:min(52vw,260px)}\n.agentbtn.mode:hover{background:#111113}\n.agentbtn.mode .mbmode{display:inline-flex;align-items:center;gap:6px;font-size:12px;font-weight:600}\n.agentbtn.mode .mbmode svg{display:block;flex:none}\n/* Plus de deuxième ligne dans le badge : l'identifiant s'annonce par un toast. */\n.agentbtn.mode .mbnotice{display:none}\n/* --- Paramètres : des zones en grisé, comme de vrais réglages --------- */\n/* Une section de réglages est une carte blanche bordée, jamais un pavé gris :\n   le gris est réservé à la LISTE des entrées (.setlistsec). */\n.setsec{background:#fff;border:1px solid var(--line);border-radius:16px;\n  padding:18px 18px 20px;margin:0 0 14px}\n.setsec h3{margin:0 0 12px;padding:0;font-size:11.5px;letter-spacing:.08em;\n  text-transform:uppercase;color:var(--ink-3)}\n.setlistsec{background:#f5f5f7;border:0;border-radius:16px;padding:6px;margin:0 0 16px}\n.setrow+.setrow{box-shadow:inset 0 1px 0 var(--line)}\n.setrow{border-radius:10px}\n.setrow:hover{background:#fff}\n/* --- Paramètres : la liste des réglages ------------------------------- */\n#setlist,#setcompte,#setcanal,#setmodels,#setwan,#settasks,#setdispo{animation:setview .22s ease both}\n@keyframes setview{from{opacity:0;transform:translateY(7px)}to{opacity:1;transform:none}}\n.setrow{display:flex;flex-wrap:nowrap;align-items:center;gap:13px;width:100%;border:1px solid transparent;\n  border-radius:13px;background:none;padding:12px 13px;cursor:pointer;text-align:left;\n  font:inherit;color:inherit;transition:background .16s ease,border-color .16s ease,transform .12s ease}\n.setrow+.setrow{margin-top:6px}\n.setrow:hover{background:#fafafb;border-color:var(--line)}\n.setrow:active{transform:scale(.985)}\n.setrow:focus-visible{outline:2px solid var(--ink);outline-offset:2px}\n.setrow .sra{transition:transform .16s ease}\n.setrow:hover .sra{transform:translateX(2px)}\n.setrow .sric{flex:none;width:38px;height:38px;border-radius:50%;overflow:hidden;display:flex;\n  align-items:center;justify-content:center;background:#f3f3f5;color:#111113;\n  transition:background .16s ease,transform .16s ease}\n.setrow:hover .sric{background:#ececef;transform:scale(1.04)}\n.setrow .sric .avat{width:38px;height:38px;font-size:15px}\n.setrow .srtx{flex:1 1 auto;min-width:0}\n.setrow .srtx b{display:block;font-size:14px;font-weight:600;letter-spacing:-.01em}\n.setrow .srtx span{display:block;font-size:12.5px;color:var(--ink-2);white-space:nowrap;\n  overflow:hidden;text-overflow:ellipsis}\n.setrow .sra{flex:none;color:var(--ink-3);display:flex;align-items:center}\n/* En haut à GAUCHE, à la même hauteur que la croix qui est en haut à droite. */\n.setbackbtn{position:absolute;top:10px;left:12px;width:34px;height:34px;margin:0;display:inline-flex;align-items:center;\n  justify-content:center;border:1px solid var(--line);border-radius:10px;background:#fff;\n  color:var(--ink-2);cursor:pointer;padding:0;transition:border-color .15s ease,transform .12s ease}\n.setbackbtn:hover{border-color:var(--ink-3);color:var(--ink)}\n.setbackbtn:active{transform:scale(.94)}\n.setver{font-size:11.5px;color:var(--ink-3);text-align:center;margin:26px 0 6px}\n/* --- Paramètres ▸ Disponibilité : un secret par ligne, testé pour de vrai - */\n.disrow{display:flex;align-items:center;gap:12px;padding:12px 0;border-top:1px solid var(--line)}\n.disrow:first-of-type{border-top:0;padding-top:2px}\n.dismain{flex:1 1 auto;min-width:0}\n.dismain b{display:block;font-size:12.8px;font-weight:600;letter-spacing:-.005em;word-break:break-all}\n.dismain em{display:block;font-style:normal;font-size:11.5px;color:var(--ink-3);margin-top:3px}\n.disbtn{flex:none;width:34px;height:34px;border-radius:50%;border:1px solid var(--line);background:#fff;\n  display:inline-flex;align-items:center;justify-content:center;cursor:pointer;padding:0;\n  color:var(--ink-2);transition:border-color .15s ease,transform .12s ease}\n.disbtn:hover{border-color:#cfcfd6;transform:scale(1.04)}\n.disbtn svg{display:block;width:14px;height:14px}\n.disbtn.ok{background:#e8f4ee;border-color:#bfe0cd;color:#1f7a4d}\n.disbtn.ko{background:#fdf0f0;border-color:#ecc9c9;color:#a33030}\n.disbtn.run{background:#f5f5f7;border-color:var(--line);color:var(--ink-3)}\n.disspin{width:14px;height:14px;border-radius:50%;border:1.8px solid #d6d6dc;border-top-color:#111113;\n  animation:disspin .8s linear infinite}\n@keyframes disspin{to{transform:rotate(360deg)}}\n.dismeta{font-size:11.5px;color:var(--ink-3);margin:14px 0 0;text-align:center}\n#dispop{position:fixed;inset:0;z-index:90;display:flex;align-items:center;justify-content:center;\n  padding:24px;background:rgba(17,17,16,.55);animation:lbin .16s ease}\n#dispop[hidden]{display:none !important}\n.dispo-card{background:#fff;border-radius:16px;max-width:420px;width:100%;padding:20px 20px 16px;\n  box-shadow:0 18px 50px rgba(0,0,0,.18)}\n.dispo-card h1{font-size:14px;font-weight:650;margin:0 0 10px;word-break:break-all}\n.dispo-card p{font-size:13px;line-height:1.55;color:var(--ink-2);margin:0 0 16px;white-space:pre-wrap}\n.dispo-act{display:flex;gap:8px;justify-content:flex-end}\n.dispo-act button{font:inherit;font-size:12.5px;padding:8px 14px;border-radius:10px;cursor:pointer;\n  border:1px solid var(--line);background:#fff;color:var(--ink-2)}\n.dispo-act button.prim{background:#111113;border-color:#111113;color:#fff}\n/* --- Paramètres ▸ Tâches : une mission par ligne, avec son état --------- */\n.taskrow{display:flex;align-items:flex-start;gap:12px;padding:13px 0;border-top:1px solid var(--line)}\n.taskrow:first-child{border-top:0;padding-top:2px}\n.taskmain{flex:1 1 auto;min-width:0}\n.taskmain b{display:block;font-size:13.5px;font-weight:600;letter-spacing:-.01em}\n.taskmain .tsub{display:block;font-size:12.5px;color:var(--ink-2);margin-top:2px;\n  white-space:nowrap;overflow:hidden;text-overflow:ellipsis}\n.taskmain em{display:block;font-style:normal;font-size:11.5px;color:var(--ink-3);margin-top:5px}\n.taskstop{flex:none;padding:7px 12px;border:1px solid var(--line);border-radius:9px;background:#fff;\n  color:#8a2c2c;font:inherit;font-size:12px;cursor:pointer;transition:border-color .15s ease,background .15s ease}\n.taskstop:hover{border-color:#e3b9b9;background:#fdf6f6}\n.taskrun{display:inline-flex;align-items:center;gap:5px;color:#1f7a4d}\n.taskrun i{width:6px;height:6px;border-radius:50%;background:#1f7a4d;animation:tpulse 1.4s ease-in-out infinite}\n@keyframes tpulse{0%,100%{opacity:.35}50%{opacity:1}}\n/* --- Mission en cours : le bouton qui l'arrête, sous le message ---------- */\n.mstop{margin:0 0 14px;padding:8px 13px;border:1px solid var(--line);border-radius:10px;\n  background:#fff;color:var(--ink-2);font:inherit;font-size:12.5px;cursor:pointer;\n  transition:border-color .15s ease,color .15s ease,transform .12s ease}\n.mstop:hover{border-color:#cfcfd6;color:var(--ink)}\n.mstop:active{transform:scale(.98)}\n/* --- Réglages : la liste arrive en douceur ---------------------------- */\n.setlistsec .setrow{animation:setrow .26s ease-out both}\n.setlistsec .setrow:nth-of-type(2){animation-delay:.05s}\n@keyframes setrow{from{opacity:0;transform:translateY(5px)}to{opacity:1;transform:none}}\n/* --- Paramètres ▸ Modèles : la liste des modèles, dessinée à la main -----\n   Un vrai <select> change d'apparence selon le navigateur et cassait la page :\n   on dessine donc les lignes nous-mêmes, avec la même écriture que le reste. */\n.mopts{display:flex;flex-direction:column;margin:16px 0 0;border:1px solid var(--line);\n  border-radius:14px;overflow:hidden;background:#fff}\n.mopt{display:flex;align-items:center;gap:12px;width:100%;padding:14px 15px;border:0;\n  border-top:1px solid var(--line);background:#fff;font:inherit;color:inherit;\n  text-align:left;cursor:pointer;transition:background .16s ease}\n.mopt:first-child{border-top:0}\n.mopt:hover{background:#fafafb}\n.mopt:focus-visible{outline:2px solid var(--ink);outline-offset:-2px}\n.mopt[aria-selected=\"true\"]{background:#f3f3f5}\n.mopt-tx{flex:1 1 auto;min-width:0}\n.mopt-tx b{display:block;font-size:13.5px;font-weight:600;letter-spacing:-.01em;line-height:1.55}\n.mopt-tx .mopt-sub{display:block;font-size:12px;color:var(--ink-3);margin-top:3px;\n  white-space:nowrap;overflow:hidden;text-overflow:ellipsis}\n.mopt-tx .setinput,.setsec .setinput{display:block;width:100%;margin:14px 0 0;padding:11px 13px;border:1px solid var(--line);\n  border-radius:12px;background:#fff;font:inherit;font-size:13px;color:var(--ink)}\n.setinput:focus{outline:none;border-color:var(--ink-3)}\n.setinput::placeholder{color:var(--ink-3)}\n.mopt-badge{display:inline-block;font-size:10.5px;font-weight:600;letter-spacing:.02em;\n  color:#1f7a4d;background:#e8f4ee;border-radius:999px;padding:2px 8px;margin-left:8px;\n  vertical-align:1.5px;line-height:1.5}\n.mopt-tick{flex:none;width:22px;height:22px;border-radius:50%;border:1px solid var(--line);\n  display:flex;align-items:center;justify-content:center;color:#fff;background:#fff;\n  transition:background .16s ease,border-color .16s ease}\n.mopt-tick svg{width:12px;height:12px;opacity:0;transition:opacity .16s ease}\n.mopt[aria-selected=\"true\"] .mopt-tick{background:#111113;border-color:#111113}\n.mopt[aria-selected=\"true\"] .mopt-tick svg{opacity:1}\n/* --- Z-Canal : la liste des canaux — un canal par ligne, en grand ------- */\n.canlist{display:block;margin:2px 0 0}\n.canrow{position:relative;display:flex;align-items:center;gap:13px;padding:15px 2px}\n/* Le trait fin entre deux canaux : il ne prend pas toute la largeur, il part\n   après le logo et s'arrête avant le bord. */\n.canrow + .canrow::before{content:\"\";position:absolute;left:47px;right:12px;top:0;height:1px;background:var(--line)}\n.canrow .cic{width:34px;height:34px;flex:none;border-radius:50%;background:#f4f4f6;\n  display:flex;align-items:center;justify-content:center}\n.canrow .cic svg{display:block}\n.canrow .cmain{flex:1;min-width:0;display:flex;align-items:baseline;gap:11px;flex-wrap:nowrap}\n.canrow .cname{font-size:16px;font-weight:600;letter-spacing:-.015em;white-space:nowrap;\n  overflow:hidden;text-overflow:ellipsis}\n.canrow .cid{flex:none;font-family:var(--mono);font-size:13px;color:var(--ink-3);background:none;padding:0}\n.canrow .copybtn{width:30px;height:30px;margin-left:2px}\n/* Le bouton « supprimer » se pose juste à droite du bouton « copier ». */\n.canrow .delbtn{width:30px;height:30px;flex:none;margin-left:2px;display:inline-flex;align-items:center;\n  justify-content:center;border:1px solid var(--line);border-radius:8px;background:#fff;\n  color:var(--ink-3);cursor:pointer;padding:0;transition:border-color .15s,color .15s,background .15s}\n.canrow .delbtn:hover{border-color:#e0b4b0;color:#c8382f;background:#fdf7f7}\n.canrow .delbtn:active{transform:scale(.94)}\n/* Les lignes de la popup sont des boutons : même dessin que celles des réglages. */\n.canpick .canrow{border:0;background:#fff;border-radius:10px;width:100%;font:inherit;\n  color:inherit;text-align:left;cursor:pointer;transition:background .15s}\n.canpick .canrow:hover{background:#f7f7f9}\n.canpick .canrow:active{transform:scale(.995)}\n.canrow .sra{flex:none;color:var(--ink-3);display:flex;align-items:center}\n.canrow.cannew .cic{background:#33333a;color:#fff}\n.canrow.cannew .cname{color:var(--ink-2);font-weight:600}\n/* --- Z-Canal : on respire un peu dans les zones grisées ----------------- */\n#setcanal .setsec{padding:14px 12px 16px}\n#setcanal .setsec h3{padding:0 2px}\n#setcanal .setnote{margin:10px 2px 0}\n#setcanal .urlbox{margin:10px 0 0}\n/* --- Supprimer un canal : la confirmation ------------------------------- */\n#canaldel{position:fixed;inset:0;z-index:150;background:rgba(38,38,44,.62);display:flex;\n  align-items:center;justify-content:center;padding:20px;animation:fadein .16s both}\n#canaldel[hidden]{display:none}\n#canaldel .delcard{background:#fff;border-radius:16px;padding:22px 22px 18px;width:min(380px,100%);\n  box-shadow:0 16px 40px rgba(17,17,19,.18),0 2px 8px rgba(17,17,19,.10);\n  animation:pop .18s ease-out both}\n#canaldel h3{margin:0 0 10px;font-size:16px;letter-spacing:-.01em}\n#canaldel p{margin:0 0 18px;font-size:13.5px;color:var(--ink-2);line-height:1.55}\n#canaldel .delrow{display:flex;gap:9px;justify-content:flex-end}\n#canaldel .delcancel,#canaldel .delok{padding:10px 18px;border-radius:10px;font:inherit;\n  font-size:13.5px;cursor:pointer;background:#fff;transition:background .15s,color .15s,border-color .15s}\n#canaldel .delcancel{border:1px solid var(--ink);color:var(--ink)}\n#canaldel .delcancel:hover{background:var(--line-2)}\n#canaldel .delok{border:1px solid #c8382f;color:#c8382f;font-weight:600}\n#canaldel .delok:hover{background:#c8382f;color:#fff}\n/* --- Popup « nouvelle discussion » : de vrais boutons en bas ------------ */\n.npbtns{display:flex;gap:8px;margin-top:16px}\n.npbtn{flex:1;padding:10px;border:1px solid var(--line);border-radius:10px;background:#fff;\n  font:inherit;font-size:13.5px;color:var(--ink);cursor:pointer;transition:border-color .15s,background .15s}\n.npbtn:hover{border-color:var(--ink-3);background:#fafafb}\n.npbtn:active{transform:scale(.97)}\n.ctag{float:right;font-size:11.5px;letter-spacing:0;text-transform:none;color:var(--ink-3)}\n/* Liste des quêtes Pollinations : une ligne par quête, le gain à droite. */\n.crqhead{color:var(--ink-2);margin:0 0 8px}\n.crqrow{display:flex;align-items:baseline;gap:10px;padding:5px 0;border-top:1px solid var(--line-2);font-size:13px;color:var(--ink)}\n.crqrow:first-of-type{border-top:0}\n.crqtx{flex:1 1 auto;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}\n.crqgn{flex:0 0 auto;font-family:var(--mono);font-size:12px;color:#1f7a4d}\n.crqmore{color:var(--ink-3);font-size:12.5px;margin-top:6px}\n/* Pastilles de la page Crédits : vert quand le tuyau est ouvert, rouge quand il est fermé. */\n.ctag.crok{color:#1f7a4d}\n.ctag.crko{color:#a33131}\n/* --- Z-Canal : l'installation, adresses entières ------------------------ */\n.urlbox{border:1px solid var(--line);border-radius:12px;padding:10px 12px 12px;margin:10px 0;background:#fff}\n.urlhead{display:flex;align-items:center;gap:8px}\n.urlab{flex:1;font-size:11px;letter-spacing:.07em;text-transform:uppercase;color:var(--ink-3)}\n.urlval{display:block;margin-top:7px;font-family:var(--mono);font-size:12px;line-height:1.62;\n  color:var(--ink);word-break:break-all;white-space:pre-wrap}\n.copybtn{width:26px;height:26px;flex:none;display:inline-flex;align-items:center;justify-content:center;\n  border:1px solid var(--line);border-radius:8px;background:#fff;color:var(--ink-2);cursor:pointer;padding:0}\n.copybtn:hover{border-color:var(--ink-3);color:var(--ink)}\n.copybtn.ok{color:#0a7d3c;border-color:#bfe3cd;background:#f4fbf6}\n/* --- Nouvelle discussion : le choix du mode --------------------------- */\n#newpop{position:fixed;inset:0;z-index:145;background:rgba(17,17,16,.45);display:flex;\n  align-items:center;justify-content:center;padding:20px;animation:fadein .18s both}\n#newpop[hidden]{display:none}\n#newpop .npc{position:relative;background:#fff;border-radius:16px;padding:20px 20px 18px;\n  width:min(440px,100%);max-height:88vh;overflow:auto;box-shadow:0 18px 50px rgba(0,0,0,.22);\n  animation:pop .2s ease-out both}\n#newpop h3{margin:0 0 4px;font-size:16px}\n#newpop .npd{margin:0 0 12px;font-size:13px;color:var(--ink-2);line-height:1.55}\n.modegrid{display:grid;grid-template-columns:repeat(3,1fr);gap:9px;margin:14px 0 6px}\n.modepick{border:1px solid var(--line);border-radius:12px;background:#fff;padding:14px 6px 11px;\n  cursor:pointer;display:flex;flex-direction:column;align-items:center;gap:9px;font-size:13px;\n  font-weight:600;color:var(--ink);transition:border-color .15s ease,transform .12s ease}\n.modepick:hover{border-color:#111113;background:#fcfcfd}\n.modepick:focus-visible{outline:2px solid var(--ink);outline-offset:2px}\n.modepick:active{transform:scale(.97)}\n.modepick svg{display:block}\n.canfree{display:flex;align-items:center;gap:8px;padding:7px 0;border-top:1px solid var(--line);font-size:13px}\n.canfree .cn{flex:1;min-width:0;display:flex;align-items:center;gap:7px;overflow:hidden;\n  text-overflow:ellipsis;white-space:nowrap}\n.canfree code{font-family:var(--mono);font-size:12px;background:#f4f4f6;padding:.15em .4em;border-radius:5px}\n/* --- pastille « non lu » → « lu ! » ---------------------------------- */\n.luchip{display:inline-flex;align-items:center;gap:4px;margin-top:9px;font-size:10.5px;font-weight:700;\n  letter-spacing:.06em;text-transform:uppercase;color:var(--ink-3);border:1px solid var(--line);\n  border-radius:999px;padding:2px 8px;background:#fff}\n.luchip.lu{color:#0a7d3c;border-color:#bfe3cd;background:#f4fbf6}\n@keyframes agtPop{0%{transform:scale(.85)}60%{transform:scale(1.06)}100%{transform:scale(1)}}\n#agentPanel{position:fixed;inset:0;z-index:120;background:#fff;display:flex;flex-direction:column;\n  animation:apIn .24s cubic-bezier(.2,.7,.3,1) both}\n@keyframes apIn{from{opacity:0;transform:translateY(14px) scale(.99)}to{opacity:1;transform:none}}\n#agentPanel[hidden],#agentViewer[hidden]{display:none}\n#agentfiles[hidden]{display:none}\n#agentPanel .ap-head,#agentViewer .av-head{display:flex;align-items:center;gap:10px;\n  padding:10px 14px;border-bottom:1px solid #ececea;background:#fff;z-index:2}\n#agentPanel .ap-head b{flex:1;font-size:13px}\n.ap-body{flex:1;min-height:0;overflow:auto;overscroll-behavior:contain}\n.ap-screen{height:44vh;min-height:200px;background:#111;display:flex;align-items:center;\n  justify-content:center;overflow:hidden}\n.ap-screen img{max-width:100%;max-height:100%;object-fit:contain;animation:fadein .3s ease}\n.ap-tree{padding:10px 12px 26px}\n.at-row{display:flex;align-items:center;gap:8px;padding:7px 11px;margin:3px 2px;\n  border-radius:10px;background:#f6f6f4;cursor:pointer;font-size:13px;color:#111113;\n  user-select:none;transition:background .15s ease,transform .15s ease;\n  animation:atIn .22s ease both}\n@keyframes atIn{from{opacity:0;transform:translateX(-8px)}to{opacity:1;transform:none}}\n.at-row:hover{background:#edecea;transform:translateX(2px)}\n.at-row:active{transform:scale(.985)}\n.at-ico{width:13px;height:13px;flex:none}\n.at-load{color:#8a8a8e;font-size:12px;padding:4px 8px}\n#agentViewer{position:fixed;inset:0;z-index:130;background:#fff;display:flex;flex-direction:column;\n  animation:apIn .24s cubic-bezier(.2,.7,.3,1) both}\n#avTitle{font-weight:600;font-size:13px;flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}\n#avBtns{display:flex;gap:6px}\n.av-btn{border:1px solid #d9d9d6;background:#fff;border-radius:8px;padding:5px 11px;\n  font-size:12px;cursor:pointer;transition:transform .12s ease,background .15s ease,color .15s ease}\n.av-btn:hover{background:#f4f4f2}\n.av-btn:active{transform:scale(.9)}\n.av-btn.pressed{background:#111113;color:#fff;border-color:#111113}\n.av-body{flex:1;min-height:0;overflow:auto;position:relative}\n.av-frame{width:100%;height:100%;border:0;display:block}\n.av-pre{white-space:pre-wrap;word-break:break-word;font-size:12px;padding:14px;margin:0;\n  font-family:ui-monospace,SFMono-Regular,Menlo,monospace;color:#333}\n.av-md{padding:14px 18px;max-width:840px;margin:0 auto}\n.av-frame,.av-pre,.av-md,.av-code{animation:avSwap .2s ease}\n@keyframes avSwap{from{opacity:0;transform:translateY(5px)}to{opacity:1;transform:none}}\n.av-frame[hidden],.av-pre[hidden],.av-md[hidden],.av-code[hidden]{display:none}\n.av-code{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px;\n  padding:12px 6px;color:#333}\n.codeline{display:flex;align-items:flex-start}\n.codeline:hover{background:#f8f8f6}\n.codeline .ln{flex:none;min-width:38px;text-align:right;padding-right:12px;color:#b9b9b2;\n  user-select:none}\n.codeline .lc{flex:1;white-space:pre-wrap;word-break:break-word}\n.tk-str{color:#1a7f37;font-style:normal}\n.tk-kw{color:#a626a4;font-style:normal;font-weight:600}\n.tk-com{color:#9d9d95;font-style:italic}\n.tk-num{color:#b76b00;font-style:normal}\n.tk-tag{color:#0b5cad;font-style:normal}\n.agent-term{margin:6px 0;animation:fadein .2s ease}\n.agent-loghead{display:flex;align-items:center;gap:6px;cursor:pointer;font-size:11px;\n  font-weight:700;color:#55555a;padding:3px 4px;user-select:none}\n.agent-loghead:hover{color:#111113}\n.agent-loghead svg{transition:transform .25s ease}\n.agent-term.folded .agent-loghead svg{transform:rotate(-90deg)}\n.agent-log{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:11px;\n  background:#f6f6f4;border-radius:10px;padding:8px 10px;margin:4px 0 6px;white-space:pre-wrap;\n  word-break:break-word;max-height:280px;overflow:auto;\n  transition:max-height .28s ease,opacity .22s ease,padding .28s ease,margin .28s ease}\n.agent-term.folded .agent-log{max-height:0;opacity:0;padding-top:0;padding-bottom:0;\n  margin-top:0;margin-bottom:0;overflow:hidden}\n.agent-step{font-weight:700;color:#111113;margin-top:6px;animation:fadein .2s ease}\n.agent-step:first-child{margin-top:0}\n.agent-out{color:#55555a;margin:2px 0 6px;animation:fadein .25s ease}\n.agent-final:empty{display:none}\n.agent-final{animation:fadein .25s ease}\n#agtToast{position:fixed;left:50%;bottom:26px;transform:translateX(-50%);z-index:200;\n  background:#111113;color:#fff;font-size:13px;padding:10px 16px;border-radius:999px;\n  box-shadow:0 6px 24px rgba(17,17,19,.3);max-width:86vw;text-align:center;\n  animation:toastIn .25s cubic-bezier(.2,.7,.3,1) both;pointer-events:none}\n#agtToast.out{animation:toastOut .3s ease both}\n@keyframes toastIn{from{opacity:0;transform:translateX(-50%) translateY(12px)}\n  to{opacity:1;transform:translateX(-50%)}}\n@keyframes toastOut{to{opacity:0;transform:translateX(-50%) translateY(10px)}}\n.chbadge{flex:none;font-size:9.5px;font-weight:700;letter-spacing:.4px;text-transform:uppercase;\n  color:#fff;background:#111113;border-radius:999px;padding:2px 7px;animation:fadein .25s ease}\n#imgmenu:not([hidden]){animation:imgIn .22s cubic-bezier(.2,.7,.3,1) both}\n@keyframes imgIn{from{opacity:0;transform:translateY(8px) scale(.95)}to{opacity:1;transform:none}}\n#send.sent{animation:sentPulse .42s cubic-bezier(.2,.7,.3,1)}\n@keyframes sentPulse{0%{transform:scale(1)}35%{transform:scale(.85)}70%{transform:scale(1.07)}100%{transform:scale(1)}}\n.pwwrap{position:relative;display:flex;align-items:center;width:100%}\n.pwwrap input{flex:1;padding-right:34px}\n.pwpeek{position:absolute;right:4px;border:0;background:none;cursor:pointer;color:var(--ink-3);\n  padding:5px;display:flex;align-items:center;transition:color .15s ease,transform .12s ease}\n.pwpeek:hover{color:var(--ink)}\n.pwpeek:active{transform:scale(.88)}\n#gpasswrap{border-top:1px solid var(--line);margin-top:12px;padding-top:12px;animation:fadein .25s ease}\n/* Images, GIF et vidéos s'affichent AJUSTÉS à l'écran : on voit l'ensemble\n   d'un coup d'œil (comme les SVG). Le bouton « Plein format » rend la taille\n   réelle, avec défilement, pour inspecter un détail. */\n.av-imgwrap{padding:12px;background:#f6f6f4;height:100%;box-sizing:border-box;\n  display:flex;align-items:center;justify-content:center;overflow:auto;\n  animation:avSwap .2s ease}\n.av-img{display:block;max-width:100%;max-height:100%;width:auto;height:auto;\n  object-fit:contain;border-radius:10px;box-shadow:0 4px 18px rgba(0,0,0,.12)}\n.av-imgwrap.full{display:block;overflow:auto;text-align:center}\n.av-imgwrap.full .av-img{max-width:none;max-height:none;margin:0 auto}\n.av-pdfwrap{padding:14px 10px 22px;text-align:center;background:#f1f1f4}\n.av-pdfpage{display:block;max-width:100%;height:auto;margin:0 auto 14px;\n  background:#fff;border-radius:4px;box-shadow:0 3px 14px rgba(0,0,0,.14)}\n.av-pdfnote{font-size:12px;color:var(--ink-3);margin:2px 0 8px}\n.av-vidwrap{padding:12px;background:#101014;height:100%;box-sizing:border-box;\n  display:flex;align-items:center;justify-content:center;overflow:auto;animation:avSwap .2s ease}\n.av-vid{display:block;max-width:100%;max-height:100%;width:auto;height:auto;\n  background:#000;border-radius:8px}\n.av-vidwrap.full{display:block;overflow:auto;text-align:center}\n.av-vidwrap.full .av-vid{max-width:none;max-height:none;margin:0 auto}\n.av-audwrap{padding:46px 20px;text-align:center}\n.av-audname{font-family:var(--mono);font-size:12px;color:var(--ink-2);margin-bottom:14px}\n.av-aud{width:min(560px,92%)}\n.av-run{white-space:pre-wrap;word-break:break-word;font-size:12.5px;margin:12px 14px 16px;padding:12px 14px;\n  background:var(--code-bg);color:var(--code-ink);border-radius:10px;font-family:var(--mono);max-height:45vh;overflow:auto}\n.av-runimg{display:block;max-width:100%;border-radius:10px;margin:0 auto 10px;background:#fff;padding:6px}\n.um-field{display:block;margin:12px 0}\n.um-field>span{display:block;font-size:12.5px;font-weight:600;color:var(--ink-2);margin-bottom:4px}\n.um-field input{width:100%;border:1px solid var(--ink);border-radius:10px;padding:8px 12px;\n  font-size:14.5px;font-family:var(--sans);color:var(--ink);background:#fff;outline:none}\n.um-field input:focus{box-shadow:0 0 0 3px rgba(17,17,19,.08)}\n.um-area .um-edit{min-height:110px}\n.av-md table{border-collapse:collapse;margin:8px 0}\n.av-md td,.av-md th{border:1px solid var(--line);padding:4px 9px;font-size:13px}\n/* --- fiche utilisateur (user.md) : seul fichier modifiable --- */\n.um-wrap{max-width:840px;margin:0 auto;padding:14px 18px}\n.um-id{border:1px solid var(--line);border-radius:12px;padding:8px 18px;background:#fafafa}\n.um-id p{margin:4px 0}\n.um-zh{font-weight:600;font-size:13px;margin:16px 0 6px;color:var(--ink-2);letter-spacing:.02em}\n.um-zone{border:1px dashed #d5d5db;border-radius:12px;padding:10px 18px;min-height:70px}\n.um-zone p{margin:6px 0}\n.um-edit{display:block;width:100%;min-height:240px;border:1px solid var(--ink);border-radius:12px;\n  padding:12px 14px;font-family:var(--sans);font-size:14.5px;line-height:1.65;color:var(--ink);\n  background:#fff;resize:vertical;outline:none}\n.um-edit:focus{box-shadow:0 0 0 3px rgba(17,17,19,.08)}\n.um-ico{display:inline-flex;align-items:center;justify-content:center;gap:6px;padding:5px 9px}\n.um-ico svg{display:block}\n</style>\n<script defer src=\"https://www.gstatic.com/firebasejs/11.6.0/firebase-app-compat.js\"></script>\n<script defer src=\"https://www.gstatic.com/firebasejs/11.6.0/firebase-auth-compat.js\"></script>\n<script defer src=\"https://www.gstatic.com/firebasejs/11.6.0/firebase-firestore-compat.js\"></script>\n<link rel=\"stylesheet\" href=\"https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css\">\n<script defer src=\"https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js\"></script>\n<script defer src=\"https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js\"></script>\n<script defer src=\"https://cdn.jsdelivr.net/npm/pdfjs-dist@3.11.174/legacy/build/pdf.min.js\"></script>\n<script defer src=\"https://cdn.jsdelivr.net/npm/jszip@3.10.1/dist/jszip.min.js\"></script>\n<script defer src=\"https://cdn.jsdelivr.net/npm/docx-preview@0.3.3/dist/docx-preview.min.js\"></script>\n<script defer src=\"https://cdn.jsdelivr.net/npm/mammoth@1.8.0/mammoth.browser.min.js\"></script>\n<script defer src=\"https://cdn.jsdelivr.net/npm/xlsx@0.18.5/dist/xlsx.full.min.js\"></script>\n</head>\n<body>\n\n<header>\n  <button id=\"burger\" class=\"burger\" type=\"button\" aria-label=\"Ouvrir le menu\" aria-expanded=\"false\">\n    <span></span><span></span><span></span>\n  </button>\n  <div class=\"brand\">\n    <svg width=\"19\" height=\"19\" viewBox=\"0 0 24 24\" aria-hidden=\"true\">\n      <path class=\"mk\" fill=\"#111113\" d=\"M12 2.6 14.2 9.8 21.4 12 14.2 14.2 12 21.4 9.8 14.2 2.6 12 9.8 9.8Z\"/>\n    </svg>\n    <span>AI Agent</span>\n  </div>\n  <div class=\"modelwrap\">\n    <button id=\"modelbtn\" type=\"button\" aria-haspopup=\"listbox\" aria-expanded=\"false\"\n      title=\"Clic : liste des modèles · appui long : réglages\"></button>\n    <div id=\"dd\" role=\"listbox\" aria-label=\"Modèles\" hidden></div>\n  </div>\n  <button id=\"agentbtn\" type=\"button\" class=\"agentbtn\" title=\"Mode agent : l'IA pilote la VM Linux via son terminal\">Agent</button>\n  <button id=\"agentfiles\" type=\"button\" class=\"iconbtn\" title=\"Panneau de l'agent : écran de la VM + fichiers\" aria-label=\"Panneau de l'agent\" hidden>\n    <svg width=\"15\" height=\"15\" viewBox=\"0 0 24 24\" fill=\"none\" stroke=\"currentColor\" stroke-width=\"1.7\" stroke-linecap=\"round\" stroke-linejoin=\"round\">\n      <path d=\"M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2Z\"/>\n    </svg>\n  </button>\n  <button class=\"iconbtn\" id=\"clearbtn\" title=\"Nouvelle conversation\" aria-label=\"Nouvelle conversation\">\n    <svg width=\"15\" height=\"15\" viewBox=\"0 0 24 24\" fill=\"none\" stroke=\"currentColor\" stroke-width=\"1.7\" stroke-linecap=\"round\">\n      <path d=\"M4 6h16M9 6V4h6v2M6.5 6l1 14h9l1-14\"/>\n    </svg>\n  </button>\n</header>\n\n<div id=\"proxywarn\" hidden>\n  <strong>Worker injoignable.</strong> Impossible de contacter\n  <code>api-ai.hugdu77777.workers.dev</code>, le relais vers Roboflow.\n  Vérifie ta connexion, ou redéploie le Worker.\n</div>\n\n<main><div class=\"wrap\" id=\"log\"></div></main>\n\n<footer>\n  <div class=\"composer\">\n    <div class=\"tools\">\n      <span class=\"pending\" id=\"pending\">\n        <img id=\"pendthumb\" alt=\"\">\n        <b id=\"pendname\"></b>\n        <button id=\"pendrm\" type=\"button\" title=\"Retirer l'image\" aria-label=\"Retirer l'image\">&times;</button>\n      </span>\n      <span class=\"hint\" id=\"hint\"></span>\n    </div>\n    <div class=\"box\">\n      <button class=\"attach\" id=\"attach\" type=\"button\" aria-pressed=\"false\" title=\"Joindre un fichier (image ou audio — l'audio sert de micro)\">\n        <svg width=\"16\" height=\"16\" viewBox=\"0 0 24 24\" fill=\"none\" stroke=\"currentColor\" stroke-width=\"1.8\" stroke-linecap=\"round\" stroke-linejoin=\"round\">\n          <path d=\"M21.44 11.05l-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66l-9.2 9.19a2 2 0 0 1-2.83-2.83l8.49-8.48\"/>\n        </svg>\n      </button>\n      <div id=\"imgmenu\" hidden>\n        <button id=\"menushare\" type=\"button\" hidden>\n          <svg width=\"13\" height=\"13\" viewBox=\"0 0 24 24\" fill=\"none\" stroke=\"currentColor\" stroke-width=\"1.9\" stroke-linecap=\"round\" stroke-linejoin=\"round\" aria-hidden=\"true\"><rect x=\"2.5\" y=\"4\" width=\"19\" height=\"13\" rx=\"2\"/><path d=\"M8 21h8M12 17v4\"/></svg>\n          partager l'écran\n        </button>\n        <button id=\"menublack\" type=\"button\" aria-pressed=\"false\">\n          <svg width=\"13\" height=\"13\" viewBox=\"0 0 24 24\" aria-hidden=\"true\"><rect x=\"3\" y=\"3\" width=\"18\" height=\"18\" rx=\"3\" fill=\"currentColor\"/></svg>\n          image noire\n        </button>\n        <button id=\"menufile\" type=\"button\">\n          <svg width=\"13\" height=\"13\" viewBox=\"0 0 24 24\" fill=\"none\" stroke=\"currentColor\" stroke-width=\"1.9\" stroke-linecap=\"round\" stroke-linejoin=\"round\" aria-hidden=\"true\">\n            <path d=\"M12 15V4M7.5 8.5 12 4l4.5 4.5M4 15v4h16v-4\"/>\n          </svg>\n          importer un fichier\n        </button>\n      </div>\n      <input type=\"file\" id=\"filepick\" multiple hidden>\n      <div id=\"prompt\" class=\"empty\" contenteditable=\"plaintext-only\" role=\"textbox\" aria-multiline=\"true\"\n           spellcheck=\"false\" data-ph=\"Demandez à l’IA…\"></div>\n      <button id=\"send\" disabled title=\"Envoyer\">\n        <svg width=\"16\" height=\"16\" viewBox=\"0 0 24 24\" fill=\"none\" stroke=\"currentColor\" stroke-width=\"2\" stroke-linecap=\"round\" stroke-linejoin=\"round\">\n          <path d=\"M5 12h13M12 5l7 7-7 7\"/>\n        </svg>\n      </button>\n    </div>\n  </div>\n</footer>\n\n<div id=\"cfg\">\n  <button id=\"cfgclose\" title=\"Fermer\" aria-label=\"Fermer\">\n    <svg width=\"16\" height=\"16\" viewBox=\"0 0 24 24\" fill=\"none\" stroke=\"currentColor\" stroke-width=\"2\" stroke-linecap=\"round\">\n      <path d=\"M6 6l12 12M18 6L6 18\"/>\n    </svg>\n  </button>\n  <div class=\"cfgwrap\">\n    <h2 id=\"cfgtitle\"></h2>\n    <p class=\"cfgsub\">Réglages propres à ce modèle. Vide = valeur par défaut du modèle. Appui long sur un autre modèle pour changer.</p>\n    <div id=\"cfgbody\"></div>\n    <button id=\"cfgreset\" type=\"button\">Réinitialiser</button>\n  </div>\n</div>\n\n<div id=\"fs\">\n  <button id=\"fsclose\" title=\"Fermer\" aria-label=\"Fermer\">\n    <svg width=\"16\" height=\"16\" viewBox=\"0 0 24 24\" fill=\"none\" stroke=\"currentColor\" stroke-width=\"2\" stroke-linecap=\"round\">\n      <path d=\"M6 6l12 12M18 6L6 18\"/>\n    </svg>\n  </button>\n  <iframe id=\"fsframe\" sandbox=\"allow-scripts allow-modals allow-forms\" title=\"Rendu HTML\"></iframe>\n</div>\n\n<script>\n/* =====================================================================\n   Modèles disponibles (slugs vérifiés côté serveur Roboflow)\n   ===================================================================== */\nconst MODELS = [\n  /* Anthropic */\n  {id:\"claude-opus-5-5\", label:\"Claude Opus 5.5\", vendor:\"anthropic\", family:\"claude\",\n   sc:[10,8,9,70], desc:\"Vaisseau amiral d'Anthropic pour le raisonnement exigeant, le code et les travaux agentiques de longue haleine ; très fort sur les gros refactors de code.\"},\n  {id:\"claude-sonnet-5-5\", label:\"Claude Sonnet 5.5\", vendor:\"anthropic\", family:\"claude\",\n   sc:[9.5,8,8,120], desc:\"Classe Sonnet d'Anthropic pour le travail quotidien bien cadré : bâtir des fonctionnalités, corriger des bugs, produire du code propre.\"},\n  {id:\"claude-fable-5-1\", label:\"Claude Fable 5.1\", vendor:\"anthropic\", family:\"claude\",\n   sc:[9,7,8,60], desc:\"Progrès sur toute la ligne vs Fable 5, surtout en codage agentique, workflows longs et travail de connaissance.\"},\n  {id:\"claude-haiku-5-5\", label:\"Claude Haiku 5.5\", vendor:\"anthropic\", family:\"claude\", via:\"Roboflow\",\n   sc:[8,6.5,8,200], desc:\"Le plus rapide de la famille Claude 5.5 : environ 200 tokens par seconde, taillé pour les gros volumes (classification, extraction, résumés courts, pré-filtrage d'agent) et les réponses quasi instantanées.\"},\n  /* OpenAI */\n  {id:\"gpt-6-astra\", hidden:true, label:\"GPT-6 Astra\", vendor:\"openai\", family:\"openai\",\n   sc:[9,9,6,40], desc:\"Modèle phare d'OpenAI pour le travail exigeant de bout en bout : analyse avancée, génie logiciel, recherche approfondie.\"},\n  {id:\"gpt-6.1-sol\", hidden:true, label:\"GPT-6.1 Sol\", vendor:\"openai\", family:\"openai\",\n   sc:[8.5,8.5,7,50], desc:\"Version améliorée de GPT-6 Sol, sous le phare Astra : codage agentique, usage d'ordinateur, documents denses.\"},\n  {id:\"gpt-6-luna\", hidden:true, label:\"GPT-6 Luna\", vendor:\"openai\", family:\"openai\",\n   sc:[6,5,6,100], desc:\"Modèle rapide et économique de la série GPT-6, pour les gros volumes et les charges sensibles à la latence.\"},\n  /* Google */\n  {id:\"gemini-3-8-flash\", hidden:true, label:\"Gemini 3.8 Flash\", vendor:\"google\", family:\"gemini\",\n   sc:[7,8,8.5,300], desc:\"Le Flash le plus intelligent de Google, en net progrès sur 3.7 en ingénierie logicielle, tâches agentiques et raisonnement multi-étapes.\"},\n  {id:\"gemini-3-7-flash\", hidden:true, label:\"Gemini 3.7 Flash\", vendor:\"google\", family:\"gemini\",\n   sc:[7,7,8,300], desc:\"Multimodal de Google pour workflows agentiques rapides, code et raisonnement complexe ; peu tokenivore.\"},\n  {id:\"gemini-3-5-flash-lite\", hidden:true, label:\"Gemini 3.5 Flash-Lite\", vendor:\"google\", family:\"gemini\",\n   sc:[5,5,5,400], desc:\"Modèle haute efficacité de Google, taillé pour les sous-agents qui exécutent des tâches ciblées dans des workflows multi-agents.\"},\n  {id:\"gemini-4-argon\", label:\"Gemini 4 Argon\", vendor:\"google\", family:\"gemini\",\n   sc:[9.5,9,9.5,110], desc:\"Le frontier de la génération 4 (sorti le 13/10/2026) : tâches longues et multi-étapes, code agentique, finance, juridique, cybersécurité ; entrée et sortie jusqu'à 1 M de tokens.\"},\n  {id:\"gemini-3-1-pro\", label:\"Gemini 3.1 Pro\", vendor:\"google\", family:\"gemini\",\n   sc:[5,5,8,120], desc:\"Modèle frontier de raisonnement de Google : ingénierie logicielle renforcée, fiabilité agentique, usage de tokens efficace.\"},\n  /* Google — API officielle Gemini (gratuite) — outils Google activables dans les paramètres */\n  {id:\"gg-gemini-3-8-flash\", label:\"Gemini 3.8 Flash\", vendor:\"google\", family:\"gtext\", via:\"Google AI API\", fb:\"gemini-3-8-flash\", fbTxt:\"roboflow\",\n   sc:[8.5,8,8.5,250], desc:\"Le Flash le plus intelligent, servi par l'API Google gratuite : ~250 TPS mesurés (jusqu'à 300), outils Google activables.\"},\n  {id:\"gg-gemini-3-7-flash\", label:\"Gemini 3.7 Flash\", vendor:\"google\", family:\"gtext\", via:\"Google AI API\", fb:\"gemini-3-7-flash\", fbTxt:\"roboflow\",\n   sc:[8,7.5,8,150], desc:\"Flash 3.7 sur l'API Google gratuite : workflows agentiques rapides, code et raisonnement complexe.\"},\n  {id:\"gg-gemini-3-6-flash\", label:\"Gemini 3.6 Flash\", vendor:\"google\", family:\"gtext\", via:\"Google AI API\",\n   sc:[7.5,7,7.5,150], desc:\"Flash 3.6 sur l'API Google gratuite : équilibré pour les tâches quotidiennes.\"},\n  {id:\"gg-gemini-3-5-flash\", label:\"Gemini 3.5 Flash\", vendor:\"google\", family:\"gtext\", via:\"Google AI API\",\n   sc:[7.5,7.5,8,150], desc:\"Flash 3.5 sur l'API Google gratuite : très bien classé sur l'Arena texte.\"},\n  {id:\"gg-gemini-3-flash\", label:\"Gemini 3 Flash\", vendor:\"google\", family:\"gtext\", via:\"Google AI API\",\n   sc:[7.5,7.5,7.5,180], desc:\"La génération 3.0 (preview) qui a fait bondir les Flash, sur l'API Google gratuite.\"},\n  {id:\"gg-gemini-2-5-flash\", label:\"Gemini 2.5 Flash\", vendor:\"google\", family:\"gtext\", via:\"Google AI API\",\n   sc:[6.5,6,6,170], desc:\"Le Flash éprouvé de la génération 2.5, sur l'API Google gratuite : rapide et économique.\"},\n  {id:\"gg-gemini-2-5-pro\", label:\"Gemini 2.5 Pro\", vendor:\"google\", family:\"gtext\", via:\"Google AI API\",\n   fb:\"gemini-2-5-pro\", fbTxt:\"roboflow\",\n   sc:[8,7.5,7.5,70], desc:\"Le seul Pro disponible en gratuit : raisonnement profond, contexte 1 M, sur l'API Google.\"},\n  {id:\"gg-gemma-4-31b\", label:\"Gemma 4 31B\", vendor:\"google\", family:\"gtext\", via:\"Google AI API\",\n   sc:[6.5,6,6.5,100], desc:\"Le modèle ouvert dense de Google (31 Md de paramètres), servi par l'API gratuite.\"},\n  {id:\"gg-gemma-4-26b-a4b\", label:\"Gemma 4 26B-A4B\", vendor:\"google\", family:\"gtext\", via:\"Google AI API\",\n   sc:[6,5.5,6,120], desc:\"Gemma 4 en MoE ouvert : 26 Md de paramètres, 4 Md actifs — rapport vitesse/qualité.\"},\n  {id:\"gg-gemini-3-5-flash-lite\", label:\"Gemini 3.5 Flash-Lite\", vendor:\"google\", family:\"gtext\", via:\"Google AI API\", fb:\"gemini-3-5-flash-lite\", fbTxt:\"roboflow\",\n   sc:[6,5.5,6.5,300], desc:\"Ultra-léger pour sous-agents et tâches ciblées, sur l'API Google gratuite.\"},\n  {id:\"gg-gemini-3-1-flash-lite\", label:\"Gemini 3.1 Flash-Lite\", vendor:\"google\", family:\"gtext\", via:\"Google AI API\",\n   sc:[5.5,5,6,300], desc:\"Le Flash-Lite de la génération 3.1, sur l'API Google gratuite.\"},\n  {id:\"gg-gemini-2-5-flash-lite\", label:\"Gemini 2.5 Flash-Lite\", vendor:\"google\", family:\"gtext\", via:\"Google AI API\",\n   sc:[5,4.5,5,350], desc:\"Le plus léger et rapide de la génération précédente, sur l'API Google gratuite.\"},\n  {id:\"gg-gemini-3-8-flash-tts\", label:\"Gemini 3.8 Flash TTS\", vendor:\"google\", family:\"gtts\", gen:\"audio\", via:\"Google AI API · audio\",\n   sc:null, desc:\"Synthèse vocale Gemini 3.8 : sortie audio naturelle, 8 voix au choix. (Pas d'outils Google en TTS.)\"},\n  {id:\"gg-gemini-3-8-flash-lite-tts\", label:\"Gemini 3.8 Flash-Lite TTS\", vendor:\"google\", family:\"gtts\", gen:\"audio\", via:\"Google AI API · audio\",\n   sc:null, desc:\"Synthèse vocale Flash-Lite : plus légère et plus rapide. (Pas d'outils Google en TTS.)\"},\n  {id:\"gg-gemini-embedding-2\", label:\"Gemini Embedding 2\", vendor:\"google\", family:\"gembed\", gen:\"embed\", via:\"Google AI API\",\n   sc:null, desc:\"Vecteurs sémantiques 3072 dimensions pour recherche, classification et RAG. (Pas d'outils Google en embedding.)\"},\n  {id:\"gg-gemini-3-5-transcribe\", label:\"Gemini 3.5 Transcribe\", vendor:\"google\", family:\"gtrans\", gen:\"transcribe\", via:\"Google AI API · transcription\",\n   sc:null, desc:\"Transcription fidèle de vos fichiers audio (MP3, WAV, OGG, FLAC…) : joignez l'audio avec le bouton fichier, il remplace le micro.\"},\n  {id:\"gg-gemini-3-5-live-translate\", label:\"Gemini 3.5 Live Translate\", vendor:\"google\", family:\"glivetr\", gen:\"live\", via:\"Google AI API · Live\",\n   sc:null, desc:\"Traduction vocale en temps réel (70+ langues, détection automatique) : joignez un fichier audio, choisissez la langue cible — transcription, traduction et voix traduite.\"},\n  {id:\"gg-gemini-3-8-live\", label:\"Gemini 3.8 Flash Live\", vendor:\"google\", family:\"glive\", gen:\"live\", via:\"Google AI API · Live\",\n   sc:[8.5,8,8.5,250], desc:\"Gemini 3.8 Flash en session temps réel (Live API) : réponse vocale immédiate ; réflexion étendue activable dans les paramètres.\"},\n  /* Groq — API gratuite (matériel GroqCloud) */\n  {id:\"groq-gpt-oss-120b\", label:\"GPT-OSS 120B\", vendor:\"openai\", family:\"groq\", via:\"Groq (gratuit)\",\n   sc:[7.5,7,7,500], desc:\"Le modèle ouvert d'OpenAI (120 Md) sur le matériel Groq : ~500 TPS, meilleur rapport qualité/vitesse du gratuit.\"},\n  {id:\"groq-gpt-oss-20b\", label:\"GPT-OSS 20B\", vendor:\"openai\", family:\"groq\", via:\"Groq (gratuit)\",\n   sc:[6,5.5,5.5,1000], desc:\"GPT-OSS 20B sur Groq : le généraliste le plus rapide (~1000 TPS), idéal pour les gros volumes.\"},\n  {id:\"groq-qwen-3-8-27b\", label:\"Qwen 3.8 27B\", vendor:\"qwen\", family:\"groq\", via:\"Groq (gratuit)\", fb:\"qwen-3-8-27b\", fbTxt:\"roboflow\",\n   sc:[6.5,6,6,500], desc:\"Le modèle ouvert d'Alibaba servi par Groq à ~500 TPS : code et raisonnement.\"},\n  /* Pollinations (API gen.pollinations.ai, compatible OpenAI) */\n  {id:\"po-gpt-6-astra\", label:\"GPT-6 Astra\", vendor:\"openai\", family:\"ptext\", via:\"Pollinations\", fb:\"gpt-6-astra\", fbTxt:\"roboflow\",\n   sc:[9,8.5,9,120], desc:\"Le modèle phare d'OpenAI servi par Pollinations : raisonnement, code, outils et vision au plus haut niveau (contexte ~1 M tokens).\"},\n  {id:\"po-gpt-6-1-sol\", label:\"GPT-6.1 Sol\", vendor:\"openai\", family:\"ptext\", via:\"Pollinations\", fb:\"gpt-6.1-sol\", fbTxt:\"roboflow\",\n   sc:[8.5,8,8.5,180], desc:\"GPT-6.1 Sol : des performances proches d'Astra pour le code et le raisonnement, à coût réduit. Vision et outils.\"},\n  {id:\"po-gpt-6-luna\", label:\"GPT-6 Luna\", vendor:\"openai\", family:\"ptext\", via:\"Pollinations\", fb:\"gpt-6-luna\", fbTxt:\"roboflow\",\n   sc:[7.5,7,7.5,280], desc:\"GPT-6 Luna : le palier haute cadence d'OpenAI — rapide et économique pour les tâches en volume. Vision et outils.\"},\n  {id:\"po-gpt-image-2\", label:\"GPT Image 2\", vendor:\"openai\", family:\"pimg\", gen:\"img\", via:\"Pollinations\",\n   sc:null, desc:\"Génération d'images d'OpenAI (GPT Image 2) via Pollinations : grande fidélité au prompt, texte net dans l'image.\"},\n  {id:\"po-mai-image-2-6\", label:\"MAI Image 2.6\", vendor:\"microsoft\", family:\"pimg\", gen:\"img\", via:\"Pollinations\",\n   sc:null, desc:\"Microsoft MAI Image 2.6 via Pollinations : images détaillées et photoréalistes (côtés multiples de 16 px, 768 à 1024 px).\"},\n  {id:\"po-mai-image-2-6-flash\", label:\"MAI Image 2.6 Flash\", vendor:\"microsoft\", family:\"pimg\", gen:\"img\", via:\"Pollinations\",\n   sc:null, desc:\"Microsoft MAI Image 2.6 Flash : la variante rapide de MAI 2.6, nettement plus véloce pour un coût moindre.\"},\n  {id:\"po-flux-2-dev-free\", label:\"Flux 2 Dev (Free)\", vendor:\"bfl\", family:\"pimg\", gen:\"img\", via:\"Pollinations (communautaire)\",\n   sc:null, desc:\"FLUX.2 [dev] open source de Black Forest Labs, hébergé par un contributeur communautaire Pollinations (scriptsnsenses-sys) : photoréalisme et ancrage du monde physique. Gratuit et illimité, limité à 8 images/min ; les prompts sont journalisés par le contributeur (datasets internes, jamais revendus ni utilisés pour entraîner). La disponibilité dépend du serveur du contributeur.\"},\n  /* Meta */\n  {id:\"muse-spark-1-3\", label:\"Muse Spark 1.3\", vendor:\"meta\", family:\"router\",\n   sc:[7,7,8.5,140], desc:\"Modèle de raisonnement multimodal de Meta pour les workflows agentiques, multi-agents et de codage de longue durée.\"},\n  {id:\"muse-wonder\", label:\"Muse Wonder\", vendor:\"meta\", family:\"router\", via:\"Roboflow\",\n   sc:[8.5,8,8,150], desc:\"Muse Wonder : la variante créative de la famille Muse (Meta) — écriture longue, reformulation, brainstorming et rendus visuels (texte + image en entrée, sortie texte).\"},\n  /* Mistral */\n  {id:\"mistral-large-4\", label:\"Mistral Large 4\", vendor:\"mistral\", family:\"mistral\", via:\"Roboflow\",\n   sc:[9,8,8,90], desc:\"Mistral Large 4 : le modèle amiral de Mistral AI — raisonnement, code, appel d'outils et multilingue (dont le français, langue d'origine de Mistral), avec la fenêtre de contexte la plus large de la famille.\"},\n  {id:\"mistral-medium-3-5\", label:\"Mistral Medium 3.5\", vendor:\"mistral\", family:\"mistral\", via:\"Roboflow\",\n   sc:[8,7,8,130], desc:\"Mistral Medium 3.5 : le compromis qualité/prix de la famille — bon raisonnement et bon code pour un coût nettement inférieur à Large, utile pour les charges soutenues.\"},\n  {id:\"mistral-small-4\", label:\"Mistral Small 4\", vendor:\"mistral\", family:\"mistral\", via:\"Roboflow\",\n   sc:[6.5,5.5,7,220], desc:\"Mistral Small 4 : le petit modèle rapide et économique — tâches simples, extraction, reformulation et pré-traitement, avec appel d'outils.\"},\n  /* Alibaba / Qwen */\n  {id:\"qwen-3-8-max\", label:\"Qwen 3.8 Max\", vendor:\"qwen\", family:\"qwen\", via:\"ModelScope · relais Roboflow\",\n   sc:[7,6,5,60], desc:\"MoE de 2,4 trillions de paramètres d'Alibaba (2.4T-A95B) ; accepte texte, image et vidéo. Servi par Roboflow, avec ModelScope en secours.\"},\n  {id:\"qwen-3-8-flash-next\", label:\"Qwen 3.8 Flash Next\", vendor:\"qwen\", family:\"qwen\",\n   sc:[6,6,4,120], desc:\"Flash multimodal nouvelle génération d'Alibaba : assistance au code, workflows agentiques, compréhension visuelle et vidéo.\"},\n  {id:\"qwen-3-8-2-4t\", hidden:true, label:\"Qwen 3.8 2.4T-A95B · ModelScope\", vendor:\"modelscope\", family:\"router\",\n   via:\"ModelScope\", sc:[7,6,5,60], desc:\"Le même Qwen 3.8 Max (2,4 T de paramètres) appelé directement chez ModelScope, avec les crédits gratuits de la plateforme. Modèle masqué : il sert de repli automatique à « Qwen 3.8 Max ».\"},\n  {id:\"qwen-3-8-27b\", hidden:true, label:\"Qwen 3.8 27B\", vendor:\"qwen\", family:\"qwen\",\n   sc:[6,5,4,60], desc:\"Modèle vision dense open-weight de Qwen pour le code, la recherche et les tâches agentiques longues.\"},\n  /* Moonshot */\n  {id:\"kimi-k3\", label:\"Kimi K3\", vendor:\"kimi\", family:\"router\",\n   sc:[8,6,4,60], desc:\"Modèle multimodal open-weight de 2,8 T de paramètres de Moonshot AI ; fort en code complexe et workflows agentiques longs.\"},\n  /* Zhipu */\n  \n  {id:\"glm-5-3-flash\", label:\"GLM 5.3 Flash\", vendor:\"glm\", family:\"router\", via:\"Roboflow · secours ModelScope\",\n   sc:[8,6.5,5,70], desc:\"Multimodal natif de Z.ai à attention hybride sparse/linéaire ; code efficace et tâches agentiques longues. Servi par Roboflow, avec ModelScope en secours.\"},\n  {id:\"glm-5-3\", label:\"GLM 5.3\", vendor:\"glm\", family:\"router\", via:\"Roboflow · secours ModelScope\",\n   sc:[9,7.5,4.5,80], desc:\"Le grand GLM de Z.ai, frère de GLM 5.3 Flash : raisonnement long, code et agents — servi par ModelScope (crédits gratuits), voie Roboflow en secours.\"},\n  /* DeepSeek */\n  {id:\"deepseek-v4-1-flash\", label:\"DeepSeek V4.1 Flash\", vendor:\"deepseek\", family:\"router\",\n   sc:[7,6,6,150], desc:\"MoE sparse de DeepSeek, premier modèle sur l'architecture CED ; 8 B de paramètres actifs en entrée, 16 B en sortie. Version 1017 du 17 octobre 2026, la plus récente.\"},\n  \n  /* Xiaomi */\n  {id:\"mimo-v2-6-pro\", label:\"Mimo V2.6 Pro\", vendor:\"xiaomi\", family:\"router\",\n   sc:[7,6,6,70], desc:\"Modèle phare de Xiaomi, plus de 1 T de paramètres, conçu pour repousser le plafond des capacités.\"},\n  {id:\"mimo-v2-6-flash\", label:\"Mimo V2.6 Flash\", vendor:\"xiaomi\", family:\"router\",\n   sc:[6,5,6,80], desc:\"MoE open-source de Xiaomi, 309 B au total / 15 B activés par token, à attention hybride.\"},\n  /* Black Forest Labs — génération d'image (Cloudflare Workers AI, gratuit) */\n  /* ModelScope — génération d'image (crédits gratuits de la plateforme) */\n  {id:\"qwen-image-2-1\", label:\"Qwen Image 2.1\", vendor:\"qwen\", family:\"msimg\", gen:\"img\", via:\"ModelScope\",\n   sc:null, edit:true, desc:\"Générateur d'images d'Alibaba (Qwen Image 2.1) : texte→image de qualité photo, bon rendu du texte dans l'image. Joins une image et il la MODIFIE (jusqu'à 10 références côté modèle ; l'app n'en envoie qu'une). Servi par ModelScope.\"},\n  {id:\"qwen-image-2-1-turbo\", label:\"Qwen Image 2.1 Turbo\", vendor:\"qwen\", family:\"msimg\", gen:\"img\", via:\"ModelScope\",\n   sc:null, edit:true, desc:\"La version rapide de Qwen Image 2.1 : même famille, génération en quelques secondes au lieu de vingt — idéale pour essayer plusieurs idées de suite. Elle édite aussi une image jointe. Servie par ModelScope.\"},\n  {id:\"wan-2-2-t2v\", label:\"Wan 2.2\", vendor:\"modelscope\", family:\"msvid\", gen:\"vid\", via:\"Wan 2.2\",\n   sc:null, desc:\"Wan 2.2 (Alibaba) transforme une image en court plan animé de quelques secondes. Sans image jointe, l'application fabrique d'abord l'image de départ avec le modèle choisi dans Paramètres ▸ Modèles ▸ Wan 2.2 (FLUX.2 klein 4B par défaut), puis l'anime.\"},\n  {id:\"flux-2-klein-9b\", label:\"FLUX.2 [klein] 9B\", vendor:\"bfl\", family:\"flux\", gen:\"img\", via:\"Workers AI\", fb:\"hf-flux-2-klein-9b\",\n   sc:null, edit:true, desc:\"Modèle de génération d'images de Black Forest Labs (9 milliards de paramètres), hébergé gratuitement par Cloudflare Workers AI. Texte→image de haute qualité, rapide. Une image jointe part comme référence d'édition (une seule, réduite à 496 px : c'est la limite de Cloudflare).\"},\n  {id:\"flux-2-klein-4b\", label:\"FLUX.2 [klein] 4B\", vendor:\"bfl\", family:\"flux\", gen:\"img\", via:\"Workers AI\", fb:\"hf-flux-2-klein-4b\",\n   sc:null, edit:true, desc:\"Version légère de FLUX.2 klein (4 milliards de paramètres), plus rapide et plus économique ; hébergée gratuitement par Cloudflare Workers AI. Édite aussi une image jointe (une seule référence, 496 px).\"},\n  {id:\"hf-flux-2-klein-9b\", hidden:true, label:\"FLUX.2 [klein] 9B · HF\", vendor:\"bfl\", family:\"kleinhf\", gen:\"img\", via:\"HF ZeroGPU\",\n   sc:null, edit:true, desc:\"FLUX.2 klein 9B servi par le Space officiel Black Forest Labs (variante KV optimisée), gratuit via Hugging Face ZeroGPU, sans token. ~13 s/image.\"},\n  {id:\"hf-flux-2-klein-4b\", hidden:true, label:\"FLUX.2 [klein] 4B · HF\", vendor:\"bfl\", family:\"kleinhf\", gen:\"img\", via:\"HF ZeroGPU\",\n   sc:null, edit:true, desc:\"FLUX.2 klein 4B sur le Space officiel Black Forest Labs, gratuit via Hugging Face ZeroGPU, sans token. Le plus rapide (~2-5 s/image).\"},\n  {id:\"hf-flux-3-klein\", label:\"FLUX.3 [klein]\", vendor:\"bfl\", family:\"klein3\", gen:\"img\", via:\"HF ZeroGPU\",\n   sc:null, edit:true, desc:\"FLUX.3 Klein, le dernier modèle image de Black Forest Labs, sur son Space officiel : jusqu'à 2K, édition avec images de référence, gratuit via Hugging Face ZeroGPU, sans token. ~10 s en 1K, ~30 s en 2K.\"},\n  {id:\"ideogram-v4\", label:\"Ideogram v4\", vendor:\"ideogram\", family:\"ideo\", gen:\"img\", via:\"HF ZeroGPU\",\n   sc:null, desc:\"Ideogram v4 (Space officiel ideogram-ai), excellent en typographie et couleurs, gratuit via Hugging Face ZeroGPU, sans token. ~20-30 s/image.\"},\n  /* IFM — passerelle OpenAI-compatible. K2 Horizon 375B A23B : un MoE de\n     375 milliards de paramètres dont 23 B actifs, raisonnement visible.\n     Aucune clé dans ce fichier : le worker détient IFM_API_KEY. */\n  {id:\"ifm-k2-horizon\", label:\"K2 Horizon 375B A23B\", vendor:\"ifm\", family:\"ifm\", via:\"IFM\",\n   sc:[9,8.5,6,45], desc:\"Modèle MoE de 375 milliards de paramètres (23 B actifs) servi par IFM : raisonnement détaillé, code et analyses longues, avec un débit rapide pour sa taille.\"},\n  /* Arena — le canal direct entre cette application et Arena : aucun service de\n     modèle derrière, tout passe par un canal Firebase dédié. Placé en fin de\n     liste : le modèle choisi par défaut au premier lancement ne change pas. */\n  /* Le modèle « arena » n'est PLUS proposé dans le sélecteur : la discussion\n     « Arena », épinglée en tête de liste, l'active d'elle-même. L'entrée reste\n     ici pour la mécanique interne du canal. */\n  {id:\"arena\", hidden:true, label:\"Arena\", vendor:\"arena\", family:\"arena\", via:\"Canal Firebase\",\n   sc:null, desc:\"Le canal direct avec Arena : la discussion « Arena », épinglée en tête de liste, envoie chaque message dans un canal Firebase dédié ; Arena les lit, répond ou y exécute des commandes. Créé par arena.ai.\"},\n];\n\n/* Relais Cloudflare Worker : il détient la clé Roboflow, ce fichier n'en\n   contient aucune. L'API Roboflow n'autorise en CORS que *.roboflow.com,\n   d'où ce relais. */\nconst API_BASE = \"https://api-ai.hugdu77777.workers.dev\";\nconst ACCESS_CODE = \"ZO7NI3Jmnup0FIf1Jb6o2H7i\";\n\n/* Marques SVG minimales (tracés maison, monochromes) */\nconst MARKS = {\n  /* Arena : la colonne du logo officiel (marque monochrome). */\n  arena: '<svg width=\"13\" height=\"13\" viewBox=\"0 -21.5 167 167\" aria-hidden=\"true\"><g class=\"mk\" fill=\"#111113\"><path d=\"M57.5364 53.3142C52.1508 53.3142 47.7887 57.6763 47.7887 63.0618V123.88H53.9963V63.0618C53.9963 61.0989 55.5902 59.505 57.5531 59.505C59.5161 59.505 61.11 61.0989 61.11 63.0618V123.88H67.3176V63.0618C67.3176 57.6763 62.9555 53.3142 57.5699 53.3142H57.5364Z\"/><path d=\"M109.121 53.3142C103.735 53.3142 99.3734 57.6763 99.3734 63.0618V123.88H105.581V63.0618C105.581 61.0989 107.175 59.505 109.138 59.505C111.101 59.505 112.695 61.0989 112.695 63.0618V123.88H118.902V63.0618C118.902 57.6763 114.54 53.3142 109.155 53.3142H109.121Z\"/><path d=\"M83.3286 53.3142C77.9431 53.3142 73.5809 57.6763 73.5809 63.0618V123.88H79.7886V63.0618C79.7886 61.0989 81.3824 59.505 83.3454 59.505C85.3083 59.505 86.9022 61.0989 86.9022 63.0618V123.88H93.1098V63.0618C93.1098 57.6763 88.7477 53.3142 83.3622 53.3142H83.3286Z\"/><path d=\"M143.782 28.035L148.077 20.5691H158.06L166.667 0H0L8.60681 20.5691H18.5894L22.8676 28.035C14.5292 30.8872 7.66727 38.135 7.66727 49.0571C7.66727 59.9792 15.5526 69.2235 27.3639 69.2235C29.9477 69.2235 32.3468 68.6867 34.4776 67.7471V123.901H40.6684V62.7978C42.6146 60.1637 43.7555 56.8754 43.8393 53.1508C43.8393 53.1005 43.8561 53.0333 43.8561 52.9662C43.8561 52.9159 43.8393 52.8823 43.8393 52.832C43.8393 52.7985 43.8561 52.7649 43.8561 52.7314C43.8561 52.681 43.8393 52.6307 43.8393 52.5636C43.7722 50.047 43.1011 47.6981 41.9267 45.7016H124.74C123.566 47.6981 122.894 50.047 122.827 52.5636C122.827 52.6307 122.811 52.681 122.811 52.7314V52.9662C122.811 52.9662 122.827 53.1005 122.827 53.1508C122.911 56.8754 124.052 60.1637 125.998 62.7978V123.901H132.189V67.7471C134.32 68.6867 136.702 69.2235 139.303 69.2235C151.097 69.2235 158.999 59.5765 158.999 49.0571C158.999 38.5377 152.121 30.8872 143.782 28.035ZM12.734 14.3615L9.31146 6.20764H157.355L153.933 14.3615H12.734ZM140.93 20.5691L137.491 26.5586H29.1759L25.7365 20.5691H140.93ZM139.303 63.0159C133.246 63.0159 129.085 58.9222 129.018 52.8991C129.052 48.6377 131.803 45.7855 135.796 45.7352H135.914C138.414 45.7352 140.46 47.7988 140.46 49.9631V50.0637C140.41 51.3221 139.403 52.2951 138.145 52.2951C136.434 52.2951 135.041 53.6877 135.041 55.399C135.041 57.1103 136.434 58.5028 138.145 58.5028C142.843 58.5028 146.685 54.6943 146.685 49.9631C146.685 49.8121 146.651 49.6611 146.634 49.5269C146.383 43.9736 141.836 39.5443 135.83 39.5443H83.3166L82.9139 39.5275H82.4945L30.8369 39.5443C24.8305 39.5443 20.2839 43.9736 20.0322 49.5269C20.0154 49.6611 19.9819 49.8121 19.9819 49.9631C19.9819 54.6943 23.8239 58.5028 28.5216 58.5028C30.2329 58.5028 31.6254 57.1103 31.6254 55.399C31.6254 53.6877 30.2329 52.2951 28.5216 52.2951C27.2633 52.2951 26.2566 51.3221 26.1895 50.0637C26.1895 50.0302 26.2063 49.9966 26.2063 49.9631C26.2063 47.7988 28.2364 45.7352 30.753 45.7352H30.8536C34.8634 45.7855 37.6149 48.6377 37.6485 52.8991C37.5814 58.9222 33.4038 63.0159 27.3639 63.0159C19.4114 63.0159 13.8749 56.5902 13.8749 49.0571C13.8749 38.89 21.8442 32.7495 31.6925 32.7495H134.974C144.822 32.7495 152.792 38.89 152.792 49.0571C152.792 56.5902 147.255 63.0159 139.303 63.0159Z\"/></g></svg>',\n  modelscope: '<svg width=\"13\" height=\"13\" viewBox=\"0 0 24 24\" aria-hidden=\"true\"><path fill=\"currentColor\" d=\"M2.667 5.333V8H0v5.333h2.667v-2.666H.5V8.5h2.166v2.166h2.666V8H8V5.333Zm0 8v5.334H8V16H5.333v-2.667Zm13.333-8V8h2.667v2.667h2.666V8.5H23.5v2.166h-2.166v2.666H24V8h-2.667V5.333Zm5.333 8h-2.666V16H16v2.667h5.333zM8 10.667v2.666h2.667v-2.666zm2.667 2.666V16h2.666v-2.667zm2.666 0H16v-2.666h-2.667z\"/></svg>',\n  microsoft: '<svg width=\"13\" height=\"13\" viewBox=\"0 0 23 23\" aria-hidden=\"true\"><path fill=\"#F25022\" d=\"M1 1h10v10H1z\"/><path fill=\"#7FBA00\" d=\"M12 1h10v10H12z\"/><path fill=\"#00A4EF\" d=\"M1 12h10v10H1z\"/><path fill=\"#FFB900\" d=\"M12 12h10v10H12z\"/></svg>',\n  anthropic: '<svg width=\"13\" height=\"13\" viewBox=\"0 0 24 24\" aria-hidden=\"true\"><path fill=\"#191918\"  d=\"M17.3041 3.541h-3.6718l6.696 16.918H24Zm-10.6082 0L0 20.459h3.7442l1.3693-3.5527h7.0052l1.3693 3.5528h3.7442L10.5363 3.5409Zm-.3712 10.2232 2.2914-5.9456 2.2914 5.9456Z\"/></svg>',\n  openai: '<svg width=\"13\" height=\"13\" viewBox=\"0 0 24 24\" aria-hidden=\"true\"><path fill=\"#111111\"  d=\"M22.2819 9.8211a5.9847 5.9847 0 0 0-.5157-4.9108 6.0462 6.0462 0 0 0-6.5098-2.9A6.0651 6.0651 0 0 0 4.9807 4.1818a5.9847 5.9847 0 0 0-3.9977 2.9 6.0462 6.0462 0 0 0 .7427 7.0966 5.98 5.98 0 0 0 .511 4.9107 6.051 6.051 0 0 0 6.5146 2.9001A5.9847 5.9847 0 0 0 13.2599 24a6.0557 6.0557 0 0 0 5.7718-4.2058 5.9894 5.9894 0 0 0 3.9977-2.9001 6.0557 6.0557 0 0 0-.7475-7.0729zm-9.022 12.6081a4.4755 4.4755 0 0 1-2.8764-1.0408l.1419-.0804 4.7783-2.7582a.7948.7948 0 0 0 .3927-.6813v-6.7369l2.02 1.1686a.071.071 0 0 1 .038.052v5.5826a4.504 4.504 0 0 1-4.4945 4.4944zm-9.6607-4.1254a4.4708 4.4708 0 0 1-.5346-3.0137l.142.0852 4.783 2.7582a.7712.7712 0 0 0 .7806 0l5.8428-3.3685v2.3324a.0804.0804 0 0 1-.0332.0615L9.74 19.9502a4.4992 4.4992 0 0 1-6.1408-1.6464zM2.3408 7.8956a4.485 4.485 0 0 1 2.3655-1.9728V11.6a.7664.7664 0 0 0 .3879.6765l5.8144 3.3543-2.0201 1.1685a.0757.0757 0 0 1-.071 0l-4.8303-2.7865A4.504 4.504 0 0 1 2.3408 7.872zm16.5963 3.8558L13.1038 8.364 15.1192 7.2a.0757.0757 0 0 1 .071 0l4.8303 2.7913a4.4944 4.4944 0 0 1-.6765 8.1042v-5.6772a.79.79 0 0 0-.407-.667zm2.0107-3.0231l-.142-.0852-4.7735-2.7818a.7759.7759 0 0 0-.7854 0L9.409 9.2297V6.8974a.0662.0662 0 0 1 .0284-.0615l4.8303-2.7866a4.4992 4.4992 0 0 1 6.6802 4.66zM8.3065 12.863l-2.02-1.1638a.0804.0804 0 0 1-.038-.0567V6.0742a4.4992 4.4992 0 0 1 7.3757-3.4537l-.142.0805L8.704 5.459a.7948.7948 0 0 0-.3927.6813zm1.0976-2.3654l2.602-1.4998 2.6069 1.4998v2.9994l-2.5974 1.4997-2.6067-1.4997Z\"/></svg>',\n  groq: '<svg width=\"13\" height=\"13\" viewBox=\"0 0 33 33\" aria-hidden=\"true\"><rect width=\"33\" height=\"33\" rx=\"7\" fill=\"#F43E01\"/><path fill=\"#fff\" d=\"m18.445 4.406-9.468 13.74 7.341.665-1.69 9.578 9.469-13.74-7.342-.664 1.69-9.579Z\"/></svg>',\n  pollinations: '<svg width=\"13\" height=\"13\" xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"15 17.6409 119.9998 114.7498\" fill=\"#0F0F0F\"><path fill-rule=\"evenodd\" d=\"M73.832 19.293a1893.685 1893.685 0 0 0-7.742 9.047c-1.93 2.273-3.274 3.898-4.266 5.156-.988 1.258-1.62 2.149-2.117 2.957-.5.805-.86 1.531-1.309 2.453-.14.285-.277.563-.402.813-.121.254-.238.48-.332.676a20.107 20.107 0 0 1-.34.652c-.008.008-.062 0-.156-.02s-.227-.047-.383-.09c-.16-.039-.351-.09-.558-.148-.207-.059-.43-.121-.668-.191-1.079-.32-1.895-.559-2.684-.743-.79-.18-1.55-.3-2.516-.382-.96-.082-2.129-.121-3.734-.145-1.602-.02-3.64-.023-6.348-.023H35.31l-2.485-.004H30.34v20.574l-.527.059-.524.058-1.055.117c-.59.067-1.203.157-1.84.274-.636.113-1.292.25-1.968.414-.676.16-1.371.348-2.086.555-.715.21-1.445.441-2.195.695-.47.16-1.024.36-1.594.574a64.38 64.38 0 0 0-1.684.649c-.523.207-.992.398-1.328.546-.336.149-.539.246-.539.274 0 .039.07.234.191.535.118.297.29.7.485 1.156.195.453.418.961.644 1.473.227.508.461 1.02.676 1.484a52.041 52.041 0 0 0 9.54 13.926 47.507 47.507 0 0 0 6.14 5.41c2.164 1.59 4.426 2.953 6.75 4.047a51.197 51.197 0 0 0 3.414 1.461c1.129.43 2.242.805 3.36 1.125 1.112.324 2.226.59 3.343.809a39.04 39.04 0 0 0 3.375.504c.84.085 1.781.136 2.777.156a51.632 51.632 0 0 0 6.203-.266 34.786 34.786 0 0 0 2.836-.394 59.177 59.177 0 0 1 2.633-.442c.149-.02.235-.023.246-.015.043.043-.02.195-.152.414a9.01 9.01 0 0 1-.582.812c-.242.313-.527.645-.813.965-.289.316-.582.617-.851.86-.57.519-1.168.968-1.809 1.355-.64.383-1.324.703-2.07.969-.746.261-1.555.464-2.441.617-.891.156-1.856.258-2.926.316-1.36.07-2.582.207-3.703.422-1.118.211-2.13.496-3.07.871-.938.375-1.801.84-2.618 1.403a15.82 15.82 0 0 0-2.351 2 14.205 14.205 0 0 0-1.489 1.765 13.256 13.256 0 0 0-1.148 1.945 15.518 15.518 0 0 0-.856 2.207 20.834 20.834 0 0 0-.59 2.543c-.058.348-.105.61-.156.817-.05.207-.105.36-.183.484a1.184 1.184 0 0 1-.325.332c-.144.106-.332.211-.578.356a9.157 9.157 0 0 0-2.543 2.183 8.797 8.797 0 0 0-1.52 2.813 8.33 8.33 0 0 0-.382 3.113 7.995 7.995 0 0 0 .852 3.098 7.95 7.95 0 0 0 1.574 2.117 9.294 9.294 0 0 0 2.168 1.566c.789.418 1.629.715 2.469.868.84.152 1.68.167 2.472.007a7.941 7.941 0 0 0 1.188-.336c.406-.144.82-.32 1.222-.515.399-.2.79-.418 1.149-.653.36-.234.691-.476.976-.73a6.93 6.93 0 0 0 1.473-1.86c.39-.71.68-1.492.855-2.308.176-.82.243-1.676.188-2.527a8.74 8.74 0 0 0-.531-2.516 5.919 5.919 0 0 0-.68-1.281c-.289-.43-.64-.856-1.031-1.258a10.34 10.34 0 0 0-1.262-1.11 8.299 8.299 0 0 0-1.379-.847c-.414-.2-.722-.352-.95-.492-.226-.137-.366-.27-.44-.43-.075-.16-.083-.352-.043-.613.039-.266.125-.598.242-1.043.226-.871.566-1.695 1.004-2.465a10.426 10.426 0 0 1 3.672-3.754c.77-.46 1.605-.836 2.492-1.102.152-.046.386-.093.675-.144a30.829 30.829 0 0 1 2.246-.273c.43-.043.876-.075 1.317-.102a41.754 41.754 0 0 0 2.773-.25 19.813 19.813 0 0 0 2.243-.406 13.283 13.283 0 0 0 1.918-.621c.609-.246 1.199-.536 1.804-.88a20.824 20.824 0 0 0 1.774-1.116c.539-.383 1.031-.786 1.511-1.227.48-.445.942-.934 1.415-1.488.468-.559.953-1.18 1.468-1.899l.309-.426.308-.43.313-.429.309-.426v19.633l-.399.262-.402.265-.797.532a11.66 11.66 0 0 0-1.567 1.21c-.44.41-.808.837-1.113 1.305-.305.465-.55.97-.746 1.535-.2.567-.351 1.192-.473 1.903a7.401 7.401 0 0 0 .024 2.574 8.502 8.502 0 0 0 2.441 4.543 7.787 7.787 0 0 0 2.157 1.496c.804.367 1.632.61 2.46.715.829.105 1.66.082 2.47-.07a7.963 7.963 0 0 0 2.343-.828 8.831 8.831 0 0 0 2.094-1.567c.406-.402.738-.789 1.011-1.18.27-.394.48-.796.641-1.234.156-.441.266-.918.336-1.473.07-.55.102-1.171.105-1.894 0-.524 0-.93-.011-1.266a6.504 6.504 0 0 0-.074-.836 3.967 3.967 0 0 0-.192-.683c-.09-.23-.207-.485-.355-.805a7.845 7.845 0 0 0-1.594-2.262 8.117 8.117 0 0 0-1.07-.879 9.342 9.342 0 0 0-1.254-.718l-.348-.164-.348-.16-.343-.165-.348-.164v-4.957l-.004-2.476v-2.48c0-1.36.004-2.657.012-3.833.008-1.176.02-2.23.035-3.11.016-.882.035-1.589.055-2.07.02-.48.043-.734.066-.707.02.028.074.102.156.208l.301.421c.121.168.258.364.406.575.145.207.305.433.461.664.38.543.735 1.027 1.09 1.468.352.438.703.832 1.063 1.2.363.367.742.707 1.148 1.039.406.328.844.652 1.332.98.777.528 1.5.965 2.223 1.328.718.364 1.437.657 2.203.89.765.235 1.582.411 2.496.544.914.137 1.93.23 3.094.3.433.024.867.055 1.289.087a40.742 40.742 0 0 1 2.148.215c.27.035.477.07.602.101.965.234 1.898.64 2.761 1.168a11.119 11.119 0 0 1 2.34 1.93 10.355 10.355 0 0 1 1.688 2.437c.43.867.707 1.781.805 2.695l.03.313.032.309.035.316.031.309-.27.14-.538.274-.274.136c-.828.426-1.558.922-2.183 1.493a7.954 7.954 0 0 0-1.54 1.91 7.731 7.731 0 0 0-.87 2.273 9.264 9.264 0 0 0-.172 2.594c.066.96.285 1.871.64 2.703a7.903 7.903 0 0 0 1.446 2.258 8.316 8.316 0 0 0 2.14 1.683c.817.45 1.727.782 2.715.98 1.309.255 2.617.126 3.836-.3a8.813 8.813 0 0 0 3.266-2.05 8.893 8.893 0 0 0 2.117-3.239c.453-1.21.613-2.527.387-3.848a9.06 9.06 0 0 0-1.504-3.644 9.344 9.344 0 0 0-1.301-1.477 8.97 8.97 0 0 0-1.61-1.16c-.18-.097-.347-.2-.507-.293-.16-.097-.305-.183-.434-.265-.125-.082-.23-.153-.312-.211a.573.573 0 0 1-.137-.121.813.813 0 0 1-.063-.184c-.027-.094-.058-.223-.097-.375a11.147 11.147 0 0 1-.11-.512c-.039-.187-.082-.39-.12-.594-.098-.5-.227-1.023-.383-1.554a19.7 19.7 0 0 0-.551-1.594c-.207-.527-.43-1.039-.668-1.512a12.868 12.868 0 0 0-.735-1.3 12.93 12.93 0 0 0-1.582-1.954 15.064 15.064 0 0 0-1.98-1.691 15.833 15.833 0 0 0-2.27-1.355 14.35 14.35 0 0 0-2.449-.934 9.33 9.33 0 0 0-.836-.188 25.72 25.72 0 0 0-1.094-.18 40.224 40.224 0 0 0-2.605-.269 51.614 51.614 0 0 1-1.281-.11 47.27 47.27 0 0 1-1.242-.128 37.65 37.65 0 0 1-1.067-.14 12.643 12.643 0 0 1-.746-.133 11.48 11.48 0 0 1-1.508-.458c-.5-.187-.996-.414-1.472-.668a12.371 12.371 0 0 1-2.606-1.847 16.117 16.117 0 0 1-1.676-1.836 5.604 5.604 0 0 1-.465-.707c-.097-.192-.125-.324-.074-.375.016-.016.117-.012.285.004.168.015.407.047.696.09.293.039.633.093 1.008.156.37.062.78.137 1.207.215a60.65 60.65 0 0 0 1.789.304c.527.079 1.011.133 1.535.172.523.04 1.09.063 1.777.078.692.016 1.504.02 2.527.024 1.22 0 2.204-.008 3.06-.031a27.474 27.474 0 0 0 2.257-.153 20.662 20.662 0 0 0 2.047-.336 48.885 48.885 0 0 0 2.426-.59 44.634 44.634 0 0 0 5.347-1.761 42.256 42.256 0 0 0 4.98-2.387 43.059 43.059 0 0 0 4.633-3.031 46.976 46.976 0 0 0 4.329-3.692 55.822 55.822 0 0 0 3.312-3.468 49.244 49.244 0 0 0 2.879-3.63 50.142 50.142 0 0 0 2.516-3.886 57.29 57.29 0 0 0 2.23-4.258c.375-.793.691-1.48.95-2.066.257-.59.46-1.078.613-1.473.144-.398.242-.7.285-.918.047-.219.039-.351-.016-.402-.078-.07-.41-.23-.902-.442a57.272 57.272 0 0 0-1.828-.73c-.692-.262-1.434-.535-2.121-.777a36.517 36.517 0 0 0-1.778-.586 54.289 54.289 0 0 0-3.531-.879 48.341 48.341 0 0 0-1.715-.332 23.874 23.874 0 0 0-1.465-.215l-.633-.07-.312-.04-.316-.035V39.301l-2.461.004h-2.461l-2.457.004h-2.461c-2.645.004-4.664.007-6.266.03-1.605.024-2.797.067-3.789.141-.992.079-1.789.192-2.598.356-.812.164-1.64.383-2.703.668a23.531 23.531 0 0 1-1.203.3c-.16.036-.289.06-.383.075-.093.016-.148.02-.156.008a29.962 29.962 0 0 1-.355-.672c-.102-.2-.22-.434-.348-.692a79.19 79.19 0 0 1-.418-.828c-.45-.902-.785-1.578-1.168-2.234-.379-.66-.805-1.297-1.422-2.117-.617-.817-1.433-1.82-2.594-3.203-1.16-1.387-2.671-3.153-4.68-5.508-1.398-1.64-2.53-2.961-3.44-4.02a272.265 272.265 0 0 0-2.122-2.441c-.523-.59-.875-.973-1.11-1.211-.237-.234-.355-.324-.402-.32-.015 0-.07.047-.156.132-.086.086-.199.207-.34.356-.136.148-.296.328-.472.523-.172.2-.364.414-.555.641m7.059 11.504a151.52 151.52 0 0 1 3.582 4.336 58.05 58.05 0 0 1 2.488 3.336c.66.965 1.16 1.793 1.516 2.515.355.72.57 1.332.66 1.868.023.14.039.25.031.347a.526.526 0 0 1-.09.262 1.205 1.205 0 0 1-.266.262c-.12.093-.285.199-.492.332-.332.21-.773.554-1.265.976-.493.426-1.043.934-1.594 1.473a43.455 43.455 0 0 0-1.625 1.672 25.487 25.487 0 0 0-1.348 1.574c-.398.512-.812 1.09-1.222 1.703a39.76 39.76 0 0 0-1.204 1.91 39.73 39.73 0 0 0-1.085 1.946c-.329.636-.622 1.25-.86 1.812-.34.809-.672 1.727-.976 2.688-.305.96-.582 1.964-.817 2.945-.234.984-.43 1.945-.562 2.816-.137.871-.215 1.653-.215 2.282-.004.609-.024 1.054-.055 1.332-.035.277-.086.386-.152.328-.067-.055-.149-.278-.246-.664a26.572 26.572 0 0 1-.336-1.66 76.29 76.29 0 0 0-.54-2.758 81.86 81.86 0 0 0-.624-2.66c-.211-.844-.43-1.637-.645-2.34a22.878 22.878 0 0 0-.617-1.801 24.472 24.472 0 0 0-.742-1.66 48.94 48.94 0 0 0-2.047-3.746c-.36-.59-.707-1.13-1.027-1.582a21.393 21.393 0 0 0-1.153-1.446 44.776 44.776 0 0 0-1.558-1.71 47.283 47.283 0 0 0-1.657-1.657c-.539-.512-1.043-.96-1.468-1.3-.172-.141-.34-.278-.489-.41a8.144 8.144 0 0 1-.402-.376 3.846 3.846 0 0 1-.27-.293c-.066-.082-.101-.144-.101-.176 0-.105.098-.378.262-.753.16-.372.386-.848.644-1.356.258-.508.547-1.05.828-1.562a24.735 24.735 0 0 1 1.582-2.508c.262-.371.543-.746.871-1.16.329-.418.7-.871 1.149-1.407.45-.535.973-1.144 1.601-1.87.235-.27.536-.614.875-1.005.336-.394.715-.832 1.114-1.293.394-.46.808-.937 1.215-1.41.402-.472.8-.933 1.168-1.363l.668-.777.668-.774.667-.777.665-.778.117.118.113.12.117.118.113.117a110.75 110.75 0 0 1 1.668 1.914l1.555 1.809c.57.664 1.184 1.383 1.813 2.12M51.227 44.149c.425.09.867.2 1.304.317.434.117.864.242 1.254.37.395.13.754.259 1.055.38.297.12.535.234.687.332.051.035.09.094.11.191.02.098.023.239.011.442-.015.199-.046.457-.097.793-.055.336-.125.75-.215 1.257-.094.524-.172.989-.23 1.446-.06.457-.106.91-.137 1.41a32.68 32.68 0 0 0-.067 1.7c-.011.648-.015 1.401-.015 2.312 0 .98.004 1.761.015 2.414.012.648.035 1.172.07 1.644.04.473.09.89.165 1.336.07.446.164.918.285 1.488.074.348.156.72.242 1.094.082.375.168.754.254 1.11.086.359.168.695.242.992.074.3.14.554.195.75.055.195.102.383.149.554.043.172.082.329.113.461.031.13.059.239.07.317.016.078.02.125.012.133a.942.942 0 0 1-.183-.079 35.955 35.955 0 0 1-1.148-.629c-.255-.144-.532-.304-.821-.472a95.651 95.651 0 0 0-2.031-1.152 44.616 44.616 0 0 0-1.723-.899 34 34 0 0 0-1.633-.75 44.224 44.224 0 0 0-1.762-.715 50.86 50.86 0 0 0-2.855-1.004 45.68 45.68 0 0 0-2.8-.793 39.106 39.106 0 0 0-2.653-.562 26.924 26.924 0 0 0-2.406-.309l-.426-.03-.426-.036-.426-.031-.426-.035V43.758h7.215c1.578.004 2.832.008 3.852.015 1.02.012 1.8.028 2.437.055a20.37 20.37 0 0 1 1.547.117c.43.051.797.117 1.196.203m64.886 7.493c0 1.23-.008 2.34-.015 3.312-.008.977-.02 1.817-.04 2.504-.015.691-.038 1.23-.062 1.602-.023.367-.05.57-.082.59a1.259 1.259 0 0 1-.281.07c-.153.031-.367.062-.617.101-.254.035-.547.078-.868.117-.32.043-.664.083-1.02.122-.765.086-1.538.199-2.323.343-.79.141-1.59.313-2.407.516-.816.203-1.652.434-2.503.7-.856.26-1.73.558-2.625.882-.735.266-1.305.477-1.801.672-.496.195-.918.375-1.348.578-.43.203-.867.43-1.398.723-.528.289-1.149.644-1.946 1.101a34.54 34.54 0 0 1-1.644.895 9.035 9.035 0 0 1-.528.242c-.128.055-.21.074-.226.059-.016-.016-.02-.082-.012-.184a5.49 5.49 0 0 1 .055-.422c.031-.172.066-.371.113-.59.043-.219.098-.457.156-.699.258-1.082.454-1.906.602-2.613.152-.703.258-1.282.332-1.867.078-.583.125-1.172.156-1.895.035-.723.059-1.578.086-2.695.024-.98.04-1.77.043-2.434.004-.66-.004-1.2-.027-1.684a17.76 17.76 0 0 0-.113-1.355 29.95 29.95 0 0 0-.223-1.445c-.152-.887-.27-1.543-.336-2.035-.066-.493-.086-.825-.04-1.059a.664.664 0 0 1 .333-.492c.18-.117.437-.207.781-.336.692-.266 1.29-.469 1.926-.63.637-.16 1.309-.276 2.152-.358.84-.082 1.852-.133 3.164-.16 1.309-.032 2.922-.044 4.965-.047l3.813-.008h1.902l1.906-.004v7.883m-25.05-2.368c.195 1.02.343 1.977.437 2.91.098.934.14 1.837.133 2.75a25.621 25.621 0 0 1-.18 2.797c-.11.961-.273 1.961-.488 3.036a39.395 39.395 0 0 1-.746 3.12 31.161 31.161 0 0 1-.946 2.829c-.355.906-.75 1.781-1.187 2.64a30.9 30.9 0 0 1-1.473 2.555c-.215.34-.597.86-1.074 1.469a76.18 76.18 0 0 1-1.625 2.012 79.82 79.82 0 0 1-1.73 2.011 30.987 30.987 0 0 1-1.387 1.489l-.258.25-.262.254-.257.253-.262.254.02-1.62.023-1.622.02-1.617.019-1.621c.02-1.414.035-2.543.058-3.465.02-.922.047-1.637.082-2.227.036-.59.082-1.054.141-1.472.059-.422.133-.797.227-1.211a38.117 38.117 0 0 1 1.086-3.91c.418-1.235.89-2.403 1.433-3.524a28.68 28.68 0 0 1 1.82-3.222 31.184 31.184 0 0 1 2.258-3.012 18.08 18.08 0 0 1 1.012-1.086c.367-.363.746-.715 1.106-1.023.355-.31.69-.575.964-.762.27-.188.48-.297.59-.297.035 0 .07.031.11.086.039.05.078.129.12.226a4.214 4.214 0 0 1 .216.75m-28.895.91c.914.915 1.738 1.86 2.488 2.86a25.52 25.52 0 0 1 2.028 3.187 30.027 30.027 0 0 1 1.636 3.645 42.64 42.64 0 0 1 1.32 4.23c.122.485.22.872.294 1.274.074.406.125.832.164 1.387.043.554.07 1.242.09 2.171.023.934.039 2.11.058 3.645.02 1.441.035 2.566.04 3.438.003.87 0 1.488-.016 1.921-.016.434-.043.68-.086.813-.04.129-.094.14-.16.101-.106-.058-.376-.324-.743-.722-.367-.399-.84-.926-1.34-1.512a82.643 82.643 0 0 1-1.539-1.832 48.185 48.185 0 0 1-1.34-1.695c-.34-.45-.722-1.031-1.117-1.688a40.534 40.534 0 0 1-1.183-2.125 43.623 43.623 0 0 1-1.075-2.207 21.82 21.82 0 0 1-.773-1.933c-.07-.215-.152-.477-.238-.762-.082-.29-.172-.602-.258-.922a49.961 49.961 0 0 1-.461-1.844c-.187-.86-.316-2.113-.395-3.508a62.606 62.606 0 0 1-.066-4.378c.027-1.434.106-2.762.227-3.735.12-.969.285-1.578.492-1.578.023 0 .097.05.21.14.11.09.259.22.438.38a44.358 44.358 0 0 1 1.305 1.25m53.23 15.39a24.897 24.897 0 0 1-.965 4.492 26.72 26.72 0 0 1-.824 2.25 33.11 33.11 0 0 1-1.058 2.305c-.344.676-.66 1.262-.98 1.805-.321.543-.641 1.035-.997 1.523-.351.485-.734.961-1.18 1.473-.445.508-.953 1.05-1.542 1.66a35.204 35.204 0 0 1-1.579 1.54c-.496.448-.976.847-1.468 1.21-.493.367-1 .703-1.551 1.031-.55.328-1.145.649-1.817.989-.84.421-1.624.761-2.445 1.03a15.58 15.58 0 0 1-2.683.626c-1.004.152-2.145.25-3.532.308-1.382.063-3.011.082-4.98.082H82.18l.03-.113.036-.113.031-.113.035-.114c.055-.199.243-.582.516-1.074.274-.496.633-1.101 1.027-1.746.399-.645.836-1.324 1.266-1.969.43-.644.852-1.25 1.219-1.746a44.604 44.604 0 0 1 5.883-6.445 41.569 41.569 0 0 1 6.812-4.977 38.048 38.048 0 0 1 7.477-3.36 34.711 34.711 0 0 1 7.886-1.585c.11-.008.22-.02.325-.027.105-.008.203-.02.293-.028.09-.008.164-.02.226-.023.063-.008.106-.016.13-.02.023-.004.038.024.05.078a.934.934 0 0 1 .023.239c.004.101 0 .226-.007.363-.008.137-.02.29-.04.45M30.934 66.89c.242 1.234.562 2.488.945 3.738a41.182 41.182 0 0 0 1.32 3.68 37.438 37.438 0 0 0 1.61 3.425 27.185 27.185 0 0 0 1.816 2.961c.45.633.992 1.313 1.594 2.004a38.232 38.232 0 0 0 1.949 2.074c.687.68 1.395 1.34 2.098 1.942.699.597 1.39 1.144 2.03 1.597.298.208.575.407.825.59.25.184.469.352.649.496.18.141.32.258.41.344.09.082.129.133.105.14-.066.028-.328-.038-.71-.167-.384-.13-.891-.324-1.458-.555a48.612 48.612 0 0 1-1.812-.777 38.066 38.066 0 0 1-7.352-4.297 42.01 42.01 0 0 1-5.043-4.422 44.733 44.733 0 0 1-4.37-5.254 45.026 45.026 0 0 1-3.575-5.941c-.125-.258-.25-.504-.356-.735-.109-.226-.207-.433-.285-.609-.078-.176-.14-.32-.18-.422-.042-.101-.058-.164-.05-.172.008-.008.062-.031.148-.066.09-.035.211-.082.36-.137.144-.05.32-.113.507-.18l.614-.21c.433-.145.988-.297 1.593-.446a39.578 39.578 0 0 1 3.805-.73c.574-.078 1.07-.125 1.434-.133l.23-.004.235-.004.23-.004.234-.004.114.57.109.57.113.567.114.57m8.199-1.87c1.383.238 2.8.59 4.226 1.039 1.426.449 2.856 1 4.27 1.636a42.064 42.064 0 0 1 4.16 2.164c1.348.801 2.648 1.68 3.879 2.621.41.317.95.778 1.547 1.32a69.069 69.069 0 0 1 1.914 1.805c.652.641 1.3 1.293 1.871 1.899a31.34 31.34 0 0 1 1.422 1.586 46.416 46.416 0 0 1 1.726 2.27c.57.796 1.125 1.613 1.622 2.39.5.781.94 1.52 1.285 2.152.343.637.593 1.172.71 1.543l.032.114.035.113.04.113.03.113h-5.511c-1.918 0-3.48-.02-4.801-.078-1.324-.058-2.41-.156-3.371-.312a15.68 15.68 0 0 1-2.63-.653c-.831-.289-1.651-.652-2.581-1.109a20.527 20.527 0 0 1-3.137-1.926 24.776 24.776 0 0 1-2.902-2.523 26.07 26.07 0 0 1-2.528-2.988 24.005 24.005 0 0 1-2.011-3.305 35.404 35.404 0 0 1-1.192-2.649 38.321 38.321 0 0 1-.968-2.66 25.394 25.394 0 0 1-.645-2.39c-.156-.72-.238-1.348-.238-1.832 0-.211 0-.371.011-.489.012-.12.032-.203.079-.257.043-.055.105-.082.203-.098.093-.016.222-.016.39-.016.094 0 .25.012.453.032.204.023.454.05.73.09.278.035.583.078.903.128.32.047.653.102.977.157m84.953.085c.336.06.809.168 1.336.297.523.13 1.101.282 1.64.434.543.152 1.051.305 1.43.43.387.125.649.222.7.273.027.023-.028.184-.149.45-.121.265-.3.632-.527 1.062-.223.433-.493.93-.79 1.457a56.9 56.9 0 0 1-.937 1.633 44.31 44.31 0 0 1-3.473 5.027 46.345 46.345 0 0 1-4.12 4.531 43.203 43.203 0 0 1-4.571 3.848 34.726 34.726 0 0 1-4.82 2.969 42.898 42.898 0 0 1-3.02 1.367c-.469.191-.89.351-1.207.457-.32.105-.535.156-.598.137-.027-.012.004-.055.079-.133.074-.074.195-.18.351-.309.153-.129.348-.281.563-.445.218-.164.465-.344.722-.531a31.987 31.987 0 0 0 3.067-2.473 30.025 30.025 0 0 0 4.992-5.887 32.018 32.018 0 0 0 1.953-3.441c.398-.809.75-1.57 1.059-2.309a30.121 30.121 0 0 0 1.48-4.39c.188-.758.363-1.547.52-2.391l.109-.59.113-.586.11-.59.109-.586h1.133c.156 0 .347.008.566.024.215.012.457.035.703.062a20.21 20.21 0 0 1 1.477.203m-79.191 55.711c.363.23.671.555.917.934.247.379.426.816.536 1.273.109.457.148.938.105 1.407a3.61 3.61 0 0 1-.371 1.336c-.156.304-.332.57-.539.808a3.303 3.303 0 0 1-1.512.996 4.104 4.104 0 0 1-.965.184 4.468 4.468 0 0 1-1.03-.043 3.427 3.427 0 0 1-1.719-.805 4.434 4.434 0 0 1-.696-.77 3.378 3.378 0 0 1-.555-1.284 4.203 4.203 0 0 1-.046-1.434c.066-.48.21-.945.43-1.36.214-.413.507-.78.87-1.058a3.654 3.654 0 0 1 1.059-.555c.383-.132.789-.207 1.2-.222.41-.02.82.023 1.214.12.39.102.766.259 1.102.473m64.375-.195c.175.086.37.23.574.406.199.176.402.38.586.598.187.215.36.438.492.652.133.211.226.407.266.567.019.07.035.168.042.285.012.113.02.25.024.398.004.145 0 .301-.004.457a8.875 8.875 0 0 1-.023.461 3.545 3.545 0 0 1-.774 1.953 3.866 3.866 0 0 1-3.625 1.368 3.584 3.584 0 0 1-1.883-.95 4.289 4.289 0 0 1-.55-.609 2.827 2.827 0 0 1-.352-.637 3.475 3.475 0 0 1-.188-.746 6.793 6.793 0 0 1-.054-.941c0-.352.027-.672.082-.961.05-.293.137-.555.25-.797.117-.242.265-.465.449-.676.188-.21.406-.41.668-.601.25-.188.547-.328.879-.434a4.168 4.168 0 0 1 1.05-.187c.364-.024.735 0 1.09.066.36.062.7.176 1 .328m-31.93.473c.177.125.368.328.556.57.183.242.367.531.527.824.16.293.297.598.394.88a2.5 2.5 0 0 1 .16.737c0 .258-.074.567-.206.895a5.551 5.551 0 0 1-.54.996 6 6 0 0 1-.742.906c-.265.27-.547.489-.816.63a2.93 2.93 0 0 1-1.024.3 3.709 3.709 0 0 1-1.117-.047 3.928 3.928 0 0 1-1.066-.37 3.194 3.194 0 0 1-.871-.669 4.034 4.034 0 0 1-.68-1.012 4.028 4.028 0 0 1-.355-1.12 3.802 3.802 0 0 1-.012-1.153 3.5 3.5 0 0 1 .344-1.094c.187-.363.375-.664.586-.91.21-.242.441-.43.71-.57.266-.14.57-.23.922-.282a6.496 6.496 0 0 1 1.239-.046c.254.011.472.027.664.046.195.024.363.055.515.098a2.284 2.284 0 0 1 .813.39\"/></svg>',\n  google: '<svg width=\"13\" height=\"13\" viewBox=\"0 0 48 48\" aria-hidden=\"true\">' +\n    '<path fill=\"#EA4335\" d=\"M24 9.5c3.54 0 6.71 1.22 9.21 3.6l6.85-6.85C35.9 2.38 30.47 0 24 0 14.62 0 6.51 5.38 2.56 13.22l7.98 6.19C12.43 13.72 17.74 9.5 24 9.5z\"/>' +\n    '<path fill=\"#4285F4\" d=\"M46.98 24.55c0-1.57-.15-3.09-.38-4.55H24v9.02h12.94c-.58 2.96-2.26 5.48-4.78 7.18l7.73 6c4.51-4.18 7.09-10.36 7.09-17.65z\"/>' +\n    '<path fill=\"#FBBC05\" d=\"M10.53 28.59c-.48-1.45-.76-2.99-.76-4.59s.27-3.14.76-4.59l-7.98-6.19C.92 16.46 0 20.12 0 24c0 3.88.92 7.54 2.56 10.78l7.97-6.19z\"/>' +\n    '<path fill=\"#34A853\" d=\"M24 48c6.48 0 11.93-2.13 15.89-5.81l-7.73-6c-2.15 1.45-4.92 2.3-8.16 2.3-6.26 0-11.57-4.22-13.47-9.91l-7.98 6.19C6.51 42.62 14.62 48 24 48z\"/></svg>',\n  kimi: '<svg width=\"13\" height=\"13\" viewBox=\"0 0 24 24\" aria-hidden=\"true\"><path fill=\"#1B1B1F\"  d=\"m1.053 16.91 9.538 2.55a21 20.981 0 0 0 .06 2.031l5.956 1.592a12 11.99 0 0 1-15.554-6.172m-1.02-5.79 11.352 3.035a21 20.981 0 0 0-.469 2.01l10.817 2.89a12 11.99 0 0 1-1.845 2.004L.658 15.918a12 11.99 0 0 1-.625-4.796m1.593-5.146L13.573 9.17a21 20.981 0 0 0-1.01 1.874l11.297 3.02a21 20.981 0 0 1-.67 2.362l-11.55-3.087L.125 10.26a12 11.99 0 0 1 1.499-4.285ZM6.067 1.58l11.285 3.016a21 20.981 0 0 0-1.688 1.719l7.824 2.091a21 20.981 0 0 1 .513 2.664L2.107 5.218a12 11.99 0 0 1 3.96-3.638M21.68 4.866 7.222 1.003A12 11.99 0 0 1 21.68 4.866\"/></svg>',\n  qwen: '<svg width=\"13\" height=\"13\" viewBox=\"0 0 80 80\" aria-hidden=\"true\"><image href=\"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAFAAAABQCAYAAACOEfKtAAAACXBIWXMAAAsTAAALEwEAmpwYAAAAAXNSR0IArs4c6QAAAARnQU1BAACxjwv8YQUAAAAOdEVYdFNvZnR3YXJlAEZpZ21hnrGWYwAABfVJREFUeAHtnFFSG0cQhv/ZhaCEF3KD5dkBixvINyAnsDgB+ASQEwSfAHwCuEGUE1gVyLN1A/RiW6l4Ne5Zi7Isza56Rj2zoyp9VRRV0kpa9fROd//dK2DLli0bjELLHHT1wQQ4xRSF7fksw/DTUN0jUVo1YOdYv9Ya13QSBysOHSHHq8lQjZAYrRmw09UFSnxweMmok+NkPFRjJESGtijxF9woPpe4RGK0YsDOke7TvwKO0OVysf9Cd5EQ0Q1oggb98/akMsOfSIjoBpxMcQ4P75uj1/lN95AIUYOIR+CwQpF7+N+jOkECxPVAoSCgFLrVYiRANA+kL9zziLz1KMoNpd6KPFrneOOTZ8Yz4JE2l26BlFHoT/5R71xeEuUS9k1bYkOeeD3LEtgEN+Bsr0ouAbahKcualG5pUnADqvXTltj0XdKkoHugVNrSAoPJg3rFOTCsByZYuzLpzfbtlQQz4Gzv62Nzec05KKQHFthsCs5BwQyYA0npdq5oxTv/HTCpK506ZCibyPlxqIakOI8oNyiwgWRT/M06btUBe8f6fO9IP1XR1PJHedOTqTJsOl2m8QabCJWJegfXvEMbIA+6JA+6AhdNfYtHNfjhPY60qX972CzOKI255RxYa0DPHG5EH3z4w/uYpFQJigihIe+jeviQe3jtJaxLL+W3+PlYX8w/YDySasy32BSmOHM53OqBsyTyBh7QG473chzOB5aq9zvF+9QDilnoz4/qwuU1dR7oXUGYgnyxe1YZM4MpjQZIkCpl0fjD1XiGJQ9cx/vmyac4+fivGi69v9lbv6TjiaXGeH8XI99+s82AUlGTXZBvMsuXsBLzjqS6Z6FYNqCW6zXQYty4Krybhq2UMz2BHmQoKPqajfkKLdI0AUZBb6wyDKmhNIAHdWmMWPVgS2tiYkpRU02tmgCj5+/pPM9cz9OexlBIhxA+fQYpTEaheONz5jxP6Tzv4IjVgLN61qm9t4J+7IDi2cxiK9HP1JZynRwXWlLTU5Hl/WmlKBdw59Il8NUa0OwFSonWsM6r60vlfdo7cD0HPhaNemAnI01McIQCjqvrzZrNLNo3z7mzN40GrCKSozqxAqfV9eGXrj7Fms0sE/gUM/Cx+sLSaY3OcRJqYFx0BsciEC/CaiqVUzlp3mV1XRGfwWEEPpYB/ydVRVIUNTmXdFoTaAantygQL8Jua+7t4CrptOZb4CggjW4OfGwDmoCSKbkKBYzV5RJyCqKqpBoCn/NwEXXqPkhJ81J1MrVd75QRC0JC52kLfO6TCYJpzarV5VDVu6GNZyjtKr3XeJt4r7dmdZnnEm902JLWOHug2VApIstWE6VfD6Zq/EccYqLSdulqcTYgddxuqIslfbuVs/w/q3f7iAhtOS8XH3MyYND9Rjl6Yai0pQHbxJarB4aUpApuWtPW8GZm0UjZBoyy32imWlO2MGtjJrYy3C8+zDKgWXGtw6ooBttUw9K5tHXPCbU5bJkCV40x+1MfkSA1/Ne65LqVO54aJrZYHqjko24j1NyxBqrYacszeYnf655jGVBDOO9bAS3Y0uetKdP7nwupULYZn2d4QUSLyvqrP26K5ROWvueE0argjLvx9sCYU6aW/Wa3q7t0Gb2HHLeTB3U2+17F4pPkVeOfcgw4Ige7Fo4162wbixMNHMbzMrnfoOEn0jnORAVVO7cW4/UhGDionfBWsh/jpMZQFLwyyS5CsaDKzAbdjecXkMBxgJyDWy0s3yf+jiVRFb9VVosq6hXOeuDuC9rQc9wJD4y/o029P/9AgFtlq8ABYfwEVfpy6gsuSJ14iTUwe2pOxvv0sPzrbOLVzxqibRNeBgzNPqUtpWDa4nP7Ahf2zYYxIePJlY5GRcl597350N6vt8WiRkWRIslLuJppLvGEdQmQtiySpAdWs4kCoyRNKooUyV7CAqMkt00qihTJGnDtCdlcPmm2keQeOI/XKIkJHI/qChFI34CuUlqEwDFP8mlMdcO2Yg54zqQqRCR5D3xmVhtbBdCKKQarxnG3bNmyZYGvqspioRhRc4oAAAAASUVORK5CYII=\" width=\"80\" height=\"80\"/></svg>',\n  glm: '<svg width=\"13\" height=\"13\" viewBox=\"0 0 30 30\" aria-hidden=\"true\"><path fill=\"#2D2D2D\" d=\"M24.51,28.51H5.49c-2.21,0-4-1.79-4-4V5.49c0-2.21,1.79-4,4-4h19.03c2.21,0,4,1.79,4,4v19.03C28.51,26.72,26.72,28.51,24.51,28.51z\"/><path fill=\"#FFFFFF\" d=\"M15.47,7.1l-1.3,1.85c-0.2,0.29-0.54,0.47-0.9,0.47h-7.1V7.09C6.16,7.1,15.47,7.1,15.47,7.1z\"/><polygon fill=\"#FFFFFF\" points=\"24.3,7.1 13.14,22.91 5.7,22.91 16.86,7.1\"/><path fill=\"#FFFFFF\" d=\"M14.53,22.91l1.31-1.86c0.2-0.29,0.54-0.47,0.9-0.47h7.09v2.33H14.53z\"/></svg>',\n  ideogram: '<svg width=\"13\" height=\"13\" viewBox=\"0 0 128 128\" aria-hidden=\"true\"><image href=\"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAIAAAACACAMAAAD04JH5AAAAdVBMVEUjJCX/+fQAAAD//////Pf///r///0UFhggISIZGxwbHR4QExULDhEAAAYXGRsABgr58+7Z1NDFwb3x6+eTkI6uqqeJhoRFRETr5eGZlpNfXl3i3dlKSUlYV1bRzMimo6BraWh1c3G3s7A1NTV/fXs9PT0tLS5I7L6LAAAKv0lEQVR4nO1aZ5PiOBDFrWDhnMEY29gY/v9PvFZy4Lb2zOzMVl0V/WVnZYWW1OG9FofDRz7ykY985CMf+chHPvKR/6EcA0AJxNIidMvxryzvw6VMz3nae65tcr0+zc9peQH3dyO/RQRUOWcUhdFboNuCG9MtPK9A/H78n4onGsIcI/wSybboQmwLI43wfnJ98UxIlyVGBZqEQt5/Qs3ySdaR5PmTZwA5faC1jdys2OMlBKXRh4/46UFz+MH1R1LJe4fMXgMuBvb4M7lyUJHxxzQ4toSrAxahOXXeescLN/eh1hV3Tq4/5Y54AfyiHM3aHatct9CnQVq1rHvhP3YJMDKH5r6v/1b75kMUDfqvUq3quzl12M9cAhREnfSkog2kHB1f7hVyDAK8k2v6MDnybkjx7RoIH0rj7fxcyTAMddOUR+mGx7JpapDhGGOU7kNK8L/RGz0f/CLhlDHOOQY9nvRRjCqZdCATgS/CY7/pUgTgf0tM8mK41h1n3MnqargNVd8wh7FRxOte8XNk2Nz0qkudOTigK64Q/6kOPrRjgrHf4QWEvhtFkRtAhvfMWA/z5B70HH2BZhCoLn4INXcwNyRjC/4fLV+cufIytjIrpQDaQn4ycwfXXHWi2aqT9k5GzsWXVfCgwls1YX+ZOjCRx6Fcp8N4sL34ZVkLbMJAc4AvXUR8PXOb5VaxxW3nbIiB8LhJhg69RrMCOZ27nU/xr9f4nUDB5xkcJ7EKeE9naZbhV8RLL/z0tJuFZNXMq7cDA+ad1cQOt24NZ7purly/ZqsGejYrCZevx5PyTQ3mjGtXuh1NO0Nfn3WgWRx2y//wiw3Dx9t2AtK/pQEmVYeut8pqZW/HC+FdOaaUayVoB6AVQJ+j6Vh23CSloN8cDHXI8IYziAPOnzUrI6BZqA4g5xMEMcClb2TEI7Xv90SGv6a/AMQYjbm+hDBbDeZNhhoe9gfncGQ5Rtp7RuwsNJXTuhV3zAkHcJj6UdpWWJX9dLCYHBw+6WQ1DyXZHSDK2bjbFUTA+M2VCLg9m1PgWawOYDYy/BoFgQIHbhBE8+bQSJXLxgY04e5biZTdG2fB3iNwJ871Oh5MZzxhBLvy/CT4+S+wATmGo6OEr40eOJgoBJxP0e/HzhKPbPZ7F05FX5wU3Md2h5kPXuSumZEbGf/HAKjPWuDAunqAXRQSNgY7FYBuUUAm48DkVRnc+BALzw3hNFWPeb7gUU0nCF1PYFiew6bn+yslUYFuryeiAbF1XyFvGf+NZWxKLvFpyijhjBRGg7ggjBOaTSf3JuMfkYOlhazvHJg25J0K8NOcP3y4V30/gBCeCo6IOAjtKzRO8lCdvBO206pEpbiKPiQSAoa+xwuYfd+78v0KxBk1gQdPNyyUGfL8KXR4p8ypAQ5o5KxUnRQzYQKgdnRWTkDckUAymYtDO0/N9vuhj7k8UWYXyUmpjfLuJGNOUiMUUynBbAkaqj6LIKwTGZ0m12YMZLC1MkOpPCv2xkJxJ5LnxDHc8iWik9YL2rK+4LF6caP2qk1VHwxr0N+QutdlG3irDM3zm5ypYw7ZzxplGGN5WZ7XGRlJyMFTDuHfTa7XMAW0kiy/4w6PfuBhxFwlAsrPZYmQab8JoMW0ROe2dT7jF+vqN5uniFaA2Ixzs0HiskmFeiZyfQMXQblNps4CisQKKfD1CTgq6+tTXsGhue97iAC6Vw24TrOHuHkFaivswxsd748X8jp8dxSyGoxkgwdsPSbOlutlKkP9qklVbdbH9zYiQg3alKjKj6z90FGY8hMs81J218zosTIWC5/dw0i5HU/S9gtk0YNHnTXnc5OVA8TWg6CZQXluIbDbzvCdN3YlEYdDr8cXj6/hcmSkoSz/hUG0buv1+t2wzHqEySDDfp3zo0CP/zJPjVT5cTuDuGusRZvaXeqEbt3o1uy+6qr1/3IB04Xb2CQY2fEMn/NuIbeMZ1UnHKi1AjYjJg/uRXZOqJOn5e0rBUyYcgt90Yq6k1kMVr5hQot3XfucUSA4ddKGtao8n941QgHdxg0pqUI98SpEGai8YgYYbrSiUL0M794soUKzdmO13UGb4jpEJa+BiBh+7A6vgYg17wXC7HV9W4bbkEa+yQV41pYCwutoW0bcKcfbr5KRYWdIm5NfJiOWW5T4wstMMrrtBcUaYvC0KLrXdOzGMQK9yDUXpJGjCY4sjT2Ej9jjNR13RZEiVt1/CRLksRJ8Hx7pogICEpiyrPJDRHydWmGdjHiGqDF0iyybQKVzu3z6kDMhbCOPvXaIjNswIA8GC0pYCn5N8DTZ+AShmJeFZOpvNHO4j/iZImG0NoxgxARMTNCs3gvJkFjO+C2CoWMS6KW4b3MXHFWQ566K5YYIM4Bnxs3Jo8ulEj4yjNfm4hFn0mwvKJWwfAmpEXLhqkJuJIQqlXKHE9L1uGtduUYWhiZH+44QrpMS8VDZa1Xd4pkXYcrk9B1i4mz6Hl1XnqOEPjR/QHzpz8gB6M2ckj9Q5ApN3+IpSCikvNNz3U0KeIeYhBtqdgxikw4l2FaFMCTnUXuJl2QUXK4hBBKvogNaBh3NA9VgpGbhTgWQacwKBDCUI2YTeQKBLJfbGpC3SfHz/yQ5VeHYkwP7G1gCiQqUe8lpNHCu87qLxERSHJ5IF0Lv+s+3AKTn6K/yzhM9sA/VRQmf82F3JALKp0DijCGxr0O6QNGsChSeH2veFYTxUpbGa2rWBQqHJROgMQQTd/bHYkx5yQmNf6kSaQOKBqKvQCBBb+usk37o911Wt0jOhdZdZ62lRIMw7QZwSvjuG5A1GllayTdFKuXDiAhrQJzzrGQ9nCLYDYKSSNjqZNUBsRP0BhXGmyJVIreyu0JzkBUd4mwykYliSLJ5k6U5sYy1A7tVRC15l525Ifbb+iVORm5voSL9PrOI5WVhzyXQXh3MaqvyC++1r72wM4fU7xKTeluqtcO3SIVPrjutV2JzsIm3pdp3119TUClzXBBRsrpc/Wi1bpgB9KZYPdPWd8R/pkudcnF/7z5bB2Xyvr3TDFyoc1/8cdaLkvT5pTcLAcPsCGzBlNHVAnPnquzKb61K7LQkn3B+sDjfvvymj7k41dic9cskMOonm/Rglouemq+tHiwFqGK1pIU32A/FfqXCs5ZlaYePHsSB7/tItyQMlwFuVklAJetT6JSh6hKDGLkjS9j184+WV5P74A9jwjlPx74o6j5LqHoH2lyrL+tTlCVjXxdFP6LOPBkH/5ueL0UUw9DIV0kpGP+aCWeOMfBpQIKhMZbvpmf5mqv78AYZdfQtqxuJkA5Ym5ZUT8RjknSyUuk9uiQZURckk9ZvePHHJ/9vAfVMzs6tnFvAWQJBxxfCl9eviU8ErWx2+PAzz+e9AvcKaBnqxKco0pHQEJ9IUYr33ofe0KChplTlD8sPGAwJMW9CUcvfYCBvivfgGi0LYXNB63mtDfpH9Qn7PH7slzxxT4pQGkBjUrBMETboI2YRKonVX3gm3SsISC4AIrU/3JEowa8tbEuRQtzIGxXZ90X4Z3JubPKxhUqbdihr8HP8nd7/bw3CbP4pl06G+i3CNDGS/fSPyQRcTAGTJ1cDsnxZJ9QFyctPnr+Royx/palk4UvTJFuKO/ydXxQKH/NAvMGYrmz5zh/OfOQjH/nIRz7ykY985CMf+XvyD8Nfl71+9iiXAAAAAElFTkSuQmCC\" width=\"128\" height=\"128\"/></svg>',\n  deepseek: '<svg width=\"13\" height=\"13\" viewBox=\"0 0 24 24\" aria-hidden=\"true\"><path fill=\"#4D6BFE\"  d=\"M23.748 4.651c-.254-.124-.364.113-.512.233-.051.04-.094.09-.137.137-.372.397-.806.657-1.373.626-.829-.046-1.537.214-2.163.848-.133-.782-.575-1.248-1.247-1.548-.352-.155-.708-.311-.955-.65-.172-.24-.219-.509-.305-.774-.055-.16-.11-.323-.293-.35-.2-.031-.278.136-.356.276-.313.572-.434 1.202-.422 1.84.027 1.436.633 2.58 1.838 3.393.137.094.172.187.129.323-.082.28-.18.553-.266.833-.055.179-.137.218-.328.14a5.5 5.5 0 0 1-1.737-1.179c-.857-.828-1.631-1.743-2.597-2.46a12 12 0 0 0-.689-.47c-.985-.957.13-1.743.387-1.836.27-.098.094-.433-.778-.428-.872.003-1.67.295-2.687.685a3 3 0 0 1-.465.136 9.6 9.6 0 0 0-2.883-.101c-1.885.21-3.39 1.1-4.497 2.622C.082 8.776-.231 10.854.152 13.02c.403 2.284 1.568 4.175 3.36 5.653 1.857 1.533 3.997 2.284 6.438 2.14 1.482-.085 3.132-.284 4.994-1.86.47.234.962.328 1.78.398.629.058 1.235-.031 1.705-.129.735-.155.684-.836.418-.961-2.155-1.004-1.682-.595-2.112-.926 1.095-1.295 2.768-3.598 3.284-6.733.05-.346.115-.834.108-1.114-.004-.171.035-.238.23-.257a4.2 4.2 0 0 0 1.545-.475c1.397-.763 1.96-2.016 2.093-3.517.02-.23-.004-.467-.247-.588M11.58 18.168c-2.088-1.642-3.101-2.183-3.52-2.16-.39.024-.32.472-.234.763.09.288.207.487.371.74.114.167.192.416-.113.603-.673.416-1.842-.14-1.897-.168-1.361-.801-2.5-1.86-3.301-3.306-.775-1.393-1.225-2.888-1.299-4.482-.02-.385.094-.522.477-.592a4.7 4.7 0 0 1 1.53-.038c2.131.311 3.946 1.264 5.467 2.774.868.86 1.525 1.887 2.202 2.89.72 1.066 1.494 2.082 2.48 2.915.348.291.626.513.892.677-.802.09-2.14.109-3.055-.615zm1.001-6.44a.306.306 0 0 1 .415-.287.3.3 0 0 1 .113.074.3.3 0 0 1 .086.214c0 .17-.136.307-.308.307a.303.303 0 0 1-.306-.307m3.11 1.596c-.2.081-.4.151-.591.16a1.25 1.25 0 0 1-.798-.254c-.274-.23-.47-.358-.551-.758a1.7 1.7 0 0 1 .015-.588c.07-.327-.007-.537-.238-.727-.188-.156-.426-.199-.689-.199a.6.6 0 0 1-.254-.078.253.253 0 0 1-.114-.358 1 1 0 0 1 .192-.21c.356-.202.767-.136 1.146.016.352.144.618.408 1.001.782.392.451.462.576.685.915.176.264.336.536.446.848.066.194-.02.353-.25.45\"/></svg>',\n  meta: '<svg width=\"13\" height=\"13\" viewBox=\"0 0 24 24\" aria-hidden=\"true\"><path fill=\"#0866FF\"  d=\"M6.915 4.03c-1.968 0-3.683 1.28-4.871 3.113C.704 9.208 0 11.883 0 14.449c0 .706.07 1.369.21 1.973a6.624 6.624 0 0 0 .265.86 5.297 5.297 0 0 0 .371.761c.696 1.159 1.818 1.927 3.593 1.927 1.497 0 2.633-.671 3.965-2.444.76-1.012 1.144-1.626 2.663-4.32l.756-1.339.186-.325c.061.1.121.196.183.3l2.152 3.595c.724 1.21 1.665 2.556 2.47 3.314 1.046.987 1.992 1.22 3.06 1.22 1.075 0 1.876-.355 2.455-.843a3.743 3.743 0 0 0 .81-.973c.542-.939.861-2.127.861-3.745 0-2.72-.681-5.357-2.084-7.45-1.282-1.912-2.957-2.93-4.716-2.93-1.047 0-2.088.467-3.053 1.308-.652.57-1.257 1.29-1.82 2.05-.69-.875-1.335-1.547-1.958-2.056-1.182-.966-2.315-1.303-3.454-1.303zm10.16 2.053c1.147 0 2.188.758 2.992 1.999 1.132 1.748 1.647 4.195 1.647 6.4 0 1.548-.368 2.9-1.839 2.9-.58 0-1.027-.23-1.664-1.004-.496-.601-1.343-1.878-2.832-4.358l-.617-1.028a44.908 44.908 0 0 0-1.255-1.98c.07-.109.141-.224.211-.327 1.12-1.667 2.118-2.602 3.358-2.602zm-10.201.553c1.265 0 2.058.791 2.675 1.446.307.327.737.871 1.234 1.579l-1.02 1.566c-.757 1.163-1.882 3.017-2.837 4.338-1.191 1.649-1.81 1.817-2.486 1.817-.524 0-1.038-.237-1.383-.794-.263-.426-.464-1.13-.464-2.046 0-2.221.63-4.535 1.66-6.088.454-.687.964-1.226 1.533-1.533a2.264 2.264 0 0 1 1.088-.285z\"/></svg>',\n  xiaomi: '<svg width=\"13\" height=\"13\" viewBox=\"0 0 24 24\" aria-hidden=\"true\"><path fill=\"#FF6900\"  d=\"M12 0C8.016 0 4.756.255 2.493 2.516.23 4.776 0 8.033 0 12.012c0 3.98.23 7.235 2.494 9.497C4.757 23.77 8.017 24 12 24c3.983 0 7.243-.23 9.506-2.491C23.77 19.247 24 15.99 24 12.012c0-3.984-.233-7.243-2.502-9.504C19.234.252 15.978 0 12 0zM4.906 7.405h5.624c1.47 0 3.007.068 3.764.827.746.746.827 2.233.83 3.676v4.54a.15.15 0 0 1-.152.147h-1.947a.15.15 0 0 1-.152-.148V11.83c-.002-.806-.048-1.634-.464-2.051-.358-.36-1.026-.441-1.72-.458H7.158a.15.15 0 0 0-.151.147v6.98a.15.15 0 0 1-.152.148H4.906a.15.15 0 0 1-.15-.148V7.554a.15.15 0 0 1 .15-.149zm12.131 0h1.949a.15.15 0 0 1 .15.15v8.892a.15.15 0 0 1-.15.148h-1.949a.15.15 0 0 1-.151-.148V7.554a.15.15 0 0 1 .151-.149zM8.92 10.948h2.046c.083 0 .15.066.15.147v5.352a.15.15 0 0 1-.15.148H8.92a.15.15 0 0 1-.152-.148v-5.352a.15.15 0 0 1 .152-.147Z\"/></svg>',\n  mistral: '<svg width=\"15\" height=\"13\" viewBox=\"0 0 24 24\" aria-hidden=\"true\"><defs><linearGradient id=\"mistralM\" x1=\"0\" y1=\"3.4\" x2=\"0\" y2=\"20.6\" gradientUnits=\"userSpaceOnUse\"><stop offset=\"0\" stop-color=\"#FFD800\"/><stop offset=\".2\" stop-color=\"#FFD800\"/><stop offset=\".2\" stop-color=\"#FFAF00\"/><stop offset=\".4\" stop-color=\"#FFAF00\"/><stop offset=\".4\" stop-color=\"#FF8205\"/><stop offset=\".6\" stop-color=\"#FF8205\"/><stop offset=\".6\" stop-color=\"#FA500F\"/><stop offset=\".8\" stop-color=\"#FA500F\"/><stop offset=\".8\" stop-color=\"#E10500\"/><stop offset=\"1\" stop-color=\"#E10500\"/></linearGradient></defs><path fill=\"url(#mistralM)\" d=\"M17.143 3.429v3.428h-3.429v3.429h-3.428V6.857H6.857V3.43H3.43v13.714H0v3.428h10.286v-3.428H6.857v-3.429h3.429v3.429h3.429v-3.429h3.428v3.429h-3.428v3.428H24v-3.428h-3.43V3.429z\"/></svg>',\n  ifm: '<svg width=\"13\" height=\"13\" viewBox=\"0 0 24 24\" aria-hidden=\"true\"><rect width=\"24\" height=\"24\" rx=\"6\" fill=\"#111827\"/><path fill=\"#fff\" d=\"M7 6.6h2.2v4.5l4-4.5h2.7l-4.2 4.7 4.4 5.9h-2.8l-3.1-4.3-1 1.1v3.2H7z\"/></svg>',\n  bfl: '<svg width=\"13\" height=\"13\" viewBox=\"32 60 196.407 140.291\" aria-hidden=\"true\"><path fill=\"#07130E\" d=\"M172.047 119.966H172.045V149.393L148.793 182.239V199.317H94.7969L94.7959 199.318H73.9072L73.9082 199.317H32L130.27 60.9777L172.047 119.966ZM228.407 199.316H195.431V182.303L172.047 149.393V119.965L228.407 199.316ZM65.1064 182.238H86.0381L130.268 119.966H151.156L106.927 182.238H127.917L172.044 119.966H151.158L130.27 90.5373L65.1064 182.238Z\"/></svg>'\n};\n</script>\n<script>\n/* =====================================================================\n   LaTeX (sous-ensemble) -> MathML Core   [rendu natif du navigateur]\n   ===================================================================== */\nconst MATH_SYM = {\n  alpha:\"α\",beta:\"β\",gamma:\"γ\",delta:\"δ\",epsilon:\"ε\",varepsilon:\"ε\",zeta:\"ζ\",eta:\"η\",\n  theta:\"θ\",vartheta:\"ϑ\",iota:\"ι\",kappa:\"κ\",lambda:\"λ\",mu:\"μ\",nu:\"ν\",xi:\"ξ\",pi:\"π\",\n  rho:\"ρ\",sigma:\"σ\",tau:\"τ\",upsilon:\"υ\",phi:\"φ\",varphi:\"φ\",chi:\"χ\",psi:\"ψ\",omega:\"ω\",\n  Gamma:\"Γ\",Delta:\"Δ\",Theta:\"Θ\",Lambda:\"Λ\",Xi:\"Ξ\",Pi:\"Π\",Sigma:\"Σ\",Upsilon:\"Υ\",\n  Phi:\"Φ\",Psi:\"Ψ\",Omega:\"Ω\",\n  infty:\"∞\",partial:\"∂\",nabla:\"∇\",hbar:\"ℏ\",ell:\"ℓ\",Re:\"ℜ\",Im:\"ℑ\",aleph:\"ℵ\",\n  cdot:\"⋅\",times:\"×\",div:\"÷\",pm:\"±\",mp:\"∓\",ast:\"∗\",star:\"⋆\",circ:\"∘\",bullet:\"∙\",\n  cap:\"∩\",cup:\"∪\",setminus:\"∖\",emptyset:\"∅\",varnothing:\"∅\",\n  in:\"∈\",notin:\"∉\",ni:\"∋\",subset:\"⊂\",subseteq:\"⊆\",supset:\"⊃\",supseteq:\"⊇\",\n  forall:\"∀\",exists:\"∃\",neg:\"¬\",lnot:\"¬\",land:\"∧\",lor:\"∨\",\n  le:\"≤\",leq:\"≤\",ge:\"≥\",geq:\"≥\",ne:\"≠\",neq:\"≠\",approx:\"≈\",equiv:\"≡\",sim:\"∼\",\n  simeq:\"≃\",cong:\"≅\",propto:\"∝\",ll:\"≪\",gg:\"≫\",\n  to:\"→\",rightarrow:\"→\",leftarrow:\"←\",Rightarrow:\"⇒\",Leftarrow:\"⇐\",\n  leftrightarrow:\"↔\",Leftrightarrow:\"⇔\",mapsto:\"↦\",longrightarrow:\"⟶\",\n  sum:\"∑\",prod:\"∏\",int:\"∫\",iint:\"∬\",iiint:\"∭\",oint:\"∮\",bigcup:\"⋃\",bigcap:\"⋂\",\n  sqrt:\"√\",angle:\"∠\",perp:\"⊥\",parallel:\"∥\",triangle:\"△\",square:\"□\",\n  ldots:\"…\",cdots:\"⋯\",vdots:\"⋮\",ddots:\"⋱\",dots:\"…\",\n  therefore:\"∴\",because:\"∵\",degree:\"°\",prime:\"′\",\n};\nconst MATH_FUNCS = new Set([\"sin\",\"cos\",\"tan\",\"cot\",\"sec\",\"csc\",\"sinh\",\"cosh\",\"tanh\",\"log\",\"ln\",\"lg\",\"exp\",\"lim\",\"limsup\",\"liminf\",\"max\",\"min\",\"sup\",\"inf\",\"det\",\"dim\",\"deg\",\"gcd\",\"arg\",\"mod\",\"Pr\"]);\nconst BIG_OPS = new Set([\"sum\",\"prod\",\"int\",\"iint\",\"iiint\",\"oint\",\"bigcup\",\"bigcap\",\"lim\"]);\nconst GREEK = new Set([\"alpha\",\"beta\",\"gamma\",\"delta\",\"epsilon\",\"varepsilon\",\"zeta\",\"eta\",\"theta\",\"vartheta\",\"iota\",\"kappa\",\"lambda\",\"mu\",\"nu\",\"xi\",\"pi\",\"rho\",\"sigma\",\"tau\",\"upsilon\",\"phi\",\"varphi\",\"chi\",\"psi\",\"omega\",\"Gamma\",\"Delta\",\"Theta\",\"Lambda\",\"Xi\",\"Pi\",\"Sigma\",\"Upsilon\",\"Phi\",\"Psi\",\"Omega\"]);\n\nfunction esc(s){return s.replace(/&/g,\"&amp;\").replace(/</g,\"&lt;\").replace(/>/g,\"&gt;\").replace(/\"/g,\"&quot;\");}\n\nfunction mathTokens(src){\n  const t=[]; let i=0;\n  while(i<src.length){\n    const c=src[i];\n    if(/\\s/.test(c)){ let j=i; while(j<src.length&&/\\s/.test(src[j]))j++; t.push({k:\"ws\",v:src.slice(i,j)}); i=j; continue; }\n    if(c===\"\\\\\"){\n      let j=i+1;\n      if(j<src.length && !/[A-Za-z]/.test(src[j])){ t.push({k:\"cmd\",v:src[j]}); i=j+1; continue; }\n      while(j<src.length && /[A-Za-z]/.test(src[j]))j++;\n      t.push({k:\"cmd\",v:src.slice(i+1,j)}); i=j; continue;\n    }\n    if(\"{}^_&\".includes(c)){ t.push({k:c,v:c}); i++; continue; }\n    if(c===\"$\"){ i++; continue; }\n    t.push({k:\"ch\",v:c}); i++;\n  }\n  return t;\n}\n\n/* Gestion des exposants / indices après coup : on repasse sur le latex brut\n   avec un mini-parseur séquentiel plus robuste. */\nconst EMPTY_ROW = \"<mrow></mrow>\";\nfunction texToMathML(latex, display){\n  try{\n    const body = texSeq(latex);\n    return '<math xmlns=\"http://www.w3.org/1998/Math/MathML\" display=\"'+(display?\"block\":\"inline\")+'\">'+body+'</math>';\n  }catch(e){\n    return '<span class=\"math-fallback\">'+esc(latex)+\"</span>\";\n  }\n}\nfunction texSeq(src){\n  const tk=mathTokens(src); let p=0;\n  const peek=()=>tk[p];\n  function group(){\n    if(peek()&&peek().k===\"{\"){ p++; const inner=list([\"}\"]); if(peek()&&peek().k===\"}\")p++; return wrap(inner); }\n    const one=atom(); return one;\n  }\n  function wrap(items){\n    items=items.filter(Boolean);\n    if(!items.length) return EMPTY_ROW;\n    if(items.length===1) return items[0];\n    return \"<mrow>\"+items.join(\"\")+\"</mrow>\";\n  }\n  function atom(){\n    const t=peek();\n    if(!t) { p++; return \"\"; }\n    p++;\n    if(t.k===\"ws\") return \"\";\n    if(t.k===\"{\"){ const inner=list([\"}\"]); if(peek()&&peek().k===\"}\")p++; return wrap(inner); }\n    if(t.k===\"cmd\") return cmd(t.v);\n    if(t.k===\"ch\"){\n      const c=t.v;\n      if(/[0-9]/.test(c)){ let n=c; while(peek()&&peek().k===\"ch\"&&/[0-9.]/.test(peek().v)) n+=tk[p++].v; return \"<mn>\"+esc(n)+\"</mn>\"; }\n      if(/[+\\-=<>*/|!,;:]/.test(c)) return \"<mo>\"+esc(c)+\"</mo>\";\n      if(\"()[]\".includes(c)) return '<mo stretchy=\"true\">'+esc(c)+\"</mo>\";\n      if(c===\".\") return \"<mo>.</mo>\";\n      return \"<mi>\"+esc(c)+\"</mi>\";\n    }\n    return \"\";\n  }\n  function cmd(name){\n    if(name===\"frac\"||name===\"dfrac\"||name===\"tfrac\"){ const a=group(),b=group(); return \"<mfrac>\"+a+b+\"</mfrac>\"; }\n    if(name===\"sqrt\"){ const a=group(); return \"<msqrt>\"+a+\"</msqrt>\"; }\n    if(name===\"text\"||name===\"textrm\"||name===\"operatorname\"||name===\"mbox\") return \"<mtext>\"+esc(braceText())+\"</mtext>\";\n    if(name===\"mathbf\"||name===\"bm\"){ const r=braceRaw(); return '<mstyle mathvariant=\"bold\">'+texSeq(r)+\"</mstyle>\"; }\n    if(name===\"mathbb\"){ const r=braceRaw(); return '<mstyle mathvariant=\"double-struck\">'+texSeq(r)+\"</mstyle>\"; }\n    if(name===\"mathit\"){ const r=braceRaw(); return \"<mstyle>\"+texSeq(r)+\"</mstyle>\"; }\n    if(name===\"vec\"){ const r=braceRaw(); return \"<mover>\"+texSeq(r)+'<mo stretchy=\"true\">→</mo></mover>'; }\n    if(name===\"hat\"){ const r=braceRaw(); return \"<mover>\"+texSeq(r)+'<mo stretchy=\"true\">^</mo></mover>'; }\n    if(name===\"bar\"||name===\"overline\"){ const r=braceRaw(); return \"<mover>\"+texSeq(r)+'<mo stretchy=\"true\">‾</mo></mover>'; }\n    if(name===\"tilde\"){ const r=braceRaw(); return \"<mover>\"+texSeq(r)+'<mo stretchy=\"true\">~</mo></mover>'; }\n    if(name===\"underbrace\"){ const r=braceRaw(); return \"<munder>\"+texSeq(r)+'<mo stretchy=\"true\">⏟</mo></munder>'; }\n    if(name===\"left\"||name===\"right\"||/^[bB](ig|igg|igg)?$/.test(name)){\n      const nx=peek(); p++;\n      const ch=nx?(nx.v||\"\"):\"\";\n      if(!ch||ch===\".\") return \"\";\n      return '<mo stretchy=\"true\">'+esc(ch)+\"</mo>\";\n    }\n    if(name===\"limits\"||name===\"nolimits\"||name===\"displaystyle\"||name===\"quad\"||name===\"qquad\"||name===\",\"||name===\";\"||name===\"!\"||name===\" \")\n      return name===\"quad\"?\"<mspace width=\\\"1em\\\"></mspace>\":name===\"qquad\"?\"<mspace width=\\\"2em\\\"></mspace>\":name===\",\"?\"<mspace width=\\\"0.17em\\\"></mspace>\":\"<mspace width=\\\"0.28em\\\"></mspace>\";\n    if(name===\"begin\"){\n      const env=braceText(); const rows=[]; let row=[];\n      for(;;){\n        row.push(wrap(list(null, true)));\n        const t=peek();\n        if(t && t.k===\"&\"){ p++; continue; }\n        if(t && t.k===\"cmd\" && t.v===\"\\\\\"){ p++; rows.push(row); row=[]; continue; }\n        if(t && t.k===\"cmd\" && t.v===\"end\"){ p++; braceText(); break; }\n        break;\n      }\n      if(row.some(c=>c && c!==EMPTY_ROW)) rows.push(row);\n      const trs=rows.filter(r=>r.some(c=>c && c!==EMPTY_ROW))\n        .map(r=>\"<mtr>\"+r.map(c=>\"<mtd>\"+(c||EMPTY_ROW)+\"</mtd>\").join(\"\")+\"</mtr>\").join(\"\");\n      const o=env===\"pmatrix\"?\"(\":env===\"bmatrix\"?\"[\":env===\"vmatrix\"?\"|\":env===\"Bmatrix\"?\"{\":\"\";\n      const c=env===\"pmatrix\"?\")\":env===\"bmatrix\"?\"]\":env===\"vmatrix\"?\"|\":env===\"Bmatrix\"?\"}\":\"\";\n      let out=\"<mtable>\"+trs+\"</mtable>\";\n      if(o||c) return \"<mrow>\"+(o?'<mo stretchy=\"true\">'+esc(o)+\"</mo>\":\"\")+out+(c?'<mo stretchy=\"true\">'+esc(c)+\"</mo>\":\"\")+\"</mrow>\";\n      return out;\n    }\n    if(MATH_FUNCS.has(name)) return '<mi mathvariant=\"normal\">'+esc(name)+\"</mi>\";\n    if(BIG_OPS.has(name)) return \"<mo>\"+esc(MATH_SYM[name]||name)+\"</mo>\";\n    if(MATH_SYM[name]){\n      const s=MATH_SYM[name];\n      if(/[+\\-=<>≤≥≠≈±×÷⋅∩∪∈∉⊂⊆⊃⊇→←⇒⇐↔⇔∧∨∓≡∼≃≅∝≪≫∖∗∘]/.test(s)) return \"<mo>\"+esc(s)+\"</mo>\";\n      return \"<mi>\"+esc(s)+\"</mi>\";\n    }\n    return '<mi mathvariant=\"normal\">'+esc(name)+\"</mi>\";\n  }\n  function braceText(){\n    if(!(peek()&&peek().k===\"{\")){ const t=peek(); p++; return t?t.v:\"\"; }\n    p++; let s=\"\",d=1;\n    while(p<tk.length){ const t=tk[p++];\n      if(t.k===\"{\")d++;\n      if(t.k===\"}\"){ d--; if(!d)break; }\n      s += t.k===\"ws\"?\" \":t.v; }\n    return s;\n  }\n  function braceRaw(){\n    if(!(peek()&&peek().k===\"{\")){ const t=peek(); p++; return t?t.v:\"\"; }\n    p++; let s=\"\",d=1;\n    while(p<tk.length){ const t=tk[p++];\n      if(t.k===\"{\")d++;\n      if(t.k===\"}\"){ d--; if(!d)break; s+=\"}\"; continue; }\n      s += t.k===\"ws\"?\" \":(t.k===\"cmd\"?\"\\\\\"+t.v:t.v); }\n    return s;\n  }\n  function list(stop, cellMode){\n    const out=[];\n    while(p<tk.length){\n      const t=peek();\n      if(stop&&stop.includes(t.k)) break;\n      if(cellMode){\n        if(t.k===\"&\") break;\n        if(t.k===\"cmd\"&&(t.v===\"\\\\\"||t.v===\"end\")) break;\n      }\n      const before=p;\n      let piece=atom();\n      /* exposants / indices */\n      let sup=null, sub=null;\n      while(peek()&&(peek().k===\"^\"||peek().k===\"_\")){\n        const s=tk[p++].k;\n        const g=group();\n        if(s===\"^\") sup=g; else sub=g;\n      }\n      if(sup&&sub) piece=\"<msubsup>\"+piece+sub+sup+\"</msubsup>\";\n      else if(sup) piece=\"<msup>\"+piece+sup+\"</msup>\";\n      else if(sub) piece=\"<msub>\"+piece+sub+\"</msub>\";\n      out.push(piece);\n      if(p===before) p++;\n    }\n    return out;\n  }\n  return wrap(list(null));\n}\n</script>\n<script>\n/* =====================================================================\n   Coloration syntaxique (HTML / SVG / XML, JS, CSS) — sans dépendance\n   ===================================================================== */\nconst JS_KEYWORDS = /^(?:const|let|var|function|return|if|else|for|while|do|switch|case|break|continue|new|class|extends|super|this|typeof|instanceof|in|of|try|catch|finally|throw|async|await|yield|import|export|from|default|null|undefined|true|false|delete|void|static|get|set)$/;\n\nfunction hlJS(src){\n  const rules=[\n    [\"com\",  /^[/][/][^\\n]*|^[/][*][\\s\\S]*?(?:[*][/]|$)/],\n    [\"str\",  /^\"(?:\\\\.|[^\"\\\\\\n])*\"?|^'(?:\\\\.|[^'\\\\\\n])*'?|^`(?:\\\\.|[^`\\\\])*`?/],\n    [\"num\",  /^0[xXbBoO][\\da-fA-F_]+n?\\b|^\\d[\\d_]*(?:\\.\\d[\\d_]*)?(?:[eE][+-]?\\d+)?n?\\b/],\n    [\"id\",   /^[A-Za-z_$][\\w$]*/],\n    [\"punc\", /^[{}()\\[\\];,.:?!<>=+\\-*/%&|^~]+/],\n    [\"ws\",   /^\\s+/],\n  ];\n  let out=\"\", i=0;\n  while(i<src.length){\n    let hit=null;\n    for(const [kind,re] of rules){\n      re.lastIndex=0;\n      const m=re.exec(src.slice(i));\n      if(m&&m.index===0){ hit=[kind,m[0]]; break; }\n    }\n    if(!hit){ out+=esc(src[i]); i++; continue; }\n    const [kind,txt]=hit; i+=txt.length;\n    if(kind===\"ws\"){ out+=txt; continue; }\n    if(kind===\"id\"){\n      const after=src.slice(i).match(/^\\s*\\(/);\n      let cls = JS_KEYWORDS.test(txt) ? \"t-kw\" : (after ? \"t-fn\" : null);\n      out += cls ? '<span class=\"'+cls+'\">'+esc(txt)+\"</span>\" : esc(txt);\n      continue;\n    }\n    const cls={com:\"t-com\",str:\"t-str\",num:\"t-num\",punc:\"t-punc\"}[kind];\n    out += '<span class=\"'+cls+'\">'+esc(txt)+\"</span>\";\n  }\n  return out;\n}\n\nfunction hlCSS(src){\n  const rules=[\n    [\"com\", /^[/][*][\\s\\S]*?(?:[*][/]|$)/],\n    [\"str\", /^\"(?:\\\\.|[^\"\\\\])*\"?|^'(?:\\\\.|[^'\\\\])*'?/],\n    [\"at\",  /^@[\\w-]+/],\n    [\"num\", /^#[0-9a-fA-F]{3,8}\\b|^-?\\d+(?:\\.\\d+)?(?:px|em|rem|%|vh|vw|vmin|vmax|s|ms|deg|fr|ch|pt)?\\b/],\n    [\"fn\",  /^[a-zA-Z-]+(?=\\()/],\n    [\"prop\",/^[-a-zA-Z]+(?=\\s*:)/],\n    [\"punc\",/^[{}();:,>~+.=\\[\\]!*^$|]+/],\n    [\"ws\",  /^\\s+/],\n  ];\n  let out=\"\",i=0;\n  while(i<src.length){\n    let hit=null;\n    for(const [kind,re] of rules){ re.lastIndex=0; const m=re.exec(src.slice(i)); if(m&&m.index===0){hit=[kind,m[0]];break;} }\n    if(!hit){ out+=esc(src[i]); i++; continue; }\n    const [kind,txt]=hit; i+=txt.length;\n    if(kind===\"ws\"){ out+=txt; continue; }\n    const cls={com:\"t-com\",str:\"t-str\",num:\"t-num\",at:\"t-kw\",fn:\"t-fn\",prop:\"t-prop\",punc:\"t-punc\"}[kind];\n    out+='<span class=\"'+cls+'\">'+esc(txt)+\"</span>\";\n  }\n  return out;\n}\n\nfunction hlXML(src){\n  let out=\"\", i=0, embed=null;   // embed = 'script' | 'style' | null\n  while(i<src.length){\n    if(embed){\n      const closeRe = new RegExp(\"</\"+embed+\"\\\\s*>\",\"i\");\n      const m = closeRe.exec(src.slice(i));\n      const chunk = m ? src.slice(i, i+m.index) : src.slice(i);\n      out += embed===\"script\" ? hlJS(chunk) : hlCSS(chunk);\n      i += chunk.length;\n      if(!m){ break; }\n      embed=null;   // la balise fermante est traitée par la suite\n    }\n    const c=src[i];\n    if(c===\"<\" && src.startsWith(\"<!\"+\"--\",i)){\n      const e=src.indexOf(\"-->\",i); const end=e<0?src.length:e+3;\n      out+='<span class=\"t-com\">'+esc(src.slice(i,end))+\"</span>\"; i=end; continue;\n    }\n    if(c===\"<\" && /^<!\\[CDATA\\[/.test(src.slice(i))){\n      const e=src.indexOf(\"]]>\",i); const end=e<0?src.length:e+3;\n      out+='<span class=\"t-str\">'+esc(src.slice(i,end))+\"</span>\"; i=end; continue;\n    }\n    if(c===\"<\" && /^<!/.test(src.slice(i))){\n      const e=src.indexOf(\">\",i); const end=e<0?src.length:e+1;\n      out+='<span class=\"t-kw\">'+esc(src.slice(i,end))+\"</span>\"; i=end; continue;\n    }\n    if(c===\"<\" && /^<\\?/.test(src.slice(i))){\n      const e=src.indexOf(\"?>\",i); const end=e<0?src.length:e+2;\n      out+='<span class=\"t-com\">'+esc(src.slice(i,end))+\"</span>\"; i=end; continue;\n    }\n    if(c===\"<\" && /^<[/]?[A-Za-z_]/.test(src.slice(i))){\n      const e=src.indexOf(\">\",i); const end=e<0?src.length:e+1;\n      const tag=src.slice(i,end);\n      const nameM=tag.match(/^<[/]?([A-Za-z_][\\w:.-]*)/);\n      const name=nameM?nameM[1].toLowerCase():\"\";\n      const closing=tag.startsWith(\"</\");\n      let html='<span class=\"t-punc\">'+esc(closing?\"</\":\"<\")+\"</span>\";\n      html+='<span class=\"t-tag\">'+esc(nameM?nameM[1]:\"\")+\"</span>\";\n      const body=tag.slice((closing?2:1)+ (nameM?nameM[1].length:0), tag.length - (tag.endsWith(\"/>\")?2:1));\n      html+=hlAttrs(body);\n      const tail=tag.endsWith(\"/>\")?\"/>\":\">\";\n      html+='<span class=\"t-punc\">'+esc(tail)+\"</span>\";\n      out+=html; i=end;\n      if(!closing && !tag.endsWith(\"/>\") && (name===\"script\"||name===\"style\")) embed=name;\n      continue;\n    }\n    const e=src.indexOf(\"<\",i); const end=e<0?src.length:e;\n    out+='<span class=\"t-txt\">'+esc(src.slice(i,end))+\"</span>\"; i=end;\n  }\n  return out;\n}\nfunction hlAttrs(s){\n  let out=\"\",i=0;\n  const rules=[\n    [\"str\",/^\"(?:[^\"\\\\]|\\\\.)*\"?|^'(?:[^'\\\\]|\\\\.)*'?/],\n    [\"attr\",/^[A-Za-z_:][\\w:.-]*/],\n    [\"eq\",/^=/],\n    [\"ws\",/^\\s+/],\n    [\"punc\",/^[^\\s=]+/],\n  ];\n  while(i<s.length){\n    let hit=null;\n    for(const [kind,re] of rules){ re.lastIndex=0; const m=re.exec(s.slice(i)); if(m&&m.index===0){hit=[kind,m[0]];break;} }\n    if(!hit){ out+=esc(s[i]); i++; continue; }\n    const [kind,txt]=hit; i+=txt.length;\n    if(kind===\"ws\"){ out+=txt; continue; }\n    const cls={str:\"t-str\",attr:\"t-attr\",eq:\"t-punc\",punc:\"t-punc\"}[kind];\n    out+='<span class=\"'+cls+'\">'+esc(txt)+\"</span>\";\n  }\n  return out;\n}\n\nfunction highlight(code, lang){\n  const l=(lang||\"\").toLowerCase();\n  if(l===\"html\"||l===\"svg\"||l===\"xml\"||l===\"xhtml\"||l===\"htm\") return hlXML(code);\n  if(l===\"css\"||l===\"scss\") return hlCSS(code);\n  if(/^(js|javascript|mjs|ts|typescript|jsx|tsx|json)$/.test(l)) return hlJS(code);\n  /* détection automatique */\n  const t=code.trim();\n  if(/^<svg[\\s>]/i.test(t)||new RegExp(\"^<\"+\"!DOCTYPE\\\\s+svg\",\"i\").test(t)) return hlXML(code);\n  if(new RegExp(\"^<\"+\"!DOCTYPE html|^<\"+\"html[\\\\s>]|^<\"+\"[a-z][\\\\w-]*[\\\\s>]\",\"i\").test(t)) return hlXML(code);\n  return esc(code);\n}\nfunction sniffLang(code, lang){\n  if(lang) return lang.toLowerCase();\n  const t=code.trim();\n  if(/^<svg[\\s>]/i.test(t)||new RegExp(\"^<\"+\"!DOCTYPE\\\\s+svg\",\"i\").test(t)) return \"svg\";\n  if(new RegExp(\"^<\"+\"!DOCTYPE html\",\"i\").test(t)||new RegExp(\"<\"+\"html[\\\\s>]\",\"i\").test(t)) return \"html\";\n  if(/^<\\?xml/i.test(t)) return \"xml\";\n  return \"\";\n}\nconst RUNNABLE = new Set([\"html\",\"svg\",\"htm\",\"xhtml\"]);\n</script>\n<script>\n/* =====================================================================\n   Markdown -> HTML  (avec maths et zones de code)\n   ===================================================================== */\nconst CODE_STORE = [];\nlet CODE_SEQ = 0;\n\nfunction renderMarkdown(src){\n  const codes=[], maths=[];\n  let s = String(src||\"\").replace(/\\r\\n?/g,\"\\n\");\n\n  /* 1. blocs de code ``` */\n  s = s.replace(/^[ \\t]*```([^\\n`]*)\\n([\\s\\S]*?)(?:^[ \\t]*```[ \\t]*$|$(?![\\s\\S]))/gm,\n    (m, lang, body) => {\n      const L = sniffLang(body, (lang||\"\").trim());\n      codes.push({lang:L, code:body.replace(/\\n$/,\"\")});\n      return \"\\u0001C\"+(codes.length-1)+\"\\u0001\\n\";\n    });\n\n  /* 2. math bloc $$...$$ et \\[...\\] */\n  s = s.replace(/\\$\\$([\\s\\S]+?)\\$\\$/g, (m,t)=>{ maths.push({tex:t.trim(),d:true});  return \"\\u0001M\"+(maths.length-1)+\"\\u0001\"; });\n  s = s.replace(/\\\\\\[([\\s\\S]+?)\\\\\\]/g, (m,t)=>{ maths.push({tex:t.trim(),d:true});  return \"\\u0001M\"+(maths.length-1)+\"\\u0001\"; });\n  /* 3. math inline $...$ et \\(...\\) */\n  s = s.replace(/\\\\\\(([\\s\\S]+?)\\\\\\)/g, (m,t)=>{ maths.push({tex:t.trim(),d:false}); return \"\\u0001M\"+(maths.length-1)+\"\\u0001\"; });\n  s = s.replace(/(^|[^\\\\$])\\$([^\\s$](?:[^$\\n]*?[^\\s$])?)\\$(?!\\d)/g,\n    (m,pre,t)=>{ maths.push({tex:t.trim(),d:false}); return pre+\"\\u0001M\"+(maths.length-1)+\"\\u0001\"; });\n\n  /* 4. code inline */\n  s = s.replace(/(`+)([\\s\\S]*?)\\1/g, (m,f,t)=>{ codes.push({lang:\"\", code:t, inline:true}); return \"\\u0001C\"+(codes.length-1)+\"\\u0001\"; });\n\n  s = esc(s);\n\n  /* 5. blocs */\n  const lines = s.split(\"\\n\");\n  let html=\"\", i=0, para=[];\n  const flushPara=()=>{ if(para.length){ html+=\"<p>\"+inline(para.join(\"\\n\"))+\"</p>\"; para=[]; } };\n\n  while(i<lines.length){\n    const ln=lines[i];\n\n    if(/^\\u0001C\\d+\\u0001\\s*$/.test(ln.trim())){\n      flushPara(); html+=ln.trim(); i++; continue;\n    }\n    if(!ln.trim()){ flushPara(); i++; continue; }\n\n    let m;\n    if((m=ln.match(/^(#{1,6})\\s+(.*)$/))){ flushPara(); const l=m[1].length; html+=\"<h\"+l+\">\"+inline(m[2].trim())+\"</h\"+l+\">\"; i++; continue; }\n    if(/^\\s*([-*_])\\s*(?:\\1\\s*){2,}$/.test(ln)){ flushPara(); html+=\"<hr>\"; i++; continue; }\n\n    if(/^\\s{0,3}(?:&gt;|>)/.test(ln)){\n      flushPara(); const buf=[];\n      while(i<lines.length && /^\\s{0,3}(?:&gt;|>)/.test(lines[i])){ buf.push(lines[i].replace(/^\\s{0,3}(?:&gt;|>)\\s?/,\"\")); i++; }\n      html+=\"<blockquote>\"+renderMarkdown(decode(buf.join(\"\\n\")))+\"</blockquote>\"; continue;\n    }\n\n    if(/^\\s*\\|.*\\|\\s*$/.test(ln) && i+1<lines.length && /^\\s*\\|[\\s:|-]+\\|\\s*$/.test(lines[i+1])){\n      flushPara();\n      const head=splitRow(ln); i+=2; const rows=[];\n      while(i<lines.length && /^\\s*\\|.*\\|\\s*$/.test(lines[i])){ rows.push(splitRow(lines[i])); i++; }\n      html+=\"<table><thead><tr>\"+head.map(c=>\"<th>\"+inline(c)+\"</th>\").join(\"\")+\"</tr></thead><tbody>\"\n          + rows.map(r=>\"<tr>\"+r.map(c=>\"<td>\"+inline(c)+\"</td>\").join(\"\")+\"</tr>\").join(\"\")\n          + \"</tbody></table>\";\n      continue;\n    }\n\n    if(/^\\s*(\\u2022|[-*+]|\\d+[.)])\\s+/.test(ln)){\n      flushPara();\n      const [consumed, listHtml] = parseList(lines, i);\n      html+=listHtml; i=consumed; continue;\n    }\n\n    para.push(ln); i++;\n  }\n  flushPara();\n\n  /* 6. restauration */\n  html = html.replace(/\\u0001C(\\d+)\\u0001/g, (m,n)=>{\n    const c=codes[+n];\n    if(c.inline) return \"<code>\"+esc(c.code)+\"</code>\";\n    const idx=CODE_SEQ++; CODE_STORE[idx]={code:c.code, lang:c.lang};\n    return zoneHTML(idx, c.lang);\n  });\n  html = html.replace(/\\u0001M(\\d+)\\u0001/g, (m,n)=>{\n    const x=maths[+n];\n    const mm=texToMathML(x.tex, x.d);\n    return x.d ? '<span class=\"math math-block\">'+mm+\"</span>\"\n                : '<span class=\"math math-inline\">'+mm+\"</span>\";\n  });\n  return html;\n}\n\nfunction decode(s){ return s.replace(/&lt;/g,\"<\").replace(/&gt;/g,\">\").replace(/&quot;/g,'\"').replace(/&amp;/g,\"&\"); }\nfunction splitRow(ln){\n  let t=ln.trim().replace(/^\\|/,\"\").replace(/\\|$/,\"\");\n  return t.split(/(?<!\\\\)\\|/).map(c=>c.trim().replace(/\\\\\\|/g,\"|\"));\n}\n\nfunction inline(s){\n  s = s.replace(/!\\[([^\\]]*)\\]\\(([^)\\s]+)(?:\\s+&quot;[^&]*&quot;)?\\)/g,\n        (m,a,u)=>'<img src=\"'+u+'\" alt=\"'+a+'\">');\n  s = s.replace(/\\[([^\\]]+)\\]\\(([^)\\s]+)(?:\\s+&quot;[^&]*&quot;)?\\)/g,\n        (m,t,u)=>'<a href=\"'+u+'\" target=\"_blank\" rel=\"noopener noreferrer\">'+t+\"</a>\");\n  /* Cases à cocher (« - [x] fait ») : affichées comme telles dans les messages. */\n  s = s.replace(/^\\[ \\]\\s+/, \"\\u2610 \");\n  s = s.replace(/^\\[([xX])\\]\\s+/, \"\\u2611 \");\n  s = s.replace(/&lt;(https?:[/][/][^\\s&]+)&gt;/g,\n        (m,u)=>'<a href=\"'+u+'\" target=\"_blank\" rel=\"noopener noreferrer\">'+u+\"</a>\");\n  s = s.replace(/(^|[\\s(])((?:https?:[/][/])[^\\s<)]+[^\\s<).,;:!?])/g,\n        (m,pre,u)=>pre+'<a href=\"'+u+'\" target=\"_blank\" rel=\"noopener noreferrer\">'+u+\"</a>\");\n  s = s.replace(/\\*\\*\\*([^*]+)\\*\\*\\*/g,\"<strong><em>$1</em></strong>\");\n  s = s.replace(/\\*\\*([\\s\\S]+?)\\*\\*/g,\"<strong>$1</strong>\");\n  s = s.replace(/__([\\s\\S]+?)__/g,\"<strong>$1</strong>\");\n  s = s.replace(/(^|[^\\w*])\\*([^*\\n]+)\\*/g,\"$1<em>$2</em>\");\n  s = s.replace(/(^|[^\\w_])_([^_\\n]+)_(?![\\w])/g,\"$1<em>$2</em>\");\n  s = s.replace(/~~([\\s\\S]+?)~~/g,\"<del>$1</del>\");\n  s = s.replace(/ {2,}\\n/g,\"<br>\\n\").replace(/\\\\\\n/g,\"<br>\\n\");\n  return s;\n}\n\nfunction indentOf(s){ return (s||\"\").replace(/\\t/g,\"  \").length; }\n\nfunction parseList(lines, start){\n  const reItem=/^(\\s*)(\\u2022|[-*+]|\\d+[.)])\\s+(.*)$/;\n  const first=lines[start].match(reItem);\n  if(!first) return [start, \"\"];\n  const baseIndent=indentOf(first[1]);\n  const ordered=/\\d/.test(first[2]);\n  first[2]=first[2].replace(\"\\u2022\",\"-\");      /* la puce « • » est une liste non ordonnée */\n  const items=[];\n  let i=start;\n\n  const deeperThan = j => j<lines.length && indentOf((lines[j].match(/^(\\s*)/)||[\"\",\"\"])[1])>baseIndent;\n\n  while(i<lines.length){\n    const ln=lines[i];\n\n    if(!ln.trim()){\n      let j=i; while(j<lines.length && !lines[j].trim()) j++;\n      if(j<lines.length){\n        const m2=lines[j].match(reItem);\n        if((m2 && indentOf(m2[1])>=baseIndent) || (!m2 && deeperThan(j))){ i=j; continue; }\n      }\n      break;\n    }\n\n    const m=ln.match(reItem);\n    const ind=m ? indentOf(m[1]) : indentOf((ln.match(/^(\\s*)/)||[\"\",\"\"])[1]);\n\n    if(m && ind===baseIndent){ items.push({text:[m[3]], children:[]}); i++; continue; }\n    if(m && ind<baseIndent) break;\n\n    if(items.length && ind>baseIndent){\n      const sub=[];\n      while(i<lines.length){\n        if(!lines[i].trim()){\n          let j=i; while(j<lines.length && !lines[j].trim()) j++;\n          if(deeperThan(j)){ sub.push(\"\"); i=j; continue; }\n          break;\n        }\n        if(indentOf((lines[i].match(/^(\\s*)/)||[\"\",\"\"])[1])<=baseIndent) break;\n        sub.push(lines[i]); i++;\n      }\n      const filled=sub.filter(s=>s.trim());\n      if(filled.length){\n        const hasItem=filled.some(s=>reItem.test(s));\n        if(!hasItem){\n          /* Ligne(s) de continuation indentée(s) : simple repli à la ligne de\n             l'élément courant — les rattacher au lieu de les perdre. */\n          for(const s of filled) items[items.length-1].text.push(s.trim());\n        }else{\n          const strip=Math.min(...filled.map(s=>indentOf((s.match(/^(\\s*)/)||[\"\",\"\"])[1])));\n          const ded=sub.map(s=>s.trim()?s.slice(strip):\"\");\n          const firstFilled=ded.findIndex(s=>s.trim());\n          const [, nested]=parseList(ded, firstFilled);\n          if(nested) items[items.length-1].children.push(nested);\n        }\n      }\n      continue;\n    }\n\n    if(items.length){ items[items.length-1].text.push(ln.trim()); i++; continue; }\n    break;\n  }\n\n  const tag=ordered?\"ol\":\"ul\";\n  const body=items.map(it=>\"<li>\"+inline(it.text.join(\"\\n\"))+it.children.join(\"\")+\"</li>\").join(\"\");\n  return [i, \"<\"+tag+\">\"+body+\"</\"+tag+\">\"];\n}\n\nfunction zoneHTML(idx, lang){\n  const runnable = RUNNABLE.has(lang);\n  return '<div class=\"zone\" data-zone=\"'+idx+'\">'\n    + '<div class=\"zone-bar\"><span class=\"zone-lang\">'+esc(lang||\"code\")+\"</span>\"\n    + '<button class=\"zbtn copy\" type=\"button\">Copier</button>'\n    + (runnable ? '<button class=\"zbtn run\" type=\"button\">Exécuter</button>' : \"\")\n    + \"</div>\"\n    + '<div class=\"zone-body\"><pre><code>'+highlight(CODE_STORE[idx].code, lang)+\"</code></pre></div>\"\n    + \"</div>\";\n}\n</script>\n<script>\n/* =====================================================================\n   Application\n   ===================================================================== */\nconst $ = (s,r=document)=>r.querySelector(s);\nconst log=$(\"#log\"), promptEl=$(\"#prompt\"), sendBtn=$(\"#send\"),\n      modelBtn=$(\"#modelbtn\"), dd=$(\"#dd\"), hint=$(\"#hint\"),\n      attachBtn=$(\"#attach\"), imgMenu=$(\"#imgmenu\"),\n      menuBlack=$(\"#menublack\"), menuFile=$(\"#menufile\"),\n      filePick=$(\"#filepick\"), pending=$(\"#pending\"),\n      pendThumb=$(\"#pendthumb\"), pendName=$(\"#pendname\"), pendRm=$(\"#pendrm\"),\n      cfg=$(\"#cfg\"), cfgClose=$(\"#cfgclose\"), cfgTitle=$(\"#cfgtitle\"),\n      cfgBody=$(\"#cfgbody\"), cfgReset=$(\"#cfgreset\"),\n      clearBtn=$(\"#clearbtn\"), fs=$(\"#fs\"), fsFrame=$(\"#fsframe\"), fsClose=$(\"#fsclose\");\n\nlet current = MODELS[0];\nlet useBlackImage = false;\nlet pendingImage = null;      /* {name, b64, kb} */\nlet pendingAudio = null;      /* {name, b64, kb, mime} */\nlet busy = false;\nlet conv = [];              /* mémoire de la conversation courante */\nlet fsOwner = null;         /* bouton ayant ouvert le plein écran */\n\n/* --- sélecteur de modèle (nom en toutes lettres, menu déroulant) ---- */\nconst VENDORS = [\n  [\"anthropic\",\"Anthropic\"],[\"openai\",\"OpenAI\"],[\"google\",\"Google\"],[\"groq\",\"Groq · API gratuite\"],\n  [\"kimi\",\"Moonshot · Kimi\"],[\"qwen\",\"Alibaba · Qwen\"],[\"glm\",\"Zhipu · GLM\"],\n  [\"deepseek\",\"DeepSeek\"],[\"mistral\",\"Mistral\"],[\"meta\",\"Meta\"],[\"xiaomi\",\"Xiaomi\"],[\"microsoft\",\"Microsoft\"],\n  [\"pollinations\",\"Pollinations\"],\n  [\"bfl\",\"Black Forest Labs\"],[\"ideogram\",\"Ideogram\"],\n  [\"modelscope\",\"ModelScope\"],\n  [\"ifm\",\"IFM\"],[\"arena\",\"Arena\"],\n];\nconst CHEV='<svg width=\"10\" height=\"10\" viewBox=\"0 0 24 24\" fill=\"none\" stroke=\"currentColor\" stroke-width=\"2.4\" stroke-linecap=\"round\" aria-hidden=\"true\"><path d=\"m5 9 7 7 7-7\"/></svg>';\nfunction renderModelBtn(){\n  modelBtn.innerHTML = MARKS[current.vendor]+'<span class=\"lbl\">'+esc(current.label)+'</span>'+CHEV;\n  modelBtn.setAttribute(\"aria-expanded\", dd.hidden?\"false\":\"true\");\n}\nfunction buildDD(){\n  dd.innerHTML=\"\";\n  for(const [vid,vlabel] of VENDORS){\n    const list=MODELS.filter(m=>m.vendor===vid && !m.hidden);\n    if(!list.length) continue;\n    const h=document.createElement(\"div\"); h.className=\"ddhead\";\n    h.innerHTML=MARKS[vid]+\"<span>\"+esc(vlabel)+\"</span>\";\n    dd.appendChild(h);\n    for(const m of list){\n      const b=document.createElement(\"button\");\n      b.type=\"button\"; b.className=\"dditem\"; b.setAttribute(\"role\",\"option\");\n      b.setAttribute(\"aria-selected\", m===current?\"true\":\"false\");\n      b.innerHTML=MARKS[m.vendor]+\"<span>\"+esc(m.label)+'</span><span class=\"mid\">'+esc(m.id)+\"</span>\";\n      b.onclick=()=>{ closeDD(); if(!busy) selectModel(m); };\n      dd.appendChild(b);\n    }\n  }\n}\nfunction openDD(){ buildDD(); dd.hidden=false; renderModelBtn(); }\nfunction closeDD(){ if(dd.hidden) return; dd.hidden=true; renderModelBtn(); }\nlet lpTimer=null, lpFired=false;\nmodelBtn.addEventListener(\"pointerdown\",()=>{\n  lpFired=false;\n  lpTimer=setTimeout(()=>{ lpFired=true; closeDD(); openSettings(current); },450);\n});\n[\"pointerup\",\"pointercancel\",\"pointerleave\"].forEach(ev=>\n  modelBtn.addEventListener(ev,()=>{ if(lpTimer) clearTimeout(lpTimer); }));\nmodelBtn.onclick=()=>{\n  if(lpFired){ lpFired=false; return; }\n  if(dd.hidden) openDD(); else closeDD();\n};\ndocument.addEventListener(\"pointerdown\",(e)=>{\n  if(!dd.hidden && !dd.contains(e.target) && !modelBtn.contains(e.target)) closeDD();\n});\ndocument.addEventListener(\"keydown\",(e)=>{ if(e.key===\"Escape\" && !dd.hidden) closeDD(); });\nfunction selectModel(m){\n  current=m;\n  renderModelBtn();\n  if(!dd.hidden) buildDD();\n  updateHint();\n  const imgGen = m.gen===\"img\";\n  const vidGen = m.gen===\"vid\";\n  /* Les modèles qui savent ÉDITER gardent le trombone : l'image jointe part\n     comme image de référence (une seule, jamais l'historique). Les autres\n     générateurs n'en ont pas besoin. */\n  const canEdit = !!(m.edit || m.id===\"po-gpt-image-2\");\n  attachBtn.style.display = (imgGen && !canEdit) ? \"none\" : \"\";\n  imgMenu.hidden = true;\n  promptEl.placeholder = vidGen ? \"Décris la scène à animer…\"\n    : (imgGen ? (canEdit ? \"Décris l’image, ou joins-en une à modifier…\" : \"Décris l’image à générer…\")\n              : \"Demandez à l’IA…\");\n}\nrenderModelBtn();\n\n/* Nouvelle discussion : la discussion en cours est déjà enregistrée au fil de\n   l'eau, on repart donc sur une vierge. Fonctionne dans les deux modes (chat et\n   agent) : seule une réponse en cours l'empêche. */\nfunction newConversation(){\n  if(busy){ alert(\"Attends la fin de la réponse en cours.\"); return; }\n  conv=[]; log.innerHTML=\"\"; emptyState(); updateHint();\n  if(typeof FB!==\"undefined\"&&FB.onNewChat) FB.onNewChat();\n  if(typeof agtNewConversation===\"function\") agtNewConversation();\n}\nclearBtn.onclick=newConversation;\n\n/* --- image noire ---------------------------------------------------- */\n/* PNG 640x640 noir pur, pré-calculé : sert de repli si canvas est indisponible. */\nconst BLACK_PNG_B64 = \"iVBORw0KGgoAAAANSUhEUgAAAoAAAAKACAIAAACDr150AAAEvklEQVR42u3BMQEAAADCoPVPbQlPoAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAHgYw44AAX2agdsAAAAASUVORK5CYII=\";\nlet blackCache = null;\nfunction blackImageB64(){\n  if(blackCache) return blackCache;\n  try{\n    const c=document.createElement(\"canvas\"); c.width=640; c.height=640;\n    const x=c.getContext(\"2d\");\n    if(!x) throw new Error(\"pas de contexte 2d\");\n    x.fillStyle=\"#000\"; x.fillRect(0,0,640,640);\n    const url=c.toDataURL(\"image/png\");\n    if(!url.startsWith(\"data:image/png;base64,\")) throw new Error(\"toDataURL invalide\");\n    blackCache=url.split(\",\")[1];\n  }catch(e){\n    blackCache=BLACK_PNG_B64;\n  }\n  return blackCache;\n}\nfunction syncBlackUI(){\n  attachBtn.classList.toggle(\"on\", useBlackImage);\n  attachBtn.setAttribute(\"aria-pressed\", useBlackImage?\"true\":\"false\");\n  menuBlack.setAttribute(\"aria-pressed\", useBlackImage?\"true\":\"false\");\n  updateHint();\n}\nattachBtn.onclick=()=>{\n  if(attachBtn.style.display===\"none\") return;\n  imgMenu.hidden=!imgMenu.hidden;\n  syncBlackUI();\n};\nmenuBlack.onclick=()=>{ useBlackImage=!useBlackImage; syncBlackUI(); imgMenu.hidden=true; };\nmenuFile.onclick=()=>{ imgMenu.hidden=true; filePick.click(); };\ndocument.addEventListener(\"click\", e=>{\n  if(!imgMenu.hidden && !imgMenu.contains(e.target) && !attachBtn.contains(e.target)) imgMenu.hidden=true;\n});\n\nfunction updateHint(){\n  /* Aucune aide textuelle au-dessus de la barre de saisie : la vignette de\n     pièce jointe suffit, et ces lignes mangeaient de la place utile. */\n  hint.textContent = \"\";\n}\nupdateHint();\n\n/* --- état vide ------------------------------------------------------ */\nfunction emptyState(){\n  if(log.children.length) return;\n  const d=document.createElement(\"div\"); d.className=\"empty\";\n  if(current.family===\"arena\"){\n    d.innerHTML=\"<p>Canal direct avec Arena.</p><p style=\\\"font-size:12.5px\\\">\"\n      +\"Écris ton message : il part aussitôt dans le canal. La réponse s'affiche ici, \"\n      +\"et les commandes exécutées apparaissent dans la zone terminale.</p>\";\n  }else{\n    d.innerHTML=\"<p>Aucun message.</p><p style=\\\"font-size:12.5px\\\">\"+esc(current.label)+\" · aucun prompt système · usage classique</p>\";\n  }\n  log.appendChild(d);\n}\nemptyState();\n\n/* --- mémoire : prompt construit à partir de l'historique ------------ */\nfunction buildPrompt(history, text){\n  if(!history.length) return text;\n  const parts = history.map(m => (m.role===\"user\" ? \"Utilisateur : \" : \"Assistant : \") + m.text);\n  parts.push(\"Utilisateur : \" + text);\n  return parts.join(\"\\n\\n\");\n}\n/* L'image jointe au message courant doit être LA référence du modèle :\n   sans cet ancrage, il répond à partir des descriptions d'anciennes images\n   présentes dans l'historique (bug « il ne voit pas la nouvelle image »). */\nfunction anchorImage(prompt, kind, snap){\n  if(kind!==\"upload\" || !snap) return prompt;\n  return prompt+\"\\n\\n[NB : l'image jointe à CE message (« \"+snap.name+\" ») est la seule à prendre en compte pour cette demande. Les images décrites ou générées plus haut dans la conversation sont d'AUTRES images — ne les confonds pas avec celle-ci.]\";\n}\n/* Images réellement envoyées au modèle : les 3 dernières de la conversation\n   (générées ET jointes) — c'est la mémoire visuelle de l'IA. Sans ça, le\n   modèle « oublie » l'image dès le message suivant. Les versions réduites\n   (small) sont calculées à la volée et mises en cache. */\n/* Transfert FIDÈLE : on envoie les octets d'origine, pas la miniature. La\n   miniature ne sert que de secours quand l'original dépasserait la limite\n   du relais (le corps de /api/chat est plafonné à 6 Mo). */\nconst SEND_IMG_MAX = 2900000;      /* ~2,2 Mo une fois décodé */\nconst SEND_ONE_MAX = 4800000;\nfunction sendB64(snap, max){\n  if(!snap) return \"\";\n  const b=String(snap.b64||\"\");\n  if(b && b.length <= (max||SEND_IMG_MAX)) return b;\n  return String(snap.small||b);\n}\nasync function collectSendImages(kind, snapshot){\n  const out=[]; const seen=new Set();\n  /* Budget global : le corps d'une requête /api/chat est plafonné à 6 Mo,\n     on garde une marge. Au-delà, la miniature prend le relais. */\n  let budget = 5200000;\n  const push=(name,b64,mime)=>{\n    let v=String(b64||\"\");\n    if(!v||seen.has(v)) return;\n    if(v.length > budget) return;\n    budget -= v.length;\n    seen.add(v);\n    out.push({name:name||\"image\", b64:v, mime:mime||\"image/png\"});\n  };\n  for(const m of conv){\n    if(m.images) for(const im of m.images) push(im.name, sendB64(im), im.mime);\n  }\n  if(typeof pendingFiles!==\"undefined\") for(const f of pendingFiles){\n    if(f.kind===\"image\") push(f.name, sendB64(f), f.mime||\"image/png\");\n  }\n  if(kind===\"upload\" && snapshot) push(snapshot.name, sendB64(snapshot), snapshot.mime||\"image/png\");\n  if(!out.length) return undefined;\n  return out.slice(-3);\n}\n/* Notes des fichiers non-image joints (nom + contenu texte si lisible). */\nfunction collectFileNotes(){\n  if(typeof pendingFiles===\"undefined\" || !pendingFiles.length) return \"\";\n  let s=\"\";\n  for(const f of pendingFiles){\n    if(f.kind!==\"file\") continue;\n    s+=\"\\n\\n[fichier joint : \"+f.name+\"]\"+(f.text?\"\\n```\\n\"+f.text+\"\\n```\":\"\");\n  }\n  return s;\n}\n/* Marqueur d'historique pour les demandes avec pièce jointe. */\nfunction markKind(text, kind, snap, asnap){\n  if(kind===\"upload\" && snap) return text+\" [image jointe : \"+snap.name+\"]\";\n  if(kind===\"audio\" && asnap) return text+\" [audio joint : \"+asnap.name+\"]\";\n  if(kind===\"black\") return text+\" [image noire jointe]\";\n  return text;\n}\n\n/* --- envoi ---------------------------------------------------------- */\n/* --- barre de saisie : le markdown s'écrit et se met en forme sur place ----\n   La barre est un champ éditable : les marqueurs restent visibles et\n   modifiables (effacer une étoile enlève le gras), mais le texte entouré\n   s'affiche aussitôt en gras, en italique, en code… Le texte du champ EST la\n   source markdown — aucun caractère n'est ajouté ni retiré au rendu — donc le\n   curseur garde exactement sa place pendant la frappe. */\n/* --- PC ou mobile ------------------------------------------------------\n   La mise en forme pendant la frappe n'existe que sur PC : sur mobile, les\n   claviers rapides (composition, suggestions) perdent des caractères quand le\n   champ est réécrit à chaque touche. Sur mobile la barre est donc un simple\n   champ d'écriture : le message s'affiche en markdown une fois envoyé. */\nconst IS_MOBILE=(function(){\n  try{\n    const ua=navigator.userAgent||\"\";\n    if(/Android|iPhone|iPad|iPod|Windows Phone|Mobile|Silk|Kindle/i.test(ua)) return true;\n    if(navigator.maxTouchPoints>1 && Math.min(screen.width||9999, screen.height||9999)<900) return true;\n  }catch(e){}\n  return false;\n})();\nconst MD_RICH=!IS_MOBILE;\nconst MD_PAIR={ \"*\":\"**\", \"_\":\"__\", \"~\":\"~~\", \"`\":\"`\" };\nconst MD_BULLET=String.fromCharCode(8226)+\" \";      /* « • » + espace */\nlet mdClosedAt=-1, mdClosedCh=\"\";\nfunction cEsc(t){ return String(t).replace(/&/g,\"&amp;\").replace(/</g,\"&lt;\").replace(/>/g,\"&gt;\"); }\nfunction cInline(line){\n  /* Les morceaux déjà mis en forme sont mis de côté (jetons \\u0001n\\u0001) :\n     sans cela, les étoiles réintroduites par le gras seraient relues par la\n     règle de l'italique et abîmeraient le rendu pendant la frappe. */\n  const tok=[];\n  const keep=(html)=>{ tok.push(html); return \"\\u0001\"+(tok.length-1)+\"\\u0001\"; };\n  let h=cEsc(line);\n  h=h.replace(/`([^`\\n]+)`/g, (m,x)=>keep('<span class=\"mdm\">`</span><code>'+x+'</code><span class=\"mdm\">`</span>'));\n  h=h.replace(/\\*\\*([^*\\n]+)\\*\\*/g, (m,x)=>keep('<span class=\"mdm\">**</span><strong>'+x+'</strong><span class=\"mdm\">**</span>'));\n  h=h.replace(/__([^_\\n]+)__/g, (m,x)=>keep('<span class=\"mdm\">__</span><strong>'+x+'</strong><span class=\"mdm\">__</span>'));\n  h=h.replace(/~~([^~\\n]+)~~/g, (m,x)=>keep('<span class=\"mdm\">~~</span><s>'+x+'</s><span class=\"mdm\">~~</span>'));\n  h=h.replace(/\\*([^*\\n]+)\\*/g, (m,x)=>keep('<span class=\"mdm\">*</span><em>'+x+'</em><span class=\"mdm\">*</span>'));\n  return h.replace(/\\u0001(\\d+)\\u0001/g, (m,n)=>tok[+n]);\n}\nfunction cRender(text){\n  const src=String(text);\n  /* Les listes sont décalées comme dans un vrai markdown : le signe reste dans\n     la marge, le texte part à droite, et les lignes suivantes s'alignent. */\n  const body=src.split(\"\\n\").map(ln=>{\n    let m;\n    if((m=/^(\\s*)(#{1,6})( +)(\\S.*)$/.exec(ln)))\n      return cEsc(m[1])+'<span class=\"mdm\">'+m[2]+'</span>'+m[3]+'<strong class=\"mdh\">'+cInline(m[4])+'</strong>';\n    if((m=/^(\\s*)(>)( ?)(.*)$/.exec(ln)))\n      return cEsc(m[1])+'<span class=\"mdm\">'+m[2]+'</span>'+m[3]+'<span class=\"mdq\">'+cInline(m[4])+'</span>';\n    /* Case à cocher : « - [ ] », mais aussi « • [x] » puisque la puce est\n       posée automatiquement après « - » + espace. Les crochets RESTENT dans le\n       texte (la barre doit contenir exactement ce qui sera envoyé) : seule la\n       couleur change, comme pour les autres signes. */\n    if((m=/^(\\s*)([-*+]|\\u2022)( +)\\[([ xX])\\]( +)(.*)$/.exec(ln)))\n      return '<span class=\"mdli\">'+cEsc(m[1])+'<span class=\"mdm\">'+m[2]+m[3]+'['+m[4]+']'+m[5]\n        +'</span>'+cInline(m[6])+'</span>';\n    if((m=/^(\\s*)(\\d+[.)])( +)(.*)$/.exec(ln)))\n      return '<span class=\"mdli\">'+cEsc(m[1])+'<span class=\"mdm\">'+m[2]+'</span>'+m[3]+cInline(m[4])+'</span>';\n    if((m=/^(\\s*)([-*+]|\\u2022)( +)(.*)$/.exec(ln)))\n      return '<span class=\"mdli\">'+cEsc(m[1])+'<span class=\"mdm\">'+m[2]+'</span>'+m[3]+cInline(m[4])+'</span>';\n    return cInline(ln);\n  }).join(\"\\n\");\n  /* Une fin de texte qui se termine par un retour à la ligne doit se VOIR :\n     sans repère, le navigateur n'affiche pas la dernière ligne vide (d'où\n     l'impression qu'Entrée ne marchait pas du premier coup). */\n  return /\\n$/.test(src) ? body+'<span class=\"mdz\">\\u200B</span>' : body;\n}\n/* Le texte du champ est la source : on la lit, on la rend, on replace le curseur. */\nfunction cText(){ return String(promptEl.textContent||\"\").replace(/\\u200B/g,\"\"); }\n/* Les nœuds de repère (fin de ligne visible) ne comptent pas dans le texte. */\nfunction cTextNodes(){\n  const out=[];\n  const w=document.createTreeWalker(promptEl, NodeFilter.SHOW_TEXT, null);\n  let t;\n  while((t=w.nextNode())){ if(t.nodeValue && t.nodeValue.indexOf(\"\\u200B\")<0) out.push(t); }\n  return out;\n}\nfunction cOffsetOf(node,off){\n  let n=0;\n  for(const t of cTextNodes()){ if(t===node) return n+off; n+=t.nodeValue.length; }\n  return n;\n}\nfunction cOffsets(){\n  const len=cText().length;\n  try{\n    const s=window.getSelection();\n    if(!s || !s.rangeCount) return {start:len,end:len,ok:false};\n    const r=s.getRangeAt(0);\n    if(!promptEl.contains(r.startContainer)) return {start:len,end:len,ok:false};\n    return {start:cOffsetOf(r.startContainer,r.startOffset), end:cOffsetOf(r.endContainer,r.endOffset), ok:true};\n  }catch(e){ return {start:len,end:len,ok:false}; }\n}\nfunction cPlace(start,end){\n  try{\n    const r=document.createRange();\n    let n=0, sN=null, sO=0, eN=null, eO=0;\n    for(const t of cTextNodes()){\n      const len=t.nodeValue.length;\n      if(sN===null && start<=n+len){ sN=t; sO=Math.max(0,start-n); }\n      if(sN!==null && end<=n+len){ eN=t; eO=Math.max(0,end-n); break; }\n      n+=len;\n    }\n    if(sN===null){ sN=promptEl; sO=promptEl.childNodes.length; eN=sN; eO=sO; }\n    r.setStart(sN,sO); r.setEnd(eN||sN, eN?eO:sO);\n    const s=window.getSelection(); s.removeAllRanges(); s.addRange(r);\n  }catch(e){}\n}\nlet cComposing=false;          /* composition IME en cours (clavier mobile/accents) */\nfunction cPaint(text, start, end){\n  const t=(text===undefined)?cText():String(text);\n  if(MD_RICH) promptEl.innerHTML=cRender(t);      /* PC : mise en forme visible */\n  else promptEl.textContent=t;                    /* mobile : texte brut */\n\n  promptEl.classList.toggle(\"empty\", t.length===0);\n  promptEl.style.height=\"auto\";\n  promptEl.style.height=Math.min(promptEl.scrollHeight,200)+\"px\";\n  if(start!==undefined) cPlace(start, end===undefined?start:end);\n  sendBtn.disabled=busy || !t.trim();\n}\n/* Compatibilité : le reste de l'app parle toujours de « value » et de\n   « placeholder » — la barre les expose donc comme avant. */\ntry{ Object.defineProperty(promptEl,\"value\",{\n  get:()=>cText(), set:(v)=>cPaint(v==null?\"\":String(v)) }); }catch(e){}\ntry{ Object.defineProperty(promptEl,\"placeholder\",{\n  get:()=>promptEl.getAttribute(\"data-ph\")||\"\",\n  set:(v)=>promptEl.setAttribute(\"data-ph\",String(v==null?\"\":v)) }); }catch(e){}\ntry{ Object.defineProperty(promptEl,\"selectionStart\",{ get:()=>cOffsets().start }); }catch(e){}\ntry{ Object.defineProperty(promptEl,\"selectionEnd\",{ get:()=>cOffsets().end }); }catch(e){}\npromptEl.setSelectionRange=(a,b)=>{ cPlace(a==null?0:a, b==null?(a==null?0:a):b); };\n\nfunction mdCount(line,p){ let n=0, idx=0; while((idx=line.indexOf(p,idx))>=0){ n++; idx+=p.length; } return n; }\n/* Un signe tapé ferme-t-il une paire ouverte sur la même ligne ? */\nfunction mdClosePair(ch){\n  const v=cText(), sel=cOffsets();\n  if(sel.start!==sel.end) return false;\n  const s=sel.start;\n  const ls=v.lastIndexOf(\"\\n\", s-1)+1, line=v.slice(ls,s);\n  const forms = MD_PAIR[ch].length===2 ? [MD_PAIR[ch], ch] : [MD_PAIR[ch]];\n  for(const p of forms){\n    const idx=line.lastIndexOf(p);\n    if(idx<0) continue;\n    if(mdCount(line,p)%2===0) continue;               /* cette paire est déjà refermée */\n    const inner=line.slice(idx+p.length);\n    if(!inner || /^\\s|\\s$/.test(inner) || inner.indexOf(ch)>=0) continue;\n    const at=s+p.length;\n    cPaint(v.slice(0,s)+p+v.slice(s), at, at);\n    mdClosedAt=at; mdClosedCh=ch;                     /* la frappe suivante du même signe est absorbée */\n    return true;\n  }\n  return false;\n}\nfunction mdEmptyPairDel(){\n  const v=cText(), sel=cOffsets();\n  if(sel.start!==sel.end) return false;\n  for(const p of [\"**\",\"__\",\"~~\",\"``\"]){\n    if(v.slice(sel.start-p.length,sel.start)===p && v.slice(sel.start,sel.start+p.length)===p){\n      cPaint(v.slice(0,sel.start-p.length)+v.slice(sel.start+p.length), sel.start-p.length, sel.start-p.length);\n      return true;\n    }\n  }\n  return false;\n}\npromptEl.addEventListener(\"input\", ()=>{\n  /* On redimensionne et on active le bouton ; sur mobile on ne touche PAS au\n     contenu du champ (le navigateur garde la main : aucune touche perdue). */\n  const t=cText();\n  promptEl.classList.toggle(\"empty\", t.length===0);\n  promptEl.style.height=\"auto\";\n  promptEl.style.height=Math.min(promptEl.scrollHeight,200)+\"px\";\n  sendBtn.disabled=busy || !t.trim();\n  if(!MD_RICH || cComposing) return;\n  /* PC : la frappe est passée par le navigateur, on rétablit la mise en forme\n     et on remet le curseur exactement là où il était. */\n  const sel=cOffsets();\n  cPaint(t, sel.start, sel.end);\n});\npromptEl.addEventListener(\"compositionstart\", ()=>{ cComposing=true; });\npromptEl.addEventListener(\"compositionend\", ()=>{\n  cComposing=false;\n  if(!MD_RICH) return;\n  const sel=cOffsets();\n  cPaint(cText(), sel.start, sel.end);\n});\npromptEl.addEventListener(\"paste\", e=>{\n  if(!MD_RICH) return;                          /* mobile : collage natif, texte brut */\n  const cd=e.clipboardData||window.clipboardData;\n  if(!cd) return;\n  const t=cd.getData(\"text/plain\");\n  if(t==null) return;\n  e.preventDefault();\n  const v=cText(), sel=cOffsets();\n  const ins=String(t).replace(/\\r\\n?/g,\"\\n\");\n  const at=sel.start+ins.length;\n  cPaint(v.slice(0,sel.start)+ins+v.slice(sel.end), at, at);\n});\npromptEl.addEventListener(\"beforeinput\", e=>{\n  if(!MD_RICH) return;                          /* mobile : le navigateur insère la ligne */\n  if(e.inputType===\"insertParagraph\" || e.inputType===\"insertLineBreak\"){\n    e.preventDefault();\n    const v=cText(), sel=cOffsets();\n    const at=sel.start+1;\n    cPaint(v.slice(0,sel.start)+\"\\n\"+v.slice(sel.end), at, at);\n  }\n});\npromptEl.addEventListener(\"keydown\", e=>{\n  /* Envoi : Ctrl+Entrée (ou Cmd+Entrée), comme le bouton. */\n  if(e.key===\"Enter\" && (e.ctrlKey||e.metaKey)){ e.preventDefault(); submit(); return; }\n  /* Mobile : aucun raccourci markdown, la frappe est laissée au navigateur. */\n  if(!MD_RICH) return;\n  const ch=e.key;\n  /* La frappe qui suit une fermeture automatique est absorbée : « **gras** »\n     reste « **gras** » quand on tape les deux étoiles de fermeture. */\n  if(ch===mdClosedCh && mdClosedAt>=0){\n    const s=cOffsets();\n    if(s.start===mdClosedAt && s.end===mdClosedAt){ mdClosedAt=-1; mdClosedCh=\"\"; e.preventDefault(); return; }\n  }\n  if(ch && MD_PAIR[ch] && !e.ctrlKey && !e.metaKey && !e.altKey){\n    const v=cText(), sel=cOffsets();\n    const p=MD_PAIR[ch];\n    if(sel.end>sel.start){                        /* texte sélectionné : on l'entoure */\n      e.preventDefault();\n      const ins=p+v.slice(sel.start,sel.end)+p;\n      cPaint(v.slice(0,sel.start)+ins+v.slice(sel.end), sel.start+p.length, sel.start+p.length+(sel.end-sel.start));\n      return;\n    }\n    if(p.length===2 && mdClosePair(ch)){ e.preventDefault(); return; }\n    return;                                        /* sinon : le signe s'écrit normalement */\n  }\n  /* Liste : « - », « * », « + » ou « 1. » puis espace → puce. */\n  if((ch===\" \"||e.key===\"Spacebar\") && !e.ctrlKey && !e.metaKey && !e.altKey){\n    const v=cText(), sel=cOffsets();\n    if(sel.start===sel.end){\n      const ls=v.lastIndexOf(\"\\n\", sel.start-1)+1, line=v.slice(ls,sel.start);\n      const m=/^(\\s*)([-*+])$/.exec(line) || /^(\\s*)(\\d+)\\.$/.exec(line);\n      if(m){\n        e.preventDefault();\n        const bullet=/^\\d+$/.test(m[2]) ? (m[1]+m[2]+\". \") : (m[1]+MD_BULLET);\n        cPaint(v.slice(0,ls)+bullet+v.slice(sel.start), ls+bullet.length, ls+bullet.length);\n        return;\n      }\n    }\n  }\n  /* Entrée : nouvelle ligne, en poursuivant la liste en cours. */\n  if(e.key===\"Enter\" && !e.ctrlKey && !e.metaKey && !e.shiftKey && !e.altKey){\n    const v=cText(), sel=cOffsets();\n    if(sel.start===sel.end){\n      const ls=v.lastIndexOf(\"\\n\", sel.start-1)+1, line=v.slice(ls,sel.start);\n      const m=/^(\\s*)(\\d+)\\. |^(\\s*)\\u2022 /.exec(line);\n      if(m){\n        e.preventDefault();\n        const num=m[2], indent=(m[1]!==undefined)?m[1]:m[3];\n        if(line.length===m[0].length){                       /* ligne vide : fin de liste */\n          cPaint(v.slice(0,ls)+v.slice(sel.start), ls, ls);\n          return;\n        }\n        const next=num ? (indent+(Number(num)+1)+\". \") : (indent+MD_BULLET);\n        const ins=\"\\n\"+next;\n        cPaint(v.slice(0,sel.start)+ins+v.slice(sel.end), sel.start+ins.length, sel.start+ins.length);\n        return;\n      }\n      e.preventDefault();\n      cPaint(v.slice(0,sel.start)+\"\\n\"+v.slice(sel.end), sel.start+1, sel.start+1);\n      return;\n    }\n  }\n  if(e.key===\"Backspace\" && mdEmptyPairDel()){ e.preventDefault(); return; }\n  mdClosedAt=-1;\n});\nsendBtn.onclick=submit;\n\nfunction addUser(text, kind, snap, asnap){\n  const el=document.createElement(\"div\"); el.className=\"msg user\";\n  const ic='<svg width=\"11\" height=\"11\" viewBox=\"0 0 24 24\"><rect x=\"3\" y=\"3\" width=\"18\" height=\"18\" rx=\"3\" fill=\"none\" stroke=\"currentColor\" stroke-width=\"2\"/><rect x=\"7\" y=\"7\" width=\"10\" height=\"10\" fill=\"currentColor\"/></svg> ';\n  const badge = kind===\"upload\" ? ic+\"toi + image\" : kind===\"black\" ? ic+\"toi + image noire\" : kind===\"audio\" ? ic+\"toi + audio\" : \"toi\";\n  el.innerHTML='<div class=\"who\">'+badge+'</div><div class=\"bubble md\"></div>';\n  el.querySelector(\".bubble\").innerHTML = renderMarkdown(text);\n  /* Contrôle visuel : la pièce jointe réellement envoyée avec CE message est\n     affichée dedans, avec son nom — plus de doute sur l'image analysée. */\n  /* Toutes les pièces jointes au même niveau ; débordement → défilement\n     horizontal de cette zone uniquement. */\n  const files=(typeof pendingFiles!==\"undefined\"&&pendingFiles.length)\n    ? pendingFiles.slice()\n    : (kind===\"upload\"&&snap ? [{kind:\"image\",name:snap.name,b64:snap.b64}]\n      : kind===\"audio\"&&asnap ? [{kind:\"audio\",name:asnap.name}] : []);\n  if(files.length){\n    let h='';\n    for(const f of files){\n      if(f.kind===\"image\"&&f.b64){\n        const cid=\"c\"+(++IMG_SEQ);\n        IMG_STORE[cid]={b64:f.b64, mime:f.mime||\"image/png\", name:f.name};\n        h+='<span class=\"attach-chip\"><img class=\"gimg\" data-iid=\"'+cid+'\" src=\"data:image/png;base64,'+(f.small||f.b64)\n          +'\" alt=\"'+esc(f.name)+'\"><span class=\"nm\">'+esc(f.name)+'</span></span>';\n      }else{\n        h+='<span class=\"attach-chip\"><span class=\"nm\">'+(f.kind===\"audio\"?\"🎙 \":\"📎 \")+esc(f.name)+'</span></span>';\n      }\n    }\n    el.querySelector(\".bubble\").insertAdjacentHTML(\"beforeend\",'<div class=\"attach\">'+h+'</div>');\n  }\n  log.appendChild(el);\n  return el;\n}\nfunction addAssistant(modelId){\n  const m=(modelId&&MODELS.find(x=>x.id===modelId))||current;\n  const el=document.createElement(\"div\"); el.className=\"msg assistant\";\n  el.innerHTML='<div class=\"who\">'+MARKS[m.vendor]+esc(m.label)+'</div>'\n             + '<div class=\"bubble md\"><span class=\"gentxt\">Generating<i>.</i><i>.</i><i>.</i></span></div>';\n  log.appendChild(el);\n  return el.querySelector(\".bubble\");\n}\nfunction scrollDown(){ const m=document.querySelector(\"main\"); m.scrollTop=m.scrollHeight; }\n\nasync function submit(){\n  const text=promptEl.value.trim();\n  if(!text) return;\n  if(busy){\n    /* Une mission tourne encore ici : elle continue en arrière-plan (c'est la\n       passerelle qui la mène) et l'on peut écrire la suite tout de suite. */\n    if(AGT_JOB_OWN){ try{ await agtDetachNow(); }catch(e){} }\n    else return;\n  }\n  if(log.querySelector(\".empty\")) log.innerHTML=\"\";\n  sendBtn.classList.remove(\"sent\"); void sendBtn.offsetWidth; sendBtn.classList.add(\"sent\");\n\n  /* Le mode agent ne concerne QUE les modèles de texte. Les autres familles\n     gardent leur propre voie, même quand le bouton « Agent » est allumé :\n     sans cela, Arena et les générateurs d'images partaient dans une mission\n     et le worker répondait « modèle inconnu ». */\n  if(CUR_CANAL&&CUR_CANAL!==\"arena\"){ await submitCanal(text); return; }\n  if(current.family===\"arena\"){ await submitArena(text); return; }\n  if(current.gen===\"img\"){ await submitImage(text); return; }\n  if(current.gen===\"vid\"){ await submitVideo(text); return; }\n  if(current.gen===\"live\"){ await submitLive(text); return; }\n  if(typeof AGENT_ON!==\"undefined\"&&AGENT_ON&&!current.gen){ await agentRun(text); return; }\n  if(current.family===\"gtext\"||current.family===\"groq\"||current.family===\"ptext\"||current.family===\"ifm\"){ await submitStreaming(text); return; }\n\n  busy=true; sendBtn.disabled=true;\n  const kind = pendingAudio ? \"audio\" : (pendingImage ? \"upload\" : (useBlackImage ? \"black\" : \"none\"));\n  const snapshot = pendingImage;\n  const asnapshot = pendingAudio;\n  addUser(text, kind, snapshot, asnapshot);\n  const bubble=addAssistant();\n  promptEl.value=\"\"; promptEl.style.height=\"auto\";\n  scrollDown();\n\n  const spec={\n    model: current.id,\n    prompt: anchorImage(buildPrompt(conv, text), kind, snapshot)+collectFileNotes(),\n    image: kind===\"upload\" ? sendB64(snapshot, SEND_ONE_MAX) : (kind===\"black\" ? blackImageB64() : WHITE_PX),\n    image_kind: kind===\"audio\" ? \"none\" : kind,\n    audio: asnapshot ? asnapshot.b64 : undefined,\n    audio_mime: asnapshot ? asnapshot.mime : undefined,\n    images: await collectSendImages(kind, snapshot),\n    params: cleanParams(current.id),\n  };\n  try{\n    const r=await fetch(API_BASE+\"/api/chat\",{\n      method:\"POST\",\n      headers:{\"Content-Type\":\"application/json\",\"X-Access-Code\":ACCESS_CODE},\n      body:JSON.stringify(spec),\n    });\n    let j={}; try{ j=await r.json(); }catch(e){}\n    if(!r.ok) throw new Error(j.error || (\"HTTP \"+r.status+\" \"+r.statusText));\n    const out=(j.output==null?\"\":String(j.output));\n    const uEnt={role:\"user\",text:markKind(text, kind, snapshot, asnapshot)};\n    if(kind===\"upload\"&&snapshot) uEnt.images=[{name:snapshot.name, b64:snapshot.b64, mime:\"image/png\"}];\n    conv.push(uEnt);\n    if(typeof FB!==\"undefined\"&&FB.afterText) FB.afterText({userText:text, kind, snap:kind===\"upload\"?snapshot:null, asnap:kind===\"audio\"?asnapshot:null, out});\n    if(j.audio){\n      conv.push({role:\"assistant\",text:out||\"(réponse audio)\"});\n      bubble.innerHTML = (out.trim()?renderMarkdown(out):\"\")\n        + renderAudioPlayer(j.audio, j.mime||\"audio/wav\", \"reponse-vocale.wav\");\n      msgActions(bubble,{getMd:()=>out, regen:(b)=>regenREST(spec,b)});\n    }else if(j.embedding){\n      conv.push({role:\"assistant\",text:\"(embedding \"+j.embedding.dim+\" dimensions)\"});\n      bubble.innerHTML = '<p><b>Embedding · '+j.embedding.dim+' dimensions</b> · norme '+j.embedding.norm+'</p>'\n        +'<pre style=\"white-space:pre-wrap;font-size:11px;padding:10px;border-radius:10px;background:#f6f6f4;color:#555\">['\n        +j.embedding.head.join(\", \")+\", …]</pre>\";\n      msgActions(bubble,{regen:(b)=>regenREST(spec,b)});\n    }else{\n      /* La passerelle a pu basculer d'elle-même sur la voie Roboflow\n         (crédits Pollinations épuisés) : on le signale discrètement. */\n      const note=j.via===\"roboflow\" ? \"roboflow\" : \"\";\n      conv.push({role:\"assistant\",text:out});\n      bubble.innerHTML = (note?'<p class=\"viafb\"><em>'+esc(note)+'</em></p>':\"\")\n        + (out.trim() ? renderMarkdown(out) : '<p style=\"color:var(--ink-3)\"><em>(réponse vide)</em></p>');\n      msgActions(bubble,{getMd:()=>out, regen:(b)=>regenREST(spec,b)});\n    }\n  }catch(err){\n    const msg=String(err && err.message ? err.message : err);\n    const fbm=current.fb ? MODELS.find(x=>x.id===current.fb) : null;\n    if(fbm){\n      /* Bascule automatique sur le canal de repli (ex. Google API → Roboflow).\n         La plateforme de repli s'affiche en italique, en tête du message. */\n      try{\n        const r2=await fetch(API_BASE+\"/api/chat\",{\n          method:\"POST\",\n          headers:{\"Content-Type\":\"application/json\",\"X-Access-Code\":ACCESS_CODE},\n          body:JSON.stringify(Object.assign({}, spec, {model:fbm.id, params:cleanParams(fbm.id)})),\n        });\n        let j2={}; try{ j2=await r2.json(); }catch(e){}\n        if(!r2.ok) throw new Error(j2.error || (\"HTTP \"+r2.status+\" \"+r2.statusText));\n        const out2=(j2.output==null?\"\":String(j2.output));\n        const uEnt2={role:\"user\",text:markKind(text, kind, snapshot, asnapshot)};\n        if(kind===\"upload\"&&snapshot) uEnt2.images=[{name:snapshot.name, b64:snapshot.b64, mime:\"image/png\"}];\n        conv.push(uEnt2);\n        conv.push({role:\"assistant\",text:out2});\n        if(typeof FB!==\"undefined\"&&FB.afterText) FB.afterText({userText:text, kind, snap:kind===\"upload\"?snapshot:null, asnap:kind===\"audio\"?asnapshot:null, out:out2, via:current.fbTxt||null});\n        const pre=current.fbTxt ? '<p class=\"viafb\"><em>'+esc(current.fbTxt)+'</em></p>' : \"\";\n        bubble.innerHTML=pre+(out2.trim()?renderMarkdown(out2):'<p style=\"color:var(--ink-3)\"><em>(réponse vide)</em></p>');\n        const spec2=Object.assign({}, spec, {model:fbm.id, params:cleanParams(fbm.id)});\n        const md2=out2;\n        msgActions(bubble,{getMd:()=>md2, regen:(b)=>regenREST(spec2,b)});\n      }catch(err2){\n        bubble.innerHTML='<div class=\"err\">'+esc(msg)+\"<hr>\"+esc(String(err2 && err2.message ? err2.message : err2))+'</div>';\n        msgActions(bubble,{regen:(b)=>regenREST(spec,b)});\n      }\n    }else{\n      const extra=/modèle inconnu/.test(msg)\n        ? \" — ce modèle existe côté interface, mais le Worker en ligne (5 modèles) doit être redéployé avec un nouveau token Cloudflare pour le reconnaître.\"\n        : \"\";\n      const isNet = err instanceof TypeError;\n      bubble.innerHTML='<div class=\"err\">'+esc(isNet\n        ? \"Impossible de joindre le Worker relais (\"+API_BASE+\") — \"+msg\n        : msg+extra)+\"</div>\";\n      msgActions(bubble,{regen:(b)=>regenREST(spec,b)});\n    }\n  }finally{\n    clearAttachments();\n    busy=false; sendBtn.disabled=!promptEl.value.trim(); scrollDown();\n  }\n}\n\n/* --- « Recommencer » : rejoue la même requête dans la même bulle ------- */\nasync function regenREST(spec, b){\n  busy=true; sendBtn.disabled=true;\n  b.innerHTML='<span class=\"gentxt\">Generating<i>.</i><i>.</i><i>.</i></span>';\n  try{\n    const r=await fetch(API_BASE+\"/api/chat\",{\n      method:\"POST\",\n      headers:{\"Content-Type\":\"application/json\",\"X-Access-Code\":ACCESS_CODE},\n      body:JSON.stringify(spec),\n    });\n    let j={}; try{ j=await r.json(); }catch(e){}\n    if(!r.ok) throw new Error(j.error || (\"HTTP \"+r.status+\" \"+r.statusText));\n    const out=(j.output==null?\"\":String(j.output));\n    b.innerHTML=out.trim()?renderMarkdown(out):'<p style=\"color:var(--ink-3)\"><em>(réponse vide)</em></p>';\n    const md=out;\n    msgActions(b,{getMd:()=>md, regen:(bb)=>regenREST(spec,bb)});\n  }catch(err){\n    b.innerHTML='<div class=\"err\">'+esc(String(err && err.message ? err.message : err))+'</div>';\n    msgActions(b,{regen:(bb)=>regenREST(spec,bb)});\n  }finally{\n    busy=false; sendBtn.disabled=!promptEl.value.trim(); scrollDown();\n  }\n}\n\n/* --- streaming SSE (modèles texte Google API, Groq et Pollinations) --- */\nasync function submitStreaming(text){\n  busy=true; sendBtn.disabled=true;\n  const kind = pendingAudio ? \"audio\" : (pendingImage ? \"upload\" : (useBlackImage ? \"black\" : \"none\"));\n  const snapshot = pendingImage;\n  const asnapshot = pendingAudio;\n  addUser(text, kind, snapshot, asnapshot);\n  const bubble=addAssistant();\n  promptEl.value=\"\"; promptEl.style.height=\"auto\"; scrollDown();\n  const spec={\n    model: current.id,\n    prompt: anchorImage(buildPrompt(conv, text), kind, snapshot)+collectFileNotes(),\n    image: kind===\"upload\" ? sendB64(snapshot, SEND_ONE_MAX) : (kind===\"black\" ? blackImageB64() : WHITE_PX),\n    image_kind: kind===\"audio\" ? \"none\" : kind,\n    audio: asnapshot ? asnapshot.b64 : undefined,\n    audio_mime: asnapshot ? asnapshot.mime : undefined,\n    images: await collectSendImages(kind, snapshot),\n    params: cleanParams(current.id),\n  };\n  const fbm = current.fb ? MODELS.find(x=>x.id===current.fb) : null;\n  const STREAM_FAMS=[\"gtext\",\"groq\",\"ptext\",\"ifm\"];\n  /* Anti-gel : au-delà de 20 000 caractères (longs SVG…), re-rendre le\n     markdown à chaque trame sature le navigateur → on affiche le texte brut\n     (textContent, quasi gratuit) pendant le flux, et on ne passe au\n     markdown complet qu'UNE fois, à la fin. */\n  const HEAVY=20000;\n  const runStream=async(model, params)=>{\n    let acc=\"\"; let paintTimer=null;\n    const paint=()=>{\n      if(paintTimer!==null) return;\n      paintTimer=setTimeout(()=>{\n        paintTimer=null;\n        if(acc.length>HEAVY){ bubble.textContent=acc; bubble.style.whiteSpace=\"pre-wrap\"; }\n        else{ bubble.innerHTML=renderMarkdown(acc); }\n        scrollDown();\n      }, 200);\n    };\n    try{\n      const r=await fetch(API_BASE+\"/api/chat/stream\",{\n        method:\"POST\",\n        headers:{\"Content-Type\":\"application/json\",\"X-Access-Code\":ACCESS_CODE},\n        body:JSON.stringify(Object.assign({}, spec, {model, params})),\n      });\n      if(!r.ok || !r.body){\n        let j={}; try{ j=await r.json(); }catch(e){}\n        throw new Error(j.error || (\"HTTP \"+r.status+\" \"+r.statusText));\n      }\n      const reader=r.body.getReader();\n      const dec=new TextDecoder();\n      let buf=\"\";\n      while(true){\n        const {done,value}=await reader.read();\n        if(done) break;\n        buf+=dec.decode(value,{stream:true});\n        let idx;\n        while((idx=buf.indexOf(\"\\n\\n\"))>=0){\n          const ev=buf.slice(0,idx); buf=buf.slice(idx+2);\n          const line=ev.split(\"\\n\").find(l=>l.indexOf(\"data:\")===0);\n          if(!line) continue;\n          const p=line.slice(5).trim();\n          if(p===\"[DONE]\") continue;\n          try{\n            const j=JSON.parse(p);\n            if(j.error) throw new Error(j.error);\n            if(j.t){ acc+=j.t; paint(); }\n          }catch(e){ if(e instanceof SyntaxError) continue; throw e; }\n        }\n      }\n      if(paintTimer!==null){ clearTimeout(paintTimer); paintTimer=null; }\n      return acc;\n    }catch(e){\n      if(paintTimer!==null){ clearTimeout(paintTimer); paintTimer=null; }\n      throw e;\n    }\n  };\n  const runRest=async(model, params)=>{\n    const r=await fetch(API_BASE+\"/api/chat\",{\n      method:\"POST\",\n      headers:{\"Content-Type\":\"application/json\",\"X-Access-Code\":ACCESS_CODE},\n      body:JSON.stringify(Object.assign({}, spec, {model, params})),\n    });\n    let j={}; try{ j=await r.json(); }catch(e){}\n    if(!r.ok) throw new Error(j.error || (\"HTTP \"+r.status+\" \"+r.statusText));\n    return (j.output==null?\"\":String(j.output));\n  };\n  let acc=null, usedFb=false, specFinal=spec;\n  try{\n    try{\n      acc=await runStream(spec.model, spec.params);\n    }catch(err){\n      if(!fbm) throw err;\n      /* Bascule automatique sur le canal de repli (ex. Pollinations → Roboflow) */\n      usedFb=true;\n      bubble.style.whiteSpace=\"\";\n      bubble.innerHTML='<span class=\"gentxt\">Generating<i>.</i><i>.</i><i>.</i></span>';\n      const pfb=cleanParams(fbm.id);\n      specFinal=Object.assign({}, spec, {model:fbm.id, params:pfb});\n      acc = STREAM_FAMS.includes(fbm.family)\n        ? await runStream(fbm.id, pfb)\n        : await runRest(fbm.id, pfb);\n    }\n    const uEnt3={role:\"user\",text:markKind(text, kind, snapshot, asnapshot)};\n    if(kind===\"upload\"&&snapshot) uEnt3.images=[{name:snapshot.name, b64:snapshot.b64, mime:\"image/png\"}];\n    conv.push(uEnt3);\n    conv.push({role:\"assistant\",text:acc});\n    if(typeof FB!==\"undefined\"&&FB.afterText) FB.afterText({userText:text, kind, snap:kind===\"upload\"?snapshot:null, asnap:kind===\"audio\"?asnapshot:null, out:acc, via:usedFb?(current.fbTxt||null):null});\n    bubble.style.whiteSpace=\"\";\n    const pre=(usedFb && current.fbTxt) ? '<p class=\"viafb\"><em>'+esc(current.fbTxt)+'</em></p>' : \"\";\n    bubble.innerHTML = pre+(acc.trim() ? renderMarkdown(acc) : '<p style=\"color:var(--ink-3)\"><em>(réponse vide)</em></p>');\n    const md=acc, sp=specFinal;\n    msgActions(bubble,{getMd:()=>md, regen:(b)=>regenStream(sp,b)});\n  }catch(err){\n    const msg=String(err && err.message ? err.message : err);\n    bubble.style.whiteSpace=\"\";\n    bubble.innerHTML='<div class=\"err\">'+esc(msg)+'</div>';\n    msgActions(bubble,{regen:(b)=>regenStream(spec,b)});\n  }finally{\n    clearAttachments();\n    busy=false; sendBtn.disabled=!promptEl.value.trim(); scrollDown();\n  }\n}\n/* « Recommencer » pour un message streamé : rejoue le flux dans la bulle. */\nasync function regenStream(spec, b){\n  busy=true; sendBtn.disabled=true;\n  b.style.whiteSpace=\"\";\n  b.innerHTML='<span class=\"gentxt\">Generating<i>.</i><i>.</i><i>.</i></span>';\n  let acc=\"\", accFb=\"\";\n  try{\n    const r=await fetch(API_BASE+\"/api/chat/stream\",{\n      method:\"POST\",\n      headers:{\"Content-Type\":\"application/json\",\"X-Access-Code\":ACCESS_CODE},\n      body:JSON.stringify(spec),\n    });\n    if(!r.ok || !r.body){\n      let j={}; try{ j=await r.json(); }catch(e){}\n      throw new Error(j.error || (\"HTTP \"+r.status+\" \"+r.statusText));\n    }\n    const reader=r.body.getReader();\n    const dec=new TextDecoder();\n    let buf=\"\";\n    while(true){\n      const {done,value}=await reader.read();\n      if(done) break;\n      buf+=dec.decode(value,{stream:true});\n      let idx;\n      while((idx=buf.indexOf(\"\\n\\n\"))>=0){\n        const ev=buf.slice(0,idx); buf=buf.slice(idx+2);\n        const line=ev.split(\"\\n\").find(l=>l.indexOf(\"data:\")===0);\n        if(!line) continue;\n        const p=line.slice(5).trim();\n        if(p===\"[DONE]\") continue;\n        try{\n          const j=JSON.parse(p);\n          if(j.error) throw new Error(j.error);\n          if(j.fb) accFb=j.fb;\n          if(j.t) acc+=j.t;\n        }catch(e){ if(e instanceof SyntaxError) continue; throw e; }\n      }\n    }\n    b.innerHTML=(accFb?'<p class=\"viafb\"><em>'+esc(accFb)+'</em></p>':\"\")\n      +(acc.trim()?renderMarkdown(acc):'<p style=\"color:var(--ink-3)\"><em>(réponse vide)</em></p>');\n    const md=(accFb?\"> \"+accFb+\"\\n\\n\":\"\")+acc;\n    msgActions(b,{getMd:()=>md, regen:(bb)=>regenStream(spec,bb)});\n  }catch(err){\n    b.innerHTML='<div class=\"err\">'+esc(String(err && err.message ? err.message : err))+(acc?\"<hr>\"+renderMarkdown(acc):\"\")+'</div>';\n    msgActions(b,{regen:(bb)=>regenStream(spec,bb)});\n  }finally{\n    busy=false; sendBtn.disabled=!promptEl.value.trim(); scrollDown();\n  }\n}\n\n/* --- sessions Live (WebSocket relayé par le Worker, API Gemini Live) -- */\nfunction pcmChunksToWav(chunks, rate){\n  let len=0; for(const c of chunks) len+=c.length;\n  const pcm=new Int16Array(len); let o=0;\n  for(const c of chunks){ pcm.set(c,o); o+=c.length; }\n  const buf=new ArrayBuffer(44+len*2); const dv=new DataView(buf);\n  const ws=(off,s)=>{ for(let i=0;i<s.length;i++) dv.setUint8(off+i, s.charCodeAt(i)); };\n  ws(0,\"RIFF\"); dv.setUint32(4,36+len*2,true); ws(8,\"WAVE\"); ws(12,\"fmt \");\n  dv.setUint32(16,16,true); dv.setUint16(20,1,true); dv.setUint16(22,1,true);\n  dv.setUint32(24,rate,true); dv.setUint32(28,rate*2,true); dv.setUint16(32,2,true); dv.setUint16(34,16,true);\n  ws(36,\"data\"); dv.setUint32(40,len*2,true);\n  const u8=new Uint8Array(buf); u8.set(new Uint8Array(pcm.buffer),44);\n  let s=\"\"; for(let i=0;i<u8.length;i+=32768) s+=String.fromCharCode.apply(null,u8.subarray(i,i+32768));\n  return btoa(s);\n}\nasync function pcmFromAudioBuffer(ab){\n  const AC=window.AudioContext||window.webkitAudioContext;\n  const ac=new AC();\n  const dec=await ac.decodeAudioData(ab.slice(0));\n  try{ ac.close(); }catch(e){}\n  const rate=16000, ch=dec.getChannelData(0), ch2=dec.numberOfChannels>1?dec.getChannelData(1):null;\n  const m=Math.max(1,Math.floor(dec.duration*rate));\n  const out=new Int16Array(m);\n  for(let i=0;i<m;i++){\n    const pos=i*dec.sampleRate/rate;\n    const i0=Math.min(Math.floor(pos),ch.length-1), i1=Math.min(i0+1,ch.length-1), f=pos-i0;\n    let v=ch[i0]+(ch[i1]-ch[i0])*f;\n    if(ch2){ const w=ch2[i0]+(ch2[i1]-ch2[i0])*f; v=(v+w)/2; }\n    out[i]=Math.max(-32768,Math.min(32767,Math.round(v*32767)));\n  }\n  return out;\n}\nfunction b64ToBytes(b64){\n  const bin=atob(b64); const arr=new Uint8Array(bin.length);\n  for(let i=0;i<bin.length;i++) arr[i]=bin.charCodeAt(i);\n  return arr;\n}\nasync function submitLive(text){\n  const fam=current.family;\n  if(fam===\"glivetr\" && !pendingAudio){\n    alert(\"Live Translate a besoin d'un fichier audio : joins-le avec le bouton fichier (icône 🎵), il sert de micro.\");\n    return;\n  }\n  busy=true; sendBtn.disabled=true;\n  const asnap=pendingAudio;\n  addUser(text, asnap?\"audio\":\"none\", null, asnap);\n  const bubble=addAssistant();\n  promptEl.value=\"\"; promptEl.style.height=\"auto\"; scrollDown();\n  const prm=cleanParams(current.id);\n  const proto=API_BASE.indexOf(\"https\")===0?\"wss://\":\"ws://\";\n  const ver=fam===\"glivetr\"?\"&v=v1alpha\":\"\";\n  const wsUrl=proto+API_BASE.replace(/^https?:[/][/]/,\"\")+\"/api/live?code=\"+ACCESS_CODE+ver;\n  const audioChunks=[]; let inTxt=\"\", outTxt=\"\", outTxtDirect=\"\", rate=24000, done=false;\n  const finish=(err)=>{\n    if(done) return; done=true;\n    clearTimeout(guard);\n    try{ ws.close(); }catch(e){}\n    busy=false; sendBtn.disabled=!promptEl.value.trim();\n    clearAttachments();\n    if(err){ bubble.innerHTML='<div class=\"err\">'+esc(err)+'</div>'; scrollDown(); return; }\n    let html=\"\";\n    if(fam===\"glivetr\"){\n      if(inTxt) html+='<p><b>Transcription :</b> '+esc(inTxt)+'</p>';\n      if(outTxt) html+='<p><b>Traduction ('+esc(String(prm.target_language||\"en\"))+') :</b> '+esc(outTxt)+'</p>';\n    }else if(outTxtDirect){\n      html+=renderMarkdown(outTxtDirect);\n    }\n    if(audioChunks.length){\n      html+=renderAudioPlayer(pcmChunksToWav(audioChunks,rate), \"audio/wav\",\n        fam===\"glivetr\" ? \"traduction.wav\" : \"reponse-live.wav\");\n    }\n    if(!html) html='<p style=\"color:var(--ink-3)\"><em>(session sans réponse)</em></p>';\n    bubble.innerHTML=html;\n    conv.push({role:\"user\",text:text});\n    conv.push({role:\"assistant\",text: fam===\"glivetr\" ? (outTxt||\"(audio traduit)\") : (outTxtDirect||\"(réponse vocale)\")});\n    scrollDown();\n  };\n  const ws=new WebSocket(wsUrl);\n  const guard=setTimeout(()=>finish(\"Session Live : délai dépassé (90 s).\"),90000);\n  ws.onopen=()=>{\n    if(fam===\"glivetr\"){\n      ws.send(JSON.stringify({setup:{model:\"models/gemini-3.5-live-translate-preview\",generationConfig:{\n        responseModalities:[\"AUDIO\"],\n        translationConfig:{targetLanguageCode:String(prm.target_language||\"en\"),echoTargetLanguage:!!prm.echo_target_language}}}}));\n    }else{\n      ws.send(JSON.stringify({setup:{model:\"models/\"+(prm.thinking?\"gemini-3.8-live-extended-thinking\":\"gemini-3.8-live\"),\n        generationConfig:{responseModalities:[\"AUDIO\"]}}}));\n    }\n  };\n  ws.onmessage=(e)=>{\n    let m; try{ m=JSON.parse(e.data); }catch(err){ return; }\n    if(m.setupComplete){ onStart(); return; }\n    const sc=m.serverContent;\n    if(!sc) return;\n    if(sc.inputTranscription&&sc.inputTranscription.text) inTxt+=sc.inputTranscription.text;\n    if(sc.outputTranscription&&sc.outputTranscription.text) outTxt+=sc.outputTranscription.text;\n    if(sc.modelTurn){\n      for(const p of (sc.modelTurn.parts||[])){\n        if(p.text) outTxtDirect+=p.text;\n        if(p.inlineData&&p.inlineData.data){\n          const mm=String(p.inlineData.mimeType||\"\"); const r=mm.match(/rate=(\\d+)/); if(r) rate=+r[1];\n          const arr=b64ToBytes(p.inlineData.data);\n          audioChunks.push(new Int16Array(arr.buffer.slice(0, arr.byteLength-(arr.byteLength%2))));\n        }\n      }\n    }\n    if(sc.turnComplete && fam!==\"glivetr\") finish();\n  };\n  ws.onerror=()=>finish(\"Connexion Live impossible (relais WebSocket du Worker).\");\n  ws.onclose=()=>{ if(!done) finish(fam===\"glivetr\"?undefined:\"Session Live fermée prématurément.\"); };\n  async function onStart(){\n    if(fam===\"glivetr\"){\n      try{\n        const pcm=await pcmFromAudioBuffer(b64ToBytes(asnap.b64).buffer);\n        const CH=3200;\n        for(let i=0;i<pcm.length && !done;i+=CH){\n          const part=pcm.subarray(i,i+CH);\n          const b=new Uint8Array(part.buffer, part.byteOffset, part.byteLength);\n          let s=\"\"; for(let k=0;k<b.length;k+=32768) s+=String.fromCharCode.apply(null,b.subarray(k,k+32768));\n          ws.send(JSON.stringify({realtimeInput:{mediaChunks:[{mimeType:\"audio/pcm;rate=16000\",data:btoa(s)}]}}));\n          await new Promise(r=>setTimeout(r,60));\n        }\n        setTimeout(()=>finish(),2500);\n      }catch(err){ finish(\"Audio illisible : \"+(err&&err.message?err.message:err)); }\n    }else{\n      ws.send(JSON.stringify({clientContent:{turns:[{role:\"user\",parts:[{text:buildPrompt(conv,text)}]}],turnComplete:true}}));\n    }\n  }\n}\n\n/* --- audio : lecture + téléchargement robustes (tous navigateurs) ----\n   Blob URL (jamais de data: URI : refusées ou limitées selon navigateurs),\n   bouton JS avec replis successifs : msSaveOrOpenBlob (vieux Edge/IE) →\n   <a download> programmé → lien « ouvrir dans un nouvel onglet » toujours\n   présent (iOS Safari et webviews où l'attribut download est ignoré). */\nconst AUDIO_STORE={}; let AUDIO_SEQ=0;\nfunction audioBlobUrl(b64, mime){\n  const blob=new Blob([b64ToBytes(b64)], {type: mime||\"audio/wav\"});\n  return {blob:blob, url:URL.createObjectURL(blob)};\n}\nfunction renderAudioPlayer(b64, mime, fname){\n  const m=mime||\"audio/wav\";\n  const made=audioBlobUrl(b64,m);\n  const id=\"a\"+(++AUDIO_SEQ);\n  AUDIO_STORE[id]={blob:made.blob, url:made.url, name:fname||\"audio.wav\", mime:m};\n  return '<audio controls preload=\"metadata\" src=\"'+made.url+'\"'\n    +' style=\"width:min(100%,340px);display:block;margin-top:8px\"></audio>'\n    +'<span style=\"display:flex;flex-wrap:wrap;gap:14px;align-items:center;margin-top:8px\">'\n    +'<button type=\"button\" class=\"auddl\" data-aid=\"'+id+'\"'\n    +' style=\"font:inherit;font-size:12.5px;padding:5px 12px;border:1px solid var(--line);border-radius:7px;background:#fff;color:var(--ink);cursor:pointer\">⬇ Télécharger</button>'\n    +'<a href=\"'+made.url+'\" target=\"_blank\" rel=\"noopener\"'\n    +' style=\"font-size:12.5px;color:var(--ink-2);text-decoration:underline\">Ouvrir dans un nouvel onglet</a>'\n    +'<button type=\"button\" class=\"auddl aback\" data-aid=\"'+id+'\"'\n    +' style=\"font:inherit;font-size:12.5px;padding:5px 12px;border:1px solid var(--line);border-radius:7px;background:#fff;color:var(--ink);cursor:pointer;display:inline-flex;align-items:center;gap:6px\">'\n    +'<span style=\"display:inline-flex;width:14px;height:14px\">'+IC_BACK+'</span> Charger dans le message suivant</button>'\n    +'</span>';\n}\nfunction saveAudio(id){\n  const a=AUDIO_STORE[id]; if(!a) return false;\n  /* Même moteur que les images et les vidéos : cascade + lien de secours. */\n  return zDownload(a.blob, a.mime, a.name);\n}\ndocument.addEventListener(\"click\", function(e){\n  const btn=e.target && e.target.closest ? e.target.closest(\"button.auddl\") : null;\n  if(!btn) return;\n  e.preventDefault();\n  saveAudio(btn.getAttribute(\"data-aid\"));\n});\n\n/* --- icônes des boutons d'action ------------------------------------- */\nconst IC_RESTART='<svg viewBox=\"0 0 24 24\" fill=\"none\" stroke=\"currentColor\" stroke-width=\"2\" stroke-linecap=\"round\" stroke-linejoin=\"round\" aria-hidden=\"true\"><path d=\"M17 2l4 4-4 4\"/><path d=\"M3 11v-1a4 4 0 0 1 4-4h14\"/><path d=\"M7 22l-4-4 4-4\"/><path d=\"M21 13v1a4 4 0 0 1-4 4H3\"/></svg>';\nconst IC_COPY='<svg viewBox=\"0 0 24 24\" fill=\"none\" stroke=\"currentColor\" stroke-width=\"2\" stroke-linecap=\"round\" stroke-linejoin=\"round\" aria-hidden=\"true\"><rect x=\"9\" y=\"9\" width=\"12\" height=\"12\" rx=\"2\"/><path d=\"M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1\"/></svg>';\nconst IC_CHECK='<svg viewBox=\"0 0 24 24\" fill=\"none\" stroke=\"currentColor\" stroke-width=\"2.6\" stroke-linecap=\"round\" stroke-linejoin=\"round\" aria-hidden=\"true\"><path d=\"M20 6L9 17l-5-5\"/></svg>';\nconst IC_DL='<svg viewBox=\"0 0 24 24\" fill=\"none\" stroke=\"currentColor\" stroke-width=\"2\" stroke-linecap=\"round\" stroke-linejoin=\"round\" aria-hidden=\"true\"><path d=\"M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4\"/><path d=\"M7 10l5 5 5-5\"/><path d=\"M12 15V3\"/></svg>';\nconst IC_BACK='<svg viewBox=\"0 0 24 24\" fill=\"none\" stroke=\"currentColor\" stroke-width=\"2\" stroke-linecap=\"round\" stroke-linejoin=\"round\" aria-hidden=\"true\"><path d=\"M9 14L4 9l5-5\"/><path d=\"M4 9h10.5a5.5 5.5 0 0 1 0 11H11\"/></svg>';\n\nfunction b64ToBlob(b64, mime){\n  const bin=atob(b64); const u8=new Uint8Array(bin.length);\n  for(let i=0;i<bin.length;i++) u8[i]=bin.charCodeAt(i);\n  return new Blob([u8],{type:mime||\"image/jpeg\"});\n}\n/* --- un seul moteur de téléchargement, pour tous les fichiers -----------\n   Pourquoi : `a.download` est ignoré sur iOS avant 13 et sur les vieux\n   Android ; les liens `data:` y sont refusés ; un <a> jamais attaché au\n   document ne déclenche rien sous Firefox ancien. On n'en choisit donc\n   aucun : on les essaie dans l'ordre, et on laisse TOUJOURS un lien\n   temporaire visible (valable dix minutes), ouvrable, copiable, que même\n   un navigateur d'avant ne peut pas rater. */\nconst ZDL_KEEP={}; let ZDL_SEQ=0;\nfunction zDlSize(n){\n  if(!n) return \"\";\n  return n>=1048576 ? (Math.round(n/104857.6)/10).toFixed(1).replace(\".\",\",\")+\" Mo\"\n                    : Math.max(1,Math.round(n/1024))+\" ko\";\n}\nfunction zDlMake(src, mime, name){\n  let blob=src, b64=\"\";\n  if(typeof src===\"string\"){ b64=src; blob=b64ToBlob(src, mime); }\n  else if(src instanceof Uint8Array) blob=new Blob([src],{type:mime||\"application/octet-stream\"});\n  else if(src && src.blob) blob=src.blob;\n  if(blob && !blob.type && mime){ try{ blob=new Blob([blob],{type:mime}); }catch(e){} }\n  return {blob:blob, b64:b64, mime:(blob&&blob.type)||mime||\"application/octet-stream\",\n          name:name||(\"fichier-\"+Date.now())};\n}\nfunction zDlUrl(b64, blob, mime){\n  try{ if(typeof URL!==\"undefined\"&&URL.createObjectURL) return URL.createObjectURL(blob); }catch(e){}\n  try{ if(window.webkitURL&&webkitURL.createObjectURL) return webkitURL.createObjectURL(blob); }catch(e){}\n  /* Très vieux moteur : le lien data: est tout ce qui reste. */\n  if(b64) return \"data:\"+(mime||\"application/octet-stream\")+\";base64,\"+b64;\n  return null;\n}\nfunction zDlCopy(text){\n  try{\n    if(navigator.clipboard&&navigator.clipboard.writeText) return navigator.clipboard.writeText(text);\n  }catch(e){}\n  return new Promise(function(res,rej){\n    try{\n      const t=document.createElement(\"textarea\");\n      t.value=text; t.setAttribute(\"readonly\",\"\");\n      t.style.position=\"fixed\"; t.style.top=\"-1000px\"; t.style.opacity=\"0\";\n      document.body.appendChild(t); t.select(); t.setSelectionRange(0,t.value.length);\n      const ok=document.execCommand&&document.execCommand(\"copy\");\n      document.body.removeChild(t);\n      ok?res(true):rej(new Error(\"copie refusée\"));\n    }catch(e){ rej(e); }\n  });\n}\n/* La feuille de secours : nom, taille, gros bouton, lien, copie. */\nfunction zDlSheet(url, f, why){\n  let box=document.getElementById(\"dlsheet\");\n  if(box){ try{ box.parentNode.removeChild(box); }catch(e){} }\n  box=document.createElement(\"div\");\n  box.id=\"dlsheet\"; box.className=\"dlmask\";\n  const touch=/iPad|iPhone|iPod/.test(navigator.userAgent||\"\");\n  box.innerHTML=\n    '<div class=\"dlcard\" role=\"dialog\" aria-modal=\"true\" aria-label=\"Téléchargement\">'+\n      '<h4>'+esc(f.name)+'</h4>'+\n      '<p class=\"dlmeta\">'+esc(f.mime||\"fichier\")+(f.size?(\" · \"+zDlSize(f.size)):\"\")+'</p>'+\n      (url\n        ? '<a class=\"dlbig\" id=\"dlbig\" href=\"'+url+'\" download=\"'+esc(f.name)+'\" rel=\"noopener\">'+IC_DL+' Enregistrer le fichier</a>'+\n          '<div class=\"dlrow\"><a id=\"dlopen\" href=\"'+url+'\" target=\"_blank\" rel=\"noopener\">Ouvrir dans un onglet</a>'+\n          '<button type=\"button\" id=\"dlcopyx\">Copier le lien</button></div>'\n        : '<p class=\"dlnote\">Ce navigateur ne sait pas préparer de lien pour ce fichier. Ouvre-le depuis l\\'aperçu, puis utilise « Enregistrer l\\'image ».</p>')+\n      '<p class=\"dlnote\">'+(why||\"\")+(touch\n        ? ' Sur iPhone ou iPad : appui long sur le bouton, puis « Télécharger le fichier lié ».'\n        : ' Si le bouton ne fait rien, touche « Ouvrir dans un onglet » puis enregistre la page.')+\n        ' Ce lien reste valable tant que la page est ouverte (dix minutes).</p>'+\n      '<button type=\"button\" class=\"dlclose\" id=\"dlclose\">Fermer</button>'+\n    '</div>';\n  document.body.appendChild(box);\n  const close=function(){ try{ box.parentNode.removeChild(box); }catch(e){} };\n  box.addEventListener(\"click\",function(e){ if(e.target===box) close(); });\n  const c=document.getElementById(\"dlclose\"); if(c) c.onclick=close;\n  const cp=document.getElementById(\"dlcopyx\");\n  if(cp) cp.onclick=function(){\n    zDlCopy(url).then(function(){ cp.textContent=\"Lien copié ✓\"; })\n      .catch(function(){ cp.textContent=\"Copie impossible\"; });\n  };\n  return box;\n}\n/* Rappel après un téléchargement lancé : le lien reste à portée de main. */\nfunction zDlHint(url, f){\n  let old=document.getElementById(\"dlhint\");\n  if(old){ try{ old.parentNode.removeChild(old); }catch(e){} }\n  const el=document.createElement(\"div\");\n  el.id=\"dlhint\"; el.className=\"dlhint\";\n  el.innerHTML='<span>Fichier prêt.</span><button type=\"button\">Rien ne s\\'ouvre ?</button>';\n  document.body.appendChild(el);\n  const kill=function(){ try{ el.classList.add(\"out\"); setTimeout(function(){ try{ el.remove(); }catch(e){} },320); }catch(e){} };\n  el.querySelector(\"button\").onclick=function(){ kill(); zDlSheet(url, f); };\n  setTimeout(kill, 9000);\n}\n/* Cascade : msSaveOrOpenBlob → <a download> attaché → feuille de secours. */\nfunction zDlGo(url, name, mime, opts){\n  opts=opts||{};\n  const nav=window.navigator||{};\n  if(nav.msSaveOrOpenBlob&&opts.blob){\n    try{ nav.msSaveOrOpenBlob(opts.blob, name); return true; }catch(e){}\n  }\n  if(!url) return false;\n  const a=document.createElement(\"a\");\n  const canDownload=(\"download\" in a);\n  let ok=false;\n  if(canDownload&&!opts.forceSheet){\n    try{\n      a.href=url; a.download=name; a.rel=\"noopener\"; a.style.display=\"none\";\n      document.body.appendChild(a); a.click();\n      setTimeout(function(){ try{ a.remove(); }catch(e){} }, 2000);\n      ok=true;\n    }catch(e){ ok=false; }\n  }\n  /* L'URL reste vivante : un blob révoqué trop tôt donne une page blanche. */\n  if(url.slice(0,5)===\"blob:\"){\n    const id=\"z\"+(++ZDL_SEQ);\n    ZDL_KEEP[id]=url;\n    setTimeout(function(){ try{ URL.revokeObjectURL(url); }catch(e){} delete ZDL_KEEP[id]; }, 600000);\n  }\n  const f={name:name, mime:mime||\"\", size:opts.size||0};\n  if(!canDownload||opts.forceSheet) zDlSheet(url, f, opts.why||\"Ce navigateur ne déclenche pas le téléchargement tout seul.\");\n  else zDlHint(url, f);\n  return ok;\n}\nfunction zDownload(src, mime, name, opts){\n  const f=zDlMake(src, mime, name);\n  if(!f.blob) return false;\n  const nav=window.navigator||{};\n  if(nav.msSaveOrOpenBlob){ try{ nav.msSaveOrOpenBlob(f.blob, f.name); return true; }catch(e){} }\n  const url=zDlUrl(f.b64, f.blob, f.mime);\n  return zDlGo(url, f.name, f.mime, Object.assign({blob:f.blob, size:f.blob.size}, opts||{}));\n}\nfunction toPngBlob(blob){\n  return new Promise((res,rej)=>{\n    const img=new Image(); const url=URL.createObjectURL(blob);\n    img.onload=()=>{\n      const c=document.createElement(\"canvas\");\n      c.width=img.naturalWidth||1; c.height=img.naturalHeight||1;\n      c.getContext(\"2d\").drawImage(img,0,0);\n      URL.revokeObjectURL(url);\n      c.toBlob(b=>b?res(b):rej(new Error(\"conversion png échouée\")),\"image/png\");\n    };\n    img.onerror=()=>{ URL.revokeObjectURL(url); rej(new Error(\"image illisible\")); };\n    img.src=url;\n  });\n}\nasync function copyImage(id){\n  const im=IMG_STORE[id]; if(!im) return false;\n  if(!(window.ClipboardItem && navigator.clipboard && navigator.clipboard.write)) return false;\n  let out=b64ToBlob(im.b64, im.mime);\n  if(im.mime!==\"image/png\") out=await toPngBlob(out);\n  await navigator.clipboard.write([new ClipboardItem({\"image/png\":out})]);\n  return true;\n}\n/* Zone d'attente pendant la génération d'image : gris arrondi, chronomètre\n   et reflets animés (voir .imgwait en CSS). */\nconst IC_CLOCK='<svg viewBox=\"0 0 24 24\" fill=\"none\" stroke=\"currentColor\" stroke-width=\"2\" stroke-linecap=\"round\" stroke-linejoin=\"round\" aria-hidden=\"true\"><circle cx=\"12\" cy=\"12\" r=\"9\"/><path d=\"M12 7v5l3 2\"/></svg>';\nfunction imgWaitHTML(){\n  return '<div class=\"imgwait\"><span class=\"clock\">'+IC_CLOCK+'<span class=\"t\">0,0 s</span></span></div>';\n}\nfunction startImgClock(bubble){\n  const el=bubble.querySelector(\".imgwait .t\");\n  if(!el) return null;\n  const t0=Date.now();\n  return setInterval(()=>{\n    el.textContent=((Date.now()-t0)/1000).toLocaleString(\"fr-FR\",{minimumFractionDigits:1,maximumFractionDigits:1})+\" s\";\n  },100);\n}\nfunction stopImgClock(iv){ if(iv) clearInterval(iv); }\n\n/* Ligne d'actions en bas de chaque message assistant : recommencer + copier.\n   Le bouton copier se mue en coche (animation CSS) pendant 1 s. */\nfunction msgActions(bubble, opts){\n  opts=opts||{};\n  const old=bubble.querySelector(\".macts\"); if(old) old.remove();\n  const row=document.createElement(\"div\"); row.className=\"macts\";\n  if(opts.regen){\n    const b=document.createElement(\"button\");\n    b.type=\"button\"; b.className=\"mact\"; b.title=\"Recommencer\"; b.setAttribute(\"aria-label\",\"Recommencer\");\n    b.innerHTML=IC_RESTART;\n    b.onclick=()=>{ if(busy) return; opts.regen(bubble); };\n    row.appendChild(b);\n  }\n  const c=document.createElement(\"button\");\n  c.type=\"button\"; c.className=\"mact\"; c.title=\"Copier\"; c.setAttribute(\"aria-label\",\"Copier\");\n  c.innerHTML=IC_COPY;\n  c.onclick=async ()=>{\n    let ok=false;\n    try{\n      ok = opts.imgId ? await copyImage(opts.imgId)\n                      : await copyText(opts.getMd ? opts.getMd() : (bubble.textContent||\"\"));\n    }catch(e){ ok=false; }\n    if(!ok) return;\n    c.innerHTML=IC_CHECK; c.title=\"Copié ✓\";\n    setTimeout(()=>{ c.innerHTML=IC_COPY; c.title=\"Copier\"; },1000);\n  };\n  row.appendChild(c);\n  bubble.appendChild(row);\n}\n\n/* --- génération d'image ---------------------------------------------- */\nlet IMG_SEQ=0; const IMG_STORE={};\nfunction renderGenImg(bubble, j, text, regen, pre, isRegen){\n  const mime=j.mime||\"image/jpeg\";\n  const id=\"i\"+(++IMG_SEQ);\n  const ext=mime===\"image/png\"?\".png\":(mime===\"image/webp\"?\".webp\":\".jpg\");\n  IMG_STORE[id]={b64:j.image, mime:mime, name:\"image-\"+id+ext, regen:regen, bubble:bubble};\n  /* L'image générée entre dans la mémoire de conversation : sans ça, le modèle\n     suivant ignore qu'une nouvelle image existe et s'appuie sur les anciennes.\n     (Sauf « recommencer » : la demande y est déjà.) */\n  if(!isRegen){\n    conv.push({role:\"user\", text:text+\" [demande de génération d'image]\"});\n    conv.push({role:\"assistant\", text:\"(image générée : \"+IMG_STORE[id].name+\" avec \"+current.label+\")\",\n      images:[{name:IMG_STORE[id].name, b64:j.image, mime:mime}]});\n  }\n  const src=\"data:\"+mime+\";base64,\"+j.image;\n  /* Une seule ligne de boutons : recommencer + copier + télécharger + retour.\n     Cliquer sur l'image l'ouvre en grand (visionneuse). */\n  bubble.innerHTML=(pre||\"\")\n    +'<img src=\"'+src+'\" alt=\"'+esc(text)+'\" class=\"gimg\" data-iid=\"'+id+'\"'\n    +' style=\"max-width:min(100%,420px);border-radius:8px;display:block;cursor:zoom-in\">'\n    +'<div class=\"imgacts\">'\n    +'<button type=\"button\" class=\"iact iregen\" data-iid=\"'+id+'\" title=\"Recommencer\" aria-label=\"Recommencer\">'+IC_RESTART+'</button>'\n    +'<button type=\"button\" class=\"iact icopy\" data-iid=\"'+id+'\" title=\"Copier\" aria-label=\"Copier l’image\">'+IC_COPY+'</button>'\n    +'<button type=\"button\" class=\"iact idl\" data-iid=\"'+id+'\" title=\"Télécharger\" aria-label=\"Télécharger l’image\">'+IC_DL+'</button>'\n    +'<button type=\"button\" class=\"iact iback\" data-iid=\"'+id+'\" title=\"Charger dans le message suivant\" aria-label=\"Charger l’image dans le message suivant\">'+IC_BACK+'</button>'\n    +'</div>';\n  return id;\n}\n/* Visionneuse : image en grand sur fond grisé, croix arrondie en haut à droite. */\nfunction openLightbox(id){\n  const im=IMG_STORE[id]; if(!im) return;\n  closeLightbox();\n  const ov=document.createElement(\"div\");\n  ov.className=\"lightbox\"; ov.id=\"lightbox\";\n  const img=document.createElement(\"img\");\n  img.src=\"data:\"+im.mime+\";base64,\"+im.b64; img.alt=im.name;\n  const x=document.createElement(\"button\");\n  x.type=\"button\"; x.className=\"lbclose\"; x.title=\"Fermer\"; x.setAttribute(\"aria-label\",\"Fermer\");\n  x.innerHTML='<svg viewBox=\"0 0 24 24\" fill=\"none\" stroke=\"currentColor\" stroke-width=\"2.4\" stroke-linecap=\"round\" aria-hidden=\"true\"><path d=\"M18 6L6 18M6 6l12 12\"/></svg>';\n  x.onclick=closeLightbox;\n  ov.appendChild(img); ov.appendChild(x);\n  ov.addEventListener(\"click\",(ev)=>{ if(ev.target===ov) closeLightbox(); });\n  document.body.appendChild(ov);\n  document.body.style.overflow=\"hidden\";\n}\nfunction closeLightbox(){\n  const ov=document.getElementById(\"lightbox\"); if(!ov) return;\n  ov.remove();\n  if(!fs.classList.contains(\"open\") && !cfg.classList.contains(\"open\")) document.body.style.overflow=\"\";\n}\ndocument.addEventListener(\"keydown\",(e)=>{ if(e.key===\"Escape\") closeLightbox(); });\n\n/* Clic sur une image générée → visionneuse. */\ndocument.addEventListener(\"click\", function(e){\n  const g = e.target && e.target.closest ? e.target.closest(\"img.gimg\") : null;\n  if(!g) return;\n  openLightbox(g.getAttribute(\"data-iid\"));\n});\n\n/* Boutons des images générées + flèche « recharger » des audios (délégation). */\ndocument.addEventListener(\"click\", function(e){\n  const t = e.target && e.target.closest ? e.target.closest(\"button.icopy, button.idl, button.iback, button.aback, button.iregen, button.vdl, button.vregen\") : null;\n  if(!t) return;\n  e.preventDefault();\n  if(t.classList.contains(\"iregen\")){\n    if(busy) return;\n    const im=IMG_STORE[t.getAttribute(\"data-iid\")];\n    if(im && im.regen && im.bubble) im.regen(im.bubble);\n    return;\n  }\n  if(t.classList.contains(\"vregen\")){\n    if(busy) return;\n    const v=VID_STORE[t.getAttribute(\"data-vid\")];\n    if(v && v.regen && v.bubble) v.regen(v.bubble);\n    return;\n  }\n  if(t.classList.contains(\"vdl\")){\n    const v=VID_STORE[t.getAttribute(\"data-vid\")]; if(!v) return;\n    zDownload(v.b64, v.mime||\"video/mp4\", v.name);\n    return;\n  }\n  if(t.classList.contains(\"icopy\")){\n    const id=t.getAttribute(\"data-iid\");\n    copyImage(id).then(ok=>{\n      if(!ok) return;\n      t.innerHTML=IC_CHECK; t.title=\"Copiée ✓\";\n      setTimeout(()=>{ t.innerHTML=IC_COPY; t.title=\"Copier\"; },1000);\n    }).catch(()=>{});\n  }else if(t.classList.contains(\"idl\")){\n    const im=IMG_STORE[t.getAttribute(\"data-iid\")]; if(!im) return;\n    zDownload(im.b64, im.mime||\"image/png\", im.name);\n  }else if(t.classList.contains(\"iback\")){\n    const im=IMG_STORE[t.getAttribute(\"data-iid\")]; if(!im) return;\n    pendingAudio=null;\n    pendingImage={name:im.name, b64:im.b64, small:im.small, tiny:im.tiny, mime:im.mime, kb:Math.max(1,Math.round(im.b64.length*0.75/1024))};\n    if(typeof setAttachImage===\"function\") setAttachImage(pendingImage); else renderPending();\n  }else if(t.classList.contains(\"aback\")){\n    const a=AUDIO_STORE[t.getAttribute(\"data-aid\")]; if(!a) return;\n    const fr=new FileReader();\n    fr.onload=()=>{\n      const s=String(fr.result||\"\"); const i=s.indexOf(\",\"); if(i<0) return;\n      pendingImage=null;\n      pendingAudio={name:a.name, b64:s.slice(i+1), kb:Math.max(1,Math.round((a.blob&&a.blob.size||0)/1024)), mime:a.mime||\"audio/wav\"};\n      if(typeof setAttachAudio===\"function\") setAttachAudio(pendingAudio); else renderPending();\n    };\n    fr.readAsDataURL(a.blob);\n  }\n});\n\nasync function submitImage(text){\n  busy=true; sendBtn.disabled=true;\n  /* Une image jointe est transmise au générateur : le worker l'utilise\n     comme base d'édition quand le modèle le permet. */\n  const snapshot = pendingImage || null;\n  const kind = snapshot ? \"upload\" : \"none\";\n  addUser(text, kind, snapshot, null);\n  const bubble=addAssistant();\n  promptEl.value=\"\"; promptEl.style.height=\"auto\"; scrollDown();\n  let clock=null;\n  const showWait=()=>{ stopImgClock(clock); bubble.innerHTML=imgWaitHTML(); clock=startImgClock(bubble); };\n  showWait();\n  /* Une seule image part : celle qui vient d'être jointe. `image` est\n     l'original (édition HF / ModelScope), `image_tiny` sa copie 496 px pour\n     Cloudflare (FLUX.2 klein exige des références plus petites que 512 px). */\n  let tiny = snapshot ? (snapshot.tiny || \"\") : \"\";\n  if (snapshot && !tiny) { try { tiny = await tinyFromB64(snapshot.b64); } catch(e) { tiny = \"\"; } }\n  const spec={\n    model: current.id,\n    prompt: text,\n    params: cleanParams(current.id),\n    image: snapshot ? snapshot.b64 : undefined,\n    image_tiny: tiny || undefined,\n  };\n  const tryGen=async(sp)=>{\n    const r=await fetch(API_BASE+\"/api/image\",{\n      method:\"POST\",\n      headers:{\"Content-Type\":\"application/json\",\"X-Access-Code\":ACCESS_CODE},\n      body:JSON.stringify(sp),\n    });\n    let j={}; try{ j=await r.json(); }catch(e){}\n    if(!r.ok) throw new Error(j.error || (\"HTTP \"+r.status+\" \"+r.statusText));\n    return j;\n  };\n  let j=null, relayErr=null, usedFb=false, pre=\"\";\n  try{\n    j=await tryGen(spec);\n  }catch(err){\n    relayErr=err;\n  }\n  const fbm = (!j && current.fb) ? MODELS.find(x=>x.id===current.fb) : null;\n  if(fbm){\n    /* Bascule automatique sur le canal de repli (ex. Cloudflare → Hugging Face).\n       Pour les Flux, aucun texte n'est affiché : la bascule est transparente. */\n    usedFb=true;\n    showWait();\n    const sp2={model:fbm.id, prompt:text, params:cleanParams(fbm.id)};\n    try{\n      j=await tryGen(sp2);\n      if(current.fbTxt) pre='<p class=\"viafb\"><em>'+esc(current.fbTxt)+'</em></p>';\n    }catch(e2){\n      const m2=String(e2 && e2.message ? e2.message : e2);\n      if(HF_CFG[fbm.id]){\n        /* Ultime repli : appel direct du Space HF depuis l'IP locale. */\n        try{\n          j=await hfLocal(HF_CFG[fbm.id], text, sp2.params);\n        }catch(e3){\n          relayErr=new Error(\"Relais : \"+String(relayErr && relayErr.message ? relayErr.message : relayErr)\n            +\" · \"+fbm.id+\" : \"+m2\n            +\" · Direct : \"+String(e3 && e3.message ? e3.message : e3));\n        }\n      }else{\n        relayErr=new Error(\"Relais : \"+String(relayErr && relayErr.message ? relayErr.message : relayErr)\n          +\" · \"+fbm.id+\" : \"+m2);\n      }\n    }\n    if(j) spec.model=fbm.id, spec.params=sp2.params;\n  }else if(relayErr && HF_CFG[current.id]){\n    /* Relais ZeroGPU (IP du Worker) saturé → appel direct depuis l'IP locale,\n       qui dispose de son propre quota anonyme Hugging Face. */\n    const m0=String(relayErr && relayErr.message ? relayErr.message : relayErr);\n    /* la zone d'attente (et son chrono) reste visible pendant l'appel direct */\n    bubble.insertAdjacentHTML(\"afterbegin\",'<div class=\"err\">Relais : '+esc(m0)+'<br>→ bascule sur appel direct depuis ton IP (quota ZeroGPU local)…</div>');\n    scrollDown();\n    try{\n      j=await hfLocal(HF_CFG[current.id], text, spec.params);\n      relayErr=null;\n    }catch(e2){\n      relayErr=new Error(\"Relais : \"+m0+\" · Direct : \"+String(e2&&e2.message?e2.message:e2));\n    }\n  }\n  stopImgClock(clock);\n  if(j){\n    const rid=renderGenImg(bubble, j, text, (b)=>regenImage(spec,b,pre), pre);\n    if(typeof FB!==\"undefined\"&&FB.afterImage) FB.afterImage({userText:text, id:rid, kind});\n  }else if(relayErr){\n    let msg=String(relayErr && relayErr.message ? relayErr.message : relayErr);\n    /* Deux pannes très lisibles, traduites en clair plutôt qu'en anglais. */\n    if(/daily free allocation|used up your daily|HTTP 429/.test(msg)){\n      msg=\"Le quota gratuit de Cloudflare pour aujourd'hui est épuisé (10 000 neurones). \"\n        +\"Choisis Qwen Image 2.1 ou FLUX.3 [klein] pour éditer cette image, ou réessaie demain : le quota se remet à zéro chaque jour.\";\n    }else if(/Space HF/.test(msg) && /event: error/.test(msg)){\n      msg=\"Le Space officiel de ce modèle répond mal pour l'instant (quota ZeroGPU ou service en maintenance). \"\n        +\"Essaie Qwen Image 2.1, qui passe par ModelScope.\";\n    }\n    const isNet = relayErr instanceof TypeError;\n    bubble.innerHTML='<div class=\"err\">'+esc(isNet\n      ? \"Impossible de joindre le Worker relais (\"+API_BASE+\") — \"+msg\n      : msg)+'</div>';\n    msgActions(bubble,{regen:(b)=>regenImage(spec,b,pre)});\n  }\n  clearAttachments();\n  busy=false; sendBtn.disabled=!promptEl.value.trim(); scrollDown();\n}\n/* « Recommencer » pour une image générée. */\nasync function regenImage(spec, b, pre){\n  busy=true; sendBtn.disabled=true;\n  b.innerHTML=imgWaitHTML();\n  const clock=startImgClock(b);\n  try{\n    const r=await fetch(API_BASE+\"/api/image\",{\n      method:\"POST\",\n      headers:{\"Content-Type\":\"application/json\",\"X-Access-Code\":ACCESS_CODE},\n      body:JSON.stringify(spec),\n    });\n    let j={}; try{ j=await r.json(); }catch(e){}\n    if(!r.ok) throw new Error(j.error || (\"HTTP \"+r.status+\" \"+r.statusText));\n    stopImgClock(clock);\n    renderGenImg(b, j, spec.prompt, (bb)=>regenImage(spec,bb,pre), pre, true);\n  }catch(err){\n    stopImgClock(clock);\n    b.innerHTML='<div class=\"err\">'+esc(String(err && err.message ? err.message : err))+'</div>';\n    msgActions(b,{regen:(bb)=>regenImage(spec,bb,pre)});\n  }finally{\n    busy=false; sendBtn.disabled=!promptEl.value.trim(); scrollDown();\n  }\n}\n\n/* --- génération de vidéo (Wan 2.2) ----------------------------------- */\n/* Modèle qui fabrique l'image de départ de Wan 2.2 — réglé dans\n   Paramètres ▸ Modèles ▸ Wan 2.2. Défaut : FLUX.2 [klein] 4B, rapide et\n   facile à animer. */\nconst WAN_POSTER_DEFAULT = \"flux-2-klein-4b\";\nfunction wanPosterModel(){\n  let v=\"\";\n  try{ v=localStorage.getItem(\"wan_poster\")||\"\"; }catch(e){}\n  return MODELS.some(m=>m.id===v && m.gen===\"img\") ? v : WAN_POSTER_DEFAULT;\n}\n/* Jeton Hugging Face facultatif : il ne quitte pas cet appareil, sauf pour\n   l'appel au Space qui anime la vidéo. */\nfunction hfTokenGet(){ try{ return localStorage.getItem(\"hf_token\")||\"\"; }catch(e){ return \"\"; } }\nfunction hfTokenSet(v){\n  const t=String(v||\"\").trim();\n  try{ if(t) localStorage.setItem(\"hf_token\", t); else localStorage.removeItem(\"hf_token\"); }catch(e){}\n  const nt=document.getElementById(\"hftokenote\"), tg=document.getElementById(\"hftag\");\n  const ok=/^hf_[A-Za-z0-9]{10,}$/.test(t);\n  if(tg){ tg.textContent = !t ? \"\" : (ok ? \"jeton enregistré\" : \"format inattendu\"); tg.className = \"ctag\" + (!t ? \"\" : (ok ? \" crok\" : \" crko\")); }\n  if(nt) nt.textContent = !t\n    ? \"Sans jeton : quota anonyme, environ une vidéo par jour.\"\n    : (ok ? \"Jeton enregistré sur cet appareil. Il ne part qu’avec la demande de vidéo.\"\n          : \"Un jeton Hugging Face commence par « hf_ ». Vérifie la copie.\");\n}\nfunction setWanPosterModel(id){\n  try{ localStorage.setItem(\"wan_poster\", id); }catch(e){}\n  const nt=document.getElementById(\"wanposternote\");\n  if(nt){\n    const m=MODELS.find(x=>x.id===id);\n    nt.textContent = m\n      ? \"Image de départ : \" + m.label + \" · \" + (m.via||\"\") + (id===WAN_POSTER_DEFAULT ? \" · réglage par défaut\" : \"\")\n      : \"\";\n  }\n}\n/* Panneau Paramètres ▸ Modèles ▸ Wan 2.2 : la liste des modèles d'image,\n   dessinée avec les mêmes lignes que le reste de l'application — un <select>\n   natif change d'allure d'un navigateur à l'autre et cassait la page. */\n/* L'ordre est choisi à la main : d'abord ceux qui animent le mieux, ensuite\n   les autres. Un tri alphabétique mettrait « GPT Image » en tête, ce qui\n   n'aiderait personne. */\nconst WAN_POSTER_ORDER = [\"flux-2-klein-4b\",\"flux-2-klein-9b\",\"qwen-image-2-1-turbo\",\"qwen-image-2-1\",\n  \"ideogram-v4\",\"po-gpt-image-2\",\"po-mai-image-2-6\",\"po-mai-image-2-6-flash\",\"po-flux-2-dev-free\"];\nfunction renderWanModels(){\n  const box=document.getElementById(\"wanposterlist\");\n  if(!box) return;\n  const cur=wanPosterModel();\n  const TICK='<svg viewBox=\"0 0 24 24\" fill=\"none\" stroke=\"currentColor\" stroke-width=\"3\" stroke-linecap=\"round\" stroke-linejoin=\"round\" aria-hidden=\"true\"><path d=\"m5 13 4 4L19 7\"/></svg>';\n  const list=MODELS.filter(m=>m.gen===\"img\" && !m.hidden).slice().sort(function(a,b){\n    const ia=WAN_POSTER_ORDER.indexOf(a.id), ib=WAN_POSTER_ORDER.indexOf(b.id);\n    return (ia<0?99:ia)-(ib<0?99:ib);\n  });\n  box.innerHTML=list.map(m=>\n    '<button type=\"button\" class=\"mopt\" role=\"option\" aria-selected=\"'+(m.id===cur)+'\" data-id=\"'+esc(m.id)+'\">'\n    +'<span class=\"mopt-tx\"><b>'+esc(m.label)\n      +(m.id===WAN_POSTER_DEFAULT?'<span class=\"mopt-badge\">recommandé</span>':\"\")+'</b>'\n    +'<span>'+esc(m.via||\"\")+'</span></span>'\n    +'<span class=\"mopt-tick\">'+TICK+'</span></button>').join(\"\");\n  box.onclick=function(e){\n    const b=(e.target && e.target.closest) ? e.target.closest(\"button.mopt\") : null;\n    if(!b) return;\n    setWanPosterModel(b.getAttribute(\"data-id\"));\n    const all=box.querySelectorAll(\"button.mopt\");\n    for(let i=0;i<all.length;i++) all[i].setAttribute(\"aria-selected\", all[i]===b ? \"true\" : \"false\");\n  };\n  setWanPosterModel(cur);\n  const ti=document.getElementById(\"hftoken\");\n  if(ti){\n    ti.value=hfTokenGet();\n    ti.onchange=function(){ hfTokenSet(ti.value); };\n    hfTokenSet(ti.value);\n  }\n}\nlet VID_SEQ=0; const VID_STORE={};\nfunction vidWaitHTML(fromUser){\n  return '<div class=\"imgwait\"><span class=\"clock\">'+IC_CLOCK+'<span class=\"t\">0,0 s</span></span></div>'\n    +'<p class=\"vidnote\">'+(fromUser\n      ? \"Wan 2.2 anime ton image — compte une minute environ.\"\n      : \"Wan 2.2 prépare une image de départ, puis l’anime — compte une minute environ.\")+'</p>';\n}\nfunction renderGenVideo(bubble, j, text, regen, pre, isRegen){\n  const mime=j.mime||\"video/mp4\";\n  const id=\"v\"+(++VID_SEQ);\n  VID_STORE[id]={b64:j.video, mime:mime, name:\"wan-2-2-\"+id+\".mp4\", poster:j.poster||\"\",\n    posterMime:j.posterMime||\"image/png\", regen:regen, bubble:bubble};\n  /* La vidéo entre dans la mémoire de conversation sous forme de texte : un\n     fichier vidéo ne peut pas être relu par un modèle de vision. */\n  if(!isRegen){\n    conv.push({role:\"user\", text:text+\" [demande de génération de vidéo]\"});\n    conv.push({role:\"assistant\", text:\"(vidéo générée : \"+VID_STORE[id].name+\" avec \"+current.label+\")\"});\n  }\n  const src=\"data:\"+mime+\";base64,\"+j.video;\n  const poster=VID_STORE[id].poster ? ' poster=\"data:'+VID_STORE[id].posterMime+';base64,'+VID_STORE[id].poster+'\"' : \"\";\n  bubble.innerHTML=(pre||\"\")\n    +'<video class=\"gvid\" data-vid=\"'+id+'\" controls loop playsinline preload=\"metadata\"'+poster\n    +' style=\"max-width:min(100%,460px);width:100%;border-radius:8px;display:block;background:#000\">'\n    +'<source src=\"'+src+'\" type=\"'+mime+'\">'\n    +'</video>'\n    +'<div class=\"imgacts\">'\n    +(regen?'<button type=\"button\" class=\"iact vregen\" data-vid=\"'+id+'\" title=\"Recommencer\" aria-label=\"Recommencer\">'+IC_RESTART+'</button>':\"\")\n    +'<button type=\"button\" class=\"iact vdl\" data-vid=\"'+id+'\" title=\"Télécharger\" aria-label=\"Télécharger la vidéo\">'+IC_DL+'</button>'\n    +'</div>';\n  return id;\n}\n/* Appel direct du Space vidéo depuis l'appareil : quand le relais a épuisé\n   son quota ZeroGPU, l'adresse de l'utilisateur a le sien. L'image de départ\n   vient du relais (elle est déjà faite), seule l'animation est rejouée. */\nconst HF_VID = { host:\"https://prithivmlmods-wan2-2-fast.hf.space\", api:\"generate_video\" };\nasync function hfVideoLocal(poster, posterMime, prompt, dur){\n  const tok=hfTokenGet();\n  const hdrs={\"Content-Type\":\"application/json\"};\n  if(tok) hdrs.Authorization=\"Bearer \"+tok;\n  const r=await fetch(HF_VID.host+\"/gradio_api/call/\"+HF_VID.api,{\n    method:\"POST\", headers:hdrs,\n    body:JSON.stringify({data:[\"data:\"+(posterMime||\"image/png\")+\";base64,\"+poster, prompt, \"4\", \"\", Number(dur||4), 4.0, 3.0, 0, true]}),\n  });\n  let ev=null;\n  try{ ev=(await r.json()).event_id; }catch(e){}\n  if(!ev) throw new Error(\"le Space vidéo n'a pas répondu (HTTP \"+r.status+\")\");\n  const sse=await (await fetch(HF_VID.host+\"/gradio_api/call/\"+HF_VID.api+\"/\"+ev,\n    tok?{headers:{Authorization:\"Bearer \"+tok}}:{})).text();\n  const du=sse.match(/data:video\\/[a-z0-9.+-]+;base64,([A-Za-z0-9+/=]+)/i);\n  if(!du){\n    const e=sse.match(/\"error\"\\s*:\\s*\"([^\"]{1,260})\"/);\n    const raw=e ? e[1] : \"\";\n    if(/ZeroGPU quota|exceeded your/i.test(raw)){\n      const t2=raw.match(/Try again in ([0-9:]+)/i);\n      throw new Error(\"Quota gratuit Hugging Face épuisé pour ton adresse\"\n        + (t2 ? \" — il se rouvre dans \"+t2[1] : \"\")\n        + \". Colle un jeton Hugging Face dans Paramètres ▸ Modèles ▸ Wan 2.2 pour un quota bien plus large.\");\n    }\n    throw new Error(raw || \"aucune vidéo dans la réponse du Space\");\n  }\n  return {video:du[1], mime:\"video/mp4\", poster:poster, posterMime:posterMime||\"image/png\"};\n}\nasync function videoCall(spec){\n  const r=await fetch(API_BASE+\"/api/video\",{\n    method:\"POST\",\n    headers:{\"Content-Type\":\"application/json\",\"X-Access-Code\":ACCESS_CODE},\n    body:JSON.stringify(spec),\n  });\n  let j={}; try{ j=await r.json(); }catch(e){}\n  if(r.ok) return j;\n  const err=new Error(j.error || (\"HTTP \"+r.status+\" \"+r.statusText));\n  err.poster=j.poster||null; err.posterMime=j.posterMime||\"image/png\";\n  throw err;\n}\n/* Relais d'abord ; si son quota est épuisé, on rejoue l'animation depuis\n   l'adresse de l'appareil avec l'image de départ que le relais a produite. */\nasync function videoWithFallback(spec){\n  try{\n    return await videoCall(spec);\n  }catch(err){\n    if(!err.poster) throw err;\n    const v=await hfVideoLocal(err.poster, err.posterMime, spec.prompt, spec.duration);\n    v.local=true;\n    return v;\n  }\n}\nasync function submitVideo(text){\n  busy=true; sendBtn.disabled=true;\n  /* Une image jointe devient la première image de la vidéo : Wan 2.2 l'anime\n     telle quelle au lieu d'en fabriquer une. */\n  const snapshot = pendingImage || null;\n  const kind = snapshot ? \"upload\" : \"none\";\n  addUser(text, kind, snapshot, null);\n  const bubble=addAssistant();\n  promptEl.value=\"\"; promptEl.style.height=\"auto\"; scrollDown();\n  bubble.innerHTML=vidWaitHTML(snapshot);\n  const clock=startImgClock(bubble);\n  const spec={model:current.id, prompt:text, duration:cleanParams(current.id).duration||4,\n    posterModel:wanPosterModel()};\n  if(hfTokenGet()) spec.hfToken=hfTokenGet();\n  if(snapshot){ spec.image=snapshot.b64; spec.imageMime=snapshot.mime||\"image/png\"; }\n  try{\n    const j=await videoWithFallback(spec);\n    stopImgClock(clock);\n    /* Voie gratuite : le Space anime une image tirée du prompt — on le dit. */\n    /* On dit d'où vient l'image de départ : c'est le maillon qui manquait\n       pour comprendre ce que la vidéo a réellement animé. */\n    const pm=MODELS.find(x=>x.id===wanPosterModel());\n    const pre=(j.via===\"hf\"||j.local)\n      ? (snapshot ? '<p class=\"viafb\"><em>wan 2.2 · ton image</em></p>'\n                  : '<p class=\"viafb\"><em>wan 2.2 · image de départ : '+esc(pm?pm.label:\"FLUX.2 [klein] 4B\")+'</em></p>')\n      : \"\";\n    const rid=renderGenVideo(bubble, j, text, (b)=>regenVideo(spec,b,pre), pre);\n    if(typeof FB!==\"undefined\"&&FB.afterVideo) FB.afterVideo({userText:text, id:rid, via:j.via});\n  }catch(err){\n    stopImgClock(clock);\n    const msg=String(err && err.message ? err.message : err);\n    const isNet = err instanceof TypeError;\n    bubble.innerHTML='<div class=\"err\">'+esc(isNet\n      ? \"Impossible de joindre le Worker relais (\"+API_BASE+\") — \"+msg\n      : msg)+'</div>';\n    msgActions(bubble,{regen:(b)=>regenVideo(spec,b,\"\")});\n  }\n  clearAttachments();\n  busy=false; sendBtn.disabled=!promptEl.value.trim(); scrollDown();\n}\nasync function regenVideo(spec, b, pre){\n  busy=true; sendBtn.disabled=true;\n  b.innerHTML=vidWaitHTML();\n  const clock=startImgClock(b);\n  try{\n    const j=await videoWithFallback(spec);\n    stopImgClock(clock);\n    const vid=renderGenVideo(b, j, spec.prompt, (bb)=>regenVideo(spec,bb,pre), pre, true);\n    /* La vidéo recommencée est enregistrée comme les autres. */\n    if(typeof FB!==\"undefined\"&&FB.afterVideo) FB.afterVideo({userText:spec.prompt, id:vid, via:j.via});\n  }catch(err){\n    stopImgClock(clock);\n    b.innerHTML='<div class=\"err\">'+esc(String(err && err.message ? err.message : err))+'</div>';\n    msgActions(b,{regen:(bb)=>regenVideo(spec,bb,pre)});\n  }finally{\n    busy=false; sendBtn.disabled=!promptEl.value.trim(); scrollDown();\n  }\n}\n\n/* --- appel direct des Spaces HF (navigateur → Space), repli du relais ----- */\nconst HF_CFG = {\n  \"hf-flux-2-klein-9b\": {\n    host:\"https://black-forest-labs-flux-klein-9b-kv.hf.space\", api:\"generate\",\n    w:[256,1024,8], h:[256,1024,8], s:[1,8,4],\n    data:(p,w,h,s,g,mode)=>[p,[],0,true,w,h,s,false],\n  },\n  \"hf-flux-2-klein-4b\": {\n    host:\"https://black-forest-labs-flux-2-klein-4b.hf.space\", api:\"infer\",\n    w:[256,1024,8], h:[256,1024,8], s:[1,50,4],\n    data:(p,w,h,s,g)=>[p,[],\"Distilled (4 steps)\",0,true,w,h,s,g===undefined?1.0:g,false],\n  },\n  \"hf-flux-3-klein\": {\n    host:\"https://black-forest-labs-flux-3-image-klein.hf.space\", api:\"generate\",\n    w:[512,2048,64], h:[512,2048,64], s:[1,50,16],\n    modes:[\"Distilled (16 steps)\",\"Base (50 steps)\"],\n    data:(p,w,h,s,g,mode,params)=>[p,[],mode||\"Distilled (16 steps)\",0,true,w,h,s===undefined?16:s,g===undefined?1.0:g,!!(params&&(params.upsample===true||params.upsample===\"oui\"))],\n  },\n  \"ideogram-v4\": {\n    host:\"https://ideogram-ai-ideogram4.hf.space\", api:\"generate\",\n    w:[512,2048,64], h:[512,2048,64], s:null,\n    modes:[\"Turbo · 12 steps\",\"Default · 20 steps\",\"Quality · 48 steps\"],\n    data:(p,w,h,s,g,mode)=>[p,mode,\"Ideogram (remote)\",w,h,0,true],\n  },\n};\nfunction hfStep(v,b,d){ let n=Number(v); if(!isFinite(n)) n=d; return Math.min(b[1], Math.max(b[0], Math.round(n/b[2])*b[2])); }\nfunction hfFindFile(o){\n  if(Array.isArray(o)){ for(const v of o){ const r=hfFindFile(v); if(r) return r; } }\n  else if(o && typeof o===\"object\"){\n    if(typeof o.url===\"string\" && o.url.indexOf(\"http\")===0) return o;\n    for(const k in o){ const r=hfFindFile(o[k]); if(r) return r; }\n  }\n  return null;\n}\nfunction bufToB64(buf){\n  const u=new Uint8Array(buf); let s=\"\";\n  for(let i=0;i<u.length;i+=32768){ s+=String.fromCharCode.apply(null, u.subarray(i,i+32768)); }\n  return btoa(s);\n}\nasync function hfLocal(cfg, text, params){\n  const w=hfStep(params.width, cfg.w, cfg.w[1]);\n  const h=hfStep(params.height, cfg.h, cfg.h[1]);\n  const s=cfg.s ? Math.min(cfg.s[1], Math.max(cfg.s[0], Number(params.num_steps)||cfg.s[2])) : undefined;\n  const g=(params.guidance===undefined||params.guidance===\"\")?undefined:Number(params.guidance);\n  const mode=cfg.modes ? (cfg.modes.indexOf(params.mode)>=0?params.mode:cfg.modes[0]) : undefined;\n  const post=await fetch(cfg.host+\"/gradio_api/call/\"+cfg.api,{\n    method:\"POST\", headers:{\"Content-Type\":\"application/json\"},\n    body:JSON.stringify({data:cfg.data(text,w,h,s,g,mode,params)}),\n  });\n  if(!post.ok) throw new Error(\"Space HF HTTP \"+post.status+\" (appel direct)\");\n  const ev=(await post.json()).event_id;\n  if(!ev) throw new Error(\"pas d'event_id (appel direct)\");\n  const sr=await fetch(cfg.host+\"/gradio_api/call/\"+cfg.api+\"/\"+ev);\n  const txt=await sr.text();\n  if(/event:\\s*error/.test(txt)){\n    let m=\"quota ZeroGPU local épuisé ou erreur du Space\";\n    const dm=txt.match(/data:\\s*(\\{.*\\})/);\n    if(dm){ try{ const j=JSON.parse(dm[1]); if(j&&j.error) m=String(j.error).slice(0,220); }catch(e){} }\n    throw new Error(m);\n  }\n  const dm2=txt.match(/^data:\\s*(\\[.*\\])\\s*$/m);\n  let file=null; if(dm2){ try{ file=hfFindFile(JSON.parse(dm2[1])); }catch(e){} }\n  if(!file||!file.url) throw new Error(\"aucune image dans la réponse du Space (appel direct)\");\n  const fr=await fetch(file.url);\n  if(!fr.ok) throw new Error(\"téléchargement de l'image HTTP \"+fr.status);\n  const buf=await fr.arrayBuffer();\n  let mime=fr.headers.get(\"content-type\")||\"\";\n  if(mime.indexOf(\"image/\")!==0) mime=/\\.webp(\\?|$)/.test(file.url)?\"image/webp\":\"image/png\";\n  return { image: bufToB64(buf), mime };\n}\n\n/* pixel 1×1 blanc, RGB sans canal alpha (l'alpha virait au violet une fois\n   réencodé en JPEG par le bloc Roboflow). L'API exige un champ image. */\nconst WHITE_PX = \"/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAMCAgMCAgMDAwMEAwMEBQgFBQQEBQoHBwYIDAoMDAsKCwsNDhIQDQ4RDgsLEBYQERMUFRUVDA8XGBYUGBIUFRT/2wBDAQMEBAUEBQkFBQkUDQsNFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBT/wAARCABAAEADASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwD9U6KKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKAP/2Q==\";   /* blanc 64x64 JPEG : certains providers rejettent le 1 px */\n\n/* --- rendu SVG isolé : seuil anti-crash + document autonome --------- */\nconst SVG_RENDER_MAX = 300000;   /* au-delà : code affiché, rendu refusé */\n/* Les balises du document hôte sont bâties par concaténation : écrites en\n   dur dans le source, elles peuvent faire sortir prématurément du bloc de\n   script sur certains navigateurs/visualiseurs et afficher le code en\n   texte — c'était le bug d'affichage « page à moitié + code en dessous ». */\nconst LT = String.fromCharCode(60), GT = String.fromCharCode(62), SL = String.fromCharCode(47);\nconst DOC_OPEN = LT+\"!DOCTYPE html\"+GT+LT+\"html\"+GT+LT+\"head\"+GT+LT+\"meta charset='utf-8'\"+GT+LT+\"style\"+GT;\nconst DOC_MID  = LT+SL+\"style\"+GT+LT+SL+\"head\"+GT+LT+\"body\"+GT;\nconst DOC_END  = LT+SL+\"body\"+GT+LT+SL+\"html\"+GT;\nfunction svgDoc(code){\n  return DOC_OPEN\n    + 'html,body{margin:0;padding:0;background:#fff}'\n    + 'body{display:flex;align-items:center;justify-content:center;min-height:100vh;padding:8px;box-sizing:border-box}'\n    + 'svg{max-width:100%;max-height:96vh;height:auto}'\n    + DOC_MID + code + DOC_END;\n}\n\n/* --- zones de code : copier / exécuter ------------------------------ */\nlog.addEventListener(\"click\", e=>{\n  const btn=e.target.closest(\".zbtn\"); if(!btn) return;\n  const zone=btn.closest(\".zone\"); if(!zone) return;\n  const entry=CODE_STORE[+zone.dataset.zone]; if(!entry) return;\n\n  if(btn.classList.contains(\"copy\")){\n    copyText(entry.code).then(ok=>{\n      const old=btn.textContent; btn.textContent=ok?\"Copié ✓\":\"Échec\";\n      setTimeout(()=>btn.textContent=old,1300);\n    });\n    return;\n  }\n  if(btn.classList.contains(\"run\")){\n    const body=zone.querySelector(\".zone-body\");\n    const done=btn.dataset.done===\"1\";\n    if(entry.lang===\"svg\"){\n      if(!done){\n        let render=zone.querySelector(\".zone-render\");\n        if(!render){\n          render=document.createElement(\"div\"); render.className=\"zone-render\";\n          /* Anti-crash : le SVG n'est JAMAIS injecté dans le DOM de la page.\n             Il est rendu dans une iframe isolée (origine opaque) : même un\n             SVG énorme ou pathologique ne peut plus geler l'onglet. */\n          const code=String(entry.code||\"\");\n          if(code.length>SVG_RENDER_MAX){\n            const p=document.createElement(\"p\");\n            p.className=\"zone-warn\";\n            p.textContent=\"SVG trop volumineux pour un rendu sûr (\"\n              +code.length.toLocaleString(\"fr-FR\")+\" caractères, seuil \"\n              +SVG_RENDER_MAX.toLocaleString(\"fr-FR\")+\"). Le code reste copiable ci-dessus.\";\n            render.appendChild(p);\n          }else{\n            const f=document.createElement(\"iframe\");\n            f.className=\"zone-frame\"; f.title=\"Rendu SVG\";\n            f.setAttribute(\"sandbox\",\"allow-scripts\");\n            f.setAttribute(\"loading\",\"lazy\");\n            f.srcdoc=svgDoc(code);\n            render.appendChild(f);\n          }\n          zone.appendChild(render);\n        }\n        body.style.display=\"none\";\n        btn.dataset.done=\"1\"; btn.textContent=\"Voir code\";\n      }else{\n        const r=zone.querySelector(\".zone-render\");\n        if(r){ const f=r.querySelector(\"iframe\"); if(f) f.srcdoc=\"\"; r.remove(); }\n        body.style.display=\"\";\n        btn.dataset.done=\"0\"; btn.textContent=\"Exécuter\";\n      }\n    }else{ /* html */\n      if(!done){\n        fsOwner=btn;\n        fsFrame.srcdoc=entry.code;\n        fs.classList.add(\"open\");\n        document.body.style.overflow=\"hidden\";\n        btn.dataset.done=\"1\"; btn.textContent=\"Voir code\";\n      }else{\n        closeFS();\n      }\n    }\n  }\n});\nfunction closeFS(){\n  fs.classList.remove(\"open\");\n  fsFrame.srcdoc=\"\";\n  document.body.style.overflow=\"\";\n  if(fsOwner){ fsOwner.dataset.done=\"0\"; fsOwner.textContent=\"Exécuter\"; fsOwner=null; }\n}\nfsClose.onclick=closeFS;\ndocument.addEventListener(\"keydown\", e=>{ if(e.key===\"Escape\" && fs.classList.contains(\"open\")) closeFS(); });\n\nfunction copyText(t){\n  if(navigator.clipboard && window.isSecureContext)\n    return navigator.clipboard.writeText(t).then(()=>true).catch(()=>fallbackCopy(t));\n  return Promise.resolve(fallbackCopy(t));\n}\nfunction fallbackCopy(t){\n  try{\n    const ta=document.createElement(\"textarea\");\n    ta.value=t; ta.style.position=\"fixed\"; ta.style.opacity=\"0\";\n    document.body.appendChild(ta); ta.select();\n    const ok=document.execCommand(\"copy\"); ta.remove(); return ok;\n  }catch(e){ return false; }\n}\n\n/* =====================================================================\n   Réglages par modèle (appui long sur une puce)\n   ===================================================================== */\nconst CFG_FIELDS = {\n  /* Arena n'a pas d'option de génération : le tableau vide évite toute\n     erreur quand on ouvre sa fiche. */\n  arena: [],\n  ifm: [\n    {k:\"max_tokens\", label:\"Tokens max en sortie\", type:\"int\", min:16, max:32768, ph:\"défaut modèle\"},\n    {k:\"temperature\", label:\"Température (0 à 2)\", type:\"num\", min:0, max:2, step:0.05, ph:\"défaut modèle\"},\n  ],\n  claude: [\n    {k:\"extended_thinking\", label:\"Réflexion étendue\", type:\"check\",\n     note:\"Activable, mais la profondeur n'est plus contrôlable depuis Claude Opus 4.7 : le thinking est « adaptive » et le budget ci-dessous est ignoré.\"},\n    {k:\"thinking_budget_tokens\", label:\"Budget de réflexion (tokens)\", type:\"int\", min:1024, max:128000, ph:\"défaut modèle\"},\n    {k:\"max_tokens\", label:\"Tokens max en sortie\", type:\"int\", min:1, max:128000, ph:\"défaut modèle\"},\n    {k:\"temperature\", label:\"Température (0 à 1)\", type:\"num\", min:0, max:1, step:0.05, ph:\"défaut modèle\",\n     note:\"Ignorée depuis Claude Opus 4.7 : Anthropic a supprimé les contrôles d'échantillonnage.\"},\n    {k:\"max_image_size\", label:\"Taille max de l'image (px)\", type:\"int\", min:64, max:8192, ph:\"1024\"},\n    {k:\"max_concurrent_requests\", label:\"Requêtes parallèles\", type:\"int\", min:1, max:64, ph:\"défaut moteur\"},\n  ],\n  openai: [\n    {k:\"reasoning_effort\", label:\"Effort de raisonnement\", type:\"select\",\n     opts:[\"low\",\"medium\",\"high\",\"xhigh\",\"max\"], ph:\"défaut modèle\"},\n    {k:\"image_detail\", label:\"Détail de l'image\", type:\"select\", opts:[\"auto\",\"high\",\"low\"], ph:\"auto\",\n     note:\"« low » réduit nettement les tokens consommés par l'image — utile quand elle ne sert à rien.\"},\n    {k:\"max_tokens\", label:\"Tokens max en sortie\", type:\"int\", min:16, max:128000, ph:\"défaut modèle\"},\n    {k:\"temperature\", label:\"Température (0 à 2)\", type:\"num\", min:0, max:2, step:0.05, ph:\"défaut modèle\"},\n    {k:\"max_concurrent_requests\", label:\"Requêtes parallèles\", type:\"int\", min:1, max:64, ph:\"défaut moteur\"},\n  ],\n  gemini: [\n    {k:\"thinking_level\", label:\"Niveau de réflexion\", type:\"select\", opts:[\"low\",\"high\"], ph:\"high (défaut)\",\n     note:\"Gemini 3 et + : « low » privilégie la latence, « high » la profondeur.\"},\n    {k:\"max_tokens\", label:\"Tokens max en sortie\", type:\"int\", min:1, max:65536, ph:\"défaut modèle\"},\n    {k:\"google_code_execution\", label:\"Exécution de code native\", type:\"check\",\n     note:\"Autorise le bloc Gemini à exécuter du code côté Google.\"},\n  ],\n  gtext: [\n    {k:\"thinking_level\", label:\"Niveau de réflexion\", type:\"select\", opts:[\"low\",\"high\"], ph:\"high (défaut)\",\n     note:\"Gemini 3 et + uniquement : « low » privilégie la latence, « high » la profondeur.\"},\n    {k:\"temperature\", label:\"Température (0 à 2)\", type:\"num\", min:0, max:2, step:0.05, ph:\"défaut modèle\"},\n    {k:\"max_tokens\", label:\"Tokens max en sortie\", type:\"int\", min:1, max:65536, ph:\"défaut modèle\"},\n    {k:\"google_search\", label:\"Outil Google Search\", type:\"check\",\n     note:\"Grounding avec Google Search : réponses à jour et sourcées.\"},\n    {k:\"google_code_execution\", label:\"Outil exécution de code\", type:\"check\",\n     note:\"Exécute du Python côté Google et exploite le résultat.\"},\n    {k:\"google_url_context\", label:\"Outil contexte d'URL\", type:\"check\",\n     note:\"Lit le contenu des URL présentes dans la conversation.\"},\n    {k:\"google_maps\", label:\"Outil Google Maps\", type:\"check\",\n     note:\"Grounding avec Google Maps : lieux, itinéraires, avis.\"},\n  ],\n  gtts: [\n    {k:\"voice\", label:\"Voix\", type:\"select\",\n     opts:[\"Kore\",\"Puck\",\"Charon\",\"Fenrir\",\"Aoede\",\"Leda\",\"Orus\",\"Zephyr\"], ph:\"Kore (défaut)\"},\n  ],\n  gembed: [],\n  gtrans: [],\n  glivetr: [\n    {k:\"target_language\", label:\"Langue cible (BCP-47)\", type:\"select\",\n     opts:[\"en\",\"fr\",\"es\",\"de\",\"it\",\"pt\",\"nl\",\"pl\",\"tr\",\"ru\",\"ar\",\"ja\",\"zh\",\"ko\",\"hi\"], ph:\"en (défaut)\"},\n    {k:\"echo_target_language\", label:\"Écho si déjà dans la langue cible\", type:\"check\",\n     note:\"Activé : répète les passages déjà dans la langue cible ; désactivé : silence.\"},\n  ],\n  glive: [\n    {k:\"thinking\", label:\"Réflexion étendue (thinking)\", type:\"check\",\n     note:\"Activé = gemini-3.8-live-extended-thinking (plus profond, plus lent).\"},\n  ],\n  ptext: [\n    {k:\"reasoning_effort\", label:\"Effort de raisonnement\", type:\"select\",\n     opts:[\"none\",\"low\",\"medium\",\"high\",\"xhigh\",\"max\"], ph:\"défaut modèle\",\n     note:\"GPT-6 Astra/Sol/Luna : de « none » (réflexion désactivée) à « max ».\"},\n    {k:\"temperature\", label:\"Température (0 à 2)\", type:\"num\", min:0, max:2, step:0.05, ph:\"défaut modèle\"},\n    {k:\"max_tokens\", label:\"Tokens max en sortie\", type:\"int\", min:1, max:128000, ph:\"défaut modèle\"},\n  ],\n  msimg: [\n    /* Deux sections, dans cet ordre : le FORMAT (ratio) puis la QUALITÉ\n       (taille demandée). Les pixels exacts partent dans `size` chez\n       ModelScope — vérifié dans la documentation de /images/generations. */\n    {k:\"format\", label:\"Format\", type:\"format\",\n     note:\"Le ratio est converti en pixels exacts au moment de l'envoi ; l'image revient sans aucune recompression.\"},\n    {k:\"tier\", label:\"Qualité\", type:\"tier\",\n     note:\"Qwen Image plafonne à 1664 px de côté : au-delà, la qualité n'ajoute plus de détail.\"},\n  ],\n\n  msvid: [\n    {k:\"duration\", label:\"Durée (secondes)\", type:\"int\", min:2, max:6, ph:\"4\",\n     note:\"Wan 2.2 rend des plans courts : 2 à 6 secondes. Plus long = plus d'attente et un fichier plus lourd.\"},\n  ],\n  pimg: [\n    {k:\"format\", label:\"Format\", type:\"format\",\n     note:\"MAI impose 768 px minimum par côté et ~1 mégapixel au total : le format est borné en conséquence.\"},\n    {k:\"tier\", label:\"Qualité\", type:\"tier\",\n     note:\"Pollinations plafonne ces modèles à 1024 px de côté.\"},\n    {k:\"quality\", label:\"Qualité du rendu\", type:\"chips\", only:\"po-gpt-image-2\",\n     opts:[\"low\",\"medium\",\"high\",\"hd\"], labels:{low:\"low\",medium:\"medium\",high:\"high\",hd:\"hd\"},\n     note:\"GPT Image 2 uniquement ; les modèles MAI et Flux ignorent ce réglage.\"},\n  ],\n\n  groq: [\n    {k:\"reasoning_effort\", label:\"Effort de raisonnement\", type:\"select\", opts:[\"low\",\"medium\",\"high\"], ph:\"défaut modèle\",\n     note:\"GPT-OSS uniquement ; ignoré par les autres modèles.\"},\n    {k:\"temperature\", label:\"Température (0 à 2)\", type:\"num\", min:0, max:2, step:0.05, ph:\"défaut modèle\"},\n    {k:\"top_p\", label:\"Top-p\", type:\"num\", min:0, max:1, step:0.05, ph:\"défaut modèle\"},\n    {k:\"max_tokens\", label:\"Tokens max en sortie\", type:\"int\", min:1, max:131072, ph:\"défaut modèle\"},\n  ],\n  mistral: [\n    /* Paramètres Mistral (API compatible OpenAI), relayés par le bloc Roboflow. */\n    {k:\"temperature\", label:\"Température (0 à 1,5)\", type:\"num\", min:0, max:1.5, step:0.05, ph:\"défaut modèle\"},\n    {k:\"top_p\", label:\"Échantillonnage top-p\", type:\"num\", min:0, max:1, step:0.05, ph:\"défaut modèle\"},\n    {k:\"max_tokens\", label:\"Tokens max en sortie\", type:\"int\", min:1, max:209715, ph:\"défaut modèle\"},\n    {k:\"presence_penalty\", label:\"Pénalité de présence (-2 à 2)\", type:\"num\", min:-2, max:2, step:0.1, ph:\"0\",\n     note:\"Positif : pousse le modèle à aborder de nouveaux sujets.\"},\n    {k:\"frequency_penalty\", label:\"Pénalité de fréquence (-2 à 2)\", type:\"num\", min:-2, max:2, step:0.1, ph:\"0\",\n     note:\"Positif : réduit les répétitions.\"},\n    {k:\"seed\", label:\"Graine aléatoire\", type:\"int\", min:0, max:2147483647, ph:\"aléatoire\",\n     note:\"Même graine + mêmes réglages = réponse reproductible (utile pour comparer).\"},\n  ],\n  router: [\n    {k:\"temperature\", label:\"Température (0 à 2)\", type:\"num\", min:0, max:2, step:0.05, ph:\"défaut modèle\"},\n    {k:\"max_tokens\", label:\"Tokens max en sortie\", type:\"int\", min:1, max:128000, ph:\"défaut modèle\"},\n    {k:\"max_concurrent_requests\", label:\"Requêtes parallèles\", type:\"int\", min:1, max:64, ph:\"défaut moteur\"},\n  ],\n  qwen: [\n    {k:\"enable_thinking\", label:\"Mode réflexion\", type:\"check\",\n     note:\"Supporté par le checkpoint Qwen 3.5 VL 2B uniquement ; ignoré ailleurs.\"},\n    {k:\"max_tokens\", label:\"Tokens max en sortie\", type:\"int\", min:1, max:65536, ph:\"défaut modèle\"},\n    {k:\"temperature\", label:\"Température (0 à 2)\", type:\"num\", min:0, max:2, step:0.05, ph:\"OpenRouter seul\",\n     note:\"Backend OpenRouter uniquement : le runtime natif n'a pas de température.\"},\n  ],\n  glm: [\n    {k:\"max_new_tokens\", label:\"Tokens max à générer\", type:\"int\", min:1, max:8192, ph:\"défaut modèle\"},\n  ],\n  flux: [\n    {k:\"format\", label:\"Format\", type:\"format\",\n     note:\"Pixels calculés depuis le ratio et la qualité (Workers AI, multiples de 64).\"},\n    {k:\"tier\", label:\"Qualité\", type:\"tier\",\n     note:\"Workers AI accepte jusqu'à 2048 px de côté.\"},\n    {k:\"num_steps\", label:\"Étapes de diffusion\", type:\"int\", min:1, max:50, ph:\"défaut modèle\",\n     note:\"Plus d'étapes = plus de détails mais plus lent. Klein est conçu pour peu d'étapes.\"},\n    {k:\"guidance\", label:\"Guidance (0 à 10)\", type:\"num\", min:0, max:10, step:0.1, ph:\"défaut modèle\",\n     note:\"Fidélité au prompt : plus la valeur est haute, plus l'image suit le texte au pied de la lettre.\"},\n  ],\n\n  kleinhf: [\n    {k:\"format\", label:\"Format\", type:\"format\",\n     note:\"Pixels calculés depuis le ratio et la qualité, côté Space HF (multiples de 8).\"},\n    {k:\"tier\", label:\"Qualité\", type:\"tier\",\n     note:\"Ces Spaces plafonnent à 1024 px de côté.\"},\n    {k:\"num_steps\", label:\"Étapes de diffusion\", type:\"int\", min:1, max:8, ph:\"4\",\n     note:\"Klein est distillé : 4 étapes suffisent (8 max sur ces Spaces).\"},\n    {k:\"guidance\", label:\"Guidance (0 à 10)\", type:\"num\", min:0, max:10, step:0.1, ph:\"1.0\",\n     note:\"Uniquement prise en compte par la variante 4B ; ignorée en 9B KV.\"},\n  ],\n\n  klein3: [\n    {k:\"format\", label:\"Format\", type:\"format\",\n     note:\"Pixels calculés depuis le ratio et la qualité (Space officiel BFL, multiples de 64).\"},\n    {k:\"tier\", label:\"Qualité\", type:\"tier\",\n     note:\"Sans compte Hugging Face, ZeroGPU plafonne à 2048 px de côté : le 4K demanderait un compte PRO.\"},\n    {k:\"upsample\", label:\"Amélioration du prompt\", type:\"chips\", def:\"non\",\n     opts:[\"non\", \"oui\"], labels:{non:\"non\", oui:\"oui\"},\n     note:\"Non (défaut) : le prompt part tel quel, mot pour mot. Oui : le Space l'enrichit avant de générer.\"},\n    {k:\"mode\", label:\"Mode\", type:\"select\", ph:\"Distilled (16 steps)\",\n     opts:[\"Distilled (16 steps)\", \"Base (50 steps)\"],\n     note:\"Distilled = 16 étapes, rapide. Base = 50 étapes, plus lent et plus fin.\"},\n    {k:\"num_steps\", label:\"Étapes de diffusion\", type:\"int\", min:1, max:50, ph:\"16\"},\n    {k:\"guidance\", label:\"Guidance (0 à 10)\", type:\"num\", min:0, max:10, step:0.1, ph:\"1.0\"},\n  ],\n  ideo: [\n    {k:\"format\", label:\"Format\", type:\"format\",\n     note:\"Pixels calculés depuis le ratio et la qualité (Ideogram, multiples de 64).\"},\n    {k:\"tier\", label:\"Qualité\", type:\"tier\",\n     note:\"Ideogram accepte jusqu'à 2048 px de côté.\"},\n    {k:\"mode\", label:\"Mode\", type:\"select\", ph:\"Turbo\",\n     opts:[\"Turbo · 12 steps\", \"Default · 20 steps\", \"Quality · 48 steps\"],\n     note:\"Turbo = le plus rapide ; Quality = le plus fidèle mais plus lent.\"},\n  ],\n\n};\n\nconst SETTINGS = {};\nMODELS.forEach(m=>{\n  let v={};\n  try{ v=JSON.parse(localStorage.getItem(\"cfg_\"+m.id)||\"{}\")||{}; }catch(e){ v={}; }\n  SETTINGS[m.id]=v;\n});\nfunction saveSettings(id){ try{ localStorage.setItem(\"cfg_\"+id, JSON.stringify(SETTINGS[id])); }catch(e){} }\n\n/* Géométrie des modèles d'image : la fiche ne demande plus des pixels bruts,\n   mais un FORMAT (ratio) puis une QUALITÉ (taille). Les pixels exacts sont\n   calculés ici, puis bornés par ce que chaque fournisseur accepte. */\nconst IMG_GEO = {\n  msimg:   { formats:[\"1:1\",\"4:3\",\"3:4\",\"3:2\",\"2:3\",\"16:9\",\"9:16\"],\n             tiers:[{v:1024,l:\"1K · 1024 px\"},{v:1328,l:\"1,5K · 1328 px\"},{v:1664,l:\"2K · 1664 px\"}],\n             def:{format:\"1:1\",tier:1328}, step:16, min:512, max:1664 },\n  pimg:    { formats:[\"1:1\",\"4:3\",\"3:4\",\"3:2\",\"2:3\",\"16:9\",\"9:16\"],\n             tiers:[{v:768,l:\"768 px\"},{v:1024,l:\"1K · 1024 px\"}],\n             def:{format:\"1:1\",tier:1024}, step:16, min:768, max:1024 },\n  flux:    { formats:[\"1:1\",\"4:3\",\"3:4\",\"3:2\",\"2:3\",\"16:9\",\"9:16\"],\n             tiers:[{v:1024,l:\"1K · 1024 px\"},{v:2048,l:\"2K · 2048 px\"}],\n             def:{format:\"1:1\",tier:1024}, step:64, min:256, max:2048 },\n  kleinhf: { formats:[\"1:1\",\"4:3\",\"3:4\",\"3:2\",\"2:3\",\"16:9\",\"9:16\"],\n             tiers:[{v:768,l:\"768 px\"},{v:1024,l:\"1K · 1024 px\"}],\n             def:{format:\"1:1\",tier:1024}, step:8, min:256, max:1024 },\n  ideo:    { formats:[\"1:1\",\"4:3\",\"3:4\",\"3:2\",\"2:3\",\"16:9\",\"9:16\"],\n             tiers:[{v:1024,l:\"1K · 1024 px\"},{v:2048,l:\"2K · 2048 px\"}],\n             def:{format:\"1:1\",tier:1024}, step:64, min:512, max:2048 },\n  klein3:  { formats:[\"1:1\",\"4:3\",\"3:4\",\"3:2\",\"2:3\",\"16:9\",\"9:16\"],\n             tiers:[{v:1024,l:\"1K · 1024 px\"},{v:1536,l:\"1,5K · 1536 px\"},{v:2048,l:\"2K · 2048 px\"}],\n             def:{format:\"1:1\",tier:1024}, step:64, min:512, max:2048 },\n};\n/* Largeur et hauteur exactes pour un ratio et une qualité donnés. */\nfunction imgSizeOf(family, format, tier){\n  const g=IMG_GEO[family];\n  if(!g) return {w:0,h:0,capped:false};\n  const parts=String(format||g.def.format).split(\":\");\n  const a=Number(parts[0])||1, b=Number(parts[1])||1;\n  const want=Number(tier)||g.def.tier;\n  const L=Math.max(g.min, Math.min(g.max, want));\n  let w,h;\n  if(a>=b){ w=L; h=L*b/a; } else { h=L; w=L*a/b; }\n  const snap=n=>Math.max(g.min, Math.min(g.max, Math.round(n/g.step)*g.step));\n  return { w:snap(w), h:snap(h), capped:(want>g.max || Math.min(w,h)<g.min) };\n}\n\n/* ne retient que les valeurs réellement renseignées */\nfunction cleanParams(id){\n  const m=MODELS.find(x=>x.id===id);\n  if(!m) return {};\n  const out={};\n  for(const f of CFG_FIELDS[m.family]){\n    if(f.only && f.only!==id) continue;\n    const v=SETTINGS[id][f.k];\n    if(v===null||v===undefined||v===\"\"||v===false) continue;\n    out[f.k]=v;\n  }\n  /* Format + qualité -> pixels exacts, bornés par le fournisseur. */\n  const g=IMG_GEO[m.family];\n  if(g){\n    const sz=imgSizeOf(m.family, out.format, out.tier);\n    out.width=sz.w; out.height=sz.h;\n  }\n  delete out.format; delete out.tier;\n  return out;\n}\nfunction countParams(id){ return Object.keys(cleanParams(id)).length; }\n\nlet cfgModel=null;\nfunction openSettings(m){\n  cfgModel=m;\n  cfgTitle.innerHTML = MARKS[m.vendor] + esc(m.label);\n  cfgBody.innerHTML=\"\";\n  /* fiche du modèle : description + scores */\n  const card=document.createElement(\"div\"); card.className=\"cfgcard\";\n  card.innerHTML='<p class=\"desc\">'+esc(m.desc||\"\")+\"</p>\"\n    +'<div class=\"stats\">'\n    +(m.gen===\"img\"\n      ? '<span><b>Génération d\\'image</b></span><span>Sortie <b>image</b></span><span><b>Gratuit</b> · '+esc(m.via||\"Workers AI\")+'</span>'\n      : m.gen===\"vid\"\n      ? '<span><b>Génération de vidéo</b></span><span>Sortie <b>vidéo 2 à 6 s</b></span><span><b>Gratuit</b> · '+esc(m.via||\"Wan 2.2\")+'</span>'\n      : m.gen===\"audio\"\n      ? '<span><b>Synthèse vocale</b></span><span>Sortie <b>audio</b></span><span><b>Gratuit</b> · '+esc(m.via||\"Google AI API\")+'</span>'\n      : m.gen===\"embed\"\n      ? '<span><b>Embeddings</b></span><span>Sortie <b>vecteur 3072-d</b></span><span><b>Gratuit</b> · '+esc(m.via||\"Google AI API\")+'</span>'\n      : m.gen===\"transcribe\"\n      ? '<span><b>Transcription</b></span><span>Entrée <b>audio</b></span><span><b>Gratuit</b> · '+esc(m.via||\"Google AI API\")+'</span>'\n      : m.gen===\"live\"\n      ? '<span><b>Session Live</b></span><span>Sortie <b>audio</b></span><span><b>Gratuit</b> · '+esc(m.via||\"Google AI API\")+'</span>'\n      : '<span>Code <b>'+(m.sc?m.sc[0]:\"–\")+'/10</b></span>'\n        +'<span>SVG <b>'+(m.sc?m.sc[1]:\"–\")+'/10</b></span>'\n        +'<span>Texte <b>'+(m.sc?m.sc[2]:\"–\")+'/10</b></span>'\n        +'<span>Vitesse <b>'+(m.sc?m.sc[3]:\"–\")+' TPS</b></span>')\n    +\"</div>\";\n  cfgBody.appendChild(card);\n  /* Ligne vivante « Image de … × … px », tenue à jour par les deux sections. */\n  const geoRow = IMG_GEO[m.family] || null;\n  let sizeEl = null;\n  const updSize = ()=>{\n    if(!sizeEl || !geoRow) return;\n    const sz=imgSizeOf(m.family, SETTINGS[m.id].format, SETTINGS[m.id].tier);\n    sizeEl.textContent = \"Image de \" + sz.w + \" × \" + sz.h + \" px\" + (sz.capped ? \" · plafonné par le fournisseur\" : \"\");\n  };\n  for(const f of CFG_FIELDS[m.family]){\n    if(f.only && f.only!==m.id) continue;      /* réglage propre à un seul modèle */\n    const row=document.createElement(\"div\"); row.className=\"field\";\n    const lab=document.createElement(\"label\");\n    const txt=document.createElement(\"span\"); txt.textContent=f.label;\n    lab.appendChild(txt);\n    let input;\n    if(f.type===\"check\"){\n      input=document.createElement(\"button\");\n      input.type=\"button\"; input.className=\"sw\";\n      input.setAttribute(\"aria-checked\", SETTINGS[m.id][f.k]?\"true\":\"false\");\n      input.setAttribute(\"aria-label\", f.label);\n      input.onclick=()=>{\n        const on = input.getAttribute(\"aria-checked\")!==\"true\";\n        input.setAttribute(\"aria-checked\", on?\"true\":\"false\");\n        SETTINGS[m.id][f.k]=on; saveSettings(m.id);\n      };\n    }else if(f.type===\"format\" || f.type===\"tier\"){\n      /* Deux listes dessinées par l'app : les ratios (avec un cadre à\n         l'échelle) puis les qualités (tailles réelles). */\n      const geo=IMG_GEO[m.family];\n      const wrap=document.createElement(\"div\");\n      wrap.className = \"resrow\" + (f.type===\"format\" ? \" fmtrow\" : \"\");\n      const cur = ()=> SETTINGS[m.id][f.k] || geo.def[f.k];\n      const paint = ()=>{\n        for(const btn of wrap.children) btn.setAttribute(\"aria-pressed\", btn.dataset.v===String(cur()) ? \"true\":\"false\");\n        updSize();\n      };\n      for(const o of geo[f.type===\"format\" ? \"formats\" : \"tiers\"]){\n        const v = (f.type===\"format\") ? o : String(o.v);\n        const btn=document.createElement(\"button\");\n        btn.type=\"button\";\n        btn.className = \"reschip\" + (f.type===\"format\" ? \" fmtchip\" : \"\");\n        btn.dataset.v=v;\n        if(f.type===\"format\"){\n          const ar=v.split(\":\").map(Number);\n          const bw=Math.round(18*Math.min(1.8,ar[0]/ar[1]));\n          const bh=Math.round(18*Math.min(1.8,ar[1]/ar[0]));\n          btn.innerHTML='<span class=\"fmtbox\" style=\"width:'+bw+'px;height:'+bh+'px\"></span><span>'+esc(v)+'</span>';\n        }else{\n          btn.textContent=o.l;\n        }\n        btn.onclick=()=>{\n          SETTINGS[m.id][f.k] = (f.type===\"tier\") ? Number(v) : v;\n          saveSettings(m.id); paint();\n        };\n        wrap.appendChild(btn);\n      }\n      row.appendChild(lab); row.appendChild(wrap);\n      if(f.note){ const n=document.createElement(\"p\"); n.className=\"note\"; n.textContent=f.note; row.appendChild(n); }\n      if(f.type===\"format\"){\n        sizeEl=document.createElement(\"p\"); sizeEl.className=\"imgsize\";\n        row.appendChild(sizeEl); paint();\n      }else{\n        paint();\n      }\n      cfgBody.appendChild(row);\n      continue;\n    }else if(f.type===\"chips\"){\n      /* Liste dessinée par l'app : une pastille par valeur, aucune liste native. */\n      const wrap=document.createElement(\"div\"); wrap.className=\"resrow\";\n      for(const o of f.opts){\n        const b=document.createElement(\"button\");\n        b.type=\"button\"; b.className=\"reschip\";\n        b.textContent=(f.labels&&f.labels[o])||o||\"Défaut\";\n        b.setAttribute(\"aria-pressed\", ((SETTINGS[m.id][f.k] ?? f.def ?? \"\")===o)?\"true\":\"false\");\n        b.onclick=()=>{\n          SETTINGS[m.id][f.k]= o||null; saveSettings(m.id);\n          for(const x of wrap.children) x.setAttribute(\"aria-pressed\", x===b?\"true\":\"false\");\n        };\n        wrap.appendChild(b);\n      }\n      row.appendChild(lab); row.appendChild(wrap);\n      if(f.note){ const n=document.createElement(\"p\"); n.className=\"note\"; n.textContent=f.note; row.appendChild(n); }\n      cfgBody.appendChild(row);\n      continue;\n    }else if(f.type===\"select\"){\n      input=document.createElement(\"select\");\n      const d=document.createElement(\"option\");\n      d.value=\"\"; d.textContent=\"— \"+f.ph+\" —\"; input.appendChild(d);\n      for(const o of f.opts){\n        const op=document.createElement(\"option\"); op.value=o; op.textContent=o; input.appendChild(op);\n      }\n      input.value = SETTINGS[m.id][f.k] || \"\";\n      input.onchange=()=>{ SETTINGS[m.id][f.k]= input.value || null; saveSettings(m.id); };\n    }else{\n      input=document.createElement(\"input\");\n      input.type=\"number\";\n      if(f.min!==undefined) input.min=f.min;\n      if(f.max!==undefined) input.max=f.max;\n      if(f.step!==undefined) input.step=f.step;\n      input.placeholder=f.ph||\"\";\n      const cur=SETTINGS[m.id][f.k];\n      if(cur!==null&&cur!==undefined) input.value=cur;\n      input.onchange=()=>{\n        const raw=input.value.trim();\n        if(raw===\"\"){ SETTINGS[m.id][f.k]=null; saveSettings(m.id); return; }\n        let n=Number(raw);\n        if(!Number.isFinite(n)){ input.value=\"\"; SETTINGS[m.id][f.k]=null; saveSettings(m.id); return; }\n        if(f.min!==undefined) n=Math.max(f.min,n);\n        if(f.max!==undefined) n=Math.min(f.max,n);\n        if(f.type===\"int\") n=Math.trunc(n);\n        input.value=n;\n        SETTINGS[m.id][f.k]=n; saveSettings(m.id);\n      };\n    }\n    lab.appendChild(input);\n    row.appendChild(lab);\n    if(f.note){\n      const n=document.createElement(\"p\"); n.className=\"note\"; n.textContent=f.note;\n      row.appendChild(n);\n    }\n    cfgBody.appendChild(row);\n  }\n  cfg.classList.add(\"open\");\n  document.body.style.overflow=\"hidden\";\n}\nfunction closeSettings(){\n  cfg.classList.remove(\"open\");\n  document.body.style.overflow=\"\";\n  cfgModel=null;\n}\ncfgClose.onclick=closeSettings;\ncfgReset.onclick=()=>{\n  if(!cfgModel) return;\n  SETTINGS[cfgModel.id]={};\n  saveSettings(cfgModel.id);\n  openSettings(cfgModel);\n};\ndocument.addEventListener(\"keydown\", e=>{\n  if(e.key===\"Escape\" && cfg.classList.contains(\"open\")) closeSettings();\n});\n\n/* =====================================================================\n   Chargement d'une image (vision réelle)\n   ===================================================================== */\npendRm.onclick=()=>{ pendingImage=null; pendingAudio=null; renderPending(); };\nfilePick.onchange=async()=>{\n  const f=filePick.files && filePick.files[0];\n  filePick.value=\"\";\n  if(!f) return;\n  const ext=(f.name.split(\".\").pop()||\"\").toLowerCase();\n  const isImg=/^image[/]/.test(f.type)||/^(png|jpe?g|gif|webp|bmp|avif|heic|heif)$/.test(ext);\n  try{\n    if(isImg){\n      /* image certaine → vision */\n      pendingAudio=null;\n      pendingImage = await fileToImage(f);\n    }else{\n      /* tout le reste passe par le canal audio (micro) : MIME audio déclaré,\n         extension audio connue, ou type inconnu/vide (fréquent sur certains\n         systèmes) — le modèle décidera s'il sait le lire. */\n      pendingImage=null;\n      pendingAudio = await fileToAudio(f);\n    }\n  }catch(err){\n    pendingImage=null; pendingAudio=null;\n    alert(\"Fichier illisible : \"+(err && err.message ? err.message : err));\n  }\n  renderPending();\n};\n\n/* Charge un fichier audio tel quel (base64) — sert de « micro » :\n   transcription (REST) et traduction live (WebSocket). */\nfunction fileToAudio(file){\n  return new Promise((resolve,reject)=>{\n    const fr=new FileReader();\n    fr.onload=()=>{\n      const s=String(fr.result||\"\");\n      const i=s.indexOf(\",\");\n      if(i<0) return reject(new Error(\"lecture impossible\"));\n      resolve({name:file.name, b64:s.slice(i+1), kb:Math.max(1,Math.round(file.size/1024)), mime:file.type||\"audio/wav\"});\n    };\n    fr.onerror=()=>reject(new Error(\"lecture impossible\"));\n    fr.readAsDataURL(file);\n  });\n}\n\nfunction renderPending(){\n  if(pendingImage){\n    pendThumb.src=\"data:image/png;base64,\"+pendingImage.b64;\n    pendName.textContent=pendingImage.name+\" · \"+pendingImage.kb+\" Ko\";\n    pending.classList.add(\"on\");\n  }else if(pendingAudio){\n    pendThumb.src=\"data:image/svg+xml,\"+encodeURIComponent('<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"40\" height=\"40\"><rect width=\"40\" height=\"40\" rx=\"8\" fill=\"#111\"/><path d=\"M16 27V13l12-2v12\" fill=\"none\" stroke=\"#fff\" stroke-width=\"2\"/><circle cx=\"14\" cy=\"27\" r=\"3\" fill=\"#fff\"/><circle cx=\"26\" cy=\"23\" r=\"3\" fill=\"#fff\"/></svg>');\n    pendName.textContent=pendingAudio.name+\" · \"+pendingAudio.kb+\" Ko (audio)\";\n    pending.classList.add(\"on\");\n  }else{\n    pending.classList.remove(\"on\");\n    pendThumb.removeAttribute(\"src\");\n  }\n  updateHint();\n}\n\n/* Réduit une image déjà en mémoire à 496 px de côté au plus (référence\n   Cloudflare). Rend une promesse de base64 JPEG. */\nfunction tinyFromB64(b64){\n  return new Promise((res,rej)=>{\n    try{\n      const img=new Image();\n      img.onload=()=>{\n        try{\n          const T=496, k=Math.min(1,T/Math.max(img.naturalWidth||1,img.naturalHeight||1));\n          const w=Math.max(1,Math.round((img.naturalWidth||1)*k)), h=Math.max(1,Math.round((img.naturalHeight||1)*k));\n          const c=document.createElement(\"canvas\"); c.width=w; c.height=h;\n          const x=c.getContext(\"2d\"); x.fillStyle=\"#fff\"; x.fillRect(0,0,w,h);\n          x.drawImage(img,0,0,w,h);\n          res(c.toDataURL(\"image/jpeg\",0.9).split(\",\")[1]);\n        }catch(e){ rej(e); }\n      };\n      img.onerror=()=>rej(new Error(\"image illisible\"));\n      img.src=\"data:image/png;base64,\"+b64;\n    }catch(e){ rej(e); }\n  });\n}\n/* L'image d'ORIGINE est conservée telle quelle (affichage + sauvegarde) :\n   seule une copie réduite (small) part vers les modèles — avant, tout était\n   ré-encodé en 1280 px, d'où la perte de qualité (et les GIF figés). */\nfunction fileToImage(file){\n  return new Promise((resolve,reject)=>{\n    const fr=new FileReader();\n    fr.onload=()=>{\n      const s=String(fr.result||\"\"); const i=s.indexOf(\",\");\n      if(i<0) return reject(new Error(\"lecture impossible\"));\n      const origB64=s.slice(i+1);\n      const mime=file.type||\"image/png\";\n      const url=URL.createObjectURL(file);\n      const img=new Image();\n      img.onload=()=>{\n        URL.revokeObjectURL(url);\n        try{\n          const MAX=1280;\n          let w=img.naturalWidth||img.width, h=img.naturalHeight||img.height;\n          if(!w||!h) return reject(new Error(\"dimensions illisibles\"));\n          if(Math.max(w,h)>MAX){\n            const r=MAX/Math.max(w,h);\n            w=Math.round(w*r); h=Math.round(h*r);\n          }\n          const c=document.createElement(\"canvas\"); c.width=w; c.height=h;\n          const x=c.getContext(\"2d\");\n          if(!x) return reject(new Error(\"canvas indisponible\"));\n          x.fillStyle=\"#fff\"; x.fillRect(0,0,w,h);\n          x.drawImage(img,0,0,w,h);\n          let small=c.toDataURL(\"image/png\").split(\",\")[1];\n          if(small.length>2000000) small=c.toDataURL(\"image/jpeg\",0.9).split(\",\")[1];\n          /* Copie minuscule (496 px max) : Cloudflare refuse les références\n             de 512 px et plus. Elle ne sert qu'à l'édition FLUX.2 klein. */\n          let tiny=\"\";\n          try{\n            const T=496, r2=Math.min(1,T/Math.max(img.naturalWidth||w,img.naturalHeight||h));\n            const tw=Math.max(1,Math.round(w*r2)), th=Math.max(1,Math.round(h*r2));\n            const c2=document.createElement(\"canvas\"); c2.width=tw; c2.height=th;\n            const x2=c2.getContext(\"2d\"); x2.fillStyle=\"#fff\"; x2.fillRect(0,0,tw,th);\n            x2.drawImage(img,0,0,tw,th);\n            tiny=c2.toDataURL(\"image/jpeg\",0.9).split(\",\")[1];\n          }catch(e){ tiny=\"\"; }\n          resolve({name:file.name, b64:origB64, mime:mime, small:small, tiny:tiny,\n            kb:Math.round(origB64.length*0.75/1024)});\n        }catch(err){ reject(err); }\n      };\n      img.onerror=()=>{ URL.revokeObjectURL(url); reject(new Error(\"fichier non reconnu comme image\")); };\n      img.src=url;\n    };\n    fr.onerror=()=>reject(new Error(\"lecture impossible\"));\n    fr.readAsDataURL(file);\n  });\n}\n\n/* --- détection du proxy au démarrage -------------------------------- */\n(function probeProxy(){\n  const warn=$(\"#proxywarn\");\n  fetch(API_BASE+\"/api/models\").then(r=>{\n    if(!r.ok) throw new Error(\"HTTP \"+r.status);\n    warn.hidden=true;\n  }).catch(()=>{ warn.hidden=false; });\n})();\n</script>\n<div id=\"sbackdrop\"></div>\n<aside id=\"sidebar\" aria-label=\"Conversations\">\n  <div class=\"sidespace\"></div>\n  <button id=\"newchat\" type=\"button\">\n    <svg width=\"14\" height=\"14\" viewBox=\"0 0 24 24\" fill=\"none\" stroke=\"currentColor\" stroke-width=\"2\" stroke-linecap=\"round\" aria-hidden=\"true\"><path d=\"M12 5v14M5 12h14\"/></svg>\n    Nouvelle discussion\n  </button>\n  <div id=\"chatlist\" class=\"chatlist\"></div>\n  <div id=\"sideuser\" class=\"sideuser\"></div>\n</aside>\n\n<div id=\"authmodal\" class=\"modal\" hidden>\n  <div class=\"modalcard\">\n    <button type=\"button\" class=\"modalx\" id=\"authclose\" aria-label=\"Fermer\" title=\"Fermer\"><svg width=\"15\" height=\"15\" viewBox=\"0 0 24 24\" fill=\"none\" stroke=\"currentColor\" stroke-width=\"2\" stroke-linecap=\"round\" aria-hidden=\"true\"><path d=\"M6 6l12 12M18 6 6 18\"/></svg></button>\n    <h2 id=\"authtitle\">Se connecter</h2>\n    <button type=\"button\" id=\"googbtn\" class=\"gbtn\">\n      <svg width=\"16\" height=\"16\" viewBox=\"0 0 48 48\" aria-hidden=\"true\"><path fill=\"#EA4335\" d=\"M24 9.5c3.54 0 6.71 1.22 9.21 3.6l6.85-6.85C35.9 2.38 30.47 0 24 0 14.62 0 6.51 5.38 2.56 13.22l7.98 6.19C12.43 13.72 17.74 9.5 24 9.5z\"/><path fill=\"#4285F4\" d=\"M46.98 24.55c0-1.57-.15-3.09-.38-4.55H24v9.02h12.94c-.58 2.96-2.26 5.48-4.78 7.18l7.73 6c4.51-4.18 7.09-10.36 7.09-17.65z\"/><path fill=\"#FBBC05\" d=\"M10.53 28.59c-.48-1.45-.76-2.99-.76-4.59s.27-3.14.76-4.59l-7.98-6.19C.92 16.46 0 20.12 0 24c0 3.88.92 7.54 2.56 10.78l7.97-6.19z\"/><path fill=\"#34A853\" d=\"M24 48c6.48 0 11.93-2.13 15.89-5.81l-7.73-6c-2.15 1.45-4.92 2.3-8.16 2.3-6.26 0-11.57-4.22-13.47-9.91l-7.98 6.19C6.51 42.62 14.62 48 24 48z\"/></svg>\n      Continuer avec Google\n    </button>\n    <div class=\"authsep\"><span>ou</span></div>\n    <label class=\"flab\"><span id=\"authemailLab\">Email ou identifiant</span>\n      <input type=\"text\" id=\"authemail\" autocomplete=\"username\" placeholder=\"Email ou identifiant…\"></label>\n    <label class=\"flab\">Mot de passe\n      <input type=\"password\" id=\"authpass\" autocomplete=\"current-password\" placeholder=\"Mot de passe…\"></label>\n    <button type=\"button\" id=\"authsubmit\" class=\"pbtn\">Se connecter</button>\n    <p class=\"authalt\"><button type=\"button\" id=\"authforgot\" class=\"linkbtn\">Mot de passe oublié ?</button></p>\n    <p class=\"authalt\" id=\"authalt\">Pas encore de compte ? <button type=\"button\" id=\"authtoggle\">Créer un compte</button></p>\n    <div id=\"gpasswrap\" hidden>\n      <p class=\"authalt\">Compte Google connecté ✓ — définis un mot de passe pour pouvoir aussi te connecter par email.</p>\n      <label class=\"flab\">Nouveau mot de passe<input type=\"password\" id=\"gpass1\" autocomplete=\"new-password\"></label>\n      <label class=\"flab\">Confirmer le mot de passe<input type=\"password\" id=\"gpass2\" autocomplete=\"new-password\"></label>\n      <button type=\"button\" id=\"gpassok\" class=\"pbtn\">Définir le mot de passe</button>\n      <p class=\"authalt\"><button type=\"button\" id=\"gpassskip\" class=\"linkbtn\">Plus tard</button></p>\n    </div>\n    <p class=\"autherr\" id=\"autherr\" hidden></p>\n  </div>\n</div>\n\n<div id=\"dispop\" hidden>\n  <div class=\"dispo-card\" role=\"dialog\" aria-modal=\"true\" aria-labelledby=\"dispo-title\">\n    <h1 id=\"dispo-title\"></h1>\n    <p id=\"dispo-msg\"></p>\n    <div class=\"dispo-act\">\n      <button type=\"button\" id=\"dispo-copy\">Copier</button>\n      <button type=\"button\" id=\"dispo-close\" class=\"prim\">Fermer</button>\n    </div>\n  </div>\n</div>\n\n<div id=\"settings\" class=\"modal\" hidden>\n  <div class=\"setcard\">\n    <button type=\"button\" class=\"setbackbtn\" id=\"setback\" title=\"Retour\" aria-label=\"Revenir aux paramètres\" hidden><svg width=\"16\" height=\"16\" viewBox=\"0 0 24 24\" fill=\"none\" stroke=\"currentColor\" stroke-width=\"1.9\" stroke-linecap=\"round\" stroke-linejoin=\"round\" aria-hidden=\"true\"><path d=\"m15 5-7 7 7 7\"/></svg></button>\n    <button type=\"button\" class=\"modalx\" id=\"setclose\" aria-label=\"Fermer\" title=\"Fermer\"><svg width=\"15\" height=\"15\" viewBox=\"0 0 24 24\" fill=\"none\" stroke=\"currentColor\" stroke-width=\"2\" stroke-linecap=\"round\" aria-hidden=\"true\"><path d=\"M6 6l12 12M18 6 6 18\"/></svg></button>\n    <div class=\"setwrap\" id=\"setlist\">\n      <h2 class=\"settitle\">Paramètres</h2>\n      <p class=\"setsub\">Compte, canaux et sécurité</p>\n      <section class=\"setsec setlistsec\">\n        <button type=\"button\" class=\"setrow\" id=\"setgoaccount\">\n          <span class=\"sric\" id=\"setrowavatar\"><svg width=\"18\" height=\"18\" viewBox=\"0 0 24 24\" fill=\"none\" stroke=\"#111113\" stroke-width=\"1.8\" stroke-linecap=\"round\" stroke-linejoin=\"round\" aria-hidden=\"true\"><circle cx=\"12\" cy=\"8.4\" r=\"3.6\"/><path d=\"M5 20c.9-3.6 3.6-5.4 7-5.4s6.1 1.8 7 5.4\"/></svg></span>\n          <span class=\"srtx\"><b>Compte</b></span>\n          <span class=\"sra\"><svg width=\"15\" height=\"15\" viewBox=\"0 0 24 24\" fill=\"none\" stroke=\"currentColor\" stroke-width=\"1.8\" stroke-linecap=\"round\" stroke-linejoin=\"round\" aria-hidden=\"true\"><path d=\"m9 5 7 7-7 7\"/></svg></span>\n        </button>\n        <button type=\"button\" class=\"setrow\" id=\"setgomodels\">\n          <span class=\"sric\"><svg width=\"18\" height=\"18\" viewBox=\"0 0 24 24\" fill=\"none\" stroke=\"#111113\" stroke-width=\"1.8\" stroke-linecap=\"round\" stroke-linejoin=\"round\" aria-hidden=\"true\"><path d=\"M12 3.6 20 8v8l-8 4.4L4 16V8l8-4.4Z\"/><path d=\"M12 12.2 20 8M12 12.2v8.2M12 12.2 4 8\"/></svg></span>\n          <span class=\"srtx\"><b>Modèles</b></span>\n          <span class=\"sra\"><svg width=\"15\" height=\"15\" viewBox=\"0 0 24 24\" fill=\"none\" stroke=\"currentColor\" stroke-width=\"1.8\" stroke-linecap=\"round\" stroke-linejoin=\"round\" aria-hidden=\"true\"><path d=\"m9 5 7 7-7 7\"/></svg></span>\n        </button>\n        <button type=\"button\" class=\"setrow\" id=\"setgotasks\">\n          <span class=\"sric\"><svg width=\"18\" height=\"18\" viewBox=\"0 0 24 24\" fill=\"none\" stroke=\"#111113\" stroke-width=\"1.8\" stroke-linecap=\"round\" stroke-linejoin=\"round\" aria-hidden=\"true\"><circle cx=\"12\" cy=\"12\" r=\"8.4\"/><path d=\"M12 7.6V12l3.1 1.9\"/></svg></span>\n          <span class=\"srtx\"><b>Tâches</b><span>Missions en cours et terminées</span></span>\n          <span class=\"sra\"><svg width=\"15\" height=\"15\" viewBox=\"0 0 24 24\" fill=\"none\" stroke=\"currentColor\" stroke-width=\"1.8\" stroke-linecap=\"round\" stroke-linejoin=\"round\" aria-hidden=\"true\"><path d=\"m9 5 7 7-7 7\"/></svg></span>\n        </button>\n        <button type=\"button\" class=\"setrow\" id=\"setgodispo\">\n          <span class=\"sric\"><svg width=\"18\" height=\"18\" viewBox=\"0 0 24 24\" fill=\"none\" stroke=\"#111113\" stroke-width=\"1.8\" stroke-linecap=\"round\" stroke-linejoin=\"round\" aria-hidden=\"true\"><path d=\"M12 3.4 5.2 6v5.4c0 4 2.9 7.3 6.8 8.6 3.9-1.3 6.8-4.6 6.8-8.6V6L12 3.4Z\"/><path d=\"m9 11.8 2.1 2.1 4-4.2\"/></svg></span>\n          <span class=\"srtx\"><b>Disponibilité</b><span>Les secrets, testés pour de vrai</span></span>\n          <span class=\"sra\"><svg width=\"15\" height=\"15\" viewBox=\"0 0 24 24\" fill=\"none\" stroke=\"currentColor\" stroke-width=\"1.8\" stroke-linecap=\"round\" stroke-linejoin=\"round\" aria-hidden=\"true\"><path d=\"m9 5 7 7-7 7\"/></svg></span>\n        </button>\n        <button type=\"button\" class=\"setrow\" id=\"setgocanal\">\n          <span class=\"sric\"><svg width=\"18\" height=\"18\" viewBox=\"0 0 24 24\" fill=\"none\" stroke=\"#111113\" stroke-width=\"1.8\" stroke-linecap=\"round\" stroke-linejoin=\"round\" aria-hidden=\"true\"><path d=\"M6.4 7.2h11.2M17.6 7.2 6.4 16.8M6.4 16.8h11.2\"/><circle cx=\"3.4\" cy=\"7.2\" r=\"1.6\"/><circle cx=\"20.6\" cy=\"16.8\" r=\"1.6\"/></svg></span>\n          <span class=\"srtx\"><b>Z-Canal</b></span>\n          <span class=\"sra\"><svg width=\"15\" height=\"15\" viewBox=\"0 0 24 24\" fill=\"none\" stroke=\"currentColor\" stroke-width=\"1.8\" stroke-linecap=\"round\" stroke-linejoin=\"round\" aria-hidden=\"true\"><path d=\"m9 5 7 7-7 7\"/></svg></span>\n        </button>\n      </section>\n      <p class=\"setver\">Version P112</p>\n    </div>\n    <div class=\"setwrap\" id=\"setcompte\" hidden>\n      <div id=\"setavatar\" class=\"setavatar\"></div>\n      <button type=\"button\" class=\"linkbtn\" id=\"setphoto\">Changer la photo de profil</button>\n      <input type=\"file\" id=\"setphotofile\" accept=\"image/*\" hidden>\n      <h2 class=\"settitle\" id=\"settitle\"></h2>\n      <p class=\"setsub\" id=\"setsub\"></p>\n      <section class=\"setsec\">\n        <h3>Profil</h3>\n        <div class=\"setrowline\"><span class=\"setlab\">Nom</span><span class=\"setval\" id=\"setnameval\"></span><button type=\"button\" class=\"setact\" id=\"setnamebtn\">Renommer</button></div>\n        <div class=\"setrowline\"><span class=\"setlab\">Email</span><span class=\"setval\" id=\"setemailval\"></span><button type=\"button\" class=\"setact\" id=\"setemailbtn\">Modifier</button></div>\n        <div class=\"setrowline\"><span class=\"setlab\">Mot de passe</span><span class=\"setval\">••••••••</span><button type=\"button\" class=\"setact\" id=\"setpassbtn\">Changer</button></div>\n      </section>\n      <section class=\"setsec\">\n        <h3>Compte Google</h3>\n        <button type=\"button\" class=\"dbtn wide\" id=\"setgoogle\">Lier un compte Google</button>\n        <p class=\"setnote\" id=\"setgooglenote\"></p>\n      </section>\n      <section class=\"setsec\">\n        <h3>Session</h3>\n        <button type=\"button\" class=\"dbtn wide\" id=\"setsignout\">Se déconnecter</button>\n      </section>\n      <section class=\"setsec\">\n        <h3>Zone dangereuse</h3>\n        <button type=\"button\" class=\"dangerbtn wide\" id=\"setdelete\">Supprimer définitivement mon compte</button>\n      </section>\n      <p class=\"autherr\" id=\"seterr\" hidden></p>\n    </div>\n    <div class=\"setwrap\" id=\"setmodels\" hidden>\n      <h2 class=\"settitle\">Modèles</h2>\n      <p class=\"setsub\">Chaque modèle qui a ses propres réglages a sa ligne. Ouvre-la pour les régler.</p>\n      <section class=\"setsec setlistsec\">\n        <button type=\"button\" class=\"setrow\" id=\"setgowan\">\n          <span class=\"sric\"><svg width=\"18\" height=\"18\" viewBox=\"0 0 24 24\" fill=\"none\" stroke=\"#111113\" stroke-width=\"1.8\" stroke-linecap=\"round\" stroke-linejoin=\"round\" aria-hidden=\"true\"><rect x=\"3.2\" y=\"6.4\" width=\"12.6\" height=\"11.2\" rx=\"2.6\"/><path d=\"m15.8 12 5-3.2v6.4l-5-3.2Z\"/></svg></span>\n          <span class=\"srtx\"><b>Wan 2.2</b><span>Image de départ · quota Hugging Face</span></span>\n          <span class=\"sra\"><svg width=\"15\" height=\"15\" viewBox=\"0 0 24 24\" fill=\"none\" stroke=\"currentColor\" stroke-width=\"1.8\" stroke-linecap=\"round\" stroke-linejoin=\"round\" aria-hidden=\"true\"><path d=\"m9 5 7 7-7 7\"/></svg></span>\n        </button>\n      </section>\n    </div>\n    <div class=\"setwrap\" id=\"setwan\" hidden>\n      <h2 class=\"settitle\">Wan 2.2</h2>\n      <p class=\"setsub\">Le modèle qui anime l’image, et le quota qui le laisse tourner.</p>\n\n      <section class=\"setsec\">\n        <h3>Image de départ</h3>\n        <p class=\"setnote\">Wan 2.2 ne part pas du texte : il part d’une image, puis il la fait bouger. Le modèle choisi ici fabrique cette image de départ quand tu n’en joins pas une. FLUX.2 [klein] 4B est le plus rapide et ses images s’animent très bien — c’est le réglage par défaut.</p>\n        <div class=\"mopts\" id=\"wanposterlist\" role=\"listbox\" aria-label=\"Modèle de l’image de départ\"></div>\n        <p class=\"setnote\" id=\"wanposternote\" style=\"margin-top:14px\"></p>\n      </section>\n\n      <section class=\"setsec\">\n        <h3>Quota Hugging Face <span class=\"ctag\" id=\"hftag\"></span></h3>\n        <p class=\"setnote\">La vidéo gratuite tourne sur ZeroGPU, qui donne environ deux minutes de GPU par jour et par adresse IP — soit une seule vidéo. Un jeton Hugging Face, même sur un compte gratuit, augmente nettement ce quota. Le jeton reste sur cet appareil : il n’est envoyé qu’au Space qui anime la vidéo, jamais enregistré ailleurs.</p>\n        <input type=\"password\" class=\"setinput\" id=\"hftoken\" placeholder=\"hf_… (facultatif)\" autocomplete=\"off\" spellcheck=\"false\">\n        <p class=\"setnote\" id=\"hftokenote\" style=\"margin-top:10px\"></p>\n      </section>\n    </div>\n    <div class=\"setwrap\" id=\"settasks\" hidden>\n      <h2 class=\"settitle\">Tâches</h2>\n      <p class=\"setsub\">Ce qui tourne sur la passerelle, même quand l’application est fermée. Chaque mission dit son modèle, ce qu’on lui a demandé, et depuis combien de temps elle vit.</p>\n      <section class=\"setsec\">\n        <h3>Missions</h3>\n        <div id=\"tasklist\"><p class=\"setnote\">Lecture…</p></div>\n        <p class=\"setnote\" id=\"taskstamp\" style=\"margin-top:14px\"></p>\n        <button type=\"button\" class=\"linkbtn\" id=\"taskrefresh\">Rafraîchir</button>\n      </section>\n    </div>\n    <div class=\"setwrap\" id=\"setdispo\" hidden>\n      <h2 class=\"settitle\">Disponibilité</h2>\n      <p class=\"setsub\">Les secrets du relais, groupés par API. Chaque ▶ lance un vrai appel avec la clé du worker : la réponse dit si elle fonctionne encore.</p>\n      <section class=\"setsec\" id=\"dissecs\">\n        <h3>Secrets</h3>\n        <div id=\"dislist\"><p class=\"setnote\">Lecture…</p></div>\n        <p class=\"setnote\" id=\"distamp\" style=\"margin-top:14px\"></p>\n        <button type=\"button\" class=\"linkbtn\" id=\"distestall\" style=\"margin-top:6px\">Tout tester</button>\n      </section>\n      <section class=\"setsec\" id=\"discredsec\">\n        <h3>Crédits</h3>\n        <div id=\"discredits\"><p class=\"setnote\">Lecture…</p></div>\n        <p class=\"setnote\" id=\"discredstamp\" style=\"margin-top:10px\"></p>\n      </section>\n    </div>\n    <div class=\"setwrap\" id=\"setcanal\" hidden>\n      <h2 class=\"settitle\">Z-Canal</h2>\n      <p class=\"setsub\">Un canal relie un agent — n'importe lequel — à l'application : il reçoit vos messages, répond, et si son mode est « agent », exécute des commandes sur la VM.</p>\n\n      <section class=\"setsec\">\n        <h3>Mes canaux <span class=\"ctag\" id=\"setcanalcount\"></span></h3>\n        <div class=\"canlist\" id=\"canallist\"></div>\n      </section>\n\n      <section class=\"setsec\">\n        <h3>Installation</h3>\n        <p class=\"setnote\">Chaque canal a sa propre adresse. L'agent s'y connecte avec l'identifiant du canal : aucun mot de passe à créer.</p>\n        <div class=\"urlbox\">\n          <span class=\"urlhead\"><span class=\"urlab\">Adresse des canaux</span>\n            <button type=\"button\" class=\"copybtn\" id=\"canalbasecopy\" title=\"Copier l'adresse\" aria-label=\"Copier l'adresse\"></button></span>\n          <code class=\"urlval\" id=\"canalbase\"></code>\n        </div>\n        <div class=\"urlbox\">\n          <span class=\"urlhead\"><span class=\"urlab\">Installer l'agent d'un canal (Python)</span>\n            <button type=\"button\" class=\"copybtn\" id=\"canalinstcopy\" title=\"Copier la commande\" aria-label=\"Copier la commande\"></button></span>\n          <code class=\"urlval\" id=\"canalinstcmd\"></code>\n        </div>\n        <div class=\"urlbox\">\n          <span class=\"urlhead\"><span class=\"urlab\">L'adresse d'un canal</span>\n            <button type=\"button\" class=\"copybtn\" id=\"canaladdrcopy\" title=\"Copier l'adresse du canal\" aria-label=\"Copier l'adresse du canal\"></button></span>\n          <code class=\"urlval\" id=\"canaladdr\"></code>\n        </div>\n        <p class=\"setnote\">Mêmes commandes pour <b>js</b>, <b>sh</b> et <b>ts</b> (<code>format=js</code>…) ; avec <code>doc=1</code> vous obtenez la documentation du canal, et sans <code>format</code> les instructions d'installation complètes. Chaque canal a son adresse complète : <b id=\"setcanalcount2\"></b>.</p>\n      </section>\n    </div>\n<div id=\"setpop\" class=\"setpop\" hidden>\n      <div class=\"setpopcard\">\n        <h3 id=\"setpoptitle\"></h3>\n        <div id=\"setpopbody\"></div>\n        <div class=\"setpoprow\">\n          <button type=\"button\" class=\"dbtn\" id=\"setpopcancel\">Annuler</button>\n          <button type=\"button\" class=\"pbtn\" id=\"setpopok\">Valider</button>\n        </div>\n        <p class=\"autherr\" id=\"setpoperr\" hidden></p>\n      </div>\n    </div>\n\n\n  </div>\n</div>\n\n<div id=\"canaldel\" hidden>\n  <div class=\"delcard\">\n    <h3>Supprimer ce canal</h3>\n    <p>Voulez-vous vraiment supprimer ce canal ? Il sera supprimé définitivement.</p>\n    <div class=\"delrow\">\n      <button type=\"button\" class=\"delcancel\" id=\"canaldelcancel\">Annuler</button>\n      <button type=\"button\" class=\"delok\" id=\"canaldelok\">Supprimer</button>\n    </div>\n  </div>\n</div>\n\n<div id=\"newpop\" hidden>\n  <div class=\"npc\">\n    <button type=\"button\" class=\"modalx\" id=\"newpclose\" aria-label=\"Fermer\" title=\"Fermer\"><svg width=\"15\" height=\"15\" viewBox=\"0 0 24 24\" fill=\"none\" stroke=\"currentColor\" stroke-width=\"2\" stroke-linecap=\"round\" aria-hidden=\"true\"><path d=\"M6 6l12 12M18 6 6 18\"/></svg></button>\n    <h3>Nouvelle discussion</h3>\n    <div id=\"npmodes\">\n      <p class=\"npd\">Choisissez le mode de cette discussion : il ne changera plus.</p>\n      <div class=\"modegrid\">\n        <button type=\"button\" class=\"modepick\" id=\"pickchat\"><svg width=\"26\" height=\"26\" viewBox=\"0 0 24 24\" fill=\"none\" stroke=\"#111113\" stroke-width=\"1.7\" stroke-linecap=\"round\" stroke-linejoin=\"round\" aria-hidden=\"true\"><path d=\"M20.5 12.4c0 4-3.8 7.2-8.5 7.2-1 0-2-.15-2.9-.43L4.2 20.6l1.3-3.5A6.9 6.9 0 0 1 3.5 12.4C3.5 8.4 7.3 5.2 12 5.2s8.5 3.2 8.5 7.2Z\"/><path d=\"M8.6 12.3h.01M12 12.3h.01M15.4 12.3h.01\"/></svg>Chat</button>\n        <button type=\"button\" class=\"modepick\" id=\"pickagent\"><svg width=\"26\" height=\"26\" viewBox=\"0 0 24 24\" fill=\"none\" stroke=\"#111113\" stroke-width=\"1.7\" stroke-linecap=\"round\" stroke-linejoin=\"round\" aria-hidden=\"true\"><rect x=\"3.2\" y=\"7.6\" width=\"17.6\" height=\"12.4\" rx=\"3\"/><path d=\"M12 3.4v4.2M8.6 12.6l-1.4 1.6 1.4 1.6M15.4 12.6l1.4 1.6-1.4 1.6\"/></svg>Agent</button>\n        <button type=\"button\" class=\"modepick\" id=\"pickcanal\"><svg width=\"26\" height=\"26\" viewBox=\"0 0 24 24\" fill=\"none\" stroke=\"#111113\" stroke-width=\"1.7\" stroke-linecap=\"round\" stroke-linejoin=\"round\" aria-hidden=\"true\"><path d=\"M6.4 7.2h11.2M17.6 7.2 6.4 16.8M6.4 16.8h11.2\"/><circle cx=\"3.4\" cy=\"7.2\" r=\"1.6\"/><circle cx=\"20.6\" cy=\"16.8\" r=\"1.6\"/></svg>Z-Canal</button>\n      </div>\n      <p class=\"npd\" id=\"npconn\" hidden>Connectez-vous : les modes « agent » et « Z-Canal » s'enregistrent sur votre compte.</p>\n    </div>\n    <div id=\"npcanalform\" hidden>\n      <p class=\"npd\">Choisissez le canal de cette discussion. Deux discussions peuvent partager le même canal : elles sont alors identiques.</p>\n      <div class=\"canlist canpick\" id=\"canalpick\"></div>\n      <div class=\"npbtns\">\n        <button type=\"button\" class=\"npbtn\" id=\"npback\">Retour</button>\n        <button type=\"button\" class=\"npbtn\" id=\"npcancel\">Annuler</button>\n      </div>\n    </div>\n    <div id=\"npcanalnew\" hidden>\n      <p class=\"npd\">Un canal relie un agent — n'importe lequel — à l'application. Son identifiant (12 caractères) est la clé de cet agent : aucun mot de passe.</p>\n      <label class=\"flab\">Nom du canal<input id=\"canalname\" placeholder=\"Mon agent\" maxlength=\"40\" autocomplete=\"off\"></label>\n      <button type=\"button\" class=\"pbtn\" id=\"canalcreate\">Créer le canal</button>\n      <p class=\"autherr\" id=\"canalerr\" hidden></p>\n      <div class=\"npbtns\">\n        <button type=\"button\" class=\"npbtn\" id=\"npbacknew\">Retour</button>\n        <button type=\"button\" class=\"npbtn\" id=\"npcancel2\">Annuler</button>\n      </div>\n    </div>\n    <div id=\"npcanalmode\" hidden>\n      <p class=\"npd\"></p>\n      <div class=\"setpoprow\">\n        <button type=\"button\" class=\"dbtn\" id=\"canalchat\">Chat</button>\n        <button type=\"button\" class=\"pbtn\" id=\"canalagent\">Agent</button>\n      </div>\n    </div>\n  </div>\n</div>\n<script>\n/* =====================================================================\n   Phase 32 — Comptes, discussions et sauvegarde (Firebase)\n   Auth Google ou email+mot de passe · Firestore (users, chats, messages)\n   Réglages plein écran · pièces jointes multiples\n   ===================================================================== */\nconst FIREBASE_CONFIG = {\n  apiKey: \"AIzaSyC58J1paO0ULhTW0RuDfUePVeuOUmG5fZY\",\n  authDomain: \"ai-agent-1234a.firebaseapp.com\",\n  projectId: \"ai-agent-1234a\",\n  storageBucket: \"ai-agent-1234a.firebasestorage.app\",\n  messagingSenderId: \"301276637514\",\n  appId: \"1:301276637514:web:6d782f12d5a86369d079b3\",\n  measurementId: \"G-EZ6432PZS5\"\n};\n\nlet fbAuth=null, fbDB=null, fbUser=null, FB_OK=false;\nlet chats=[];            /* [{id,title,model,updatedAt}]                     */\nlet chatId=null;         /* discussion courante (null = pas encore enregistrée) */\nlet msgSeq=0;\nlet pendingFiles=[];     /* pièces jointes du message en préparation          */\nlet userName=\"\", userPhoto=\"\", userColor=\"#555\";\nlet authMode=\"login\", chatPop=null;\n\nconst $fb=(s)=>document.querySelector(s);\n\n/* --- Firebase : initialisation différée (SDK en defer) ---------------- */\nfunction initFirebase(){\n  if(!window.firebase || !FIREBASE_CONFIG.apiKey) return;   /* mode dégradé */\n  try{\n    firebase.initializeApp(FIREBASE_CONFIG);\n    fbAuth=firebase.auth();\n    fbDB=firebase.firestore();\n    FB_OK=true;\n  }catch(e){ console.warn(\"Firebase :\", e); return; }\n  /* reste connecté entre deux visites de l'app */\n  fbAuth.setPersistence(firebase.auth.Auth.Persistence.LOCAL).catch(()=>{});\n  fbAuth.onAuthStateChanged(async(u)=>{\n    /* Compte unique autorisé : tout autre compte est refusé poliment. */\n    if(u && String(u.email||\"\").toLowerCase()!==\"hugdu77777@gmail.com\"){\n      showAuthErr(\"L'IA n'est pas disponible pour vous pour l'instant !\");\n      try{ await fbAuth.signOut(); }catch(e){}\n      return;\n    }\n    fbUser=u;\n    if(u){\n      $fb(\"#authmodal\").hidden=true;\n      await ensureUserDoc();\n      await loadChats();\n      /* Appel quotidien ModelScope : lancé tout seul, sans jamais retarder\n         l'app (il tourne en arrière-plan et se contente d'enregistrer). */\n      msDailyRun().catch(()=>{});\n      /* Canal Arena : écoute en temps réel + présence signalée au canal.\n         Protégé : un souci de canal ne doit jamais gêner l'application. */\n      try{ if(arenaWatchSafe()) arenaHeartbeat(); }catch(e){ console.warn(\"canal arena :\",e); }\n    }else{\n      chats=[]; chatId=null; msgSeq=0; userName=\"\"; userPhoto=\"\";\n      $fb(\"#authmodal\").hidden=false;\n      if(typeof agtSyncUI===\"function\") agtSyncUI();\n    }\n    renderSideUser(); renderChatList();\n    if(typeof agtSyncUI===\"function\") agtSyncUI();\n  });\n}\ndocument.addEventListener(\"DOMContentLoaded\", ()=>{\n  initFirebase();\n  renderSideUser(); renderChatList();\n});\n\n/* --- barre latérale : burger ⇄ croix ---------------------------------- */\nfunction openSide(){\n  $fb(\"#sidebar\").classList.add(\"open\"); $fb(\"#sbackdrop\").classList.add(\"on\");\n  $fb(\"#burger\").classList.add(\"open\"); $fb(\"#burger\").setAttribute(\"aria-expanded\",\"true\");\n}\nfunction closeSide(){\n  $fb(\"#sidebar\").classList.remove(\"open\"); $fb(\"#sbackdrop\").classList.remove(\"on\");\n  $fb(\"#burger\").classList.remove(\"open\"); $fb(\"#burger\").setAttribute(\"aria-expanded\",\"false\");\n  closeChatPop();\n}\ndocument.addEventListener(\"keydown\",(e)=>{\n  if(e.key===\"Escape\"){ closeSide(); $fb(\"#authmodal\").hidden=true; $fb(\"#settings\").hidden=true; }\n});\n\n/* --- liste des discussions + menu ••• ---------------------------------- */\nfunction renderChatList(){\n  const list=$fb(\"#chatlist\");\n  list.innerHTML=\"\";\n  const note=document.createElement(\"p\");\n  note.style.cssText=\"font-size:12.5px;color:var(--ink-3);padding:12px 10px;line-height:1.5\";\n  if(!fbUser){ note.textContent=\"Connecte-toi pour enregistrer tes discussions et les retrouver sur tous tes appareils.\"; list.appendChild(note); return; }\n  /* Les discussions écartées (doublons de l'ancien modèle « arena ») ne\n     s'affichent pas : il n'existe qu'une seule discussion « Arena ». */\n  const shown=chats.filter(c=>!c.hidden);\n  if(!shown.length){ note.textContent=\"Aucune discussion enregistrée pour le moment.\"; list.appendChild(note); return; }\n  /* La discussion « Arena » est TOUJOURS en tête, épinglée, avec sa marque. */\n  const ordered=shown.slice().sort((a,b)=>(b.arena?1:0)-(a.arena?1:0));\n  for(const c of ordered){\n    const row=document.createElement(\"div\");\n    row.className=\"chitem\"+(c.id===chatId?\" sel\":\"\")+(c.arena?\" pinned\":\"\");\n    if(c.arena){\n      const mk=document.createElement(\"span\");\n      mk.className=\"chmark\"; mk.title=\"Discussion Arena · canal direct\";\n      mk.innerHTML=MARKS.arena;\n      row.appendChild(mk);\n    }\n    const t=document.createElement(\"button\");\n    t.type=\"button\"; t.className=\"cht\"; t.textContent=c.title; t.title=c.title;\n    t.onclick=()=>{ if(c.id!==chatId) loadChat(c.id); else closeSide(); };\n    row.appendChild(t);\n    if(c.arena&&ARENA_DOT){\n      const dot=document.createElement(\"span\");\n      dot.className=\"chdot\"; dot.title=\"Nouveau message d'Arena\";\n      row.appendChild(dot);\n    }\n    if(c.agent){\n      const badge=document.createElement(\"span\");\n      badge.className=\"chbadge\"; badge.textContent=\"Agent\"; badge.title=\"Discussion en mode agent\";\n      row.appendChild(badge);\n    }\n    if(c.job&&c.job.state===\"run\"){\n      const jb=document.createElement(\"span\");\n      jb.className=\"chjob\"; jb.title=\"Mission en cours : elle avance toute seule, même app fermée\";\n      jb.innerHTML=\"<i></i>en cours\";\n      row.appendChild(jb);\n    }\n    const more=document.createElement(\"button\");\n    more.type=\"button\"; more.className=\"chmore\"; more.textContent=\"•••\";\n    more.setAttribute(\"aria-label\",\"Options : \"+c.title);\n    more.onclick=(e)=>{ e.stopPropagation(); openChatPop(more, c); };\n    row.appendChild(more);\n    list.appendChild(row);\n  }\n}\nfunction closeChatPop(){ if(chatPop){ chatPop.remove(); chatPop=null; } }\nfunction openChatPop(anchor, c){\n  closeChatPop();\n  const pop=document.createElement(\"div\"); pop.className=\"chpop\";\n  const ren=document.createElement(\"button\"); ren.type=\"button\"; ren.textContent=\"Renommer\";\n  ren.onclick=()=>{ closeChatPop(); renameChat(c); };\n  const del=document.createElement(\"button\"); del.type=\"button\"; del.className=\"del\"; del.textContent=\"Supprimer\";\n  del.onclick=()=>{ closeChatPop(); deleteChat(c); };\n  pop.appendChild(ren); pop.appendChild(del);\n  document.body.appendChild(pop);\n  const r=anchor.getBoundingClientRect();\n  pop.style.top=Math.max(8, Math.min(r.bottom+4, innerHeight-pop.offsetHeight-8))+\"px\";\n  pop.style.left=Math.max(8, Math.min(r.right-pop.offsetWidth, innerWidth-pop.offsetWidth-8))+\"px\";\n  chatPop=pop;\n  setTimeout(()=>document.addEventListener(\"click\", ()=>closeChatPop(), {once:true}), 0);\n}\nfunction renameChat(c){\n  const t=prompt(\"Nouveau nom de la discussion :\", c.title);\n  if(t==null) return;\n  const nt=t.trim().slice(0,60); if(!nt) return;\n  c.title=nt; renderChatList();\n  if(FB_OK&&fbUser) chatsRef().doc(c.id).update({title:nt, updatedAt:Date.now()}).catch(e=>console.warn(e));\n}\nasync function purgeChat(id){\n  const mref=fbDB.collection(\"users\").doc(fbUser.uid).collection(\"chats\").doc(id).collection(\"messages\");\n  const snap=await mref.get();\n  let batch=fbDB.batch(), n=0;\n  /* Les morceaux (sous-collection « parts ») partent avec le message :\n     Firestore ne supprime pas les sous-collections tout seul. */\n  for(const d of snap.docs){\n    if(d.data() && d.data().parts){\n      const ps=await d.ref.collection(\"parts\").get();\n      for(const q of ps.docs){ batch.delete(q.ref); if(++n%400===0){ await batch.commit(); batch=fbDB.batch(); } }\n    }\n    batch.delete(d.ref); if(++n%400===0){ await batch.commit(); batch=fbDB.batch(); }\n  }\n  await batch.commit();\n  await chatsRef().doc(id).delete();\n}\nasync function deleteChat(c){\n  if(!confirm(\"Supprimer définitivement « \"+c.title+\" » ?\")) return;\n  chats=chats.filter(x=>x.id!==c.id);\n  if(chatId===c.id){ chatId=null; msgSeq=0; }\n  renderChatList();\n  if(FB_OK&&fbUser){ try{ await purgeChat(c.id); }catch(e){ console.warn(\"delete :\", e); } }\n}\nfunction chatsRef(){ return fbDB.collection(\"users\").doc(fbUser.uid).collection(\"chats\"); }\n\n/* --- nouvelle discussion / chargement ---------------------------------- */\nfunction onNewChat(){\n  /* Une mission affichée ici continue en arrière-plan : elle a sa discussion. */\n  if(AGT_JOB_OWN) agtDetachNow().catch(()=>{});\n  chatId=null; msgSeq=0; renderChatList();\n}\nasync function loadChats(){\n  if(!FB_OK||!fbUser) return;\n  try{\n    const snap=await chatsRef().orderBy(\"updatedAt\",\"desc\").limit(100).get();\n    chats=snap.docs.map(d=>Object.assign({id:d.id}, d.data()));\n  }catch(e){ chats=[]; console.warn(\"chats :\", e); }\n  /* Les missions en cours connues de CET appareil : le marqueur Firestore fait\n     foi (il est visible de partout), mais une écriture qui a échoué ne doit pas\n     faire disparaître une mission du registre. */\n  try{\n    const reg=agtJobsLoad();\n    for(const c of chats){\n      const l=reg[c.id];\n      if(l&&(!c.job||!c.job.state)) c.job={id:l.id, state:\"run\", at:l.at, mission:l.mission||\"\", model:l.model||\"\"};\n    }\n  }catch(e){}\n  /* La discussion « Arena » existe toujours, et une seule : elle est créée ici\n     si besoin, et les doublons éventuels sont écartés de la liste. */\n  try{ const a=await arenaChat(); if(a) await arenaDedupe(a); }\n  catch(e){ console.warn(\"canal arena :\", e); }\n  /* Réparation : une discussion de canal masquée par l'ancien dédoublonnage\n     revient dans la liste (elle n'a jamais été un doublon). */\n  for(const c of chats){\n    if(!c.canal || c.hidden!==true) continue;\n    c.hidden=false;\n    try{ await chatsRef().doc(c.id).set({hidden:false},{merge:true}); }catch(e){}\n  }\n  /* Les canaux du compte, puis l'écoute de ceux qui sont reliés à une discussion. */\n  if(typeof canalLoad===\"function\"){\n    try{ await canalLoad(); canalWatchAll(); }catch(e){ console.warn(\"canaux :\", e); }\n  }\n  renderChatList();\n}\n/* ================= l'historique d'une discussion de canal ================\n   Une discussion de canal n'enregistre RIEN à elle : ni ses messages, ni des\n   copies. Tout vit dans le canal (Firebase), et la discussion s'ouvre depuis\n   lui. Elle se retrouve donc entière sur n'importe quel appareil, même après\n   une fermeture complète de l'application — et deux discussions du même canal\n   racontent forcément la même chose. */\nconst CANAL_HIST = {};          /* dernier historique lu, par canal */\nasync function canalHistoryLoad(cid){\n  const ref = canalMsgsOf(cid);\n  if (!ref) return 0;\n  /* Le même historique est relu à chaque ouverture : on le garde en mémoire\n     trois minutes (les nouveaux messages, eux, arrivent en direct). */\n  let msgs = null;\n  const c = CANAL_HIST[cid];\n  if (c && Date.now() - c.at < 180000) msgs = c.msgs;\n  if (!msgs) {\n    let snap = null;\n    try { snap = await ref.orderBy(\"ts\", \"asc\").limit(120).get(); }\n    catch (e) { console.warn(\"canal (historique) :\", e); return 0; }\n    msgs = snap.docs.map((d) => Object.assign({}, d.data() || {}));\n    CANAL_HIST[cid] = { at: Date.now(), msgs: msgs };\n  }\n  /* UN message par échange : les commandes et leurs sorties s'empilent dans la\n     même bulle, et le texte de l'agent la termine — comme en direct. */\n  let n = 0, cur = null;\n  const ouvrir = function () {\n    if (cur) return cur;\n    const b = addAssistant(\"arena\");\n    /* addAssistant pose l'indicateur « Generating… » : il ne fait pas partie du\n       message et ne doit pas revenir à l'ouverture (c'est ce qui rallumait\n       l'animation sur tous les messages avec une commande). */\n    const sp = b.querySelector(\".gentxt\"); if (sp) sp.remove();\n    const built = agtTermEl(b, null, true);\n    cur = { bubble: b, log: built.log };\n    return cur;\n  };\n  for (const m of msgs) {\n    const txt = String(m.text || \"\");\n    if (m.role === \"user\") {\n      if (!txt) continue;\n      addUser(txt, \"none\", null, null);\n      conv.push({ role: \"user\", text: txt });\n      n++; cur = null; continue;\n    }\n    if (m.role === \"agent\") {\n      const p = arenaParse(txt);\n      const act = p && p.call ? String(p.call.action || \"\").toLowerCase() : \"\";\n      if (p.kind === \"call\" && p.call && act !== \"final\") {\n        const kind = (act === \"fetch\") ? \"fetch\" : \"bash\";\n        const cmd = (kind === \"fetch\") ? String(p.call.url || \"\") : String(p.call.command || \"\");\n        agtStepEl(ouvrir().log, cmd, kind);\n        n++; continue;\n      }\n      const out = (p && p.text) ? p.text : (p.kind === \"call\" ? \"\" : txt);\n      if (!out) continue;\n      if (cur) {\n        const fin = document.createElement(\"div\"); fin.className = \"agent-final\";\n        fin.innerHTML = renderMarkdown(out);\n        cur.bubble.appendChild(fin);\n        msgActions(cur.bubble, { getMd: () => out });\n      } else {\n        const b = addAssistant(\"arena\");\n        b.innerHTML = renderMarkdown(out);\n        msgActions(b, { getMd: () => out });\n      }\n      conv.push({ role: \"assistant\", text: out });\n      n++; cur = null; continue;\n    }\n    if (m.role === \"tool\" && txt) {\n      /* La sortie d'une commande se range dans la bulle de l'échange. */\n      const lg = ouvrir().log;\n      const lines = txt.split(\"\\n\");\n      let cmd = \"\", exit = null, out = txt;\n      if (/^\\$ /.test(lines[0] || \"\")) {\n        cmd = lines[0].slice(2);\n        let i = 1;\n        const ex = /^\\[exit\\]\\s*(.*)$/.exec(lines[i] || \"\");\n        if (ex) { exit = ex[1]; i++; }\n        out = lines.slice(i).join(\"\\n\");\n      }\n      if (cmd) agtStepEl(lg, cmd, \"bash\");\n      if (typeof agtOutEl === \"function\") agtOutEl(lg, out, \"\", (exit === null ? 0 : (Number(exit) || 0)));\n      else { const dv = document.createElement(\"div\"); dv.className = \"agent-out\"; dv.textContent = out; lg.appendChild(dv); }\n      n++;\n    }\n  }\n  return n;\n}\nasync function loadChat(id){\n  if(!FB_OK||!fbUser) return;\n  const c=chats.find(x=>x.id===id); if(!c) return;\n  /* On quitte la discussion où une mission s'affichait : la mission continue\n     toute seule sur la passerelle (et reste marquée « en cours »). */\n  if(AGT_JOB_OWN&&AGT_JOB_CHAT&&AGT_JOB_CHAT!==id){\n    try{ await agtDetachNow(); }catch(e){}\n  }\n  closeChatPop();\n  try{\n    /* Une discussion de canal n'a pas de messages à elle : inutile d'aller en\n       chercher (une lecture économisée à chaque ouverture). */\n    const parCanal=!!(c.canal && c.canal!==\"arena\");\n    const snap=parCanal?{docs:[],size:0}:await chatsRef().doc(id).collection(\"messages\").orderBy(\"order\").get();\n    const mref=chatsRef().doc(id).collection(\"messages\");\n    conv=[]; log.innerHTML=\"\"; chatId=id; msgSeq=0;\n  CANAL_MISSION=null; ARENA_WAIT=null;\n    /* Ouvrir la discussion « Arena » efface son point : on a vu le message. */\n    if(c.arena && ARENA_DOT){ ARENA_DOT=false; renderChatList(); }\n    const mm=MODELS.find(m=>m.id===c.model);\n    /* Une ancienne discussion qui portait encore le modèle « arena » reprend le\n       modèle par défaut : le canal ne passe plus que par la discussion épinglée. */\n    if(mm && mm.family===\"arena\" && !c.arena){ selectModel(MODELS[0]); }\n    else if(mm && mm!==current) selectModel(mm);\n    /* La discussion retrouve son mode : chat, agent ou Z-Canal. */\n    if(typeof AGENT_ON!==\"undefined\"){\n      AGENT_ON=!!c.agent;\n      agtBtn.classList.toggle(\"on\",AGENT_ON);\n      if(typeof CUR_MODE!==\"undefined\"){\n        CUR_CANAL=(c.canal&&c.canal!==\"arena\")?c.canal:null;\n        CUR_MODE=c.canal?\"canal\":(c.agent?\"agent\":\"chat\");\n        if(c.canal){ const am=MODELS.find(m=>m.id===\"arena\"); if(am) selectModel(am); canalWatchOne(c.canal); }\n        if(c.dot){ c.dot=false; }\n        renderModeBadge();\n      }\n      agtSyncUI();\n    }\n    /* Discussion de canal : son contenu EST le canal. On l'affiche depuis\n       Firebase, sans rien relire dans la discussion (elle n'en garde aucune\n       copie). */\n    if(c.canal && c.canal!==\"arena\"){\n      const n=await canalHistoryLoad(c.canal);\n      if(!n) emptyState();\n      renderChatList(); closeSide(); scrollDown();\n      if(typeof agtMissionBanner===\"function\") agtMissionBanner(id);\n      return;\n    }\n    let lastA=null, lastAMd=\"\";\n    for(const d of snap.docs){\n      const m=await readDocChunked(mref, d.data()); msgSeq=Math.max(msgSeq, m.order||0);\n      if(m.role===\"user\"){\n        const imgs=(m.images||[]).filter(i=>!i.tooBig&&i.b64&&!/^video\\//.test(i.mime||\"\"));\n        const au=(m.audio&&m.audio.b64)?{name:m.audio.name, b64:m.audio.b64, mime:m.audio.mime||\"audio/wav\",\n          kb:Math.max(1,Math.round(m.audio.b64.length*0.75/1024))}:null;\n        pendingFiles=imgs.map(i=>({kind:\"image\",name:i.name,b64:i.b64}))\n          .concat(m.audio?[{kind:\"audio\",name:m.audio.name}]:[])\n          .concat((m.files||[]).map(f=>({kind:\"file\",name:f.name})));\n        addUser(m.text||\"\", m.kind||\"none\", imgs[0]?{name:imgs[0].name,b64:imgs[0].b64}:null, au);\n        pendingFiles=[];\n        const ent={role:\"user\", text:m.text||\"\"};\n        if(imgs.length) ent.images=imgs.map(i=>({name:i.name,b64:i.b64,mime:i.mime}));\n        conv.push(ent);\n      }else{\n        const b=addAssistant(m.modelName);   /* renvoie directement la bulle */\n        /* Mission d'agent : zone terminal repliable reconstruite telle\n           quelle, exactement comme avant de quitter la page. */\n        const legacy=(m.steps&&m.steps.length)?null:agtSplitLegacyMission(m.text||\"\");\n        if((m.steps&&m.steps.length)||legacy){\n          const stp=(m.steps&&m.steps.length)?m.steps:[{raw:legacy.trace}];\n          const fin=(m.steps&&m.steps.length)?(m.text||\"\"):legacy.finalText;\n          agtRenderMission(b, stp, fin, true);\n          msgActions(b,{getMd:()=>agtTraceMd(stp, fin)});\n          lastA=b; lastAMd=fin;\n          conv.push({role:\"assistant\", text:fin});\n          continue;\n        }\n        /* le marqueur interne « (image générée : …) » n'est pas un message */\n        const isMarker=(m.text||\"\").indexOf(\"(image générée\")===0;\n        let html=((m.images&&m.images.length)&&isMarker)?\"\":renderMarkdown(m.text||\"\");\n        for(const im of (m.images||[])){\n          const imMime=im.mime||\"image/jpeg\";\n          /* Une vidéo se relit dans un lecteur, pas dans une balise image. */\n          if(/^video\\//.test(imMime)){\n            if(im.tooBig||!im.b64){ html+='<p style=\"font-size:12.5px;color:var(--ink-3)\">🎬 '+esc(im.name||\"vidéo\")+' (trop lourde pour la sauvegarde)</p>'; continue; }\n            const vid=\"v\"+(++VID_SEQ);\n            VID_STORE[vid]={b64:im.b64, mime:imMime, name:im.name||\"video.mp4\", poster:\"\", regen:null, bubble:b};\n            html+='<video class=\"gvid\" data-vid=\"'+vid+'\" controls loop playsinline preload=\"metadata\"'\n              +' style=\"max-width:min(100%,460px);width:100%;border-radius:8px;display:block;background:#000;margin-top:8px\">'\n              +'<source src=\"data:'+imMime+';base64,'+im.b64+'\" type=\"'+imMime+'\"></video>'\n              +'<div class=\"imgacts\"><button type=\"button\" class=\"iact vdl\" data-vid=\"'+vid+'\" title=\"Télécharger\" aria-label=\"Télécharger la vidéo\">'+IC_DL+'</button></div>';\n            continue;\n          }\n          if(im.tooBig||!im.b64){ html+='<p style=\"font-size:12.5px;color:var(--ink-3)\">🖼 '+esc(im.name)+' (trop lourde pour la sauvegarde)</p>'; continue; }\n          const mime=im.mime||\"image/jpeg\";\n          const cid=\"i\"+(++IMG_SEQ);\n          IMG_STORE[cid]={b64:im.b64, mime:mime, name:im.name, regen:null, bubble:b};\n          html+='<img src=\"data:'+mime+';base64,'+im.b64+'\" alt=\"'+esc(im.name)+'\" class=\"gimg\" data-iid=\"'+cid\n            +'\" style=\"max-width:min(100%,420px);border-radius:8px;display:block;cursor:zoom-in;margin-top:8px\">'\n            +'<div class=\"imgacts\">'\n            +'<button type=\"button\" class=\"iact icopy\" data-iid=\"'+cid+'\" title=\"Copier\" aria-label=\"Copier l’image\">'+IC_COPY+'</button>'\n            +'<button type=\"button\" class=\"iact idl\" data-iid=\"'+cid+'\" title=\"Télécharger\" aria-label=\"Télécharger l’image\">'+IC_DL+'</button>'\n            +'<button type=\"button\" class=\"iact iback\" data-iid=\"'+cid+'\" title=\"Charger dans le message suivant\" aria-label=\"Charger l’image dans le message suivant\">'+IC_BACK+'</button>'\n            +'</div>';\n        }\n        if(m.audio&&m.audio.b64) html+=renderAudioPlayer(m.audio.b64, m.audio.mime, m.audio.name);\n        b.innerHTML=html;\n        if(!isMarker) msgActions(b,{getMd:()=>m.text||\"\"});   /* copier sur chaque réponse */\n        lastA=b; lastAMd=isMarker?\"\":(m.text||\"\");\n        const ent={role:\"assistant\", text:m.text||\"\"};\n        const imgs=(m.images||[]).filter(i=>!i.tooBig&&i.b64&&!/^video\\//.test(i.mime||\"\"));\n        if(imgs.length) ent.images=imgs.map(i=>({name:i.name,b64:i.b64,mime:i.mime}));\n        conv.push(ent);\n      }\n    }\n    if(c.canal) canalChipsFromHistory(c);   /* pastilles « non lu » / « lu ! » */\n    if(lastA) msgActions(lastA,{getMd:()=>lastAMd, regen:()=>rerunLastLoaded()});\n    if(!snap.size) emptyState();\n    renderChatList(); closeSide(); scrollDown();\n    /* Une mission tourne peut-être encore pour cette discussion : on l'affiche\n       en direct au lieu de laisser croire qu'elle a disparu. */\n    if(typeof agtMissionBanner===\"function\") agtMissionBanner(id);\n  }catch(e){ console.warn(\"loadChat :\", e); alert(\"Impossible de charger cette discussion.\"); }\n}\n\n/* Relance le dernier échange d'une discussion chargée : on retire la paire\n   user/assistant et on rejoue submit() avec le texte d'origine. */\nfunction rerunLastLoaded(){\n  if(busy) return;\n  let ui=conv.length-1; while(ui>=0 && conv[ui].role!==\"user\") ui--;\n  if(ui<0) return;\n  const u=conv[ui];\n  conv.length=ui;\n  const ubs=[...log.querySelectorAll(\".msg.user\")]; if(ubs.length) ubs[ubs.length-1].remove();\n  const abs=[...log.querySelectorAll(\".msg.assistant\")]; if(abs.length) abs[abs.length-1].remove();\n  pendingFiles=(u.images||[]).map(im=>({kind:\"image\",name:im.name,b64:im.b64}));\n  pendingImage=pendingFiles.length?{name:pendingFiles[0].name,b64:pendingFiles[0].b64}:null;\n  pendingAudio=null;\n  promptEl.value=u.text;\n  submit();\n}\n/* --- sauvegarde Firestore ---------------------------------------------- */\nfunction loadImgEl(src){\n  return new Promise((res,rej)=>{ const im=new Image();\n    im.onload=()=>res(im); im.onerror=()=>rej(new Error(\"image illisible\")); im.src=src; });\n}\n/* Sauvegarde SANS PERTE. Un document Firestore est plafonné à ~1 Mio, mais\n   l'app découpe déjà les messages volumineux en morceaux (writeDocChunked) :\n   plus aucune raison de recompresser. On enregistre donc les OCTETS EXACTS,\n   avec leur type d'origine — l'image relue, zoomée ou téléchargée est\n   rigoureusement celle qui a été reçue (même règle pour les vidéos). */\nconst STORE_MAX = 22000000;          /* au-delà, même découpé ce serait absurde */\nfunction storeExact(b64, mime){\n  if(!b64) return null;\n  if(b64.length > STORE_MAX) return null;\n  return {b64:b64, mime:mime||\"image/png\"};\n}\n/* Nom conservé pour les appels existants : la fonction ne rétrécit plus rien. */\nconst shrinkForStore = storeExact;\nfunction capAudio(a){\n  const mime=a.mime||\"audio/wav\";\n  return a.b64.length<=900000 ? {name:a.name, mime:mime, b64:a.b64} : {name:a.name, mime:mime, tooBig:true};\n}\nasync function ensureChat(firstText){\n  if(chatId) return chatId;\n  chatId=\"c\"+Date.now().toString(36)+Math.random().toString(36).slice(2,7);\n  msgSeq=0;\n  const now=Date.now();\n  const title=(firstText||\"\").replace(/\\s+/g,\" \").trim().slice(0,38) || \"Nouvelle discussion\";\n  const ag=(typeof AGENT_ON!==\"undefined\"&&AGENT_ON);\n  try{\n    await chatsRef().doc(chatId).set({title:title, model:current.id, agent:ag, createdAt:now, updatedAt:now});\n    chats.unshift({id:chatId, title:title, model:current.id, agent:ag, createdAt:now, updatedAt:now});\n    renderChatList();\n  }catch(e){ console.warn(\"ensureChat :\", e); }\n  return chatId;\n}\n/* Un document Firestore ne peut pas dépasser ~1 Mio : une mission d'agent\n   complète (sorties entières, telles quelles) dépasse forcément cette\n   limite. Les champs volumineux partent donc en morceaux dans une\n   sous-collection « parts », puis sont recollés au chargement — rien n'est\n   perdu, rien n'est coupé. */\nconst DOC_MAX=450000, PART_LEN=400000;\nasync function writeDocChunked(mref, id, doc){\n  const small={}, big={};\n  for(const k of Object.keys(doc)){\n    if(k===\"role\"||k===\"order\"||k===\"createdAt\"||k===\"modelName\"||k===\"kind\"||k===\"agent\") small[k]=doc[k];\n    else big[k]=doc[k];\n  }\n  const json=JSON.stringify(big);\n  if(json.length<=DOC_MAX){ await mref.doc(id).set(doc); return; }\n  const parts=[];\n  for(let i=0;i<json.length;i+=PART_LEN) parts.push(json.slice(i,i+PART_LEN));\n  await mref.doc(id).set(Object.assign({}, small, {parts:parts.length}));\n  const pref=mref.doc(id).collection(\"parts\");\n  let batch=fbDB.batch(), n=0;\n  for(let i=0;i<parts.length;i++){\n    batch.set(pref.doc(\"p\"+i), {d:parts[i]});\n    if(++n%400===0){ await batch.commit(); batch=fbDB.batch(); }\n  }\n  await batch.commit();\n}\n/* Recolle un message stocké en morceaux (transparent pour le reste du code). */\nasync function readDocChunked(mref, m){\n  if(!m||!m.parts) return m;\n  const snap=await mref.doc(\"m\"+m.order).collection(\"parts\").get();\n  const arr=[];\n  for(const d of snap.docs) arr[+String(d.id).slice(1)]=d.data().d||\"\";\n  try{ return Object.assign({}, m, JSON.parse(arr.join(\"\"))); }catch(e){ return m; }\n}\n/* Le numéro d'ordre d'un nouveau message suit ce qui est DÉJÀ écrit dans la\n   discussion : une mission a pu déposer ses messages pendant que la page était\n   fermée, et deux messages ne doivent jamais partager le même numéro (le\n   second écraserait le premier). */\nasync function fbSeqSync(cid){\n  if(!cid) return msgSeq;\n  try{\n    const last=await chatsRef().doc(cid).collection(\"messages\").orderBy(\"order\",\"desc\").limit(1).get();\n    const v=last.empty?0:(Number(last.docs[0].data().order)||0);\n    if(cid===chatId) msgSeq=Math.max(msgSeq,v);\n    return v;\n  }catch(e){ return msgSeq; }\n}\nasync function commitPair(uDoc, aDoc){\n  const mref=chatsRef().doc(chatId).collection(\"messages\");\n  await writeDocChunked(mref, \"m\"+uDoc.order, uDoc);\n  await writeDocChunked(mref, \"m\"+aDoc.order, aDoc);\n  await chatsRef().doc(chatId).update({updatedAt:Date.now(), model:current.id});\n}\n/* appelé après chaque échange texte (bloc principal) */\nasync function afterText(x){\n  if(!FB_OK||!fbUser) return;\n  const files=pendingFiles.slice();\n  try{\n    await ensureChat(x.userText);\n    await fbSeqSync(chatId);                 /* on repart de ce qui est écrit */\n    const uDoc={role:\"user\", text:x.userText||\"\", kind:x.kind||\"none\", modelName:current.id,\n      order:++msgSeq, createdAt:Date.now(), images:[]};\n    const imgs=[];\n    if(x.kind===\"upload\"&&x.snap) imgs.push({name:x.snap.name, b64:x.snap.b64, mime:x.snap.mime||\"image/png\"});\n    for(const f of files){ if(f.kind===\"image\"&&!imgs.some(i=>i.b64===f.b64)) imgs.push({name:f.name, b64:f.b64, mime:f.mime||\"image/png\"}); }\n    for(const im of imgs.slice(0,4)){\n      const sm=storeExact(im.b64, im.mime||\"image/png\");\n      uDoc.images.push(sm?{name:im.name, b64:sm.b64, mime:sm.mime}:{name:im.name, mime:im.mime||\"image/png\", tooBig:true});\n    }\n    if(x.kind===\"audio\"&&x.asnap) uDoc.audio=capAudio(x.asnap);\n    const fl=[];\n    for(const f of files){ if(f.kind===\"file\") fl.push({name:f.name, text:(f.text||\"\").slice(0,20000)}); }\n    if(fl.length) uDoc.files=fl;\n    const aDoc={role:\"assistant\", text:x.out||\"\", modelName:x.via||current.id,\n      order:++msgSeq, createdAt:Date.now()};\n    await commitPair(uDoc, aDoc);\n  }catch(e){ console.warn(\"sauvegarde :\", e); fbSaveWarn(e); }\n}\n/* appelé après chaque génération d'image */\nasync function afterImage(x){\n  if(!FB_OK||!fbUser) return;\n  try{\n    await ensureChat(x.userText);\n    const im=IMG_STORE[x.id]; if(!im) return;\n    const sm=storeExact(im.b64, im.mime);\n    const uDoc={role:\"user\", text:x.userText||\"\", kind:\"imggen\", modelName:current.id,\n      order:++msgSeq, createdAt:Date.now()};\n    const aDoc={role:\"assistant\", text:\"(image générée : \"+im.name+\" avec \"+current.label+\")\",\n      modelName:current.id, order:++msgSeq, createdAt:Date.now(),\n      images:[sm?{name:im.name, b64:sm.b64, mime:sm.mime}:{name:im.name, mime:im.mime, tooBig:true}]};\n    await commitPair(uDoc, aDoc);\n  }catch(e){ console.warn(\"sauvegarde image :\", e); fbSaveWarn(e); }\n}\n/* appelé après chaque génération de vidéo */\nasync function afterVideo(x){\n  if(!FB_OK||!fbUser) return;\n  try{\n    await ensureChat(x.userText);\n    const v=VID_STORE[x.id]; if(!v) return;\n    /* La vidéo est enregistrée elle aussi, OCTET POUR OCTET : le message\n       découpé (writeDocChunked) accepte plusieurs mégaoctets. Une fois\n       rechargée, elle se relit dans son lecteur, comme au premier affichage. */\n    const vs=storeExact(v.b64, v.mime||\"video/mp4\");\n    const uDoc={role:\"user\", text:x.userText||\"\", kind:\"vidgen\", modelName:current.id,\n      order:++msgSeq, createdAt:Date.now()};\n    const aDoc={role:\"assistant\", text:\"(vidéo générée : \"+v.name+\" avec \"+current.label+\")\",\n      modelName:current.id, order:++msgSeq, createdAt:Date.now(),\n      images:[vs?{name:v.name, b64:vs.b64, mime:vs.mime}:{name:v.name, mime:v.mime||\"video/mp4\", tooBig:true}]};\n    await commitPair(uDoc, aDoc);\n  }catch(e){ console.warn(\"sauvegarde vidéo :\", e); fbSaveWarn(e); }\n}\n/* Mission d'agent : étapes structurées (commande → sortie) pour un rendu\n   identique au rechargement, sur n'importe quel appareil. */\nasync function afterAgent(x){\n  if(!FB_OK||!fbUser) return;\n  /* Une mission est rattachée à la discussion où elle a été lancée : si\n     l'utilisateur a changé de discussion entre-temps, le résultat va quand\n     même dans la bonne — c'est ce qui faisait « disparaître » la mission. */\n  const back=(x.chatId && x.chatId!==chatId)?{id:chatId, seq:msgSeq}:null;\n  try{\n    if(back) chatId=x.chatId;\n    await ensureChat(x.userText);\n    /* Toujours repartir de ce qui est écrit : la passerelle a pu enregistrer\n       la mission elle-même (application fermée, autre appareil). */\n    await fbSeqSync(chatId);\n    const uDoc={role:\"user\", text:x.userText||\"\", kind:\"none\", modelName:current.id,\n      order:++msgSeq, createdAt:Date.now(), images:[]};\n    const aDoc={role:\"assistant\", text:x.out||\"\", modelName:current.id,\n      order:++msgSeq, createdAt:Date.now(), steps:(x.steps||[]).slice(0,400), agent:true};\n    await commitPair(uDoc, aDoc);\n  }catch(e){ console.warn(\"sauvegarde agent :\", e); fbSaveWarn(e); }\n  finally{ if(back){ chatId=back.id; msgSeq=back.seq; } }\n}\n/* Une sauvegarde qui échoue ne doit pas passer inaperçue (c'est ce qui\n   faisait disparaître des messages en silence). */\nfunction fbSaveWarn(e){\n  try{\n    if(typeof agtToast===\"function\") agtToast(\"Sauvegarde impossible : \"+String(e&&e.message||e));\n    else console.warn(\"sauvegarde :\", e);\n  }catch(x){}\n}\n/* profil utilisateur : users/{uid} */\nasync function ensureUserDoc(){\n  if(!FB_OK||!fbUser) return;\n  try{\n    const ref=fbDB.collection(\"users\").doc(fbUser.uid);\n    const snap=await ref.get();\n    const def=(fbUser.email||\"\").split(\"@\")[0] || fbUser.displayName || \"Utilisateur\";\n    if(!snap.exists){\n      await ref.set({displayName:def, email:fbUser.email||\"\", photoURL:fbUser.photoURL||\"\",\n        createdAt:Date.now(), updatedAt:Date.now()});\n      userName=def; userPhoto=fbUser.photoURL||\"\";\n    }else{\n      const d=snap.data()||{};\n      userName=d.displayName||def; userPhoto=d.photoURL||fbUser.photoURL||\"\";\n    }\n    userColor=avatarColor(fbUser.email||userName);\n  }catch(e){ console.warn(\"profil :\", e); }\n}\n\n/* --- ModelScope : appel quotidien (une fois par jour) ------------------\n   Le jeton ModelScope sert une fois par jour à s'authentifier auprès de la\n   plateforme (identité + solde de crédits). L'état du jour vit dans Firestore\n   (users/{uid}.msDaily) sous la forme { day: \"AAAA-MM-JJ\", ok: true|false } :\n     • par défaut la journée est « non » : au chargement de l'app, si l'état\n       enregistré ne porte pas la date du jour, il est réécrit à « non » ;\n     • l'appel part alors tout seul ; s'il réussit, l'état passe à « oui » ;\n     • le lendemain, le « oui » de la veille est périmé (sa date n'est plus\n       celle du jour) : l'app le repasse à « non » et refait l'appel.\n   Une seule fois par jour : si la journée est déjà « oui », rien ne repart. */\nlet MS_DAILY=null;                    /* état du jour : {day, ok, at, …} */\nfunction msTodayKey(){\n  const d=new Date();\n  return d.getFullYear()+\"-\"+String(d.getMonth()+1).padStart(2,\"0\")+\"-\"+String(d.getDate()).padStart(2,\"0\");\n}\nasync function msDailyRun(){\n  if(!FB_OK||!fbUser||!fbDB) return null;\n  const ref=fbDB.collection(\"users\").doc(fbUser.uid);\n  const today=msTodayKey();\n  let st=null;\n  try{ const snap=await ref.get(); st=((snap&&snap.data())||{}).msDaily||null; }catch(e){ st=null; }\n  if(st&&st.day===today&&st.ok===true){ MS_DAILY=st; return st; }   /* déjà « oui » aujourd'hui */\n  const base={day:today, ok:false, at:Date.now(),\n    tries:((st&&st.day===today)?(Number(st.tries)||0):0)+1};\n  MS_DAILY=base;\n  try{ await ref.set({msDaily:base},{merge:true}); }catch(e){}      /* « non » pour aujourd'hui */\n  let j={};\n  try{\n    const r=await fetch(API_BASE+\"/api/ms/daily\",{method:\"POST\",\n      headers:{\"Content-Type\":\"application/json\",\"X-Access-Code\":ACCESS_CODE},body:\"{}\"});\n    j=await r.json();\n  }catch(e){ j={ok:false,error:String(e&&e.message||e)}; }\n  const done=Object.assign({},base,{\n    ok:j.ok===true, token:j.token||\"\", identity:j.identity||\"\",\n    balance:j.balance||null, error:j.error||\"\",\n    /* Noms de modèles confirmés par ModelScope (lecture seule, zéro crédit) :\n       « ok » = le nom configuré existe chez ModelScope, « inconnu » = à revoir. */\n    models:j.models||null,\n  });\n  MS_DAILY=done;\n  try{ await ref.set({msDaily:done},{merge:true}); }catch(e){}      /* « oui » si l'appel a réussi */\n  if(typeof console!==\"undefined\"&&console.log){\n    if(done.models){\n      const okN=Object.keys(done.models).filter(k=>done.models[k]===\"ok\").length;\n      const tot=Object.keys(done.models).length;\n      console.log(\"ModelScope — noms de modèles confirmés : \"+okN+\"/\"+tot);\n    }\n    console.log(\"ModelScope (quotidien) :\",done.ok?(\"connecté\"+(done.identity?(\" · \"+done.identity):\"\")+(done.balance?(\" · solde \"+done.balance.available):\"\")):(\"non — \"+(done.error||\"échec\")));\n  }\n  return done;\n}\n\n/* --- Canal Arena : le modèle « arena » --------------------------------\n   Deux sens de circulation, un seul document Firebase (chan/main) :\n\n     chan/main           état du canal (agent connecté ou non, dernier échange)\n     chan/main/box/{id}  les messages : role « user » (l'application),\n                         « arena » (Arena) ou « tool » (résultat de commande),\n                         kind « text » ou « call » (JSON d'action).\n\n   • Ce que l'utilisateur écrit au modèle « arena » part dans le canal.\n   • Arena écrit dedans de son côté (script z-canal.py) : son message arrive ici\n     par écoute temps réel. Un JSON d'action (bash / fetch / final) est EXÉCUTÉ\n     sur la VM et le résultat repart dans le canal — il n'est jamais affiché\n     comme du texte. Un message normal s'affiche exactement comme une réponse\n     d'IA (markdown, boutons), dans la discussion « Arena ».\n   • Rien n'est perdu quand l'utilisateur est sur une autre discussion : le\n     message est enregistré dans la discussion « Arena », sans notification. */\nlet ARENA_UNSUB=null, ARENA_DEV_T=null, ARENA_LAST_USER=null;\n/* Un message d'Arena arrivé pendant qu'une autre discussion est ouverte laisse\n   un simple point sur la ligne épinglée — aucune fenêtre, aucun bandeau. */\nlet ARENA_DOT=false;\n/* La bulle « Envoi… » de la discussion Arena : la réponse d'Arena la remplace\n   au lieu d'en ajouter une troisième. */\nlet ARENA_WAIT=null;\n/* Le message unique d'un échange de canal : toutes les commandes de l'agent\n   s'empilent dans le MÊME message, et ce message ne se termine que lorsque\n   l'agent écrit du texte (ou que l'utilisateur envoie autre chose) —\n   exactement comme une mission d'agent classique. */\nlet CANAL_MISSION=null;\n/* Le plus grand numéro d'ordre déjà écrit dans la discussion Arena : deux\n   messages ne peuvent jamais partager le même, sinon le second écraserait le\n   premier (l'identifiant du document en découle). */\nlet ARENA_ORDER=0;\n/* Ids déjà traités dans cette page : un même message ne s'exécute jamais deux\n   fois, même si l'écoute le redonne avant que « done » soit écrit. */\nconst ARENA_DONE=new Set();\nfunction arenaBoxRef(){\n  if(!FB_OK||!fbDB) return null;\n  return fbDB.collection(\"chan\").doc(\"main\").collection(\"box\");\n}\nfunction arenaMainRef(){\n  if(!FB_OK||!fbDB) return null;\n  return fbDB.collection(\"chan\").doc(\"main\");\n}\n/* Aucune notification ne s'affiche : un message d'Arena arrivé pendant qu'une\n   autre discussion est ouverte est simplement enregistré dans la discussion\n   « Arena » (il y attend). Rien ne recouvre l'écran. */\n/* ---- marqueur de présence : l'app dit au canal qu'elle est ouverte ---- */\nasync function arenaDevice(online){\n  const ref=arenaMainRef();\n  if(!ref) return;\n  try{ await ref.set({arenaApp:{online:!!online, at:Date.now()}},{merge:true}); }catch(e){}\n}\nfunction arenaHeartbeat(){\n  if(ARENA_DEV_T) clearInterval(ARENA_DEV_T);\n  if(!FB_OK) return;\n  arenaDevice(true);\n  ARENA_DEV_T=setInterval(()=>arenaDevice(true), 60000);\n  window.addEventListener(\"pagehide\",()=>{ arenaDevice(false); });\n  window.addEventListener(\"beforeunload\",()=>{ arenaDevice(false); });\n}\n/* ---- écriture/lecture du canal, par le worker (le jeton reste chez lui) ---- */\n/* Un appel réseau borné : si le worker ne répond pas, on le dit — mais on ne\n   bloque jamais l'interface. */\nfunction arenaTimeout(ms){\n  const c=new AbortController();\n  const t=setTimeout(()=>c.abort(), ms);\n  return {signal:c.signal, stop:()=>clearTimeout(t)};\n}\nasync function arenaSend(text, kind, req, note){\n  const g=arenaTimeout(15000);\n  try{\n    const r=await fetch(API_BASE+\"/api/arena/send\",{signal:g.signal,\n      method:\"POST\",\n      headers:{\"Content-Type\":\"application/json\",\"X-Access-Code\":ACCESS_CODE},\n      body:JSON.stringify({text:String(text||\"\"), kind:kind||\"text\", req:req||\"\", note:note||\"\"}),\n    });\n    const j=await r.json().catch(()=>({}));\n    if(!r.ok) return {ok:false, error:(j&&j.error)||(\"HTTP \"+r.status)};\n    return j||{ok:false, error:\"réponse illisible\"};\n  }catch(e){\n    const m=String(e&&e.message?e.message:e);\n    return {ok:false, error:/abort/i.test(m)?\"canal : délai dépassé\":m};\n  }finally{ g.stop(); }\n}\n/* ---- ce qu'on fait d'un message reçu ---- */\nfunction arenaParse(text){\n  const t=String(text||\"\").trim();\n  if(t.charAt(0)!==\"{\") return {kind:\"text\", text:String(text||\"\")};\n  let o=null;\n  try{ o=JSON.parse(t); }catch(e){ return {kind:\"text\", text:String(text||\"\")}; }\n  if(!o||typeof o!==\"object\") return {kind:\"text\", text:String(text||\"\")};\n  const a=String(o.action||\"\").toLowerCase();\n  if(a!==\"bash\"&&a!==\"fetch\"&&a!==\"final\") return {kind:\"text\", text:String(text||\"\")};\n  return {kind:\"call\", call:o, text:String(o.message||\"\")};\n}\nasync function arenaExecCall(call){\n  const action=String(call.action||\"\").toLowerCase();\n  if(action===\"bash\"){\n    const cmd=String(call.command||\"\");\n    let r={};\n    const g=arenaTimeout(120000);          /* une commande reste bornée */\n    try{\n      const er=await fetch(API_BASE+\"/api/agent/exec\",{\n        method:\"POST\", signal:g.signal,\n        headers:{\"Content-Type\":\"application/json\",\"X-Access-Code\":ACCESS_CODE},\n        body:JSON.stringify({command:cmd}),\n      });\n      r=await er.json().catch(()=>({}));\n    }catch(e){ r={ok:false, stderr:String(e&&e.message?e.message:e)}; }\n    finally{ g.stop(); }\n    const so=String(r.stdout||\"\"), se=String(r.stderr||\"\");\n    const ex=(r.exit_code==null?0:r.exit_code);\n    const txt=\"$ \"+cmd+\"\\n[exit] \"+(r.exit_code==null?\"?\":r.exit_code)+\"\\n\"+(so||se||\"(aucune sortie)\");\n    return {ok:!!r.ok, text:txt, exit:ex,\n      step:{kind:\"bash\", cmd:cmd, out:so, err:se, exit:ex}};\n  }\n  /* fetch : lecture d'une page par la VM (Chrome sans écran). */\n  const url=String(call.url||\"\").trim();\n  const fmode=(String(call.mode||\"text\").toLowerCase()===\"html\")?\"html\":\"text\";\n  let j={};\n  const g=arenaTimeout(120000);\n  try{\n    const fr=await fetch(API_BASE+\"/api/agent/fetch\",{\n      method:\"POST\", signal:g.signal,\n      headers:{\"Content-Type\":\"application/json\",\"X-Access-Code\":ACCESS_CODE},\n      body:JSON.stringify({url:url, mode:fmode}),\n    });\n    j=await fr.json().catch(()=>({}));\n  }catch(e){ j={ok:false, error:String(e&&e.message?e.message:e)}; }\n  finally{ g.stop(); }\n  const head=(j.ok?(\"HTTP \"+(j.status==null?\"?\":j.status)):\"ÉCHEC\")\n    +(j.title?(\" · \"+j.title):\"\")+(j.chars?(\" · \"+j.chars+\" caractères\"):\"\");\n  const txt=url+\"\\n\"+head+\"\\n\\n\"+String(j.content==null?(j.error||\"\"):j.content);\n  return {ok:!!j.ok, text:txt, exit:j.ok?0:1,\n    step:{kind:\"fetch\", cmd:\"fetch [\"+fmode+\"] \"+url, out:txt, err:(j.ok?\"\":String(j.error||head)), exit:(j.ok?0:1)}};\n}\n/* ---- la discussion « Arena » : UNE seule, épinglée en tête ---- */\n/* Elle est créée une fois puis retrouvée à chaque fois (jamais de doublon) :\n   c'est elle qui porte le canal, et rien d'autre. */\nasync function arenaChat(){\n  if(!FB_OK||!fbUser||!fbDB) return null;\n  if(typeof chats===\"undefined\"||!Array.isArray(chats)) return null;\n  let found=chats.find(c=>c.arena===true&&c.pinned===true)\n    || chats.find(c=>c.arena===true)\n    || chats.find(c=>c.model===\"arena\"&&!c.hidden)     /* une ancienne : on l'adopte */\n    || null;\n  if(found){\n    if(found.arena!==true||found.pinned!==true||found.agent!==true){\n      found.arena=true; found.pinned=true; found.agent=true;\n      try{ await chatsRef().doc(found.id).set({arena:true, pinned:true, agent:true},{merge:true}); }catch(e){}\n    }\n    return found;\n  }\n  const id=\"c\"+Date.now().toString(36)+Math.random().toString(36).slice(2,7);\n  const now=Date.now();\n  const neuve={id:id, title:\"Arena\", model:\"arena\", agent:true, arena:true, pinned:true,\n    createdAt:now, updatedAt:now};\n  try{\n    await chatsRef().doc(id).set({title:\"Arena\", model:\"arena\", agent:true, arena:true, pinned:true,\n      createdAt:now, updatedAt:now});\n  }catch(e){ return null; }\n  chats.unshift(neuve);\n  renderChatList();\n  return neuve;\n}\n/* Les discussions qui portaient encore le modèle « arena » avant l'épinglage\n   deviendraient des doublons : elles quittent la liste (rien n'est supprimé\n   dans Firestore). */\nasync function arenaDedupe(main){\n  if(!main) return;\n  /* Une discussion de CANAL n'est jamais un doublon de la discussion Arena :\n     elle a son propre contenu (le canal). Les anciennes portaient encore le\n     modèle « arena » et se faisaient donc masquer — elles disparaissaient de\n     la liste à la fermeture de l'application. */\n  const extra=chats.filter(c=>c!==main && !c.canal && (c.arena===true||c.model===\"arena\"));\n  if(!extra.length) return;\n  let change=false;\n  for(const c of extra){\n    if(c.hidden===true) continue;\n    c.hidden=true; change=true;\n    if(FB_OK&&fbUser){ try{ await chatsRef().doc(c.id).set({hidden:true},{merge:true}); }catch(e){} }\n  }\n  if(change) renderChatList();\n}\n/* ---- enregistrement dans la discussion « Arena » ----\n   `order` vient de l'horloge : les messages restent dans l'ordre même si\n   l'utilisateur est en train de lire une autre discussion pendant ce temps. */\nasync function arenaCommit(chat, docs){\n  if(!FB_OK||!chat) return;\n  const mref=chatsRef().doc(chat.id).collection(\"messages\");\n  /* Le numéro d'ordre suit l'heure RÉELLE du message — celle du canal quand\n     elle est connue : une réponse écrite pendant une absence se replace donc à\n     sa vraie place dans l'historique au lieu de s'ajouter à la fin. */\n  const plancher=(chatId===chat.id)?msgSeq:0;\n  for(const d of docs){\n    let o=Number(d.order)||Date.now();\n    const mini=Math.max(ARENA_ORDER, plancher);\n    if(o<=mini) o=mini+1;\n    d.order=o; ARENA_ORDER=o;\n    await writeDocChunked(mref, \"m\"+o, d);\n  }\n  try{ await chatsRef().doc(chat.id).update({updatedAt:Date.now()}); }catch(e){}\n}\n/* Enregistrement « au mieux » : une panne de Firestore ne doit jamais empêcher\n   d'envoyer un message ni bloquer l'application. */\nasync function arenaSaveUser(text, req){\n  try{\n    const chat=await arenaChat();\n    if(!chat) return null;\n    const now=Date.now();\n    await arenaCommit(chat, [{role:\"user\", text:String(text||\"\"), kind:\"none\", modelName:\"arena\", order:now, createdAt:now, images:[], req:String(req||\"\")}]);\n    return chat;\n  }catch(e){ console.warn(\"canal arena (enregistrement) :\", e); return null; }\n}\n/* Une commande exécutée est enregistrée comme un message : à la réouverture,\n   la zone terminale est reconstruite à l'identique. */\nasync function arenaSaveStep(chat, step, at){\n  if(!chat||!step) return;\n  try{\n    const now=Date.now();\n    const t=(typeof at===\"number\" && at>0 && at<=now+60000)?at:now;\n    await arenaCommit(chat, [{role:\"assistant\", text:\"\", modelName:\"arena\",\n      order:t, createdAt:now, steps:[step]}]);\n  }catch(e){ console.warn(\"canal arena (commande) :\", e); }\n}\nasync function arenaSaveReply(chat, text, at){\n  if(!chat) return;\n  try{\n    const now=Date.now();\n    const t=(typeof at===\"number\" && at>0 && at<=now+60000)?at:now;\n    await arenaCommit(chat, [{role:\"assistant\", text:String(text||\"\"), modelName:\"arena\", order:t, createdAt:now}]);\n  }catch(e){ console.warn(\"canal arena (enregistrement) :\", e); }\n}\nasync function arenaMark(id, state){\n  const box=arenaBoxRef(); if(!box) return;\n  try{ await box.doc(id).set({state:state, doneAt:Date.now()},{merge:true}); }catch(e){}\n}\n/* ---- réception : l'utilisateur voit la réponse comme une réponse d'IA ---- */\nasync function arenaDeliver(d, id){\n  if(ARENA_DONE.has(id)) return;\n  ARENA_DONE.add(id);\n  try{ await arenaDeliver1(d, id); }\n  catch(e){ console.warn(\"canal arena (traitement) :\", e); }\n}\nasync function arenaDeliver1(d, id){\n  const parsed=arenaParse(d.text);\n  const chat=await arenaChat();\n  const inChat=!!(chat&&chatId===chat.id);\n  /* 1. Une action (bash / fetch) est EXÉCUTÉE, jamais affichée ; son résultat\n        repart dans le canal pour Arena, et s'affiche ici en zone repliable. */\n  if(parsed.kind===\"call\" && parsed.call.action!==\"final\"){\n    const res=await arenaExecCall(parsed.call);\n    /* La commande s'affiche dans la discussion Arena, avec sa sortie, dans la\n       zone terminale (mêmes briques qu'une mission d'agent) — et elle y est\n       ENREGISTRÉE : on la retrouve en rouvrant la discussion, même plus tard. */\n    if(inChat&&chat&&chatId===chat.id){\n      const b=addAssistant(\"arena\");\n      agtRenderMission(b, [res.step], \"\", true);\n      scrollDown();\n    }\n    if(chat) await arenaSaveStep(chat, res.step, Number(d.ts)||0);\n    await arenaSend(res.text, \"tool\", id, \"résultat de commande\");\n    await arenaMark(id, \"done\");\n    return;\n  }\n  /* 2. Un message (texte, ou action « final ») s'affiche comme une réponse. */\n  const out=parsed.text;\n  if(!out) { await arenaMark(id,\"done\"); return; }\n  if(inChat && chat){\n    /* La réponse prend la place de la bulle « Envoi… » : elle apparaît donc\n       exactement là où l'utilisateur l'attend, dès qu'elle est générée. */\n    const b=(ARENA_WAIT&&ARENA_WAIT.isConnected)?ARENA_WAIT:addAssistant(\"arena\");\n    ARENA_WAIT=null;\n    b.innerHTML=renderMarkdown(out);\n    msgActions(b,{getMd:()=>out});\n    conv.push({role:\"assistant\", text:out});\n    scrollDown();\n    await arenaSaveReply(chat, out, Number(d.ts)||0);\n  }else{\n    /* Sur une autre discussion : on enregistre, et la ligne épinglée gagne un\n       simple point (aucune fenêtre ne s'ouvre, rien ne recouvre l'écran). */\n    await arenaSaveReply(chat, out, Number(d.ts)||0);\n    if(!ARENA_DOT){ ARENA_DOT=true; renderChatList(); }\n  }\n  await arenaMark(id, \"done\");\n}\n/* ---- écoute du canal (temps réel) ---- */\nfunction arenaWatch(){\n  const box=arenaBoxRef();\n  if(!box) return false;\n  arenaStop();\n  try{\n    /* On écoute TOUT le canal et on filtre ici : aucune requête composée,\n       donc aucun index Firestore à créer (une requête à deux filtres peut\n       être refusée et donner l'impression que l'application ne répond plus). */\n    ARENA_UNSUB=box.onSnapshot(snap=>{\n      snap.docChanges().forEach(ch=>{\n        if(ch.type===\"modified\"){\n          const dm=ch.doc.data()||{};\n          if(dm.role===\"user\"&&typeof chipState===\"function\") chipState(\"arena\", ch.doc.id, dm.state);\n          return;\n        }\n        if(ch.type!==\"added\") return;\n        const d=ch.doc.data()||{};\n        if(d.role!==\"arena\"||d.state!==\"new\") return;\n        arenaDeliver(d, ch.doc.id).catch(e=>console.warn(\"canal arena :\",e));\n      });\n    }, err=>console.warn(\"canal arena :\", err && err.message));\n  }catch(e){ console.warn(\"canal arena :\",e); return false; }\n  return true;\n}\nfunction arenaWatchSafe(){\n  try{ return arenaWatch(); }catch(e){ console.warn(\"canal arena :\",e); return false; }\n}\nfunction arenaStop(){\n  if(ARENA_UNSUB){ try{ ARENA_UNSUB(); }catch(e){} ARENA_UNSUB=null; }\n}\n/* ---- envoi depuis l'application (modèle « arena » sélectionné) ---- */\nasync function submitArena(text){\n  busy=true; sendBtn.disabled=true;\n  try{\n    const kind = pendingAudio ? \"audio\" : (pendingImage ? \"upload\" : (useBlackImage ? \"black\" : \"none\"));\n    addUser(text, kind, pendingImage, pendingAudio);\n    const b=addAssistant(\"arena\");\n    ARENA_WAIT=b;\n    promptEl.value=\"\"; promptEl.style.height=\"auto\"; scrollDown();\n    b.innerHTML='<span class=\"gentxt\">Envoi<i>.</i><i>.</i><i>.</i></span>';\n    conv.push({role:\"user\", text:text});\n    /* 1. l'enregistrement local (au mieux, jamais bloquant) puis l'envoi. */\n    const j=await arenaSend(text, \"text\", \"\", \"\");\n    const dev=!!(j&&j.device);\n    const chat=await arenaSaveUser(text, (j&&j.id)||\"\");\n    if(j&&j.id&&typeof chipAttach===\"function\") chipAttach(\"arena\", j.id);\n    if(!j||!j.ok){\n      b.innerHTML='<div class=\"err\">Canal injoignable : '+esc((j&&j.error)||\"raison inconnue\")+'</div>';\n    }else if(dev){\n      b.innerHTML='<p style=\"color:var(--ink-2);font-size:13.5px\">Message transmis à Arena — la réponse arrivera ici même.</p>';\n    }else{\n      b.innerHTML='<p style=\"color:var(--ink-2);font-size:13.5px\">Message envoyé à Arena. Il n\\'est pas connecté en ce moment : la réponse apparaîtra ici dès qu\\'il reprendra le canal.</p>';\n    }\n    ARENA_LAST_USER=new Date().toISOString();\n    if(chat) renderChatList();\n  }catch(err){\n    /* Quoi qu'il arrive, l'application reste utilisable. */\n    try{\n      const b2=addAssistant(\"arena\");\n      b2.innerHTML='<div class=\"err\">Canal : '+esc(String(err&&err.message?err.message:err))+'</div>';\n    }catch(e){}\n  }finally{\n    busy=false; sendBtn.disabled=!promptEl.value.trim(); scrollDown();\n  }\n}\n\n/* --- authentification --------------------------------------------------- */\nfunction authErr(e){\n  const map={\n    \"auth/invalid-email\":\"Email invalide.\",\n    \"auth/user-not-found\":\"Aucun compte avec cet email.\",\n    \"auth/wrong-password\":\"Mot de passe incorrect.\",\n    \"auth/invalid-credential\":\"Email ou mot de passe incorrect.\",\n    \"auth/email-already-in-use\":\"Cet email est déjà utilisé.\",\n    \"auth/weak-password\":\"Mot de passe trop faible (6 caractères minimum).\",\n    \"auth/too-many-requests\":\"Trop de tentatives — réessaie dans quelques minutes.\",\n    \"auth/popup-closed-by-user\":\"Fenêtre Google fermée avant la fin.\",\n    \"auth/popup-blocked\":\"Le navigateur a bloqué la fenêtre Google.\",\n    \"auth/requires-recent-login\":\"Reconnecte-toi puis relance cette action.\",\n    \"auth/network-request-failed\":\"Réseau indisponible.\",\n    \"auth/operation-not-allowed\":\"Ce mode de connexion n'est pas activé sur le projet Firebase.\",\n    \"auth/account-exists-with-different-credential\":\"Un compte existe déjà avec cet email — saisis son mot de passe pour lier les deux méthodes.\",\n    \"auth/credential-already-in-use\":\"Ce compte Google est déjà rattaché à un autre compte.\",\n    \"auth/provider-already-linked\":\"Google est déjà lié à ce compte.\"\n  };\n  return map[e&&e.code] || (e&&e.message) || \"Erreur inconnue.\";\n}\nfunction showAuthErr(msg){ const el=$fb(\"#autherr\"); el.textContent=msg; el.hidden=!msg; }\n/* --- raccourci d'identifiant ---------------------------------------------\n   L'app mémorise, à chaque connexion réussie, un ou deux raccourcis pour le\n   compte (« hugdu77777 » plutôt que toute l'adresse). À la connexion, si ce\n   qui est saisi n'est pas une adresse, le raccourci est traduit en email.\n   Le tableau vit dans la passerelle (durable), donc il suit d'un appareil à\n   l'autre, et rien n'est stocké côté navigateur. */\nfunction aliasKeys(u){\n  const out=[];\n  const em=String((u&&u.email)||\"\").toLowerCase();\n  const pre=em.split(\"@\")[0];\n  const dn=String((u&&u.displayName)||\"\").toLowerCase().replace(/[^a-z0-9._-]/g,\"\");\n  for(const k of [pre, dn]) if(k && k.length>=3 && out.indexOf(k)<0) out.push(k);\n  return out;\n}\nasync function aliasSave(name, email){\n  try{\n    await fetch(API_BASE+\"/api/alias\",{ method:\"POST\",\n      headers:{\"Content-Type\":\"application/json\",\"X-Access-Code\":ACCESS_CODE},\n      body:JSON.stringify({name:name, email:email}) });\n  }catch(e){ /* sans réseau : le raccourci sera mémorisé une autre fois */ }\n}\nfunction aliasRemember(u){\n  if(!u) return;\n  const em=String(u.email||\"\");\n  if(!em) return;\n  for(const k of aliasKeys(u)) aliasSave(k, em);\n}\nasync function aliasLookup(name){\n  try{\n    const r=await fetch(API_BASE+\"/api/alias?name=\"+encodeURIComponent(String(name).toLowerCase()),\n      { headers:{\"X-Access-Code\":ACCESS_CODE} });\n    const j=await r.json();\n    return (j&&j.ok&&j.email)?String(j.email):\"\";\n  }catch(e){ return \"\"; }\n}\nfunction setErr(msg){ const el=$fb(\"#seterr\"); el.textContent=msg; el.hidden=!msg; }\nfunction setAuthMode(m){\n  authMode=m;\n  const login=(m===\"login\");\n  /* Le champ accepte le raccourci (identifiant) à la connexion, et une\n     adresse email à la création du compte. */\n  const em=$fb(\"#authemail\");\n  em.type = login ? \"text\" : \"email\";\n  em.setAttribute(\"autocomplete\", login ? \"username\" : \"email\");\n  em.placeholder = login ? \"Email ou identifiant…\" : \"Email…\";\n  const lab=$fb(\"#authemailLab\"); if(lab) lab.textContent = login ? \"Email ou identifiant\" : \"Email\";\n  $fb(\"#authpass\").placeholder = \"Mot de passe…\";\n  $fb(\"#authtitle\").textContent = m===\"login\" ? \"Se connecter\" : \"Créer un compte\";\n  $fb(\"#authsubmit\").textContent = m===\"login\" ? \"Se connecter\" : \"Créer un compte\";\n  $fb(\"#authtoggle\").textContent = m===\"login\" ? \"Créer un compte\" : \"J'ai déjà un compte\";\n  $fb(\"#authalt\").firstChild.textContent = m===\"login\" ? \"Pas encore de compte ? \" : \"Déjà inscrit ? \";\n  showAuthErr(\"\");\n}\nfunction openAuth(){ setAuthMode(\"login\"); $fb(\"#gpasswrap\").hidden=true; $fb(\"#authmodal\").hidden=false; }\n\n/* --- avatar + utilisateur en bas du menu ------------------------------- */\nfunction avatarColor(seed){\n  const pal=[\"#e0562d\",\"#c2410c\",\"#0e7490\",\"#15803d\",\"#7c3aed\",\"#b45309\",\"#be185d\",\"#1d4ed8\",\"#4d7c0f\",\"#9333ea\"];\n  let h=0; const s=seed||\"?\";\n  for(let i=0;i<s.length;i++) h=(h*31+s.charCodeAt(i))>>>0;\n  return pal[h%pal.length];\n}\nfunction profileName(){\n  return userName || (fbUser ? (fbUser.displayName || (fbUser.email||\"\").split(\"@\")[0]) : \"\") || \"Utilisateur\";\n}\nfunction avatarHTML(name, photo, size){\n  const s=size||30;\n  if(photo) return '<img class=\"avat\" src=\"'+esc(photo)+'\" alt=\"\" style=\"width:'+s+'px;height:'+s+'px\">';\n  const L=((name||\"?\").trim().charAt(0)||\"?\").toUpperCase();\n  return '<span class=\"avat\" style=\"width:'+s+'px;height:'+s+'px;background:'\n    +(fbUser?userColor:avatarColor(name))+'\">'+esc(L)+'</span>';\n}\nfunction renderSideUser(){\n  const su=$fb(\"#sideuser\");\n  if(fbUser){\n    su.innerHTML=avatarHTML(profileName(), userPhoto||fbUser.photoURL||\"\")\n      +'<span class=\"un\">'+esc(profileName())+'</span>';\n    su.title=\"Paramètres du compte\";\n    su.onclick=openAccountSettings;\n  }else{\n    su.innerHTML='<div class=\"sideauth\" style=\"width:100%\"><button type=\"button\" class=\"pri\">Se connecter / Créer un compte</button></div>';\n    su.title=\"\";\n    su.onclick=(e)=>{ e.stopPropagation(); if(FB_OK) openAuth(); else alert(\"Firebase n'est pas configuré (clé web manquante) : la connexion est indisponible.\"); };\n  }\n}\n\n/* --- réglages plein écran + popups -------------------------------------- */\nfunction openAccountSettings(){\n  if(!fbUser){ openAuth(); return; }\n  fillSettings();\n  $fb(\"#settings\").hidden=false;\n}\nfunction fillSettings(){\n  $fb(\"#settitle\").textContent=profileName();\n  $fb(\"#setsub\").textContent=fbUser.email||\"\";\n  $fb(\"#setnameval\").textContent=profileName();\n  $fb(\"#setemailval\").textContent=fbUser.email||\"(aucun)\";\n  const hasG=(fbUser.providerData||[]).some(p=>p.providerId===\"google.com\");\n  $fb(\"#setgoogle\").textContent=hasG?\"Ouvrir mon compte Google\":\"Lier un compte Google\";\n  $fb(\"#setgooglenote\").textContent=hasG\n    ? \"Compte Google lié à ce profil.\"\n    : \"Permet de te connecter sans mot de passe, et ouvrira ton compte Google une fois lié.\";\n  renderSetAvatar();\n  setErr(\"\");\n  hidePop();\n}\nfunction renderSetAvatar(){\n  const av=$fb(\"#setavatar\");\n  const photo=userPhoto||fbUser.photoURL||\"\";\n  if(photo) av.innerHTML='<img src=\"'+esc(photo)+'\" alt=\"\" style=\"width:100%;height:100%;object-fit:cover\">';\n  else av.innerHTML='<span style=\"font:600 44px/1 var(--sans);color:#fff\">'\n    +esc(((profileName()||\"?\").trim().charAt(0)||\"?\").toUpperCase())+'</span>';\n  av.style.background=photo?\"#eee\":userColor;\n}\nlet popMode=null;\nfunction hidePop(){ $fb(\"#setpop\").hidden=true; popMode=null; }\nfunction popErr(m){ const e=$fb(\"#setpoperr\"); e.textContent=m; e.hidden=!m; }\nfunction openPop(mode){\n  popMode=mode;\n  const body=$fb(\"#setpopbody\"); body.innerHTML=\"\";\n  const T=$fb(\"#setpoptitle\");\n  const mk=(type,id,lab,ph)=>{\n    const l=document.createElement(\"label\"); l.className=\"flab\"; l.textContent=lab;\n    const i=document.createElement(\"input\"); i.type=type; i.id=id; i.placeholder=ph||\"\";\n    if(type===\"email\") i.autocomplete=\"email\";\n    if(type===\"password\"){\n      i.autocomplete=(id===\"popcurpass\"?\"current-password\":\"new-password\");\n      l.appendChild(i); body.appendChild(l);\n      if(typeof pwPeek===\"function\") pwPeek(i);\n      return i;\n    }\n    l.appendChild(i); body.appendChild(l); return i;\n  };\n  if(mode===\"name\"){ T.textContent=\"Renommer le profil\"; mk(\"text\",\"popname\",\"Nouveau nom\",profileName()); }\n  const popHasPass=(fbUser.providerData||[]).some(p=>p.providerId===\"password\");\n  if(mode===\"email\"){ T.textContent=\"Changer d'adresse email\";\n    mk(\"email\",\"popemail\",\"Nouvelle adresse\",fbUser.email||\"\");\n    if(popHasPass) mk(\"password\",\"popcurpass\",\"Mot de passe actuel (confirmation)\"); }\n  if(mode===\"pass\"){ T.textContent=popHasPass?\"Changer de mot de passe\":\"Définir un mot de passe\";\n    if(popHasPass) mk(\"password\",\"popcurpass\",\"Mot de passe actuel\");\n    mk(\"password\",\"popnewpass\",popHasPass?\"Nouveau mot de passe\":\"Mot de passe\");\n    mk(\"password\",\"popnewpass2\",\"Confirmer le nouveau mot de passe\"); }\n  if(mode===\"del\"){ T.textContent=\"Supprimer le compte\";\n    body.insertAdjacentHTML(\"beforeend\",\n      '<p style=\"font-size:12.5px;color:var(--ink-2);margin:0 0 10px\">Action définitive : profil, discussions et messages seront effacés.</p>');\n    const delHasPass=(fbUser.providerData||[]).some(p=>p.providerId===\"password\");\n    if(delHasPass) mk(\"password\",\"popcurpass\",\"Mot de passe (confirmation)\");\n    else body.insertAdjacentHTML(\"beforeend\",\n      '<p style=\"font-size:12.5px;color:var(--ink-2);margin:0 0 10px\">Ce compte a été créé avec Google : il sera supprimé directement, sans confirmation supplémentaire.</p>'); }\n  popErr(\"\");\n  $fb(\"#setpop\").hidden=false;\n  const f=body.querySelector(\"input\"); if(f) f.focus();\n}\nasync function reauth(pass){\n  const cred=firebase.auth.EmailAuthProvider.credential(fbUser.email, pass);\n  await fbUser.reauthenticateWithCredential(cred);\n}\n\n/* --- pièces jointes multiples -------------------------------------------- */\nfunction syncMirrors(){\n  const im=pendingFiles.find(f=>f.kind===\"image\");\n  pendingImage=im?{name:im.name, b64:im.b64, small:im.small, mime:im.mime, kb:im.kb||0}:null;\n  if(!pendingFiles.some(f=>f.kind===\"audio\")) pendingAudio=null;\n}\nfunction setAttachImage(im){\n  /* comme un chargement depuis l'appareil : l'image se place à côté */\n  if(!pendingFiles.some(f=>f.kind===\"image\"&&f.b64===im.b64)){\n    if(pendingFiles.filter(f=>f.kind===\"image\").length>=4){ alert(\"Maximum 4 images par message.\"); return; }\n    pendingFiles.push({kind:\"image\", name:im.name, b64:im.b64, small:im.small, tiny:im.tiny, mime:im.mime, kb:im.kb||0});\n  }\n  syncMirrors(); renderPending();\n}\nfunction setAttachAudio(a){\n  pendingFiles=[{kind:\"audio\", name:a.name}];\n  pendingImage=null; syncMirrors(); renderPending();\n  pendingAudio=a; renderPending();\n}\nfunction clearAttachments(){\n  pendingFiles=[]; pendingImage=null; pendingAudio=null; renderPending();\n}\n/* Une puce par pièce jointe ; défilement horizontal de la zone seule. */\nfunction renderPending(){\n  const pend=$fb(\"#pending\");\n  pend.innerHTML=\"\";\n  for(const f of pendingFiles){\n    const chip=document.createElement(\"span\"); chip.className=\"pchip\";\n    if(f.kind===\"image\"){\n      const img=document.createElement(\"img\"); img.alt=\"\";\n      img.src=\"data:image/png;base64,\"+(f.small||f.b64);\n      chip.appendChild(img);\n    }else{\n      const ico=document.createElement(\"span\");\n      ico.textContent=(f.kind===\"audio\"?\"🎙\":\"📎\"); chip.appendChild(ico);\n    }\n    const nm=document.createElement(\"span\"); nm.className=\"pn\";\n    nm.textContent=f.name+(f.kb?\" · \"+f.kb+\" Ko\":\"\");\n    nm.title=f.name;\n    chip.appendChild(nm);\n    const x=document.createElement(\"button\"); x.type=\"button\"; x.textContent=\"×\";\n    x.title=\"Retirer \"+f.name; x.setAttribute(\"aria-label\",\"Retirer \"+f.name);\n    x.onclick=()=>{ pendingFiles=pendingFiles.filter(z=>z!==f);\n      if(f.kind===\"audio\") pendingAudio=null; syncMirrors(); renderPending(); };\n    chip.appendChild(x);\n    pend.appendChild(chip);\n  }\n  pend.classList.toggle(\"on\", pendingFiles.length>0);\n  updateHint();\n}\n\n/* --- câblage des panneaux ------------------------------------------------- */\n(function wire(){\n  $fb(\"#burger\").onclick=()=>{ $fb(\"#sidebar\").classList.contains(\"open\") ? closeSide() : openSide(); };\n  $fb(\"#sbackdrop\").onclick=closeSide;\n  $fb(\"#newchat\").onclick=()=>{ newConversation(); closeSide(); };\n\n  $fb(\"#authclose\").onclick=()=>{ $fb(\"#authmodal\").hidden=true; };\n  $fb(\"#authtoggle\").onclick=()=>setAuthMode(authMode===\"login\"?\"signup\":\"login\");\n  /* Fusion Google ⇄ compte email existant : si un doublon est détecté,\n     on guide l'utilisateur pour lier les deux méthodes sur UN seul compte. */\n  let pendingLink=null;\n  $fb(\"#googbtn\").onclick=async()=>{\n    if(!FB_OK) return showAuthErr(\"Firebase n'est pas configuré (clé web manquante).\");\n    showAuthErr(\"\");\n    try{\n      const res=await fbAuth.signInWithPopup(new firebase.auth.GoogleAuthProvider());\n      const email=(res.user&&res.user.email)||\"\";\n      let methods=[];\n      try{\n        if(email&&fbAuth.fetchSignInMethodsForEmail) methods=await fbAuth.fetchSignInMethodsForEmail(email)||[];\n      }catch(e){}\n      const hasPassOnUser=(res.user.providerData||[]).some(p=>p.providerId===\"password\");\n      if(email&&methods.includes(\"password\")&&!hasPassOnUser){\n        /* Un compte mot de passe existe déjà avec cet email : le compte Google\n           qui vient d'être créé est un doublon → on propose la fusion. */\n        const cred=firebase.auth.GoogleAuthProvider.credentialFromResult\n          ? firebase.auth.GoogleAuthProvider.credentialFromResult(res) : null;\n        pendingLink={email:email, cred:cred, gUser:res.user};\n        setAuthMode(\"login\");\n        $fb(\"#authemail\").value=email;\n        $fb(\"#authpass\").value=\"\";\n        showAuthErr(\"Un compte existe déjà avec \"+email+\". Saisis son mot de passe : le doublon Google sera supprimé et Google sera lié à ton compte — les deux méthodes fonctionneront.\");\n        $fb(\"#authpass\").focus();\n        return;\n      }\n      if(!hasPassOnUser){\n        /* Compte Google tout neuf : proposer de définir un mot de passe pour\n           avoir directement les deux méthodes (email+mdp et Google). */\n        setTimeout(()=>{\n          $fb(\"#authmodal\").hidden=false;\n          $fb(\"#gpasswrap\").hidden=false;\n          const g1=$fb(\"#gpass1\");\n          if(g1&&typeof pwPeek===\"function\"){ pwPeek(g1); pwPeek($fb(\"#gpass2\")); }\n          if(g1) g1.focus();\n        },400);\n      }\n    }catch(e){\n      if(e.code===\"auth/account-exists-with-different-credential\"&&e.email){\n        pendingLink={email:e.email, cred:e.credential||null, gUser:null};\n        setAuthMode(\"login\");\n        $fb(\"#authemail\").value=e.email;\n        $fb(\"#authpass\").value=\"\";\n        showAuthErr(\"Un compte existe déjà avec cet email. Saisis son mot de passe pour lier Google et utiliser les deux méthodes.\");\n        $fb(\"#authpass\").focus();\n        return;\n      }\n      showAuthErr(authErr(e));\n    }\n  };\n  $fb(\"#authforgot\").onclick=async()=>{\n    if(!FB_OK) return showAuthErr(\"Firebase n'est pas configuré (clé web manquante).\");\n    const email=$fb(\"#authemail\").value.trim();\n    if(!email) return showAuthErr(\"Saisis d'abord ton email ci-dessus.\");\n    try{\n      await fbAuth.sendPasswordResetEmail(email);\n      showAuthErr(\"Email de réinitialisation envoyé ✓\");\n    }catch(e){ showAuthErr(authErr(e)); }\n  };\n  $fb(\"#authsubmit\").onclick=async()=>{\n    if(!FB_OK) return showAuthErr(\"Firebase n'est pas configuré (clé web manquante).\");\n    let email=$fb(\"#authemail\").value.trim(), pass=$fb(\"#authpass\").value;\n    if(!email||!pass) return showAuthErr(\"Email (ou identifiant) et mot de passe requis.\");\n    /* Raccourci : « hugdu77777 » au lieu de toute l'adresse. Le raccourci est\n       mémorisé par la passerelle à chaque connexion réussie. */\n    if(authMode===\"login\" && email.indexOf(\"@\")<0){\n      showAuthErr(\"Recherche de l'identifiant…\");\n      const found=await aliasLookup(email);\n      if(!found){\n        showAuthErr(\"Identifiant « \"+email+\" » inconnu. Connecte-toi une fois avec ton adresse email : le raccourci sera mémorisé.\");\n        return;\n      }\n      email=found;\n    }\n    showAuthErr(\"\");\n    try{\n      if(authMode===\"login\"){\n        /* Fusion en attente : supprimer le doublon Google (encore frais) AVANT\n           de basculer sur le compte mot de passe, puis lier Google dessus. */\n        if(pendingLink&&pendingLink.gUser){\n          try{ await pendingLink.gUser.delete(); }catch(e){ console.warn(\"doublon :\", e); }\n        }\n        const u=(await fbAuth.signInWithEmailAndPassword(email,pass)).user;\n        aliasRemember(u);            /* le raccourci est prêt pour la prochaine fois */\n        if(pendingLink){\n          const cred=pendingLink.cred;\n          pendingLink=null;\n          if(cred){\n            try{ await u.linkWithCredential(cred); showAuthErr(\"\"); }\n            catch(e){ showAuthErr(\"Connecté ✓ — mais Google n'a pas pu être lié : \"+authErr(e)); }\n          }\n        }\n      }\n      else{\n        const cred=await fbAuth.createUserWithEmailAndPassword(email,pass);\n        await cred.user.updateProfile({displayName:email.split(\"@\")[0]});\n        aliasRemember(cred.user);\n      }\n      $fb(\"#authmodal\").hidden=true;\n    }catch(e){ showAuthErr(authErr(e)); }\n  };\n\n  /* Définir un mot de passe juste après une connexion Google */\n  $fb(\"#gpassok\").onclick=async()=>{\n    if(!fbUser) return showAuthErr(\"Connecte-toi d'abord.\");\n    const p1=$fb(\"#gpass1\").value, p2=$fb(\"#gpass2\").value;\n    if(p1.length<6) return showAuthErr(\"Mot de passe trop faible (6 caractères minimum).\");\n    if(p1!==p2) return showAuthErr(\"Les deux mots de passe ne correspondent pas.\");\n    try{\n      const cred=firebase.auth.EmailAuthProvider.credential(fbUser.email, p1);\n      await fbUser.linkWithCredential(cred);\n      $fb(\"#gpass1\").value=\"\"; $fb(\"#gpass2\").value=\"\";\n      $fb(\"#gpasswrap\").hidden=true; $fb(\"#authmodal\").hidden=true;\n    }catch(e){ showAuthErr(authErr(e)); }\n  };\n  $fb(\"#gpassskip\").onclick=()=>{ $fb(\"#gpasswrap\").hidden=true; $fb(\"#authmodal\").hidden=true; };\n\n  $fb(\"#setclose\").onclick=()=>{ $fb(\"#settings\").hidden=true; };\n  $fb(\"#setphoto\").onclick=()=>$fb(\"#setphotofile\").click();\n  $fb(\"#setphotofile\").onchange=async()=>{\n    const inp=$fb(\"#setphotofile\");\n    const f=inp.files&&inp.files[0]; inp.value=\"\";\n    if(!f) return;\n    try{\n      const im=await fileToImage(f);\n      const el=await loadImgEl(\"data:image/png;base64,\"+im.b64);\n      const S=256;\n      let w=el.naturalWidth||el.width, h=el.naturalHeight||el.height;\n      const r=Math.min(S/w, S/h, 1);\n      const c=document.createElement(\"canvas\");\n      c.width=Math.max(1,Math.round(w*r)); c.height=Math.max(1,Math.round(h*r));\n      const x=c.getContext(\"2d\"); x.fillStyle=\"#fff\"; x.fillRect(0,0,c.width,c.height);\n      x.drawImage(el,0,0,c.width,c.height);\n      userPhoto=c.toDataURL(\"image/jpeg\",0.85);\n      if(FB_OK&&fbUser) await fbDB.collection(\"users\").doc(fbUser.uid)\n        .set({photoURL:userPhoto, updatedAt:Date.now()},{merge:true});\n      renderSetAvatar(); renderSideUser();\n    }catch(e){ setErr(\"Photo illisible.\"); }\n  };\n  $fb(\"#setnamebtn\").onclick=()=>openPop(\"name\");\n  $fb(\"#setemailbtn\").onclick=()=>openPop(\"email\");\n  $fb(\"#setpassbtn\").onclick=()=>openPop(\"pass\");\n  $fb(\"#setpopcancel\").onclick=hidePop;\n  $fb(\"#setpopok\").onclick=async()=>{\n    if(!fbUser) return popErr(\"Connecte-toi d'abord.\");\n    try{\n      if(popMode===\"name\"){\n        const n=$fb(\"#popname\").value.trim();\n        if(!n) return popErr(\"Nom vide.\");\n        await fbUser.updateProfile({displayName:n});\n        userName=n;\n        if(FB_OK) await fbDB.collection(\"users\").doc(fbUser.uid)\n          .set({displayName:n, updatedAt:Date.now()},{merge:true});\n      }\n      if(popMode===\"email\"){\n        const cpEl=$fb(\"#popcurpass\");\n        const e2=$fb(\"#popemail\").value.trim(), cp=cpEl?cpEl.value:\"\";\n        if(!e2) return popErr(\"Adresse vide.\");\n        if(cpEl){\n          if(!cp) return popErr(\"Mot de passe actuel requis.\");\n          await reauth(cp);\n        }\n        await fbUser.updateEmail(e2);\n        if(FB_OK) await fbDB.collection(\"users\").doc(fbUser.uid)\n          .set({email:e2, updatedAt:Date.now()},{merge:true});\n      }\n      if(popMode===\"pass\"){\n        const cpEl=$fb(\"#popcurpass\");\n        const cp=cpEl?cpEl.value:\"\", n1=$fb(\"#popnewpass\").value, n2=$fb(\"#popnewpass2\").value;\n        if(n1.length<6) return popErr(\"Mot de passe trop faible (6 caractères minimum).\");\n        if(n1!==n2) return popErr(\"Les deux mots de passe ne correspondent pas.\");\n        if(cpEl){\n          if(!cp) return popErr(\"Mot de passe actuel requis.\");\n          await reauth(cp);\n        }\n        await fbUser.updatePassword(n1);\n      }\n      if(popMode===\"del\"){\n        const delHasPass=(fbUser.providerData||[]).some(p=>p.providerId===\"password\");\n        if(delHasPass){\n          const cpEl=$fb(\"#popcurpass\");\n          const cp=cpEl?cpEl.value:\"\";\n          if(!cp) return popErr(\"Mot de passe requis.\");\n          await reauth(cp);\n        }\n        /* Compte Google : suppression directe, sans vérification (voulue ainsi). */\n        for(const c of chats.slice()){ try{ await purgeChat(c.id); }catch(e){} }\n        await fbDB.collection(\"users\").doc(fbUser.uid).delete();\n        await fbUser.delete();\n        hidePop();\n        $fb(\"#settings\").hidden=true;\n        return;\n      }\n      renderSideUser(); fillSettings(); hidePop();\n    }catch(e){ popErr(authErr(e)); }\n  };\n  $fb(\"#setgoogle\").onclick=async()=>{\n    if(!fbUser) return setErr(\"Connecte-toi d'abord.\");\n    const hasG=(fbUser.providerData||[]).some(p=>p.providerId===\"google.com\");\n    if(hasG){ window.open(\"https://myaccount.google.com\",\"_blank\",\"noopener\"); return; }\n    try{ await fbUser.linkWithPopup(new firebase.auth.GoogleAuthProvider()); setErr(\"\"); fillSettings(); }\n    catch(e){ setErr(authErr(e)); }\n  };\n  $fb(\"#setsignout\").onclick=async()=>{\n    try{ await fbAuth.signOut(); }catch(e){}\n    $fb(\"#settings\").hidden=true;\n  };\n  $fb(\"#setdelete\").onclick=()=>{ if(fbUser) openPop(\"del\"); };\n\n  /* Plusieurs fichiers à la fois : images (max 4), un audio (micro),\n     fichiers texte (contenu transmis au modèle). */\n  filePick.onchange=async()=>{\n    const list=Array.from(filePick.files||[]);\n    filePick.value=\"\";\n    for(const f of list){\n      if(pendingFiles.length>=6){ alert(\"Maximum 6 pièces jointes par message.\"); break; }\n      const ext=(f.name.split(\".\").pop()||\"\").toLowerCase();\n      const isImg=/^image[/]/.test(f.type)||/^(png|jpe?g|gif|webp|bmp|avif|heic|heif)$/.test(ext);\n      const isAud=/^audio[/]/.test(f.type)||/^(mp3|wav|ogg|oga|m4a|aac|flac|opus|weba|webm|amr)$/.test(ext);\n      try{\n        if(isImg && pendingFiles.filter(x=>x.kind===\"image\").length<4){\n          const im=await fileToImage(f);\n          pendingFiles.push({kind:\"image\", name:im.name, b64:im.b64, kb:im.kb});\n        }else if(isAud && !pendingAudio){\n          pendingAudio=await fileToAudio(f);\n          pendingFiles.push({kind:\"audio\", name:pendingAudio.name});\n        }else{\n          let text=null;\n          if(f.size<=102400 && (/^text[/]/.test(f.type)\n             || /^(txt|md|csv|json|js|py|html|css|xml|yml|yaml|log|ts|jsx|sh)$/i.test(ext)\n             || !f.type)){\n            try{ text=typeof f.text===\"function\" ? await f.text() : null; }catch(e){ text=null; }\n          }\n          pendingFiles.push({kind:\"file\", name:f.name, text:text||undefined,\n            kb:Math.max(1,Math.round(f.size/1024))});\n        }\n      }catch(err){ alert(\"Fichier illisible : \"+(err&&err.message?err.message:err)); }\n    }\n    syncMirrors(); renderPending();\n  };\n})();\n\n/* Points d'entrée appelés par le bloc principal. */\nconst FB={ afterText:afterText, afterImage:afterImage, afterVideo:afterVideo, onNewChat:onNewChat, afterAgent:afterAgent };\n\n</script>\n\n<!-- ============ MODE AGENT (phase 37) ============ -->\n<div id=\"agentPanel\" hidden>\n  <div class=\"ap-head\">\n    <b>Agent · VM Linux</b>\n    <button id=\"agentPanelClose\" class=\"iconbtn\" type=\"button\" aria-label=\"Fermer le panneau\">\n      <svg width=\"14\" height=\"14\" viewBox=\"0 0 24 24\" fill=\"none\" stroke=\"currentColor\" stroke-width=\"2\" stroke-linecap=\"round\"><path d=\"M5 5l14 14M19 5L5 19\"/></svg>\n    </button>\n  </div>\n  <div class=\"ap-body\">\n    <div class=\"ap-screen\"><img id=\"agentScreen\" alt=\"Écran de la VM en direct\"></div>\n    <div class=\"ap-tree\" id=\"agentTree\"></div>\n  </div>\n</div>\n\n<div id=\"agentViewer\" hidden>\n  <div class=\"av-head\">\n    <span id=\"avTitle\"></span>\n    <span id=\"avBtns\"></span>\n    <button id=\"agentViewerClose\" class=\"iconbtn\" type=\"button\" aria-label=\"Fermer\">\n      <svg width=\"14\" height=\"14\" viewBox=\"0 0 24 24\" fill=\"none\" stroke=\"currentColor\" stroke-width=\"2\" stroke-linecap=\"round\"><path d=\"M5 5l14 14M19 5L5 19\"/></svg>\n    </button>\n  </div>\n  <div class=\"av-body\" id=\"avBody\"></div>\n</div>\n\n<script>\n/* ================= MODE AGENT (phase 38) =============================== */\nconst AGENT_EMAIL=\"hugdu77777@gmail.com\";\nlet AGENT_ON=false, AGENT_BUSY=false, AGENT_MD=null, AGENT_INFOS=null, AGT_TIMER=null, AGT_OBJURL=null, AGT_TOAST_T=null;\nconst AGENT_USER=\"user\";\nconst agtBtn=document.getElementById(\"agentbtn\");\nconst agtFilesBtn=document.getElementById(\"agentfiles\");\nconst agtPanel=document.getElementById(\"agentPanel\");\nconst agtViewer=document.getElementById(\"agentViewer\");\n\nfunction agtHeaders(extra){\n  return Object.assign({\"X-Access-Code\":ACCESS_CODE}, extra||{});\n}\nfunction agtLogged(){\n  return (typeof fbUser!==\"undefined\"&&!!fbUser\n    &&String(fbUser.email||\"\").toLowerCase()===AGENT_EMAIL);\n}\n/* --- missions en arrière-plan : mémoire locale, suivi, reprise ----------\n   La mission est exécutée par la passerelle (worker) et non par le\n   navigateur : elle continue donc même si l'app est fermée. L'app garde son\n   identifiant et s'y raccroche au chargement suivant. */\nlet AGT_JOB_OWN=null;        /* mission affichée en direct sur cette page */\nlet AGT_JOB_CHAT=null;       /* sa discussion */\nlet AGT_DETACH=false;        /* on lâche la mission : elle continue seule */\n/* Registre local des missions : plusieurs peuvent tourner en même temps, une\n   par discussion (le marqueur qui fait foi, lui, est dans Firestore, donc il\n   est visible depuis n'importe quel appareil). */\nfunction agtJobsLoad(){\n  try{\n    const s=localStorage.getItem(\"agt_jobs\"); const m=s?JSON.parse(s):{};\n    return (m&&typeof m===\"object\")?m:{};\n  }catch(e){ return {}; }\n}\nfunction agtJobsSave(m){ try{ localStorage.setItem(\"agt_jobs\", JSON.stringify(m||{})); }catch(e){} }\nfunction agtJobPut(job){\n  if(!job||!job.id||!job.chat) return;\n  const m=agtJobsLoad();\n  m[job.chat]={id:job.id, chat:job.chat, mission:job.mission||\"\", model:job.model||\"\", at:job.at||Date.now()};\n  agtJobsSave(m);\n}\nfunction agtJobDrop(chat){\n  if(!chat) return;\n  const m=agtJobsLoad();\n  if(m[chat]){ delete m[chat]; agtJobsSave(m); }\n}\nfunction agtJobLocal(chat){ const m=agtJobsLoad(); return (chat&&m[chat])?m[chat]:null; }\n/* Les missions que cet appareil a lancées se déclarent à la passerelle : une\n   tâche née avant le registre devient ainsi listable — donc arrêtable — même\n   si l'app ne la suit plus. */\nasync function agtJobsDeclare(){\n  let jobs=[];\n  try{\n    const m=agtJobsLoad();\n    jobs=Object.keys(m).map(function(k){ return m[k]; })\n      .filter(function(j){ return !!(j&&j.id); })\n      .map(function(j){ return {id:String(j.id), chat:String(j.chat||\"\"), model:String(j.model||\"\"),\n        mission:String(j.mission||\"\"), at:Number(j.at||0)||0, state:\"run\"}; });\n  }catch(e){ return; }\n  if(!jobs.length) return;\n  try{\n    await fetch(API_BASE+\"/api/agent/jobs/sync\",{method:\"POST\",\n      headers:agtHeaders({\"Content-Type\":\"application/json\"}),\n      body:JSON.stringify({jobs:jobs})});\n  }catch(e){}\n}\n/* Lâcher la mission affichée : elle continue sur la passerelle et son résultat\n   arrivera dans sa discussion. */\nasync function agtDetachNow(){\n  if(!AGT_JOB_OWN) return;\n  AGT_DETACH=true;\n  for(let i=0;i<60&&AGT_JOB_OWN;i++) await new Promise(r=>setTimeout(r,100));\n  AGT_DETACH=false;\n}\n/* Le marqueur « en cours » de la discussion, écrit dans Firestore : il dit à\n   tous les appareils qu'une mission tourne là. */\nasync function agtSetJobMark(cid, job, state){\n  if(!cid) return;\n  const c=(typeof chats!==\"undefined\")?chats.find(x=>x.id===cid):null;\n  const mark={id:String((job&&job.id)||(c&&c.job&&c.job.id)||\"\"), state:state||\"run\",\n    at:Date.now(), mission:String((job&&job.mission)||(c&&c.job&&c.job.mission)||\"\"),\n    model:String((job&&job.model)||(c&&c.job&&c.job.model)||\"\")};\n  if(c){ c.job=mark; renderChatList(); }\n  if(typeof FB===\"undefined\"||!FB_OK||!fbUser) return;\n  try{ await chatsRef().doc(cid).set({job:mark},{merge:true}); }catch(e){}\n}\n/* La passerelle range le résultat elle-même : on vérifie qu'elle l'a fait. */\nasync function agtDelivered(cid){\n  if(!cid||typeof FB===\"undefined\"||!FB_OK||!fbUser) return false;\n  for(let i=0;i<4;i++){\n    let ok=true;\n    try{\n      const d=await chatsRef().doc(cid).get();\n      const j=(d.data()||{}).job;\n      if(j&&j.state===\"done\") return true;\n    }catch(e){ ok=false; }        /* Firebase injoignable : inutile d'attendre */\n    if(!ok) return false;\n    if(i<3) await new Promise(r=>setTimeout(r,3000));\n  }\n  return false;\n}\n/* Rien ne doit se perdre : si la passerelle n'a pas pu ranger le résultat,\n   c'est la page qui l'enregistre — une seule fois. */\nasync function agtEnsureSaved(info, mission, stepsArr, finalText){\n  const cid=String((info&&info.chat)||\"\");\n  if(!cid) return;\n  if(await agtDelivered(cid)){ agtJobDrop(cid); return; }\n  if(typeof FB!==\"undefined\"&&FB.afterAgent){\n    try{ await FB.afterAgent({userText:mission, steps:stepsArr, out:finalText, chatId:cid}); }catch(e){}\n  }\n  agtJobDrop(cid);\n  await agtSetJobMark(cid, info, finalText?\"done\":\"error\");\n}\nasync function agtJobState(id,since){\n  try{\n    /* `since` : ne recevoir que les étapes pas encore affichées — une mission\n       très longue ne retransfère donc jamais tout son déroulé. */\n    const r=await fetch(API_BASE+\"/api/agent/job?id=\"+encodeURIComponent(id)\n      +(since?(\"&since=\"+encodeURIComponent(since)):\"\"),{headers:agtHeaders()});\n    const j=await r.json();\n    return (j&&j.ok&&j.job)?j.job:null;\n  }catch(e){ return null; }\n}\n/* Suit une mission jusqu'à la fin : chaque nouvelle étape est confiée à\n   onStep() pour l'affichage, puis on attend la suivante. */\nasync function agtJobFollow(id, ui, initial){\n  ui=ui||{};\n  let seen=0, first=(initial||null);\n  for(;;){\n    /* On peut lâcher la mission à tout moment (changement de discussion,\n       nouveau message) : elle continue sur la passerelle. */\n    if(ui.detached && ui.detached()) return {detached:true, seen:seen};\n    const st=first||await agtJobState(id,seen);\n    first=null;\n    if(!st) return {lost:true, seen:seen};\n    const list=st.steps||[];\n    /* La passerelle renvoie les étapes à partir de `stepsFrom` (celles déjà\n       affichées ne sont plus transmises) ; `stepsTotal` donne le nombre total.\n       Si elle renvoie tout le déroulé (passerelle d'une version antérieure),\n       on reprend simplement après ce qui est déjà affiché : jamais de doublon. */\n    const from=(st.stepsFrom===undefined)?0:(Number(st.stepsFrom)||0);\n    let i=Math.max(0,seen-from);\n    const L=(list.length>i)?ui.ensureTerm():null;\n    for(;i<list.length;i++){ if(ui.onStep) ui.onStep(list[i]||{}, L); }\n    seen=from+list.length;\n    if(st.stepsTotal!==undefined) seen=Math.max(seen,Number(st.stepsTotal)||0);\n    if(ui.afterPoll) ui.afterPoll();\n    if(st.done) return {done:true, final:String(st.final||\"\"), error:String(st.error||\"\"), seen:seen};\n    await new Promise(r=>setTimeout(r,1500));\n  }\n}\n/* Reprise au rechargement : une mission lancée juste avant la fermeture de\n   l'app a continué sur la passerelle ; on la réaffiche et on l'enregistre. */\n/* ---- attacher une mission en cours à sa discussion ---------------------\n   Une mission vit sur la passerelle : rouvrir sa discussion la montre en\n   direct, où qu'on soit et quel que soit l'appareil. Rien n'est réenregistré\n   ici — la passerelle écrit le résultat dans SA discussion (agtEnsureSaved\n   sert de repli si elle n'a pas pu). */\n/* Arrêter une mission : la passerelle tranche tout de suite (l'étape en\n   cours, elle, ne s'interrompt pas — mais elle ne sera plus jamais suivie). */\nasync function agtStopJob(id, cid){\n  if(!id) return;\n  if(!confirm(\"Arrêter cette mission ?\")) return;\n  let ok=false;\n  try{\n    const r=await fetch(API_BASE+\"/api/agent/job/stop?id=\"+encodeURIComponent(id),{method:\"POST\",headers:agtHeaders()});\n    const j=await r.json().catch(()=>({}));\n    ok=!!(j&&j.ok);\n  }catch(e){}\n  agtJobDrop(cid||AGT_JOB_CHAT||\"\");\n  agtSetJobMark(cid||AGT_JOB_CHAT,null,\"done\").catch(()=>{});\n  agtToast(ok?\"Mission arrêtée.\":\"Arrêt impossible : la passerelle n'a pas répondu.\");\n}\n\nasync function agtAttachJob(info, cid){\n  if(!info||!info.id||!cid) return false;\n  if(AGT_JOB_OWN) return false;                 /* une seule mission affichée */\n  const st0=await agtJobState(info.id);\n  if(!st0){\n    /* La passerelle ne connaît plus cette mission : le marqueur ne doit pas\n       rester « en cours » pour toujours. */\n    agtJobDrop(cid);\n    const c0=(typeof chats!==\"undefined\")?chats.find(x=>x.id===cid):null;\n    if(c0&&c0.job){ c0.job=null; renderChatList(); }\n    return false;\n  }\n  const mission=String(info.mission||st0.mission||\"(mission)\");\n  AGT_JOB_OWN=info.id; AGT_JOB_CHAT=cid;\n  if(typeof AGENT_ON!==\"undefined\"&&!AGENT_ON){\n    /* Une mission rattachée à sa discussion : cette discussion est en mode\n       agent (c'est le mode qui commande l'en-tête, plus un interrupteur). */\n    if(typeof CUR_MODE!==\"undefined\") CUR_MODE=\"agent\";\n    AGENT_ON=true; try{ agtBtn.classList.add(\"on\"); }catch(e){} agtSyncUI();\n  }\n  addUser(mission,\"none\",null,null);\n  const bubble=addAssistant();\n  const stopBtn=document.createElement(\"button\");\n  stopBtn.type=\"button\"; stopBtn.className=\"mstop\"; stopBtn.textContent=\"Arrêter la mission\";\n  stopBtn.onclick=function(){ agtStopJob(info.id, cid); };\n  bubble.appendChild(stopBtn);\n  const finalEl=document.createElement(\"div\");\n  finalEl.className=\"agent-final\";\n  bubble.appendChild(finalEl);\n  let term=null, logEl=null;\n  const ensureTerm=()=>{\n    if(!term){ const built=agtTermEl(bubble,finalEl,false); term=built.term; logEl=built.log; }\n    return logEl;\n  };\n  const stepsArr=[];\n  const r=await agtJobFollow(info.id,{\n    ensureTerm:ensureTerm,\n    onStep:(st,L)=>{\n      const kind=(st.kind===\"fetch\")?\"fetch\":\"bash\";\n      agtStepEl(L,String(st.cmd||\"\"),kind);\n      agtOutEl(L,String(st.out||\"\"),String(st.err||\"\"),kind===\"fetch\"?0:(st.exit||0));\n      stepsArr.push({kind:kind, cmd:String(st.cmd||\"\"), out:String(st.out||\"\"),\n        err:String(st.err||\"\"), exit:st.exit||0});\n      agtSmart();\n    },\n    afterPoll:agtSmart,\n    detached:()=>AGT_DETACH\n  }, st0);\n  AGT_JOB_OWN=null; AGT_JOB_CHAT=null;\n  if(term) term.classList.add(\"folded\");\n  agtSmart();\n  if(r.detached) return true;\n  const inf={id:info.id, chat:cid, mission:mission};\n  if(r.lost){\n    agtJobDrop(cid);\n    finalEl.innerHTML=renderMarkdown(\"*(mission interrompue : la passerelle ne la connaît plus)*\");\n    await agtSetJobMark(cid, inf, \"error\");\n    return true;\n  }\n  const finalText=String(r.final||\"\")||(r.error?(\"*(\"+r.error+\")*\"):\"(mission sans texte final)\");\n  finalEl.innerHTML=renderMarkdown(finalText);\n  msgActions(bubble,{getMd:()=>agtTraceMd(stepsArr, finalText)});\n  try{ conv.push({role:\"user\",text:mission}); conv.push({role:\"assistant\",text:finalText}); }catch(e){}\n  if(r.done) await agtEnsureSaved(inf, mission, stepsArr, finalText);\n  return true;\n}\n/* Ouvrir une discussion dont la mission tourne encore : on s'y raccroche. */\nfunction agtMissionBanner(cid){\n  if(!cid||AGT_JOB_OWN) return;                 /* déjà affichée sur la page */\n  const c=(typeof chats!==\"undefined\")?chats.find(x=>x.id===cid):null;\n  if(!c) return;\n  const live=(c.job&&c.job.state===\"run\")\n    ? {id:c.job.id, chat:cid, mission:c.job.mission||\"\", model:c.job.model||\"\"} : null;\n  const info=live||agtJobLocal(cid);\n  if(!info||!info.id) return;\n  agtAttachJob(info, cid).catch(e=>{\n    AGT_JOB_OWN=null; AGT_JOB_CHAT=null; console.warn(\"mission :\", e);\n  });\n}\n/* De temps en temps, on rafraîchit les marqueurs « en cours » : une mission\n   peut s'être terminée pendant qu'on regardait ailleurs. */\nasync function agtJobsRefresh(){\n  if(typeof FB===\"undefined\"||!FB_OK||!fbUser) return;\n  if(typeof document!==\"undefined\"&&document.hidden) return;\n  let change=false;\n  for(const c of chats.slice()){\n    if(!c.job||c.job.state!==\"run\") continue;\n    try{\n      const d=await chatsRef().doc(c.id).get();\n      const j=(d.data()||{}).job;\n      if(!j||j.state!==\"run\"){\n        c.job=j||null; change=true;\n        if(j&&j.state===\"done\") agtJobDrop(c.id);\n      }\n    }catch(e){}\n  }\n  if(change) renderChatList();\n}\n/* Nouvelle discussion depuis le mode agent : aucune mission ne tourne\n   (AGENT_BUSY est vérifié), on remet juste l'interface à plat. */\nfunction agtNewConversation(){\n  if(typeof AGENT_BUSY!==\"undefined\"&&AGENT_BUSY) return;\n  const t=document.getElementById(\"agtToast\"); if(t) t.classList.add(\"out\");\n}\nfunction agtToast(msg){\n  let t=document.getElementById(\"agtToast\");\n  if(!t){ t=document.createElement(\"div\"); t.id=\"agtToast\"; document.body.appendChild(t); }\n  t.classList.remove(\"out\");\n  t.textContent=msg;\n  if(AGT_TOAST_T) clearTimeout(AGT_TOAST_T);\n  AGT_TOAST_T=setTimeout(()=>{ t.classList.add(\"out\"); },3000);\n}\nfunction agtSyncUI(){\n  const ok=agtLogged();\n  if(AGENT_ON&&!ok){ AGENT_ON=false; agtBtn.classList.remove(\"on\"); }\n  agtFilesBtn.hidden=!(AGENT_ON&&ok);\n  /* La corbeille (nouvelle discussion) disparaît en mode agent. */\n  if(typeof clearBtn!==\"undefined\"&&clearBtn) clearBtn.hidden=!!(AGENT_ON&&ok);\n  const ms=document.getElementById(\"menushare\");\n  if(ms) ms.hidden=!(AGENT_ON&&ok);\n}\nfunction agtCopyText(t){\n  try{\n    if(navigator.clipboard&&navigator.clipboard.writeText&&window.isSecureContext){\n      navigator.clipboard.writeText(t).catch(()=>agtCopyFallback(t));\n    } else agtCopyFallback(t);\n  }catch(e){ agtCopyFallback(t); }\n}\nfunction agtCopyFallback(t){\n  try{\n    const ta=document.createElement(\"textarea\");\n    ta.value=t; ta.style.position=\"fixed\"; ta.style.opacity=\"0\";\n    document.body.appendChild(ta); ta.select();\n    document.execCommand(\"copy\"); ta.remove();\n  }catch(e){}\n}\n\n/* --- œil : afficher/masquer les mots de passe --- */\nfunction pwPeek(input){\n  if(!input||input.dataset.peek) return;\n  input.dataset.peek=\"1\";\n  const w=document.createElement(\"span\");\n  w.className=\"pwwrap\";\n  input.parentNode.insertBefore(w,input);\n  w.appendChild(input);\n  const b=document.createElement(\"button\");\n  b.type=\"button\"; b.className=\"pwpeek\"; b.title=\"Afficher le mot de passe\";\n  b.setAttribute(\"aria-label\",\"Afficher le mot de passe\");\n  b.innerHTML='<svg width=\"15\" height=\"15\" viewBox=\"0 0 24 24\" fill=\"none\" stroke=\"currentColor\" stroke-width=\"1.8\" stroke-linecap=\"round\" stroke-linejoin=\"round\"><path d=\"M2 12s3.5-6.5 10-6.5S22 12 22 12s-3.5 6.5-10 6.5S2 12 2 12Z\"/><circle cx=\"12\" cy=\"12\" r=\"2.6\"/></svg>';\n  b.addEventListener(\"click\",e=>{\n    e.preventDefault();\n    const on=input.type===\"password\";\n    input.type=on?\"text\":\"password\";\n    b.title=on?\"Masquer le mot de passe\":\"Afficher le mot de passe\";\n    b.style.color=on?\"var(--ink)\":\"\";\n  });\n  w.appendChild(b);\n}\npwPeek(document.getElementById(\"authpass\"));\n\n/* --- la zone noire (phases 69-70) : elle MONTRE le mode de la discussion --\n   Fini l'interrupteur allumé/éteint, et fini le clic-surprise : c'est un\n   affichage pur. Le mode se choisit à la création (chat, agent ou Z-Canal),\n   depuis la poubelle en mode chat ou depuis « Nouvelle discussion » dans le\n   menu, et il ne change plus ensuite. */\nfunction agtBtnModeClick(){\n  agtToast((typeof modeBadgeInfo===\"function\")?modeBadgeInfo().label:\"Mode\");\n}\n/* Aucun écouteur : la zone noire est un affichage. */\nagtFilesBtn.addEventListener(\"click\",()=>agtOpenPanel());\ndocument.getElementById(\"attach\").addEventListener(\"click\",()=>agtSyncUI());\n\n/* --- partager l'écran de la VM (menu image, mode agent seulement) --- */\ndocument.getElementById(\"menushare\").addEventListener(\"click\",async()=>{\n  document.getElementById(\"imgmenu\").hidden=true;\n  agtToast(\"Capture de l'écran de la VM…\");\n  try{\n    const r=await fetch(API_BASE+\"/api/agent/screenshot\",{headers:agtHeaders()});\n    if(!r.ok) throw new Error(\"HTTP \"+r.status);\n    const b=await r.blob();\n    const url=URL.createObjectURL(b);\n    const img=await new Promise((res,rej)=>{\n      const i=new Image(); i.onload=()=>res(i); i.onerror=()=>rej(new Error(\"image illisible\")); i.src=url;\n    });\n    const W=1280, sc=Math.min(1,W/(img.naturalWidth||img.width||W));\n    const c=document.createElement(\"canvas\");\n    c.width=Math.max(1,Math.round((img.naturalWidth||img.width)*sc));\n    c.height=Math.max(1,Math.round((img.naturalHeight||img.height)*sc));\n    c.getContext(\"2d\").drawImage(img,0,0,c.width,c.height);\n    try{URL.revokeObjectURL(url);}catch(e){}\n    const b64=String(c.toDataURL(\"image/jpeg\",0.82)).split(\",\")[1]||\"\";\n    if(!b64) throw new Error(\"capture vide\");\n    pendingFiles.push({kind:\"image\", name:\"ecran-vm.jpg\", b64:b64, kb:Math.max(1,Math.round(b64.length*0.75/1024))});\n    syncMirrors(); renderPending();\n    agtToast(\"Écran partagé — il partira avec ton prochain message ✓\");\n  }catch(e){ agtToast(\"Capture impossible : \"+(e&&e.message?e.message:e)); }\n});\n\n/* --- panneau : écran VM + explorateur, défilement unique --- */\nfunction agtOpenPanel(){\n  agtSyncUI();\n  if(!AGENT_ON||!agtLogged()){ agtToast(\"Veuillez vous connecter pour utiliser le mode agent.\"); return; }\n  agtPanel.hidden=false;\n  agtPollScreen();\n  agtLoadDir(AGENT_USER, document.getElementById(\"agentTree\"), 0, true);\n  if(!AGT_TIMER) AGT_TIMER=setInterval(agtPollScreen, 3000);\n}\nfunction agtClosePanel(){\n  agtPanel.hidden=true;\n  if(AGT_TIMER){ clearInterval(AGT_TIMER); AGT_TIMER=null; }\n  if(AGT_OBJURL){ try{URL.revokeObjectURL(AGT_OBJURL);}catch(e){} AGT_OBJURL=null; }\n}\ndocument.getElementById(\"agentPanelClose\").addEventListener(\"click\",agtClosePanel);\n\nasync function agtPollScreen(){\n  if(agtPanel.hidden||document.visibilityState!==\"visible\") return;\n  try{\n    const r=await fetch(API_BASE+\"/api/agent/screenshot\",{headers:agtHeaders()});\n    if(!r.ok) return;\n    const b=await r.blob();\n    const u=URL.createObjectURL(b);\n    const img=document.getElementById(\"agentScreen\");\n    const old=AGT_OBJURL;\n    img.src=u; AGT_OBJURL=u;\n    if(old) setTimeout(()=>{ try{URL.revokeObjectURL(old);}catch(e){} },1500);\n  }catch(e){}\n}\n\nconst AGT_ICO_DIR='<svg class=\"at-ico\" viewBox=\"0 0 24 24\" fill=\"none\" stroke=\"#8a6d1a\" stroke-width=\"1.7\" stroke-linejoin=\"round\"><path d=\"M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2Z\" fill=\"#f6e7b4\"/></svg>';\nconst AGT_ICO_FILE='<svg class=\"at-ico\" viewBox=\"0 0 24 24\" fill=\"none\" stroke=\"#555\" stroke-width=\"1.7\" stroke-linejoin=\"round\"><path d=\"M6 3h8l4 4v14H6Z\"/><path d=\"M14 3v4h4\"/></svg>';\n/* Logos de langages (couleurs officielles) */\nconst AGT_LANG_ICONS={\n  html:'<svg class=\"at-ico\" viewBox=\"0 0 128 128\"><path fill=\"#E44D26\" d=\"M19.037 113.876L9.032 1.661h109.936l-10.016 112.198-45.019 12.48z\"/><path fill=\"#F16529\" d=\"M64 116.8l36.378-10.086 8.559-95.878H64z\"/><path fill=\"#EBEBEB\" d=\"M64 52.455H45.788L44.53 38.361H64V24.599H29.489l.33 3.692 3.382 37.927H64zm0 35.743l-.061.017-15.327-4.14-.979-10.975H33.816l1.928 21.609 28.193 7.826.063-.017z\"/><path fill=\"#fff\" d=\"M63.952 52.455v13.763h16.947l-1.597 17.849-15.35 4.143v14.319l28.215-7.82.207-2.325 3.234-36.233.335-3.696h-3.708zm0-27.856v13.762h33.244l.276-3.092.628-6.978.329-3.692z\"/></svg>',\n  css:'<svg class=\"at-ico\" viewBox=\"0 0 128 128\"><path fill=\"#1572B6\" d=\"M18.814 114.123L8.76 1.352h110.48l-10.064 112.754-45.243 12.543-45.119-12.526z\"/><path fill=\"#33A9DC\" d=\"M64.001 117.062l36.559-10.136 8.601-96.354h-45.16v106.49z\"/><path fill=\"#fff\" d=\"M64.001 51.429h18.302l1.264-14.163H64.001V23.435h34.682l-.332 3.711-3.4 38.114h-30.95V51.429z\"/><path fill=\"#EBEBEB\" d=\"M64.083 87.349l-.061.018-15.403-4.159-.985-11.031H33.752l1.937 21.717 28.331 7.863.063-.018v-14.39z\"/><path fill=\"#fff\" d=\"M81.127 64.675l-1.666 18.522-15.426 4.164v14.39l28.354-7.858.208-2.337 2.406-26.881H81.127z\"/><path fill=\"#EBEBEB\" d=\"M64.048 23.435v13.831H30.64l-.277-3.108-.63-7.012-.331-3.711h34.646zm-.047 27.996v13.831H48.792l-.277-3.108-.631-7.012-.33-3.711h16.447z\"/></svg>',\n  js:'<svg class=\"at-ico\" viewBox=\"0 0 128 128\"><path fill=\"#F0DB4F\" d=\"M1.408 1.408h125.184v125.185H1.408z\"/><path fill=\"#323330\" d=\"M116.347 96.736c-.917-5.711-4.641-10.508-15.672-14.981-3.832-1.761-8.104-3.022-9.377-5.926-.452-1.69-.512-2.642-.226-3.665.821-3.32 4.784-4.355 7.925-3.403 2.023.678 3.938 2.237 5.093 4.724 5.402-3.498 5.391-3.475 9.163-5.879-1.381-2.141-2.118-3.129-3.022-4.045-3.249-3.629-7.676-5.498-14.756-5.355l-3.688.477c-3.534.893-6.902 2.748-8.877 5.235-5.926 6.724-4.236 18.492 2.975 23.335 7.104 5.332 17.54 6.545 18.873 11.531 1.297 6.104-4.486 8.08-10.234 7.378-4.236-.881-6.592-3.034-9.139-6.949-4.688 2.713-4.688 2.713-9.508 5.485 1.143 2.499 2.344 3.63 4.26 5.795 9.068 9.198 31.76 8.746 35.83-5.176.165-.478 1.261-3.666.38-8.581zM69.462 58.943H57.753l-.048 30.272c0 6.438.333 12.34-.714 14.149-1.713 3.558-6.152 3.117-8.175 2.427-2.059-1.012-3.106-2.451-4.319-4.485-.333-.584-.583-1.036-.667-1.071l-9.52 5.83c1.583 3.249 3.915 6.069 6.902 7.901 4.462 2.678 10.459 3.499 16.731 2.059 4.082-1.189 7.604-3.652 9.448-7.401 2.666-4.915 2.094-10.864 2.07-17.444.06-10.735.001-21.468.001-32.237z\"/></svg>',\n  ts:'<svg class=\"at-ico\" viewBox=\"0 0 128 128\"><path fill=\"#fff\" d=\"M22.67 47h99.67v73.67H22.67z\"/><path data-name=\"original\" fill=\"#007acc\" d=\"M1.5 63.91v62.5h125v-125H1.5zm100.73-5a15.56 15.56 0 017.82 4.5 20.58 20.58 0 013 4c0 .16-5.4 3.81-8.69 5.85-.12.08-.6-.44-1.13-1.23a7.09 7.09 0 00-5.87-3.53c-3.79-.26-6.23 1.73-6.21 5a4.58 4.58 0 00.54 2.34c.83 1.73 2.38 2.76 7.24 4.86 8.95 3.85 12.78 6.39 15.16 10 2.66 4 3.25 10.46 1.45 15.24-2 5.2-6.9 8.73-13.83 9.9a38.32 38.32 0 01-9.52-.1 23 23 0 01-12.72-6.63c-1.15-1.27-3.39-4.58-3.25-4.82a9.34 9.34 0 011.15-.73L82 101l3.59-2.08.75 1.11a16.78 16.78 0 004.74 4.54c4 2.1 9.46 1.81 12.16-.62a5.43 5.43 0 00.69-6.92c-1-1.39-3-2.56-8.59-5-6.45-2.78-9.23-4.5-11.77-7.24a16.48 16.48 0 01-3.43-6.25 25 25 0 01-.22-8c1.33-6.23 6-10.58 12.82-11.87a31.66 31.66 0 019.49.26zm-29.34 5.24v5.12H56.66v46.23H45.15V69.26H28.88v-5a49.19 49.19 0 01.12-5.17C29.08 59 39 59 51 59h21.83z\"/></svg>',\n  react:'<svg class=\"at-ico\" viewBox=\"0 0 128 128\"><g fill=\"#61DAFB\"><circle cx=\"64\" cy=\"64\" r=\"11.4\"/><path d=\"M107.3 45.2c-2.2-.8-4.5-1.6-6.9-2.3.6-2.4 1.1-4.8 1.5-7.1 2.1-13.2-.2-22.5-6.6-26.1-1.9-1.1-4-1.6-6.4-1.6-7 0-15.9 5.2-24.9 13.9-9-8.7-17.9-13.9-24.9-13.9-2.4 0-4.5.5-6.4 1.6-6.4 3.7-8.7 13-6.6 26.1.4 2.3.9 4.7 1.5 7.1-2.4.7-4.7 1.4-6.9 2.3C8.2 50 1.4 56.6 1.4 64s6.9 14 19.3 18.8c2.2.8 4.5 1.6 6.9 2.3-.6 2.4-1.1 4.8-1.5 7.1-2.1 13.2.2 22.5 6.6 26.1 1.9 1.1 4 1.6 6.4 1.6 7.1 0 16-5.2 24.9-13.9 9 8.7 17.9 13.9 24.9 13.9 2.4 0 4.5-.5 6.4-1.6 6.4-3.7 8.7-13 6.6-26.1-.4-2.3-.9-4.7-1.5-7.1 2.4-.7 4.7-1.4 6.9-2.3 12.5-4.8 19.3-11.4 19.3-18.8s-6.8-14-19.3-18.8zM92.5 14.7c4.1 2.4 5.5 9.8 3.8 20.3-.3 2.1-.8 4.3-1.4 6.6-5.2-1.2-10.7-2-16.5-2.5-3.4-4.8-6.9-9.1-10.4-13 7.4-7.3 14.9-12.3 21-12.3 1.3 0 2.5.3 3.5.9zM81.3 74c-1.8 3.2-3.9 6.4-6.1 9.6-3.7.3-7.4.4-11.2.4-3.9 0-7.6-.1-11.2-.4-2.2-3.2-4.2-6.4-6-9.6-1.9-3.3-3.7-6.7-5.3-10 1.6-3.3 3.4-6.7 5.3-10 1.8-3.2 3.9-6.4 6.1-9.6 3.7-.3 7.4-.4 11.2-.4 3.9 0 7.6.1 11.2.4 2.2 3.2 4.2 6.4 6 9.6 1.9 3.3 3.7 6.7 5.3 10-1.7 3.3-3.4 6.6-5.3 10zm8.3-3.3c1.5 3.5 2.7 6.9 3.8 10.3-3.4.8-7 1.4-10.8 1.9 1.2-1.9 2.5-3.9 3.6-6 1.2-2.1 2.3-4.2 3.4-6.2zM64 97.8c-2.4-2.6-4.7-5.4-6.9-8.3 2.3.1 4.6.2 6.9.2 2.3 0 4.6-.1 6.9-.2-2.2 2.9-4.5 5.7-6.9 8.3zm-18.6-15c-3.8-.5-7.4-1.1-10.8-1.9 1.1-3.3 2.3-6.8 3.8-10.3 1.1 2 2.2 4.1 3.4 6.1 1.2 2.2 2.4 4.1 3.6 6.1zm-7-25.5c-1.5-3.5-2.7-6.9-3.8-10.3 3.4-.8 7-1.4 10.8-1.9-1.2 1.9-2.5 3.9-3.6 6-1.2 2.1-2.3 4.2-3.4 6.2zM64 30.2c2.4 2.6 4.7 5.4 6.9 8.3-2.3-.1-4.6-.2-6.9-.2-2.3 0-4.6.1-6.9.2 2.2-2.9 4.5-5.7 6.9-8.3zm22.2 21l-3.6-6c3.8.5 7.4 1.1 10.8 1.9-1.1 3.3-2.3 6.8-3.8 10.3-1.1-2.1-2.2-4.2-3.4-6.2zM31.7 35c-1.7-10.5-.3-17.9 3.8-20.3 1-.6 2.2-.9 3.5-.9 6 0 13.5 4.9 21 12.3-3.5 3.8-7 8.2-10.4 13-5.8.5-11.3 1.4-16.5 2.5-.6-2.3-1-4.5-1.4-6.6zM7 64c0-4.7 5.7-9.7 15.7-13.4 2-.8 4.2-1.5 6.4-2.1 1.6 5 3.6 10.3 6 15.6-2.4 5.3-4.5 10.5-6 15.5C15.3 75.6 7 69.6 7 64zm28.5 49.3c-4.1-2.4-5.5-9.8-3.8-20.3.3-2.1.8-4.3 1.4-6.6 5.2 1.2 10.7 2 16.5 2.5 3.4 4.8 6.9 9.1 10.4 13-7.4 7.3-14.9 12.3-21 12.3-1.3 0-2.5-.3-3.5-.9zM96.3 93c1.7 10.5.3 17.9-3.8 20.3-1 .6-2.2.9-3.5.9-6 0-13.5-4.9-21-12.3 3.5-3.8 7-8.2 10.4-13 5.8-.5 11.3-1.4 16.5-2.5.6 2.3 1 4.5 1.4 6.6zm9-15.6c-2 .8-4.2 1.5-6.4 2.1-1.6-5-3.6-10.3-6-15.6 2.4-5.3 4.5-10.5 6-15.5 13.8 4 22.1 10 22.1 15.6 0 4.7-5.8 9.7-15.7 13.4z\"/></g></svg>',\n  md:'<svg class=\"at-ico\" viewBox=\"0 0 128 128\"><path d=\"M11.95 24.348c-5.836 0-10.618 4.867-10.618 10.681v57.942c0 5.814 4.782 10.681 10.617 10.681h104.102c5.835 0 10.617-4.867 10.617-10.681V35.03c0-5.814-4.783-10.681-10.617-10.681H14.898l-.002-.002H11.95zm-.007 9.543h104.108c.625 0 1.076.423 1.076 1.14v57.94c0 .717-.453 1.14-1.076 1.14H11.949c-.623 0-1.076-.423-1.076-1.14V35.029c0-.715.451-1.135 1.07-1.138z\" font-weight=\"400\" font-family=\"sans-serif\"/><path d=\"M20.721 84.1V43.9H32.42l11.697 14.78L55.81 43.9h11.696v40.2H55.81V61.044l-11.694 14.78-11.698-14.78V84.1H20.722zm73.104 0L76.28 64.591h11.697V43.9h11.698v20.69h11.698zm0 0\"/></svg>',\n  py:'<svg class=\"at-ico\" viewBox=\"0 0 128 128\"><linearGradient id=\"zp-python-original-a\" gradientUnits=\"userSpaceOnUse\" x1=\"70.252\" y1=\"1237.476\" x2=\"170.659\" y2=\"1151.089\" gradientTransform=\"matrix(.563 0 0 -.568 -29.215 707.817)\"><stop offset=\"0\" stop-color=\"#5A9FD4\"/><stop offset=\"1\" stop-color=\"#306998\"/></linearGradient><linearGradient id=\"zp-python-original-b\" gradientUnits=\"userSpaceOnUse\" x1=\"209.474\" y1=\"1098.811\" x2=\"173.62\" y2=\"1149.537\" gradientTransform=\"matrix(.563 0 0 -.568 -29.215 707.817)\"><stop offset=\"0\" stop-color=\"#FFD43B\"/><stop offset=\"1\" stop-color=\"#FFE873\"/></linearGradient><path fill=\"url(#zp-python-original-a)\" d=\"M63.391 1.988c-4.222.02-8.252.379-11.8 1.007-10.45 1.846-12.346 5.71-12.346 12.837v9.411h24.693v3.137H29.977c-7.176 0-13.46 4.313-15.426 12.521-2.268 9.405-2.368 15.275 0 25.096 1.755 7.311 5.947 12.519 13.124 12.519h8.491V67.234c0-8.151 7.051-15.34 15.426-15.34h24.665c6.866 0 12.346-5.654 12.346-12.548V15.833c0-6.693-5.646-11.72-12.346-12.837-4.244-.706-8.645-1.027-12.866-1.008zM50.037 9.557c2.55 0 4.634 2.117 4.634 4.721 0 2.593-2.083 4.69-4.634 4.69-2.56 0-4.633-2.097-4.633-4.69-.001-2.604 2.073-4.721 4.633-4.721z\" transform=\"translate(0 10.26)\"/><path fill=\"url(#zp-python-original-b)\" d=\"M91.682 28.38v10.966c0 8.5-7.208 15.655-15.426 15.655H51.591c-6.756 0-12.346 5.783-12.346 12.549v23.515c0 6.691 5.818 10.628 12.346 12.547 7.816 2.297 15.312 2.713 24.665 0 6.216-1.801 12.346-5.423 12.346-12.547v-9.412H63.938v-3.138h37.012c7.176 0 9.852-5.005 12.348-12.519 2.578-7.735 2.467-15.174 0-25.096-1.774-7.145-5.161-12.521-12.348-12.521h-9.268zM77.809 87.927c2.561 0 4.634 2.097 4.634 4.692 0 2.602-2.074 4.719-4.634 4.719-2.55 0-4.633-2.117-4.633-4.719 0-2.595 2.083-4.692 4.633-4.692z\" transform=\"translate(0 10.26)\"/><radialGradient id=\"zp-python-original-c\" cx=\"1825.678\" cy=\"444.45\" r=\"26.743\" gradientTransform=\"matrix(0 -.24 -1.055 0 532.979 557.576)\" gradientUnits=\"userSpaceOnUse\"><stop offset=\"0\" stop-color=\"#B8B8B8\" stop-opacity=\".498\"/><stop offset=\"1\" stop-color=\"#7F7F7F\" stop-opacity=\"0\"/></radialGradient><path opacity=\".444\" fill=\"url(#zp-python-original-c)\" d=\"M97.309 119.597c0 3.543-14.816 6.416-33.091 6.416-18.276 0-33.092-2.873-33.092-6.416 0-3.544 14.815-6.417 33.092-6.417 18.275 0 33.091 2.872 33.091 6.417z\"/></svg>',\n  sh:'<svg class=\"at-ico\" viewBox=\"0 0 24 24\" fill=\"#4EAA25\"><path d=\"M21.038,4.9l-7.577-4.498C13.009,0.134,12.505,0,12,0c-0.505,0-1.009,0.134-1.462,0.403L2.961,4.9 C2.057,5.437,1.5,6.429,1.5,7.503v8.995c0,1.073,0.557,2.066,1.462,2.603l7.577,4.497C10.991,23.866,11.495,24,12,24 c0.505,0,1.009-0.134,1.461-0.402l7.577-4.497c0.904-0.537,1.462-1.529,1.462-2.603V7.503C22.5,6.429,21.943,5.437,21.038,4.9z M15.17,18.946l0.013,0.646c0.001,0.078-0.05,0.167-0.111,0.198l-0.383,0.22c-0.061,0.031-0.111-0.007-0.112-0.085L14.57,19.29 c-0.328,0.136-0.66,0.169-0.872,0.084c-0.04-0.016-0.057-0.075-0.041-0.142l0.139-0.584c0.011-0.046,0.036-0.092,0.069-0.121 c0.012-0.011,0.024-0.02,0.036-0.026c0.022-0.011,0.043-0.014,0.062-0.006c0.229,0.077,0.521,0.041,0.802-0.101 c0.357-0.181,0.596-0.545,0.592-0.907c-0.003-0.328-0.181-0.465-0.613-0.468c-0.55,0.001-1.064-0.107-1.072-0.917 c-0.007-0.667,0.34-1.361,0.889-1.8l-0.007-0.652c-0.001-0.08,0.048-0.168,0.111-0.2l0.37-0.236 c0.061-0.031,0.111,0.007,0.112,0.087l0.006,0.653c0.273-0.109,0.511-0.138,0.726-0.088c0.047,0.012,0.067,0.076,0.048,0.151 l-0.144,0.578c-0.011,0.044-0.036,0.088-0.065,0.116c-0.012,0.012-0.025,0.021-0.038,0.028c-0.019,0.01-0.038,0.013-0.057,0.009 c-0.098-0.022-0.332-0.073-0.699,0.113c-0.385,0.195-0.52,0.53-0.517,0.778c0.003,0.297,0.155,0.387,0.681,0.396 c0.7,0.012,1.003,0.318,1.01,1.023C16.105,17.747,15.736,18.491,15.17,18.946z M19.143,17.859c0,0.06-0.008,0.116-0.058,0.145 l-1.916,1.164c-0.05,0.029-0.09,0.004-0.09-0.056v-0.494c0-0.06,0.037-0.093,0.087-0.122l1.887-1.129 c0.05-0.029,0.09-0.004,0.09,0.056V17.859z M20.459,6.797l-7.168,4.427c-0.894,0.523-1.553,1.109-1.553,2.187v8.833 c0,0.645,0.26,1.063,0.66,1.184c-0.131,0.023-0.264,0.039-0.398,0.039c-0.42,0-0.833-0.114-1.197-0.33L3.226,18.64 c-0.741-0.44-1.201-1.261-1.201-2.142V7.503c0-0.881,0.46-1.702,1.201-2.142l7.577-4.498c0.363-0.216,0.777-0.33,1.197-0.33 c0.419,0,0.833,0.114,1.197,0.33l7.577,4.498c0.624,0.371,1.046,1.013,1.164,1.732C21.686,6.557,21.12,6.411,20.459,6.797z\"/></svg>',\n  json:'<svg class=\"at-ico\" viewBox=\"0 0 24 24\" fill=\"#111\"><path d=\"M12.043 23.968c.479-.004.953-.029 1.426-.094a11.805 11.805 0 003.146-.863 12.404 12.404 0 003.793-2.542 11.977 11.977 0 002.44-3.427 11.794 11.794 0 001.02-3.476c.149-1.16.135-2.346-.045-3.499a11.96 11.96 0 00-.793-2.788 11.197 11.197 0 00-.854-1.617c-1.168-1.837-2.861-3.314-4.81-4.3a12.835 12.835 0 00-2.172-.87h-.005c.119.063.24.132.345.201.12.074.239.146.351.225a8.93 8.93 0 011.559 1.33c1.063 1.145 1.797 2.548 2.218 4.041.284.982.434 1.998.495 3.017.044.743.044 1.491-.047 2.229-.149 1.27-.554 2.51-1.228 3.596a7.475 7.475 0 01-1.903 2.084c-1.244.928-2.877 1.482-4.436 1.114a3.916 3.916 0 01-.748-.258 4.692 4.692 0 01-.779-.45 6.08 6.08 0 01-1.244-1.105 6.507 6.507 0 01-1.049-1.747 7.366 7.366 0 01-.494-2.54c-.03-1.273.225-2.553.854-3.67a6.43 6.43 0 011.663-1.918c.225-.178.464-.333.704-.479l.016-.007a5.121 5.121 0 00-1.441-.12 4.963 4.963 0 00-1.228.24c-.359.12-.704.27-1.019.45a6.146 6.146 0 00-.733.494c-.211.18-.42.36-.615.555-1.123 1.153-1.768 2.682-2.022 4.256-.15.973-.15 1.96-.091 2.95.105 1.395.391 2.787.945 4.062a8.518 8.518 0 001.348 2.173 8.14 8.14 0 003.132 2.23 7.934 7.934 0 002.113.54c.074.015.149.015.209.015zm-2.934-.398a4.102 4.102 0 01-.45-.228 8.5 8.5 0 01-2.038-1.534c-1.094-1.137-1.827-2.566-2.247-4.08a15.184 15.184 0 01-.495-3.172 12.14 12.14 0 01.046-2.082c.135-1.257.495-2.501 1.124-3.58a6.889 6.889 0 011.783-2.053 6.23 6.23 0 011.633-.9 5.363 5.363 0 013.522-.045c.029 0 .029 0 .045.03.015.015.045.015.06.03.045.016.104.045.165.074.239.12.479.271.704.42a6.294 6.294 0 012.097 2.502c.42.914.615 1.934.631 2.938.014 1.079-.18 2.157-.645 3.146a6.42 6.42 0 01-2.638 2.832c.09.03.18.045.271.075.225.044.449.074.688.074 1.468.045 2.892-.66 3.94-1.647.195-.18.375-.375.54-.585.225-.27.435-.54.614-.823.239-.375.435-.75.614-1.154a8.112 8.112 0 00.509-1.664c.196-1.004.211-2.022.149-3.026-.135-2.022-.673-4.045-1.842-5.724a9.054 9.054 0 00-.555-.719 9.868 9.868 0 00-1.063-1.034 8.477 8.477 0 00-1.363-.915 9.927 9.927 0 00-1.692-.598l-.3-.06c-.209-.03-.42-.044-.634-.06a8.453 8.453 0 00-1.015.016c-.704.045-1.412.16-2.112.337C5.799 1.227 2.863 3.566 1.3 6.67A11.834 11.834 0 00.238 9.801a11.81 11.81 0 00-.104 3.775c.12 1.02.374 2.023.778 2.977.227.57.511 1.124.825 1.648 1.094 1.783 2.683 3.236 4.51 4.24.688.39 1.408.69 2.157.944.226.074.45.15.689.21z\"/></svg>',\n  tex:'<svg class=\"at-ico\" viewBox=\"0 0 24 24\" fill=\"#008080\"><path d=\"M2.176 2.814c.233.42.476.78.73 1.09.247-.013 1.132.456 1.312.523.508.282 1.063.63 1.567.966.505.337.96.662 1.272.9.156.12.278.218.352.286a.483.483 0 01.078.082.08.08 0 01.01.021.06.06 0 01-.004.047.057.057 0 01-.04.03.077.077 0 01-.028 0c-.057 0-.203-.163-.497-.415a23.474 23.474 0 00-2.759-1.827c-.504-.28-.956-.542-1.264-.613a2.322 2.322 0 00-.36-.025 2.706 2.706 0 00-.788.133c.494.414.91.716 1.28.949-.57-.182-1.182-.21-1.902.133.526.329.967.567 1.354.745 1.103.156 2.258.696 3.224 1.309.483.307.904.615 1.219.867.157.125.29.237.39.328.098.091.174.154.197.21.03.073-.019.104-.084.058-.032-.022-.088-.102-.184-.191a7.35 7.35 0 00-.384-.327c-.312-.25-.729-.552-1.209-.857-.893-.562-2.232-1.013-3.173-1.397-.602-.11-1.225-.06-1.906.39.449.2.837.349 1.182.463.812 0 1.892.365 2.935.922 1.042.556 2.04 1.214 2.523 1.774.066.077-.016.126-.074.07-.52-.495-1.463-1.204-2.498-1.756-.639-.337-2.153-1.01-2.886-1.01l.004.002c-.567.02-1.13.195-1.679.716.477.118.885.196 1.244.249-.44.088-.87.3-1.289.722.324.07.616.122.882.162-.328.159-.639.404-.923.78.373.03.703.042 1 .044-.36.166-.696.43-.996.85.533.027.98.025 1.364.003-.422.172-.812.464-1.145.968.662.01 1.188-.022 1.628-.076l-.006.002c.99-.073 2.297.127 2.962.847.052.057-.024.118-.072.074-.648-.58-1.493-.827-2.89-.921h-.002c-.543.149-1.046.446-1.46 1.074.536.008.982-.013 1.366-.05-.469.257-.873.644-1.139 1.306.483-.092.888-.19 1.237-.292-.363.265-.668.636-.873 1.194.324-.072.612-.146.871-.221a2.519 2.519 0 00-.513 1.095c.352-.13.655-.254.926-.377-.257.3-.453.681-.55 1.19.495-.199.899-.388 1.238-.568-.31.333-.543.76-.635 1.356a11.816 11.816 0 001.442-.744c-.433.362-.764.843-.879 1.587.788-.348 1.339-.663 1.767-.955-.184.372-.282.806-.235 1.348.762-.584 1.243-1.056 1.602-1.473-.024.269-.003.56.077.884.546-.939 1.089-1.212 1.65-1.526-.895.451-.762.79-.762 1.184.683-.72 1.635-1.482 1.927-1.96-.39.585-.547 1.14-.65 1.63-1.993 1.054-3.207 1.329-4.568 1.75.528.194 1.093.383.859.652l-.624.622c.399-.124.805-.3 1.158-.059-.035.327-.447.492-.8.683.621-.224.756-.172.92-.12.081.393-.203.603-.388.862 1.565-1.19 3.606-2.13 5.044-2.522 2.022-.681 4.63-1.389 5.339-3.115l.712-2.847-.004.004c-.111-.034-.246-.063-.35-.133a.651.651 0 01-.235-.297c-.252.065-.44.03-.56-.088-.117-.117-.167-.296-.203-.491-.203.041-.362.016-.467-.077-.116-.101-.17-.26-.198-.444l-.008-.039.037-.015a.842.842 0 00.302-.194.257.257 0 00.07-.225l-.006-.037.03-.016c.163-.093.345-.169.428-.28a.274.274 0 00.053-.21.88.88 0 00-.155-.357l-.027-.04.04-.027c.118-.09.244-.179.308-.26.032-.04.048-.076.047-.11 0-.033-.015-.07-.064-.117l-.098-.094.135.006c.213.01.395-.007.538-.053a.504.504 0 00.274-.197c-.007-.033-.02-.063-.02-.098a.484.484 0 01.967 0c0 .044-.015.084-.026.125.177.014.347.01.507-.06l.002.001.035-.013c.236-.085.334.045.72-.456-1.69-2.19-4.157-.635-4.977 1.622-.21.576-1.405.578-1.751 0-1.37-2.95-5.53-6.068-9.07-7.218zm.86 2.145c.906.293 1.913.782 2.77 1.328.43.273.813.543 1.114.779.301.236.566.473.62.575.054.102 0 .14-.082.06-.081-.078-.303-.32-.6-.553-.298-.234-.68-.505-1.106-.777-.775-.49-1.982-.958-2.716-1.412zm-1.7 2.7c1.116.014 2.35.447 3.434.997.541.275 1.023.567 1.395.83.372.263.672.524.734.657.061.134-.02.13-.087.055a4.401 4.401 0 00-.704-.626 11.47 11.47 0 00-1.385-.826C3.76 8.264 2.439 7.82 1.336 7.66zm14.916.772a.381.381 0 100 .762.381.381 0 000-.762zM1.7 8.478c.822.072 1.72.368 2.534.75 1.086.509 2.035 1.158 2.434 1.667.035.045-.014.131-.08.062-.428-.44-1.322-1.131-2.397-1.635-.913-.421-2.282-.87-3.262-.78.251-.03.497-.088.771-.064zm16.339.01c-.366.475-.53.423-.703.464.094.43.35.586.585.77l-.06.012c2.315-.447 4.186-.286 6.139-.236l-5.961-1.01zm-.178 1.246h-.002l-.004.016.006-.016zm-.625-.757c-.183.074-.373.076-.563.059a.477.477 0 01-.42.26.483.483 0 01-.435-.278.609.609 0 01-.274.188c-.139.045-.308.057-.493.055.02.035.054.068.055.104a.273.273 0 01-.069.174c-.073.092-.189.17-.295.248.087.141.137.26.149.362a.39.39 0 01-.07.284c-.106.14-.288.21-.439.293a.374.374 0 01-.09.268.89.89 0 01-.297.198c.027.156.074.283.154.354.086.076.211.103.425.047l.055-.014.01.055c.033.207.088.385.187.483.1.099.244.135.503.055l.049-.015.016.048c.05.142.12.223.209.282.087.06.247.112.358.147.798-.869 1.525-1.772 1.884-2.86-.225-.177-.506-.338-.609-.797zm-16.23.386c1.165-.08 2.283.196 3.202.626.92.43 1.658.939 1.974 1.307.075.087-.019.12-.072.072a8.187 8.187 0 00-1.947-1.29c-.904-.414-2.193-.644-3.157-.715zm.864.802c.61.02 1.24.155 1.806.352.756.262 1.421.614 1.747.98.045.05-.007.127-.074.069-.349-.304-.961-.693-1.706-.951-.574-.195-1.613-.369-2.268-.397.197-.022.292-.06.495-.053zm1.05 1.788c.423.034.886.133 1.341.407.043.026.049.136-.049.09-.856-.402-1.326-.49-2.457-.31.386-.128.74-.221 1.164-.187zm-.04.788c.4-.035.784-.002 1.297.204.044.018.08.126-.033.094-.857-.243-1.167-.328-2.287.104.28-.229.622-.366 1.023-.402zm1.285.687c.317-.023.635-.026.934.006.052.006.055.105-.006.102a7.87 7.87 0 00-1.837.115c-.243.046-.423.043-1.405.458.287-.233.794-.452 1.385-.56a8.91 8.91 0 01.93-.12zm1.28.49c.099.003.062.104.006.103-.728-.01-1.304.132-1.875.295a9.78 9.78 0 00-1.318.525c.283-.23.713-.457 1.291-.622.579-.166 1.248-.326 1.896-.302zm.528.398c.036-.005.105.084.018.1-.73.137-1.244.267-1.794.454-.216.074-.58.207-1.243.587.26-.269.656-.492 1.213-.68.558-.19 1.196-.37 1.806-.46zm.311.507c.075-.012.097.087.02.102-1.217.241-1.76.556-2.54 1.144.504-.523 1.297-1.051 2.52-1.246zm.595.448c.087-.013.11.087.021.1-.872.13-1.477.553-2.255 1.33.295-.493 1.004-1.24 2.234-1.43zm.372.39c.046-.006.114.073.023.1a2.634 2.634 0 00-.669.3c-.182.118-.3.2-.597.507.111-.245.296-.434.542-.59.247-.157.509-.293.7-.317z\"/></svg>',\n  svglogo:'<svg class=\"at-ico\" viewBox=\"0 0 24 24\"><rect x=\"1.5\" y=\"1.5\" width=\"21\" height=\"21\" rx=\"4\" fill=\"#FFB13B\"/><text x=\"12\" y=\"15.5\" text-anchor=\"middle\" font-size=\"7\" font-weight=\"700\" fill=\"#3A3A3A\" font-family=\"sans-serif\">svg</text></svg>',\n  img:'<svg class=\"at-ico\" viewBox=\"0 0 24 24\"><rect x=\"2.5\" y=\"4\" width=\"19\" height=\"16\" rx=\"2.5\" fill=\"#E8F0E9\" stroke=\"#4C8C5A\" stroke-width=\"1.3\"/><circle cx=\"8.6\" cy=\"9.6\" r=\"1.6\" fill=\"#F2B134\"/><path d=\"M4.5 18l4.8-5 3.2 3.2 2.6-2.4 4.4 4.2H4.5Z\" fill=\"#4C8C5A\"/></svg>',\n  vid:'<svg class=\"at-ico\" viewBox=\"0 0 24 24\"><rect x=\"2\" y=\"5\" width=\"20\" height=\"14\" rx=\"2.5\" fill=\"#2B2B33\"/><path d=\"M10.4 9.2l5 2.8-5 2.8V9.2Z\" fill=\"#F5A623\"/><circle cx=\"5.4\" cy=\"7.6\" r=\".9\" fill=\"#888\"/><circle cx=\"18.6\" cy=\"16.4\" r=\".9\" fill=\"#888\"/></svg>',\n  aud:'<svg class=\"at-ico\" viewBox=\"0 0 24 24\"><circle cx=\"12\" cy=\"12\" r=\"10.2\" fill=\"#1DB954\"/><path d=\"M7.6 9.4c3-.8 6-.6 8.6.8M8.2 12.3c2.4-.6 4.8-.5 6.9.6M8.8 15c1.8-.5 3.6-.4 5.2.4\" fill=\"none\" stroke=\"#fff\" stroke-width=\"1.5\" stroke-linecap=\"round\"/></svg>',\n  pdf:'<svg class=\"at-ico\" viewBox=\"0 0 24 24\"><path fill=\"#E2453C\" d=\"M5.5 2h9L19 6.5V22h-13.5Z\"/><path fill=\"#fff\" d=\"M14 2v5h5\"/><text x=\"12\" y=\"17.5\" text-anchor=\"middle\" font-size=\"6.5\" font-weight=\"700\" fill=\"#fff\" font-family=\"sans-serif\">PDF</text></svg>',\n  doc:'<svg class=\"at-ico\" viewBox=\"0 0 24 24\"><path fill=\"#2B579A\" d=\"M5.5 2h9L19 6.5V22h-13.5Z\"/><path fill=\"#fff\" d=\"M14 2v5h5\"/><text x=\"12\" y=\"17.5\" text-anchor=\"middle\" font-size=\"5.6\" font-weight=\"700\" fill=\"#fff\" font-family=\"sans-serif\">DOC</text></svg>',\n  ppt:'<svg class=\"at-ico\" viewBox=\"0 0 24 24\"><path fill=\"#D24726\" d=\"M5.5 2h9L19 6.5V22h-13.5Z\"/><path fill=\"#fff\" d=\"M14 2v5h5\"/><text x=\"12\" y=\"17.5\" text-anchor=\"middle\" font-size=\"5.6\" font-weight=\"700\" fill=\"#fff\" font-family=\"sans-serif\">PPT</text></svg>',\n  xls:'<svg class=\"at-ico\" viewBox=\"0 0 24 24\"><path fill=\"#217346\" d=\"M5.5 2h9L19 6.5V22h-13.5Z\"/><path fill=\"#fff\" d=\"M14 2v5h5\"/><text x=\"12\" y=\"17.5\" text-anchor=\"middle\" font-size=\"5.2\" font-weight=\"700\" fill=\"#fff\" font-family=\"sans-serif\">XLS</text></svg>',\n  tex:'<svg class=\"at-ico\" viewBox=\"0 0 24 24\"><rect x=\"1.5\" y=\"3.5\" width=\"21\" height=\"17\" rx=\"3\" fill=\"#1B1B1F\"/><text x=\"12\" y=\"16\" text-anchor=\"middle\" font-size=\"8\" font-weight=\"700\" fill=\"#4FA3D1\" font-family=\"serif\">TeX</text></svg>'\n};\nconst AGT_EXT_MAP={htm:\"html\",html:\"html\",css:\"css\",js:\"js\",mjs:\"js\",cjs:\"js\",ts:\"ts\",\n  jsx:\"react\",tsx:\"react\",md:\"md\",markdown:\"md\",py:\"py\",sh:\"sh\",bash:\"sh\",json:\"json\",svg:\"svglogo\",\n  png:\"img\",jpg:\"img\",jpeg:\"img\",gif:\"img\",webp:\"img\",bmp:\"img\",avif:\"img\",ico:\"img\",\n  mp4:\"vid\",webm:\"vid\",ogv:\"vid\",mov:\"vid\",m4v:\"vid\",\n  mp3:\"aud\",wav:\"aud\",ogg:\"aud\",oga:\"aud\",m4a:\"aud\",flac:\"aud\",aac:\"aud\",opus:\"aud\",\n  pdf:\"pdf\",docx:\"doc\",doc:\"doc\",pptx:\"ppt\",ppt:\"ppt\",xlsx:\"xls\",xls:\"xls\",tex:\"tex\"};\nconst AGT_IMG_EXT={png:1,jpg:1,jpeg:1,gif:1,webp:1,bmp:1,avif:1,ico:1};\nfunction agtFileIcon(name){\n  const ext=(String(name).split(\".\").pop()||\"\").toLowerCase();\n  const key=AGT_EXT_MAP[ext];\n  return (key&&AGT_LANG_ICONS[key])||AGT_ICO_FILE;\n}\n\nasync function agtLoadDir(p, container, depth, fresh){\n  if(fresh) container.innerHTML='<div class=\"at-load\">Chargement…</div>';\n  let j={};\n  try{\n    const r=await fetch(API_BASE+\"/api/agent/files?path=\"+encodeURIComponent(p),{headers:agtHeaders()});\n    j=await r.json();\n  }catch(e){ j={ok:false}; }\n  if(fresh) container.innerHTML=\"\";\n  if(!j.ok){ container.innerHTML='<div class=\"at-load\">Dossier inaccessible</div>'; return; }\n  const entries=(j.entries||[]).slice().sort((a,b)=>(b.isDirectory?1:0)-(a.isDirectory?1:0)||a.name.localeCompare(b.name));\n  if(!entries.length){ container.innerHTML='<div class=\"at-load\">Dossier vide</div>'; return; }\n  entries.forEach((e,idx)=>{\n    const row=document.createElement(\"div\");\n    row.className=\"at-row\";\n    row.style.paddingLeft=(11+depth*18)+\"px\";\n    row.style.animationDelay=(Math.min(idx,10)*22)+\"ms\";\n    row.innerHTML=(e.isDirectory?AGT_ICO_DIR:agtFileIcon(e.name))+\"<span>\"+esc(e.name)+\"</span>\";\n    container.appendChild(row);\n    const full=p+\"/\"+e.name;\n    if(e.isDirectory){\n      const sub=document.createElement(\"div\");\n      sub.hidden=true;\n      container.appendChild(sub);\n      let loaded=false;\n      row.addEventListener(\"click\",()=>{\n        sub.hidden=!sub.hidden;\n        if(!sub.hidden&&!loaded){ loaded=true; agtLoadDir(full, sub, depth+1, true); }\n      });\n    }else{\n      row.addEventListener(\"click\",()=>agtOpenFile(full, e.name));\n    }\n  });\n}\n\n/* --- visionneuse de fichiers --- */\nfunction agtCloseViewer(){\n  agtViewer.hidden=true;\n  if(typeof agtRevBlobs===\"function\") agtRevBlobs();\n}\ndocument.getElementById(\"agentViewerClose\").addEventListener(\"click\",agtCloseViewer);\n\nfunction agtColorLine(raw){\n  const rx=/([/][/][^\\n]*|#[!][/][^\\n]*)|(\"(?:[^\"\\\\]|\\\\.)*\"|'(?:[^'\\\\]|\\\\.)*')|\\b(const|let|var|function|return|if|else|for|while|await|async|new|class|import|export|from|def|print|echo|sudo|apt|npm|node|cd|mv|cp|mkdir|cat|ls|rm|python3)\\b|(\\b\\d+(?:\\.\\d+)?\\b)|(<[/]?[a-zA-Z][\\w-]*)/g;\n  let out=\"\", last=0, m;\n  while((m=rx.exec(raw))!==null){\n    out+=esc(raw.slice(last,m.index));\n    const cls=m[1]?\"tk-com\":m[2]?\"tk-str\":m[3]?\"tk-kw\":m[4]?\"tk-num\":\"tk-tag\";\n    out+='<i class=\"'+cls+'\">'+esc(m[0])+\"</i>\";\n    last=m.index+m[0].length;\n  }\n  out+=esc(raw.slice(last));\n  return out;\n}\nfunction agtHighlight(code){\n  return String(code).replace(/\\t/g,\"  \").split(\"\\n\").map((ln,i)=>\n    '<div class=\"codeline\"><span class=\"ln\">'+(i+1)+'</span><span class=\"lc\">'+agtColorLine(ln)+\"</span></div>\"\n  ).join(\"\");\n}\n\n/* ---------------- visionneuse de fichiers (phase 45) -------------------- */\nfunction agtRevBlobs(){\n  (agtViewer._blobs||[]).forEach(u=>{ try{URL.revokeObjectURL(u);}catch(e){} });\n  agtViewer._blobs=[];\n  if(agtViewer._blob){ try{URL.revokeObjectURL(agtViewer._blob);}catch(e){} agtViewer._blob=null; }\n}\nfunction agtBlobUrl(blob){\n  if(!agtViewer._blobs) agtViewer._blobs=[];\n  const u=URL.createObjectURL(blob);\n  agtViewer._blobs.push(u);\n  return u;\n}\nfunction agtB64Blob(b64,mime){\n  const bin=atob(String(b64||\"\").replace(/[^A-Za-z0-9+/=]/g,\"\"));\n  const arr=new Uint8Array(bin.length);\n  for(let i=0;i<bin.length;i++) arr[i]=bin.charCodeAt(i);\n  return new Blob([arr],{type:mime||\"application/octet-stream\"});\n}\n/* --- lecture des fichiers de la VM --------------------------------------\n   Une requête ne doit JAMAIS rester en attente indéfiniment : au-delà du\n   délai, on l'abandonne (AbortController) et l'interface reprend la main.\n   Pour un fichier lourd, la lecture se fait par tranches de quelques Mo,\n   servies par la VM : l'aperçu se remplit au fur et à mesure, avec le\n   pourcentage, au lieu de tourner sans fin. */\nconst AGT_READ_TIMEOUT=45000;            /* lecture normale (texte) */\nconst AGT_READ_TIMEOUT_B64=90000;        /* lecture normale (base64) */\nconst AGT_CHUNK=6*1024*1024;             /* taille d'une tranche */\nconst AGT_MAX_READ=120*1024*1024;        /* au-delà : téléchargement conseillé */\nfunction agtMo(n){\n  n=Number(n)||0;\n  if(n>=1073741824) return (n/1073741824).toFixed(2)+\" Go\";\n  if(n>=1048576) return (n/1048576).toFixed(1)+\" Mo\";\n  if(n>=1024) return Math.round(n/1024)+\" Ko\";\n  return n+\" o\";\n}\nasync function agtFetchJson(url,ms,opts){\n  const ctl=(typeof AbortController!==\"undefined\")?new AbortController():null;\n  let to=null;\n  if(ctl&&ms) to=setTimeout(()=>{ try{ ctl.abort(); }catch(e){} },ms);\n  try{\n    const r=await fetch(url,Object.assign({},opts||{},{headers:agtHeaders(),signal:ctl?ctl.signal:undefined}));\n    return await r.json();\n  }catch(e){\n    const ab=(e&&(e.name===\"AbortError\"||String(e.message||\"\").indexOf(\"abort\")>=0));\n    return {ok:false,error:ab?(\"délai dépassé (\"+Math.round(ms/1000)+\" s) — le fichier est peut-être trop lourd pour un envoi en un seul bloc\"):String(e&&e.message||e)};\n  }finally{ if(to) clearTimeout(to); }\n}\nasync function agtReadFileChunked(path,onProgress){\n  const parts=[]; let chars=0, total=0, n=0;\n  for(let guard=0;guard<2000;guard++){\n    const j=await agtFetchJson(API_BASE+\"/api/agent/file?path=\"+encodeURIComponent(path)\n      +\"&chunk=\"+n+\"&size=\"+AGT_CHUNK, AGT_READ_TIMEOUT);\n    if(!j.ok) throw new Error(String(j.error||\"lecture impossible\"));\n    total=Number(j.total)||total;\n    if(total>AGT_MAX_READ) throw new Error(\"Fichier de \"+agtMo(total)+\" : trop lourd pour l'aperçu — utilisez « Télécharger ».\");\n    const c=String(j.content||\"\");\n    if(c){ parts.push(c); chars+=c.length; }\n    if(onProgress) onProgress(chars,total);\n    if(j.eof||!c) break;\n    n++;\n  }\n  return {content:parts.join(\"\"),total:total};\n}\nfunction agtFromB64Utf8(b64){\n  const bin=atob(String(b64||\"\").replace(/[^A-Za-z0-9+/=]/g,\"\"));\n  const arr=new Uint8Array(bin.length);\n  for(let i=0;i<bin.length;i++) arr[i]=bin.charCodeAt(i);\n  try{ return new TextDecoder(\"utf-8\").decode(arr); }catch(e){}\n  let out=\"\",i=0;\n  while(i<arr.length){\n    const b=arr[i++];\n    if(b<0x80){ out+=String.fromCharCode(b); continue; }\n    let cp=0, extra=0;\n    if(b>=0xf0){ cp=b&7; extra=3; } else if(b>=0xe0){ cp=b&15; extra=2; } else { cp=b&31; extra=1; }\n    for(let k=0;k<extra&&i<arr.length;k++) cp=(cp<<6)|(arr[i++]&63);\n    if(cp>0xffff){ cp-=0x10000; out+=String.fromCharCode(0xd800|(cp>>10),0xdc00|(cp&1023)); }\n    else out+=String.fromCharCode(cp);\n  }\n  return out;\n}\n\nfunction agtKatex(el){\n  try{\n    if(window.renderMathInElement&&/\\$/.test(String(el.textContent||\"\"))){\n      window.renderMathInElement(el,{delimiters:[{left:\"$$\",right:\"$$\",display:true},{left:\"$\",right:\"$\",display:false}],throwOnError:false});\n    }\n  }catch(e){}\n}\nconst AGT_VID_EXT={mp4:1,webm:1,ogv:1,mov:1,m4v:1};\nconst AGT_AUD_EXT={mp3:1,wav:1,ogg:1,oga:1,m4a:1,flac:1,aac:1,opus:1};\nfunction agtCat(ext){\n  if(ext===\"svg\") return \"svg\";\n  if(AGT_IMG_EXT[ext]) return \"img\";\n  if(AGT_VID_EXT[ext]) return \"vid\";\n  if(AGT_AUD_EXT[ext]) return \"aud\";\n  if(ext===\"pdf\") return \"pdf\";\n  if(ext===\"docx\") return \"docx\";\n  if(ext===\"pptx\"||ext===\"ppt\") return \"pptx\";\n  if(ext===\"xlsx\"||ext===\"xls\") return \"xlsx\";\n  if(ext===\"html\"||ext===\"htm\") return \"html\";\n  if(ext===\"md\"||ext===\"markdown\") return \"md\";\n  if(ext===\"tex\") return \"tex\";\n  if(ext===\"js\"||ext===\"mjs\"||ext===\"cjs\") return \"js\";\n  if(ext===\"py\") return \"py\";\n  if(ext===\"sh\"||ext===\"bash\") return \"sh\";\n  return \"txt\";\n}\n\n/* HTML multi-fichiers : les .css/.js/images locaux référencés sont récupérés\n   depuis la VM et inlinés. L'iframe reste en blob + sandbox sans same-origin :\n   aucune URL relative ne peut charger l'interface (pas d'« IA dans l'IA »). */\nasync function agtInlineHtml(path,html){\n  const dir=String(path).slice(0,String(path).lastIndexOf(\"/\")+1);\n  const doc=new DOMParser().parseFromString(html,\"text/html\");\n  const norm=(href)=>{\n    const parts=(dir+String(href)).split(\"/\");\n    const out=[];\n    for(const seg of parts){\n      if(seg===\"\"||seg===\".\") continue;\n      if(seg===\"..\"){ out.pop(); continue; }\n      out.push(seg);\n    }\n    return out.join(\"/\");\n  };\n  const vmFile=async(rel,b64)=>{\n    try{\n      const r=await fetch(API_BASE+\"/api/agent/file?path=\"+encodeURIComponent(rel)+(b64?\"&format=base64\":\"\"),{headers:agtHeaders()});\n      const j=await r.json();\n      return j.ok?String(j.content||\"\"):null;\n    }catch(e){ return null; }\n  };\n  for(const link of [...doc.querySelectorAll('link[rel=\"stylesheet\"][href]')]){\n    const href=link.getAttribute(\"href\")||\"\";\n    if(/^(https?:|data:|blob:)/i.test(href)) continue;\n    const css=await vmFile(norm(href),false);\n    if(css!==null){ const st=doc.createElement(\"style\"); st.textContent=css; link.replaceWith(st); }\n  }\n  for(const sc of [...doc.querySelectorAll(\"script[src]\")]){\n    const src=sc.getAttribute(\"src\")||\"\";\n    if(/^(https?:|data:|blob:)/i.test(src)) continue;\n    const js=await vmFile(norm(src),false);\n    if(js!==null){ const ns=doc.createElement(\"script\"); ns.textContent=js; sc.replaceWith(ns); }\n  }\n  for(const im of [...doc.querySelectorAll(\"img[src]\")]){\n    const src=im.getAttribute(\"src\")||\"\";\n    if(/^(https?:|data:|blob:)/i.test(src)) continue;\n    const b=await vmFile(norm(src),true);\n    if(b!==null){\n      const e2=(src.split(\".\").pop()||\"png\").toLowerCase();\n      im.setAttribute(\"src\",\"data:image/\"+(e2===\"svg\"?\"svg+xml\":(e2===\"jpg\"?\"jpeg\":e2))+\";base64,\"+b);\n    }\n  }\n  return LT+\"!DOCTYPE html\"+GT+\"\\n\"+doc.documentElement.outerHTML;\n}\n\n/* Exécution JS locale : iframe jetable, sandbox allow-scripts SEUL (pas de\n   same-origin), console capturée via postMessage et affichée. */\nfunction agtJsRun(code,out){\n  out.hidden=false;\n  out.textContent=\"Exécution…\";\n  const lines=[];\n  /* Balises construites par concaténation : la séquence fermante n'apparaît\n     JAMAIS littéralement dans le source (sinon le navigateur — ou un transit\n     qui mange les antislashs — referme le bloc de script de l'application). */\n  const T_OPEN=\"<\"+\"script>\", T_CLOSE=\"<\"+\"/\"+\"script>\";\n  const harness=LT+\"!DOCTYPE html\"+GT+LT+\"html\"+GT+LT+\"head\"+GT+LT+'meta charset=\"utf-8\"'+GT+LT+SL+\"head\"+GT+LT+\"body\"+GT+T_OPEN\n    +'(function(){var send=function(lvl,args){try{parent.postMessage({__agt:1,lvl:lvl,text:Array.prototype.map.call(args,function(a){try{return typeof a===\"string\"?a:JSON.stringify(a)}catch(e){return String(a)}}).join(\" \")},\"*\")}catch(e){}};'\n    +'[\"log\",\"info\",\"warn\",\"error\",\"debug\"].forEach(function(k){console[k]=function(){send(k===\"error\"?\"error\":\"log\",arguments)}});'\n    +'window.onerror=function(m){send(\"error\",[String(m)])};'\n    +'window.addEventListener(\"unhandledrejection\",function(e){send(\"error\",[String(e&&e.reason)])});})();'\n    +T_CLOSE+T_OPEN+'try{\\n'+String(code)+'\\n}catch(e){parent.postMessage({__agt:1,lvl:\"error\",text:String(e&&e.message||e)},\"*\")}'\n    +'parent.postMessage({__agt:1,done:1},\"*\");'+T_CLOSE+LT+SL+\"body\"+GT+LT+SL+\"html\"+GT;\n  const f=document.createElement(\"iframe\");\n  f.setAttribute(\"sandbox\",\"allow-scripts\");\n  f.style.display=\"none\";\n  let done=false;\n  const paint=()=>{ out.textContent=lines.length?lines.join(\"\\n\"):\"(aucune sortie)\"; };\n  const onMsg=(e)=>{\n    const d=e.data;\n    if(!d||!d.__agt||done) return;\n    if(d.done){ finish(); return; }\n    lines.push((d.lvl===\"error\"?\"⚠ \":\"\")+String(d.text));\n    paint();\n  };\n  const finish=()=>{\n    if(done) return;\n    done=true;\n    clearTimeout(to);\n    window.removeEventListener(\"message\",onMsg);\n    if(f.parentNode) f.parentNode.removeChild(f);\n    paint();\n  };\n  window.addEventListener(\"message\",onMsg);\n  const to=setTimeout(()=>{ lines.push(\"⏱ (arrêt après 10 s)\"); finish(); },10000);\n  /* srcdoc et non blob: : une iframe sandboxée (origine opaque) refuse de\n     charger une URL blob créée par la page — le script ne s'exécutait pas. */\n  f.srcdoc=harness;\n  (document.getElementById(\"avBody\")||document.body).appendChild(f);\n}\n\n/* « Lancer sur la VM » : le code est écrit dans ~/.agtconv puis exécuté via\n   le terminal de la VM (node/python3/bash) ; la sortie est renvoyée.\n   Python : matplotlib en Agg, la dernière figure est renvoyée en PNG. */\nasync function agtRunOnVm(lang,code,out,btn){\n  out.hidden=false;\n  out.textContent=\"Exécution sur la VM…\";\n  if(btn) btn.disabled=true;\n  try{\n    const id=Date.now()+\"_\"+Math.floor(Math.random()*1e6);\n    let src=String(code==null?\"\":code);\n    const ext=lang===\"js\"?\"js\":(lang===\"py\"?\"py\":\"sh\");\n    const fileAbs=\"/home/kasm-user/.agtconv/run_\"+id+\".\"+ext;\n    if(lang===\"py\"){\n      /* matplotlib est OPTIONNEL : sans lui, le script doit quand même\n         s'exécuter (l'import faisait échouer tout le programme). */\n      src='try:\\n    import matplotlib\\n    matplotlib.use(\"Agg\")\\nexcept Exception:\\n    pass\\n'+src\n        +'\\ntry:\\n'\n        +'    import matplotlib.pyplot as _plt, base64 as _b64, io as _io\\n'\n        +'    _figs=_plt.get_fignums()\\n'\n        +'    if _figs:\\n'\n        +'        _buf=_io.BytesIO()\\n'\n        +'        _plt.figure(_figs[-1]).savefig(_buf, format=\"png\", dpi=110, bbox_inches=\"tight\")\\n'\n        +'        print(\"AGT_IMG:\"+_b64.b64encode(_buf.getvalue()).decode())\\n'\n        +'except Exception:\\n    pass\\n';\n    }\n    const runner=lang===\"js\"\n      ? 'NODE=$(ls /home/kasm-user/.nvm/versions/node/*/bin/node 2>/dev/null | sort -V | tail -1); if [ -n \"$NODE\" ]; then timeout 170 \"$NODE\" \"'+fileAbs+'\"; else echo \"(Node indisponible sur la VM)\"; fi'\n      : (lang===\"py\" ? 'python3 \"'+fileAbs+'\"' : 'bash \"'+fileAbs+'\"');\n    const b64=agtB64Utf8(src);\n    const cmd=\"mkdir -p /home/kasm-user/.agtconv && printf '%s' '\"+b64+\"' | base64 -d > \"+fileAbs\n      +\" && cd /home/kasm-user && \"+(lang===\"js\"?\"\":\"timeout 170 \")+runner+\" 2>&1; rm -f \"+fileAbs;\n    const r=await fetch(API_BASE+\"/api/agent/exec\",{\n      method:\"POST\",headers:agtHeaders({\"Content-Type\":\"application/json\"}),\n      body:JSON.stringify({command:cmd,timeout:180})\n    });\n    const j=await r.json();\n    let so=String(j.stdout||\"\");\n    const se=String(j.stderr||\"\");\n    /* Node n'est pas installé sur la VM : on exécute le JS localement. */\n    if(lang===\"js\" && /Node indisponible/.test(so+se)){\n      out.textContent=\"\";\n      agtJsRun(code,out);\n      return;\n    }\n    let img=null;\n    const m=so.match(/AGT_IMG:([A-Za-z0-9+/=]+)/);\n    if(m){ img=m[1]; so=so.replace(/AGT_IMG:[A-Za-z0-9+/=]+/g,\"\").trim(); }\n    out.innerHTML=\"\";\n    if(img){\n      const im=document.createElement(\"img\");\n      im.className=\"av-runimg\";\n      im.src=\"data:image/png;base64,\"+img;\n      out.appendChild(im);\n    }\n    const tx=document.createElement(\"div\");\n    tx.textContent=(so||se||\"(aucune sortie)\")+(j.exit_code?\"  [exit \"+j.exit_code+\"]\":\"\");\n    out.appendChild(tx);\n  }catch(e){\n    out.textContent=\"Exécution impossible : \"+String(e&&e.message||e);\n  }finally{\n    if(btn) btn.disabled=false;\n  }\n}\n\n/* Présentation (pptx/ppt) : convertie en PDF par LibreOffice dans la VM,\n   puis affichée. Repli : message + téléchargement. */\nasync function agtPptxToPdf(path,name,content,body,mkBtn,dlBtn){\n  const div=document.createElement(\"div\"); div.className=\"av-md\";\n  body.appendChild(div);\n  div.innerHTML='<div class=\"at-load\">Conversion de la présentation via la VM…</div>';\n  const abs=\"/home/kasm-user/\"+String(path).replace(/^[/]+/,\"\");\n  if(!/['\"]/.test(abs)){\n    const outdir=\"/home/kasm-user/.agtconv\";\n    const base=name.replace(/\\.[^.]+$/,\"\")+\".pdf\";\n    try{\n      const er=await fetch(API_BASE+\"/api/agent/exec\",{\n        method:\"POST\",headers:agtHeaders({\"Content-Type\":\"application/json\"}),\n        body:JSON.stringify({timeout:180,command:\"mkdir -p \"+outdir+\" && rm -f \"+outdir+\"/*\"\n          +\" && (which soffice >/dev/null 2>&1 || sudo -n apt-get install -y -qq --no-install-recommends libreoffice-impress libreoffice-writer >/dev/null 2>&1);\"\n          +\" LD_LIBRARY_PATH=/usr/lib/libreoffice/program timeout 170 soffice --headless -env:UserInstallation=file:///tmp/lo_agt --convert-to pdf --outdir \"+outdir+\" '\"+abs+\"' >/dev/null 2>&1;\"\n          +\" [ -s '\"+outdir+\"/\"+base+\"' ] && echo OKPDF || echo NOPDF\"})\n      });\n      const jr=await er.json();\n      if(String(jr.stdout||\"\").includes(\"OKPDF\")){\n        const r2=await fetch(API_BASE+\"/api/agent/file?path=\"+encodeURIComponent(\".agtconv/\"+base)+\"&format=base64\",{headers:agtHeaders()});\n        const j2=await r2.json();\n        if(j2.ok){\n          const f=document.createElement(\"iframe\"); f.className=\"av-frame\";\n          const u=agtBlobUrl(agtB64Blob(j2.content,\"application/pdf\"));\n          f.src=u;\n          body.innerHTML=\"\"; body.appendChild(f);\n          dlBtn(u,base);\n          return;\n        }\n      }\n    }catch(e){}\n  }\n  div.innerHTML='<p>Conversion de la présentation impossible (LibreOffice indisponible sur la VM pour le moment). Utilisez le bouton Télécharger pour l’ouvrir sur votre appareil.</p>';\n}\n\nasync function agtOpenFile(path, name){\n  const title=document.getElementById(\"avTitle\");\n  const body=document.getElementById(\"avBody\");\n  const bar=document.getElementById(\"avBtns\");\n  title.textContent=name;\n  bar.innerHTML=\"\";\n  body.innerHTML='<div class=\"at-load\">Chargement…</div>';\n  agtRevBlobs();\n  agtViewer.hidden=false;\n  const ext=(name.split(\".\").pop()||\"\").toLowerCase();\n  const cat=agtCat(ext);\n  const needB64=(cat===\"img\"||cat===\"svg\"||cat===\"vid\"||cat===\"aud\"||cat===\"pdf\"||cat===\"docx\"||cat===\"pptx\"||cat===\"xlsx\");\n  let j={};\n  try{\n    j=await agtFetchJson(API_BASE+\"/api/agent/file?path=\"+encodeURIComponent(path)+(needB64?\"&format=base64\":\"\"),\n      needB64?AGT_READ_TIMEOUT_B64:AGT_READ_TIMEOUT);\n  }catch(e){ j={ok:false,error:String(e&&e.message||e)}; }\n  let content=\"\";\n  if(j.ok){\n    content=String(j.content==null?\"\":j.content);\n  }else{\n    /* Lecture directe impossible (fichier lourd, ou délai dépassé) : on\n       reprend par tranches, en affichant la progression. */\n    try{\n      const prog=document.createElement(\"div\");\n      prog.className=\"at-load\";\n      prog.textContent=\"Chargement…\";\n      body.innerHTML=\"\"; body.appendChild(prog);\n      const r=await agtReadFileChunked(path,(done,total)=>{\n        prog.textContent=\"Chargement… \"+agtMo(Math.round(done*3/4))\n          +(total?(\" / \"+agtMo(total)+\" (\"+Math.min(100,Math.round((done*3/4)*100/Math.max(1,total)))+\" %)\"):\"\");\n      });\n      content=r.content;\n      if(!needB64&&content){ try{ content=agtFromB64Utf8(content); }catch(e){} }\n      body.innerHTML=\"\";\n    }catch(e){\n      body.innerHTML='<div class=\"at-load\">'+esc(String(e&&e.message||e))+\"</div>\";\n      return;\n    }\n  }\n  const mkBtn=(label,fn)=>{\n    const b=document.createElement(\"button\");\n    b.type=\"button\"; b.className=\"av-btn\"; b.textContent=label;\n    b.addEventListener(\"click\",()=>{ fn(b); });\n    bar.appendChild(b);\n    return b;\n  };\n  const dlBtn=(url,fname)=>mkBtn(\"Télécharger\",()=>{\n    /* Le lien peut être un blob d'aperçu : on le passe au moteur commun,\n       qui garde une feuille de secours si le navigateur ne suit pas. */\n    zDlGo(url, fname, \"\", {why:\"Le fichier est prêt.\"});\n  });\n  const codeToggle=(mainEl,content2)=>{\n    const code=document.createElement(\"div\");\n    code.className=\"av-code\";\n    code.innerHTML=agtHighlight(content2);\n    code.hidden=true;\n    body.appendChild(code);\n    let showCode=false;\n    mkBtn(\"Voir le code\",(b)=>{\n      showCode=!showCode;\n      mainEl.hidden=showCode; code.hidden=!showCode;\n      b.classList.toggle(\"pressed\",showCode);\n      b.textContent=showCode?\"Aperçu\":\"Voir le code\";\n    });\n  };\n  body.innerHTML=\"\";\n  if(path===AGENT_USER+\"/user.md\"){\n    /* Fiche utilisateur : seul fichier modifiable depuis l'interface. */\n    agtRenderUserMd(content);\n    return;\n  }\n  if(cat===\"img\"){\n    /* Ajusté à l'écran par défaut (l'image entière est visible) ; « Plein\n       format » affiche la taille réelle, la zone défile alors. */\n    const wrap=document.createElement(\"div\"); wrap.className=\"av-imgwrap\";\n    const im=document.createElement(\"img\"); im.className=\"av-img\"; im.alt=name;\n    im.src=\"data:image/\"+(ext===\"jpg\"?\"jpeg\":(ext===\"ico\"?\"x-icon\":ext))+\";base64,\"+content;\n    wrap.appendChild(im);\n    body.appendChild(wrap);\n    dlBtn(im.src,name);\n    let full=false;\n    const fb=mkBtn(\"Plein format\",(b)=>{\n      full=!full;\n      wrap.classList.toggle(\"full\",full);\n      b.classList.toggle(\"pressed\",full);\n      b.textContent=full?\"Ajuster à l'écran\":\"Plein format\";\n    });\n    fb.title=\"Afficher l'image à sa taille réelle\";\n    im.addEventListener(\"click\",()=>fb.click());\n    return;\n  }\n  if(cat===\"svg\"){\n    /* Le fichier arrive en base64 : il est DÉCODÉ en texte d'abord.\n       (Avant, le base64 brut partait dans l'iframe -> aperçu blanc, et\n       « Voir le code » affichait du base64.) */\n    const text=agtDecodeUtf8(content);\n    const f=document.createElement(\"iframe\");\n    f.className=\"av-frame\";\n    f.setAttribute(\"sandbox\",\"allow-scripts\");\n    f.srcdoc=svgDoc(text);\n    body.appendChild(f);\n    codeToggle(f,text);\n    dlBtn(agtBlobUrl(new Blob([text],{type:\"image/svg+xml\"})),name);\n    mkBtn(\"Copier\",()=>agtCopyText(text));\n    return;\n  }\n  if(cat===\"vid\"){\n    const wrap=document.createElement(\"div\"); wrap.className=\"av-vidwrap\";\n    const v=document.createElement(\"video\"); v.className=\"av-vid\"; v.controls=true;\n    const u=agtBlobUrl(agtB64Blob(content,\"video/\"+(ext===\"mov\"?\"quicktime\":ext)));\n    v.src=u;\n    wrap.appendChild(v);\n    body.appendChild(wrap);\n    dlBtn(u,name);\n    let fullV=false;\n    const fbv=mkBtn(\"Plein format\",(b)=>{\n      fullV=!fullV;\n      wrap.classList.toggle(\"full\",fullV);\n      b.classList.toggle(\"pressed\",fullV);\n      b.textContent=fullV?\"Ajuster à l'écran\":\"Plein format\";\n    });\n    fbv.title=\"Afficher la vidéo à sa taille réelle\";\n    return;\n  }\n  if(cat===\"aud\"){\n    const wrap=document.createElement(\"div\"); wrap.className=\"av-audwrap\";\n    const lab=document.createElement(\"div\"); lab.className=\"av-audname\"; lab.textContent=name;\n    const a=document.createElement(\"audio\"); a.className=\"av-aud\"; a.controls=true;\n    const u=agtBlobUrl(agtB64Blob(content,\"audio/\"+(ext===\"m4a\"?\"mp4\":(ext===\"oga\"?\"ogg\":ext))));\n    a.src=u;\n    wrap.appendChild(lab); wrap.appendChild(a);\n    body.appendChild(wrap);\n    dlBtn(u,name);\n    return;\n  }\n  if(cat===\"pdf\"){\n    /* Rendu CANVAS via pdf.js : une iframe PDF reste blanche sur de\n       nombreux navigateurs (Android, Firefox…), d'où « page blanche ». */\n    const u=agtBlobUrl(agtB64Blob(content,\"application/pdf\"));\n    dlBtn(u,name);\n    const wrap=document.createElement(\"div\"); wrap.className=\"av-pdfwrap\";\n    body.appendChild(wrap);\n    if(window.pdfjsLib&&window.pdfjsLib.getDocument){\n      wrap.innerHTML='<div class=\"at-load\">Rendu du PDF…</div>';\n      try{\n        try{ window.pdfjsLib.GlobalWorkerOptions.workerSrc=\"https://cdn.jsdelivr.net/npm/pdfjs-dist@3.11.174/legacy/build/pdf.worker.min.js\"; }catch(e2){}\n        const data=agtB64Blob(content,\"application/pdf\").arrayBuffer();\n        const doc=await window.pdfjsLib.getDocument({data:new Uint8Array(data),disableAutoFetch:true}).promise;\n        wrap.innerHTML=\"\";\n        const total=Math.min(doc.numPages,60);\n        for(let n=1;n<=total;n++){\n          const page=await doc.getPage(n);\n          const vp=page.getViewport({scale:1.6});\n          const cv=document.createElement(\"canvas\");\n          cv.className=\"av-pdfpage\";\n          cv.width=Math.round(vp.width); cv.height=Math.round(vp.height);\n          const cx=cv.getContext(\"2d\");\n          await page.render({canvasContext:cx,viewport:vp}).promise;\n          wrap.appendChild(cv);\n        }\n        if(doc.numPages>total){\n          const note=document.createElement(\"p\");\n          note.className=\"av-pdfnote\";\n          note.textContent=\"Affichage limité aux \"+total+\" premières pages (\"+doc.numPages+\" au total).\";\n          wrap.appendChild(note);\n        }\n      }catch(e){\n        wrap.innerHTML=\"<p>Aperçu du PDF indisponible ici. \"\n          +\"Utilisez le bouton Télécharger (le PDF est intact).</p>\";\n      }\n    }else{\n      wrap.innerHTML=\"<p>Aperçu du PDF indisponible (librairie de rendu non chargée). \"\n        +\"Utilisez le bouton Télécharger pour l’ouvrir sur votre appareil.</p>\";\n    }\n    return;\n  }\n  if(cat===\"docx\"){\n    const u=agtBlobUrl(agtB64Blob(content,\"application/vnd.openxmlformats-officedocument.wordprocessingml.document\"));\n    dlBtn(u,name);\n    const div=document.createElement(\"div\"); div.className=\"av-md av-docx\";\n    body.appendChild(div);\n    /* docx-preview respecte la mise en page ET les couleurs du document Word\n       (mammoth, lui, ne rendait qu'un texte gris, sans styles). */\n    if(window.docx&&window.docx.renderAsync){\n      div.innerHTML='<div class=\"at-load\">Mise en page du document Word…</div>';\n      try{\n        const blob=agtB64Blob(content,\"application/vnd.openxmlformats-officedocument.wordprocessingml.document\");\n        div.innerHTML=\"\";\n        await window.docx.renderAsync(blob, div, null,\n          {className:\"docx\",inWrapper:true,breakPages:true,ignoreLastRenderedPageBreak:true});\n        div.style.background=\"#fff\";\n      }catch(e){ div.innerHTML=\"<p>Document Word illisible.</p>\"; }\n    }else if(window.mammoth&&window.mammoth.convertToHtml){\n      div.innerHTML='<div class=\"at-load\">Conversion du document Word…</div>';\n      try{\n        const ab=await agtB64Blob(content,\"\").arrayBuffer();\n        const res=await window.mammoth.convertToHtml({arrayBuffer:ab});\n        div.innerHTML=res.value||\"<p>(document vide)</p>\";\n      }catch(e){ div.innerHTML=\"<p>Document Word illisible.</p>\"; }\n    }else{\n      div.innerHTML=\"<p>Aperçu Word indisponible ici (librairie de conversion non chargée). Utilisez le bouton Télécharger pour l’ouvrir sur votre appareil.</p>\";\n    }\n    return;\n  }\n  if(cat===\"pptx\"){\n    const u=agtBlobUrl(agtB64Blob(content,\"application/vnd.openxmlformats-officedocument.presentationml.presentation\"));\n    dlBtn(u,name);\n    await agtPptxToPdf(path,name,content,body,mkBtn,dlBtn);\n    return;\n  }\n  if(cat===\"xlsx\"){\n    const u=agtBlobUrl(agtB64Blob(content,\"application/vnd.openxmlformats-officedocument.spreadsheetml.sheet\"));\n    dlBtn(u,name);\n    const div=document.createElement(\"div\"); div.className=\"av-md\";\n    body.appendChild(div);\n    if(window.XLSX&&window.XLSX.read){\n      div.innerHTML='<div class=\"at-load\">Lecture du tableur…</div>';\n      try{\n        const ab=await agtB64Blob(content,\"\").arrayBuffer();\n        const wb=window.XLSX.read(new Uint8Array(ab),{type:\"array\"});\n        let h=\"\";\n        wb.SheetNames.slice(0,4).forEach(sn=>{\n          h+=\"<h3>\"+esc(sn)+\"</h3>\"+window.XLSX.utils.sheet_to_html(wb.Sheets[sn]);\n        });\n        div.innerHTML=h||\"<p>(tableur vide)</p>\";\n      }catch(e){ div.innerHTML=\"<p>Tableur illisible.</p>\"; }\n    }else{\n      div.innerHTML=\"<p>Aperçu tableur indisponible ici (librairie non chargée). Utilisez le bouton Télécharger.</p>\";\n    }\n    return;\n  }\n  if(cat===\"html\"){\n    /* Le HTML lié à ses .css/.js locaux : fichiers frères inlinés depuis la VM. */\n    let full=content;\n    try{ full=await agtInlineHtml(path,content); }catch(e){}\n    const f=document.createElement(\"iframe\");\n    f.className=\"av-frame\";\n    f.setAttribute(\"sandbox\",\"allow-scripts allow-forms allow-modals allow-popups\");\n    /* srcdoc : les blob: ne se chargent pas dans une iframe sandboxée\n       (origine opaque) — l'aperçu restait donc vide. */\n    f.srcdoc=full;\n    body.appendChild(f);\n    codeToggle(f,content);\n    mkBtn(\"Copier le code\",()=>agtCopyText(content));\n    return;\n  }\n  if(cat===\"md\"){\n    const div=document.createElement(\"div\");\n    div.className=\"av-md\";\n    div.innerHTML=renderMarkdown(content);\n    agtKatex(div);\n    const pre=document.createElement(\"pre\");\n    pre.className=\"av-pre\"; pre.textContent=content; pre.hidden=true;\n    body.appendChild(div); body.appendChild(pre);\n    let raw=false;\n    mkBtn(\"Texte brut\",(b)=>{\n      raw=!raw;\n      div.hidden=raw; pre.hidden=!raw;\n      b.classList.toggle(\"pressed\",raw);\n      b.textContent=raw?\"Aperçu\":\"Texte brut\";\n    });\n    mkBtn(\"Copier\",()=>agtCopyText(content));\n    return;\n  }\n  if(cat===\"tex\"){\n    const div=document.createElement(\"div\");\n    div.className=\"av-md\";\n    let rendered=false;\n    if(window.renderMathInElement){\n      try{\n        const holder=document.createElement(\"div\");\n        holder.textContent=\"$$\"+content+\"$$\";\n        div.appendChild(holder);\n        window.renderMathInElement(div,{delimiters:[{left:\"$$\",right:\"$$\",display:true}],throwOnError:false});\n        rendered=true;\n      }catch(e){ rendered=false; }\n    }\n    if(!rendered){\n      div.innerHTML='<p class=\"um-zh\">Aperçu LaTeX indisponible ici (KaTeX non chargé) — voir le texte brut.</p>';\n    }\n    const pre=document.createElement(\"pre\");\n    pre.className=\"av-pre\"; pre.textContent=content;\n    pre.hidden=rendered; div.hidden=!rendered;\n    body.appendChild(div); body.appendChild(pre);\n    let raw=!rendered;\n    mkBtn(\"Texte brut\",(b)=>{\n      raw=!raw;\n      div.hidden=raw; pre.hidden=!raw;\n      b.classList.toggle(\"pressed\",raw);\n      b.textContent=raw?\"Aperçu\":\"Texte brut\";\n    });\n    mkBtn(\"Copier\",()=>agtCopyText(content));\n    return;\n  }\n  if(cat===\"js\"||cat===\"py\"||cat===\"sh\"){\n    const pre=document.createElement(\"pre\");\n    pre.className=\"av-pre\";\n    pre.innerHTML=agtHighlight(content);\n    const out=document.createElement(\"pre\");\n    out.className=\"av-run\"; out.hidden=true;\n    body.appendChild(pre); body.appendChild(out);\n    if(cat===\"js\"){\n      mkBtn(\"Exécuter\",()=>agtJsRun(content,out));\n      mkBtn(\"Lancer sur la VM\",(b)=>agtRunOnVm(\"js\",content,out,b));\n    }else if(cat===\"py\"){\n      mkBtn(\"Lancer sur la VM\",(b)=>agtRunOnVm(\"py\",content,out,b));\n    }else{\n      mkBtn(\"Lancer sur la VM\",(b)=>agtRunOnVm(\"sh\",content,out,b));\n    }\n    mkBtn(\"Copier\",()=>agtCopyText(content));\n    return;\n  }\n  const pre=document.createElement(\"pre\");\n  pre.className=\"av-pre\"; pre.textContent=content;\n  body.appendChild(pre);\n  mkBtn(\"Copier\",()=>agtCopyText(content));\n}\n\n/* --- fiche utilisateur (user.md) ---------------------------------------\n   Seul fichier que l'utilisateur peut modifier dans son interface :\n   crayon pour éditer (nom, e-mail, âge, métier, infos, instructions),\n   ✕ annule, ✓ enregistre. Protégé côté VM : recréé s'il est supprimé. */\nfunction agtB64Utf8(s){\n  /* Encodage UTF-8 → b64 sans TextEncoder (compatible partout, y c. jsdom). */\n  s=String(s==null?\"\":s);\n  let bin=\"\";\n  for(let i=0;i<s.length;i++){\n    let c=s.charCodeAt(i);\n    if(c<0x80){ bin+=String.fromCharCode(c); }\n    else if(c<0x800){ bin+=String.fromCharCode(0xc0|(c>>6),0x80|(c&63)); }\n    else if(c>=0xd800&&c<0xdc00&&i+1<s.length){\n      const cp=0x10000+((c-0xd800)<<10)+(s.charCodeAt(++i)-0xdc00);\n      bin+=String.fromCharCode(0xf0|(cp>>18),0x80|((cp>>12)&63),0x80|((cp>>6)&63),0x80|(cp&63));\n    }\n    else{ bin+=String.fromCharCode(0xe0|(c>>12),0x80|((c>>6)&63),0x80|(c&63)); }\n  }\n  return btoa(bin);\n}\nfunction agtDecodeUtf8(b64){\n  /* base64 -> texte UTF-8, sans TextDecoder (compatible jsdom). */\n  let bin=\"\";\n  try{ bin=atob(String(b64||\"\").replace(/[^A-Za-z0-9+/=]/g,\"\")); }catch(e){ return \"\"; }\n  let out=\"\", i=0;\n  while(i<bin.length){\n    const c=bin.charCodeAt(i++);\n    if(c<0x80){ out+=String.fromCharCode(c); continue; }\n    if(c<0xe0){ out+=String.fromCharCode(((c&31)<<6)|(bin.charCodeAt(i++)&63)); continue; }\n    if(c<0xf0){ out+=String.fromCharCode(((c&15)<<12)|((bin.charCodeAt(i++)&63)<<6)|(bin.charCodeAt(i++)&63)); continue; }\n    const cp=((c&7)<<18)|((bin.charCodeAt(i++)&63)<<12)|((bin.charCodeAt(i++)&63)<<6)|(bin.charCodeAt(i++)&63);\n    const v=cp-0x10000;\n    out+=String.fromCharCode(0xd800+(v>>10),0xdc00+(v&1023));\n  }\n  return out;\n}\nconst AGT_UM_HINT_INFOS=\"(Écrivez ici ce que vous voulez que l'agent sache sur vous : préférences, habitudes, projets en cours.)\";\nconst AGT_UM_HINT_INSTR=\"(Vos instructions personnelles pour l'agent : ton, format, ce qu'il doit toujours faire ou ne jamais faire.)\";\nfunction agtParseUserMd(c){\n  c=String(c==null?\"\":c);\n  const field=(label)=>{\n    const m=c.match(new RegExp(\"\\\\*\\\\*\"+label+\"\\\\s*:\\\\*\\\\*\\\\s*(.*)\",\"m\"));\n    let v=m?String(m[1]||\"\").trim():\"\";\n    if(v===\"(non renseigné)\") v=\"\";\n    return v;\n  };\n  const zone=(head)=>{\n    const i=c.indexOf(\"## \"+head);\n    if(i<0) return \"\";\n    let s=c.slice(i+head.length+3);\n    const j=s.indexOf(\"\\n## \");\n    if(j>=0) s=s.slice(0,j);\n    s=s.replace(/^\\s+/,\"\").replace(/\\s+$/,\"\");\n    if(/^\\([\\s\\S]*\\)$/.test(s)) s=\"\";   /* indice par défaut entre parenthèses */\n    return s;\n  };\n  return {nom:field(\"Nom\"),email:field(\"Adresse e-mail\"),age:field(\"Âge\"),metier:field(\"Métier\"),\n          infos:zone(\"Informations supplémentaires\"),instructions:zone(\"Instructions personnalisées\")};\n}\nfunction agtBuildUserMd(f){\n  const fl=(l,v)=>\"- **\"+l+\" :** \"+(String(v==null?\"\":v).trim()||\"(non renseigné)\");\n  return \"# user.md — Fiche utilisateur\\n\\n\"\n    +fl(\"Nom\",f.nom)+\"\\n\"+fl(\"Adresse e-mail\",f.email)+\"\\n\"+fl(\"Âge\",f.age)+\"\\n\"+fl(\"Métier\",f.metier)+\"\\n\\n\"\n    +\"## Informations supplémentaires\\n\\n\"+(String(f.infos||\"\").trim()||AGT_UM_HINT_INFOS)+\"\\n\\n\"\n    +\"## Instructions personnalisées\\n\\n\"+(String(f.instructions||\"\").trim()||AGT_UM_HINT_INSTR)+\"\\n\";\n}\nasync function agtSaveUserMd(text){\n  try{\n    const b64=agtB64Utf8(text);\n    const r=await fetch(API_BASE+\"/api/agent/exec\",{\n      method:\"POST\",headers:agtHeaders({\"Content-Type\":\"application/json\"}),\n      body:JSON.stringify({command:\"printf '%s' '\"+b64+\"'\"\n        +\" | base64 -d > /home/kasm-user/user/user.md\"\n        +\" && cp -f /home/kasm-user/user/user.md /home/kasm-user/.user.md.bak && echo SAVED\"})\n    });\n    const j=await r.json();\n    return String(j.stdout||\"\").includes(\"SAVED\");\n  }catch(e){ return false; }\n}\nfunction agtRenderUserMd(content){\n  const body=document.getElementById(\"avBody\");\n  const bar=document.getElementById(\"avBtns\");\n  const SV_PENCIL='<svg width=\"14\" height=\"14\" viewBox=\"0 0 24 24\" fill=\"none\" stroke=\"currentColor\" stroke-width=\"2\" stroke-linecap=\"round\" stroke-linejoin=\"round\"><path d=\"M17 3a2.83 2.83 0 1 1 4 4L7.5 20.5 2 22l1.5-5.5Z\"/><path d=\"M15 5l4 4\"/></svg>';\n  const SV_CROSS='<svg width=\"14\" height=\"14\" viewBox=\"0 0 24 24\" fill=\"none\" stroke=\"currentColor\" stroke-width=\"2\" stroke-linecap=\"round\"><path d=\"M5 5l14 14M19 5L5 19\"/></svg>';\n  const SV_CHECK='<svg width=\"14\" height=\"14\" viewBox=\"0 0 24 24\" fill=\"none\" stroke=\"currentColor\" stroke-width=\"2.5\" stroke-linecap=\"round\" stroke-linejoin=\"round\"><path d=\"M20 6L9 17l-5-5\"/></svg>';\n  let cur=String(content==null?\"\":content);\n  const iconBtn=(svg,label,fn)=>{\n    const b=document.createElement(\"button\");\n    b.type=\"button\"; b.className=\"av-btn um-ico\";\n    b.setAttribute(\"aria-label\",label); b.title=label;\n    b.innerHTML=svg;\n    b.addEventListener(\"click\",()=>{ fn(b); });\n    bar.appendChild(b);\n    return b;\n  };\n  const showView=()=>{\n    bar.innerHTML=\"\";\n    iconBtn(SV_PENCIL,\"Modifier\",()=>startEdit());\n    const wrap=document.createElement(\"div\"); wrap.className=\"um-wrap\";\n    const div=document.createElement(\"div\"); div.className=\"av-md\";\n    div.innerHTML=renderMarkdown(cur);\n    wrap.appendChild(div);\n    body.innerHTML=\"\"; body.appendChild(wrap);\n  };\n  const startEdit=()=>{\n    const f=agtParseUserMd(cur);\n    bar.innerHTML=\"\";\n    iconBtn(SV_CROSS,\"Annuler\",()=>showView());\n    iconBtn(SV_CHECK,\"Enregistrer\",(b)=>save(b));\n    const wrap=document.createElement(\"div\"); wrap.className=\"um-wrap\";\n    const mkField=(label,val,ph)=>{\n      const lab=document.createElement(\"label\"); lab.className=\"um-field\";\n      const s=document.createElement(\"span\"); s.textContent=label;\n      const i=document.createElement(\"input\"); i.type=\"text\"; i.value=val||\"\"; i.placeholder=ph||\"\";\n      lab.appendChild(s); lab.appendChild(i);\n      wrap.appendChild(lab);\n      return i;\n    };\n    const mkArea=(label,val,ph)=>{\n      const lab=document.createElement(\"label\"); lab.className=\"um-field um-area\";\n      const s=document.createElement(\"span\"); s.textContent=label;\n      const ta=document.createElement(\"textarea\"); ta.className=\"um-edit\"; ta.value=val||\"\"; ta.placeholder=ph||\"\";\n      lab.appendChild(s); lab.appendChild(ta);\n      wrap.appendChild(lab);\n      return ta;\n    };\n    const F={};\n    F.inNom=mkField(\"Nom\",f.nom,\"Votre nom\");\n    F.inMail=mkField(\"Adresse e-mail\",f.email,\"vous@exemple.com\");\n    F.inAge=mkField(\"Âge\",f.age,\"ex. : 34\");\n    F.inMet=mkField(\"Métier\",f.metier,\"ex. : développeur\");\n    F.taInf=mkArea(\"Informations supplémentaires\",f.infos,\"Préférences, habitudes, projets…\");\n    F.taIns=mkArea(\"Instructions personnalisées\",f.instructions,\"Vos instructions pour l'agent…\");\n    wrap._fields=F;\n    body.innerHTML=\"\"; body.appendChild(wrap);\n    try{ F.inNom.focus(); }catch(e){}\n  };\n  const save=async(b)=>{\n    const w=body.querySelector(\".um-wrap\");\n    if(!w||!w._fields){ showView(); return; }\n    const F=w._fields;\n    const nc=agtBuildUserMd({nom:F.inNom.value,email:F.inMail.value,age:F.inAge.value,\n      metier:F.inMet.value,infos:F.taInf.value,instructions:F.taIns.value});\n    b.disabled=true;\n    const ok=await agtSaveUserMd(nc);\n    b.disabled=false;\n    if(ok){\n      cur=nc;\n      agtToast(\"Fiche enregistrée.\");\n      showView();\n      try{ agtLoadDir(AGENT_USER, document.getElementById(\"agentTree\"), 0, true); }catch(e){}\n    }else{\n      agtToast(\"Enregistrement impossible (VM injoignable).\");\n    }\n  };\n  showView();\n}\n\n\ndocument.addEventListener(\"keydown\",e=>{\n  if(e.key!==\"Escape\") return;\n  if(!agtViewer.hidden){ agtCloseViewer(); return; }\n  if(!agtPanel.hidden){ agtClosePanel(); }\n});\n\n/* --- rendu des missions : UN SEUL chemin de construction du DOM -------\n   Les mêmes fonctions servent PENDANT la mission et au RECHARGEMENT :\n   l'affichage d'une discussion est donc rigoureusement identique dans les\n   deux cas (zone terminal repliable comprise). */\nfunction agtTermEl(bubble, beforeEl, folded){\n  const term=document.createElement(\"div\"); term.className=\"agent-term\";\n  const head=document.createElement(\"div\"); head.className=\"agent-loghead\";\n  head.innerHTML='<svg width=\"10\" height=\"10\" viewBox=\"0 0 24 24\" fill=\"none\" stroke=\"currentColor\" stroke-width=\"2.4\" stroke-linecap=\"round\" stroke-linejoin=\"round\"><path d=\"M6 9l6 6 6-6\"/></svg><span>Terminal de la VM</span>';\n  head.addEventListener(\"click\",()=>term.classList.toggle(\"folded\"));\n  const log=document.createElement(\"div\"); log.className=\"agent-log\";\n  term.appendChild(head); term.appendChild(log);\n  if(folded) term.classList.add(\"folded\");\n  if(beforeEl && beforeEl.parentNode===bubble) bubble.insertBefore(term, beforeEl);\n  else bubble.appendChild(term);\n  return {term:term, log:log};\n}\nfunction agtStepEl(logEl, cmd, kind){\n  const st=document.createElement(\"div\"); st.className=\"agent-step\";\n  /* Les étapes « fetch » s'affichent telles quelles (fetch [html] https://…) ;\n     les commandes shell gardent le préfixe « $ ». */\n  st.textContent=(kind===\"fetch\"?\"\":\"$ \")+String(cmd==null?\"\":cmd);\n  logEl.appendChild(st);\n  return st;\n}\nfunction agtOutEl(logEl, so, se, exit){\n  const ol=document.createElement(\"div\"); ol.className=\"agent-out\";\n  ol.textContent=(so||se||\"(aucune sortie)\")+(exit?\"  [exit \"+exit+\"]\":\"\");\n  logEl.appendChild(ol);\n  logEl.scrollTop=logEl.scrollHeight;\n  return ol;\n}\n/* Version markdown (bouton Copier) : texte final + trace du terminal. */\nfunction agtTraceMd(steps, finalText){\n  const tr=(steps||[]).map(s=>(s.kind===\"fetch\"?\"\":\"$ \")+s.cmd+\"\\n\"+(s.out||s.err||\"(aucune sortie)\")\n    +(s.exit&&s.kind!==\"fetch\"?\"  [exit \"+s.exit+\"]\":\"\")).join(\"\\n\\n\");\n  return String(finalText||\"\")+(tr.trim()\n    ? (\"\\n\\n---\\n\\n**Terminal de cette mission :**\\n\\n```\\n\"+tr.trim()+\"\\n```\")\n    : \"\");\n}\n/* Missions enregistrées AVANT la phase 50 : la trace du terminal était\n   stockée en markdown dans le texte. On la ressort pour retrouver la même\n   zone repliable, au lieu du gros bloc de code. */\nfunction agtSplitLegacyMission(text){\n  const t=String(text||\"\");\n  const mk=\"\\n\\n---\\n\\n**Terminal de cette mission :**\\n\\n```\";\n  const i=t.indexOf(mk);\n  if(i<0) return null;\n  let tr=t.slice(i+mk.length);\n  const j=tr.lastIndexOf(\"\\n```\");\n  if(j>=0) tr=tr.slice(0,j);\n  return {finalText:t.slice(0,i), trace:tr};\n}\n/* Mission rechargée depuis Firestore : même bulle, même zone repliable. */\nfunction agtRenderMission(bubble, steps, finalText, folded){\n  /* La bulle vient d'être créée avec l'indicateur d'attente : il ne fait pas\n     partie du message et ne doit pas réapparaître au rechargement. */\n  const sp=bubble.querySelector(\".gentxt\"); if(sp) sp.remove();\n  const built=agtTermEl(bubble, null, folded!==false);\n  (steps||[]).forEach(s=>{\n    if(s&&s.raw){                       /* ancienne trace, d'un seul bloc */\n      const d=document.createElement(\"div\"); d.className=\"agent-out\";\n      d.textContent=String(s.raw); built.log.appendChild(d); return;\n    }\n    agtStepEl(built.log, s.cmd, s.kind);\n    agtOutEl(built.log, s.out, s.err, (s.kind===\"fetch\"?0:s.exit));\n  });\n  const finalEl=document.createElement(\"div\");\n  finalEl.className=\"agent-final\";\n  finalEl.innerHTML=(finalText&&String(finalText).trim())\n    ? renderMarkdown(String(finalText))\n    : '<p style=\"color:var(--ink-3)\"><em>(mission sans texte final)</em></p>';\n  bubble.appendChild(finalEl);\n  return built;\n}\n\n/* Au chargement, si une mission tournait quand l'app a été fermée, on s'y\n   raccroche (quelques essais le temps que la connexion soit établie). */\n(function agtJobsTick(n){\n  setTimeout(()=>{\n    try{\n      if(typeof agtLogged===\"function\"&&agtLogged()){\n        /* Plus de reprise « dans le vide » : une mission vivante s'affiche\n           quand on ouvre SA discussion (elle est marquée « en cours » dans la\n           liste). Ici on ne fait que rafraîchir ces marqueurs. */\n        agtJobsRefresh();\n        setInterval(()=>{ try{ agtJobsRefresh(); }catch(e){} }, 45000);\n        return;\n      }\n    }catch(e){ /* pas encore connecté */ }\n    if(n<7) agtJobsTick(n+1);\n  }, 2000);\n})(0);\n\n/* --- boucle d'agent : JSON ⇄ bash --- */\n/* Lecture d'un appel d'outil. Un modèle renvoie presque toujours du JSON, mais\n   pas toujours propre : retours à la ligne bruts dans la chaîne (interdit en\n   JSON), guillemets internes non échappés, réponse coupée en pleine commande.\n   Sans réparation, ce JSON s'affichait comme un message au lieu d'être exécuté\n   (mêmes correctifs que dans la passerelle). */\nfunction agtJsonRepair(t){\n  let out=\"\", str=false, esc=false;\n  for(let i=0;i<t.length;i++){\n    const c=t[i];\n    if(esc){ out+=c; esc=false; continue; }\n    if(c===\"\\\\\"){ out+=c; esc=true; continue; }\n    if(c==='\"'){ str=!str; out+=c; continue; }\n    if(str&&(c===\"\\n\"||c===\"\\r\"||c===\"\\t\")){ out+= c===\"\\n\"?\"\\\\n\":(c===\"\\r\"?\"\\\\r\":\"\\\\t\"); continue; }\n    out+=c;\n  }\n  return out;\n}\nfunction agtJsonBody(body){\n  let out=\"\";\n  for(let i=0;i<body.length;i++){\n    const c=body[i];\n    if(c!==\"\\\\\"){ out+=c; continue; }\n    const n=body[++i];\n    if(n===\"n\") out+=\"\\n\";\n    else if(n===\"r\") out+=\"\\r\";\n    else if(n===\"t\") out+=\"\\t\";\n    else if(n===\"u\"){ out+=String.fromCharCode(parseInt(body.slice(i+1,i+5),16)||32); i+=4; }\n    else out+= (n===undefined?\"\\\\\":n);\n  }\n  return out;\n}\nfunction agtLooseCall(s){\n  const am=/\"action\"\\s*:\\s*\"?([a-zA-Z]+)\"?/.exec(s);\n  if(!am) return null;\n  const action=am[1].toLowerCase();\n  const key = action===\"fetch\" ? \"url\" : (action===\"final\" ? \"message\" : \"command\");\n  const km=new RegExp('\"'+key+'\"\\\\s*:\\\\s*\"').exec(s);\n  const o={action:action};\n  if(km){\n    let i=km.index+km[0].length, body=\"\", esc=false;\n    for(;i<s.length;i++){\n      const c=s[i];\n      if(esc){ body+=\"\\\\\"+c; esc=false; continue; }\n      if(c===\"\\\\\"){ esc=true; continue; }\n      if(c==='\"'){\n        const rest=s.slice(i+1).replace(/^\\s+/,\"\");\n        if(rest===\"\"||rest[0]===\",\"||rest[0]===\"}\") break;\n        body+='\"'; continue;\n      }\n      body+=c;\n    }\n    o[key]=agtJsonBody(body);\n    if(!String(o[key]).trim()) return null;\n  }else if(action!==\"final\") return null;\n  const mm=/\"mode\"\\s*:\\s*\"([a-zA-Z]+)\"/.exec(s);\n  if(mm) o.mode=mm[1];\n  return o;\n}\nfunction agtParseCall(s){\n  s=String(s||\"\");\n  const cand=[];\n  const fences=s.match(/```[a-zA-Z]*\\s*([\\s\\S]*?)```/g);\n  if(fences) fences.forEach(f=>cand.push(f.replace(/^```[a-zA-Z]*/,\"\").replace(/```$/,\"\").trim()));\n  cand.push(s.trim());\n  const i=s.indexOf(\"{\");\n  if(i>=0){\n    const k=s.lastIndexOf(\"}\");\n    if(k>i) cand.push(s.slice(i,k+1));\n  }\n  for(const c of cand){\n    for(const v of [c, agtJsonRepair(c)]){\n      try{\n        const o=JSON.parse(v);\n        if(o&&typeof o===\"object\"&&(o.action||o.command)){\n          if(!o.action&&o.command) o.action=\"bash\";\n          return o;\n        }\n      }catch(e){}\n    }\n  }\n  return agtLooseCall(s);\n}\nfunction agtNearBottom(){\n  const m=document.querySelector(\"main\");\n  return m.scrollHeight-m.scrollTop-m.clientHeight<160;\n}\nfunction agtSmart(){ if(agtNearBottom()) scrollDown(); }\n\nasync function agentRun(text){\n  if(AGENT_BUSY) return;\n  if(!agtLogged()){ agtToast(\"Veuillez vous connecter pour utiliser le mode agent.\"); return; }\n  AGENT_BUSY=true; busy=true; sendBtn.disabled=true;\n  let bubble=null;\n  try{\n    const shot=(typeof pendingImage!==\"undefined\"&&pendingImage&&pendingImage.b64)?pendingImage:null;\n    addUser(text,\"none\",null,null);\n    bubble=addAssistant();\n    promptEl.value=\"\"; promptEl.style.height=\"auto\"; scrollDown();\n    if(shot){\n      try{ pendingFiles=[]; pendingAudio=null; syncMirrors(); renderPending(); }catch(e){}\n    }\n    const finalEl=document.createElement(\"div\");\n    finalEl.className=\"agent-final\";\n    bubble.appendChild(finalEl);          /* invisible tant que vide */\n    let term=null, logEl=null;\n    const killWait=()=>{ const g=bubble.querySelector(\".gentxt\"); if(g) g.remove(); };\n    const ensureTerm=()=>{\n      if(!term){\n        killWait();\n        const built=agtTermEl(bubble, finalEl, false);\n        term=built.term; logEl=built.log;\n      }\n      return logEl;\n    };\n\n    /* Les deux fichiers de la VM sont récupérés UNE fois et envoyés dans le\n       premier message ; l'IA peut les relire elle-même à tout moment. */\n    if(AGENT_MD===null){\n      try{\n        const r=await fetch(API_BASE+\"/api/agent/file?path=\"+encodeURIComponent(AGENT_USER+\"/agent.md\"),{headers:agtHeaders()});\n        const j=await r.json();\n        AGENT_MD=j.ok?String(j.content||\"\"):\"\";\n      }catch(e){ AGENT_MD=\"\"; }\n    }\n    if(AGENT_INFOS===null){\n      try{\n        const r=await fetch(API_BASE+\"/api/agent/file?path=\"+encodeURIComponent(AGENT_USER+\"/infos.md\"),{headers:agtHeaders()});\n        const j=await r.json();\n        AGENT_INFOS=j.ok?String(j.content||\"\"):\"\";\n      }catch(e){ AGENT_INFOS=\"\"; }\n    }\n\n    const intro=[\n      \"Tu es un agent IA capable de piloter une VM Ubuntu 22.04 (Docker, bureau KASM) via son terminal.\",\n      \"\",\n      \"Voici le fichier agent.md contenant toutes tes instructions système :\",\n      AGENT_MD||\"(agent.md indisponible)\",\n      \"\",\n      \"Et voici le contenu du fichier infos.md, qui détaille ce que l'utilisateur peut faire et voit sur son interface :\",\n      AGENT_INFOS||\"(infos.md indisponible)\",\n      \"\",\n      \"Tu peux relire ces fichiers à tout moment (cat /home/kasm-user/user/agent.md, cat /home/kasm-user/user/infos.md) pour aider l'utilisateur ou le comprendre.\",\n      \"Outre bash, tu disposes de l'outil fetch pour lire une page web (Chrome headless sur la VM) : {\\\"action\\\":\\\"fetch\\\",\\\"url\\\":\\\"https://…\\\",\\\"mode\\\":\\\"html|text\\\"} — voir agent.md.\",\n      shot?\"Une capture d'écran actuelle de la VM est jointe à ce message.\":\"\"\n    ].join(\"\\n\");\n\n    const rappel='Tu es l\\'agent VM. Réponds par un seul objet JSON : {\"action\":\"bash\",\"command\":...}, '\n      +'{\"action\":\"fetch\",\"url\":\"https://…\",\"mode\":\"html\"|\"text\"} ou {\"action\":\"final\",\"message\":...}.';\n\n    /* Mémoire de la conversation : les échanges précédents sont transmis\n       à l'agent (borné pour rester léger). */\n    const hist=(typeof conv!==\"undefined\"&&conv.length)\n      ? conv.slice(-12).map(m=>\"[\"+(m.role===\"user\"?\"utilisateur\":\"assistant\")+\"] \"\n          +String(m.text||\"\")).join(\"\\n\")\n      : \"\";\n\n    let tr=\"\", steps=0, done=false, lastRaw=\"\", traceMd=\"\", finalMd=\"\";\n    const stepsArr=[];                    /* étapes structurées (Firestore) */\n\n    /* --- Voie principale : la mission est exécutée par la passerelle, qui\n       continue même si l'app est fermée. L'app ne fait que suivre l'avancée. */\n    let job=null, detached=false;\n    let jobChat=(typeof chatId!==\"undefined\")?chatId:null;\n    try{\n      /* La discussion est créée AVANT la mission, et sans course : la mission\n         lui est rattachée pour de bon. Son résultat ira donc chez elle — même\n         si l'on change de discussion, même si l'on ferme l'application. */\n      if(typeof FB!==\"undefined\"&&FB_OK&&fbUser&&!jobChat&&typeof ensureChat===\"function\"){\n        try{ jobChat=await ensureChat(text); }catch(e){ jobChat=null; }\n      }\n      const sr=await fetch(API_BASE+\"/api/agent/job/start\",{\n        method:\"POST\",\n        headers:agtHeaders({\"Content-Type\":\"application/json\"}),\n        body:JSON.stringify({model:current.id, mission:text, intro:intro, rappel:rappel, hist:hist,\n          chat:jobChat||\"\", uid:(typeof fbUser!==\"undefined\"&&fbUser)?fbUser.uid:\"\",\n          image:shot?(shot.small||shot.b64):WHITE_PX, image_kind:shot?\"upload\":\"none\",\n          params:{temperature:0.2}})\n      });\n      const sj=await sr.json();\n      if(sj&&sj.job) job={id:String(sj.job), chat:jobChat, deliver:!!sj.deliver,\n        model:current.id, mission:text, at:Date.now(), durable:!!sj.durable};\n    }catch(e){ /* passerelle sans missions : la boucle locale prend le relais */ }\n    /* La mission est notée dans SA discussion (marqueur « en cours », visible\n       de tout appareil) et dans le registre de cette page. */\n    if(job&&job.chat){ agtJobPut(job); try{ await agtSetJobMark(job.chat, job, \"run\"); }catch(e){} }\n\n    if(job){\n      AGT_JOB_OWN=job.id; AGT_JOB_CHAT=job.chat||null;\n      const follow=await agtJobFollow(job.id,{\n        ensureTerm:ensureTerm,\n        onStep:(st,L)=>{\n          const kind=(st.kind===\"fetch\")?\"fetch\":\"bash\";\n          agtStepEl(L,String(st.cmd||\"\"),kind);\n          agtOutEl(L,String(st.out||\"\"),String(st.err||\"\"),kind===\"fetch\"?0:(st.exit||0));\n          stepsArr.push({kind:kind, cmd:String(st.cmd||\"\"), out:String(st.out||\"\"),\n            err:String(st.err||\"\"), exit:st.exit||0});\n          tr+=\"\\n[commande] \"+String(st.cmd||\"\")\n            +\"\\n[exit] \"+(st.exit==null?\"?\":st.exit)\n            +\"\\n[stdout]\\n\"+String(st.out||\"\")+(st.err?\"\\n[stderr]\\n\"+st.err:\"\")+\"\\n\";\n          traceMd+=((kind===\"fetch\")?String(st.cmd||\"\"):\"$ \"+String(st.cmd||\"\"))+\"\\n\"\n            +(String(st.out||\"\")||String(st.err||\"\")||\"(aucune sortie)\")\n            +(st.exit?\"  [exit \"+st.exit+\"]\":\"\")+\"\\n\\n\";\n          agtSmart();\n        },\n        afterPoll:agtSmart,\n        detached:()=>AGT_DETACH\n      });\n      AGT_JOB_OWN=null; AGT_JOB_CHAT=null;\n      if(follow.detached){\n        /* Lâchée : elle continue sur la passerelle, sa discussion porte le\n           marqueur « en cours », et l'on peut faire autre chose. */\n        detached=true;\n        if(term) term.classList.add(\"folded\");\n        killWait();\n        agtSmart();\n        agtToast(\"Mission laissée en arrière-plan — son résultat arrivera dans sa discussion.\");\n      }else if(follow.done){\n        killWait();\n        finalMd=String(follow.final||\"\");\n        if(finalMd) finalEl.innerHTML=renderMarkdown(finalMd);\n        if(follow.error){\n          traceMd+=\"(\"+follow.error+\")\\n\";\n          if(!finalMd) finalEl.innerHTML=renderMarkdown(\"*(\"+follow.error+\")*\");\n        }\n        done=true;\n      }\n      /* follow.lost : la passerelle ne connaît plus la mission (isolate\n         recyclé) → la boucle locale ci-dessous reprend la mission. */\n    }\n\n    while(!detached&&steps<1000&&!done){   /* filet de sécurité très large, pas une limite pratique */\n      const prompt=steps===0\n        ? intro\n          +(hist?\"\\n\\n## Conversation précédente (contexte)\\n\"+hist:\"\")\n          +\"\\n\\nVoici le premier message de l'utilisateur :\\n\"+text\n        : rappel\n          +(hist?\"\\n\\n## Conversation précédente (contexte)\\n\"+hist:\"\")\n          +\"\\n\\n## Mission\\n\"+text\n          +(tr?\"\\n\\n## Historique des commandes\\n\"+tr:\"\")\n          +\"\\n\\nRéponds par un seul objet JSON.\";\n      const r=await fetch(API_BASE+\"/api/chat\",{\n        method:\"POST\",\n        headers:agtHeaders({\"Content-Type\":\"application/json\"}),\n        body:JSON.stringify({model:current.id,prompt,image:shot?(shot.small||shot.b64):WHITE_PX,image_kind:shot?\"upload\":\"none\",params:{temperature:0.2}})\n      });\n      let j={}; try{ j=await r.json(); }catch(e){}\n      if(!r.ok) throw new Error(j.error||(\"HTTP \"+r.status));\n      const out=String(j.output==null?\"\":j.output);\n      lastRaw=out;\n      const call=agtParseCall(out);\n      if(!call){ killWait(); finalMd=out; finalEl.innerHTML=renderMarkdown(out); done=true; break; }\n      if(call.action===\"final\"){ killWait(); finalMd=String(call.message||\"\"); finalEl.innerHTML=renderMarkdown(finalMd); done=true; break; }\n      if(call.action===\"bash\"){\n        const L=ensureTerm();\n        const cmd=String(call.command||\"\");\n        agtStepEl(L,cmd,\"bash\");\n        agtSmart();\n        let res={};\n        try{\n          const er=await fetch(API_BASE+\"/api/agent/exec\",{\n            method:\"POST\",\n            headers:agtHeaders({\"Content-Type\":\"application/json\"}),\n            body:JSON.stringify({command:cmd})\n          });\n          res=await er.json();\n        }catch(e){ res={ok:false,stderr:String(e&&e.message||e)}; }\n        const so=String(res.stdout||\"\");       /* sortie COMPLÈTE, telle quelle */\n        const se=String(res.stderr||\"\");\n        agtOutEl(L,so,se,res.exit_code);\n        stepsArr.push({kind:\"bash\", cmd:cmd, out:so, err:se, exit:res.exit_code||0});\n        tr+=\"\\n[commande] \"+cmd\n          +\"\\n[exit] \"+(res.exit_code==null?\"?\":res.exit_code)\n          +\"\\n[stdout]\\n\"+so+(se?\"\\n[stderr]\\n\"+se:\"\")+\"\\n\";\n        traceMd+=\"$ \"+cmd+\"\\n\"+(so||se||\"(aucune sortie)\")\n          +(res.exit_code?\"  [exit \"+res.exit_code+\"]\":\"\")+\"\\n\\n\";\n        agtSmart();\n      }else if(call.action===\"fetch\"){\n        /* Outil 2 : lecture web par Chrome headless sur la VM. */\n        const L=ensureTerm();\n        const url=String(call.url||\"\").trim();\n        const fmode=(String(call.mode||\"text\").toLowerCase()===\"html\")?\"html\":\"text\";\n        const label=\"fetch [\"+fmode+\"] \"+url;\n        agtStepEl(L,label,\"fetch\");\n        agtSmart();\n        let fj={};\n        try{\n          const fbody={url:url, mode:fmode};\n          if(call.wait_ms!=null) fbody.wait_ms=Number(call.wait_ms)||0;\n          if(call.timeout_ms!=null) fbody.timeout=Math.round((Number(call.timeout_ms)||45000)/1000);\n          if(call.selector) fbody.selector=String(call.selector);\n          const fr=await fetch(API_BASE+\"/api/agent/fetch\",{\n            method:\"POST\",\n            headers:agtHeaders({\"Content-Type\":\"application/json\"}),\n            body:JSON.stringify(fbody)\n          });\n          fj=await fr.json();\n        }catch(e){ fj={ok:false,error:String(e&&e.message||e)}; }\n        const head=(fj.ok?\"HTTP \"+(fj.status==null?\"?\":fj.status):\"ÉCHEC\")\n          +(fj.title?\" · \"+fj.title:\"\")\n          +(fj.chars?\" · \"+fj.chars+\" caractères\":\"\")\n          +(fj.ms?\" · \"+fj.ms+\" ms\":\"\")\n          +(fj.path?\"\\n(copie enregistrée : \"+fj.path+\")\":\"\");\n        const full=String(fj.content==null?\"\":fj.content);\n        const shown=fj.ok?(head+(full?\"\\n\\n\"+full:\"\")):(String(fj.error||head));\n        agtOutEl(L,shown,\"\",fj.ok?0:1);\n        stepsArr.push({kind:\"fetch\", cmd:label, out:shown, err:fj.ok?\"\":String(fj.error||\"\"), exit:fj.ok?0:1});\n        tr+=\"\\n[fetch \"+fmode+\"] \"+url\n          +\"\\n[statut] \"+(fj.status==null?\"?\":fj.status)+\" · \"+(fj.title||\"\")\n          +\"\\n[contenu complet]\\n\"+full+\"\\n\";\n        traceMd+=label+\"\\n\"+(full||String(fj.error||\"\"))+\"\\n\\n\";\n        agtSmart();\n      }else{\n        killWait(); finalEl.innerHTML=renderMarkdown(out); finalMd=out;\n        done=true;\n      }\n      steps++;\n    }\n    if(detached) return;      /* la passerelle finit et enregistre : rien à faire ici */\n    if(!done){\n      killWait();\n      finalMd=lastRaw||\"\";\n      finalEl.innerHTML='<p><em>(Mission interrompue : trop d\\'étapes d\\'affilée)</em></p>'+(lastRaw?renderMarkdown(lastRaw):\"\");\n    }\n    if(term) term.classList.add(\"folded\");      /* zone terminal repliée en fin de mission */\n    killWait();\n    /* On enregistre le markdown D'ORIGINE (pas le texte rendu) : au\n       rechargement, la réponse est re-rendue à l'identique, avec sa mise en\n       forme — c'est ce qui manquait (« les messages arrivés sans rien »). */\n    const finalText=(finalMd||\"\").trim() || String(finalEl.textContent||\"(mission agent)\");\n    /* Les étapes sont enregistrées EN CLAIR dans Firestore : en rouvrant la\n       discussion (3 h plus tard, sur un autre appareil) la zone terminal\n       repliable est reconstruite à l'identique. */\n    const savedText=agtTraceMd(stepsArr, finalText);\n    conv.push({role:\"user\",text});\n    conv.push({role:\"assistant\",text:finalText});\n    msgActions(bubble,{getMd:()=>savedText});   /* bouton copier uniquement */\n    /* Qui enregistre ? La passerelle, quand elle a la discussion : c'est ce\n       qui fait qu'une mission finie pendant que l'app est fermée est quand\n       même rangée au bon endroit. On vérifie qu'elle l'a fait ; sinon — et\n       seulement là — la page s'en charge. */\n    let savedHere=false;\n    if(job&&job.deliver&&job.chat){\n      try{ savedHere=await agtDelivered(job.chat); }catch(e){ savedHere=false; }\n    }\n    if(!savedHere&&typeof FB!==\"undefined\"&&FB.afterAgent){\n      try{ FB.afterAgent({userText:text, steps:stepsArr, out:finalText, chatId:(job&&job.chat)||null}); }catch(e){}\n    }\n    if(job&&job.chat){\n      agtJobDrop(job.chat);\n      try{ await agtSetJobMark(job.chat, job, \"done\"); }catch(e){}\n      if(job.chat!==chatId){\n        const c=(typeof chats!==\"undefined\")?chats.find(x=>x.id===job.chat):null;\n        agtToast(\"Mission terminée — le résultat est dans « \"+((c&&c.title)||\"sa discussion\")+\" ».\");\n      }\n    }\n  }catch(err){\n    const b=bubble||addAssistant();\n    const g=b.querySelector(\".gentxt\"); if(g) g.remove();\n    b.innerHTML='<div class=\"err\">'+esc(String(err&&err.message?err.message:err))+'</div>';\n  }finally{\n    busy=false; sendBtn.disabled=false; AGENT_BUSY=false; agtSmart();\n  }\n}\n</script>\n<script>\n/* =====================================================================\n   Phase 69 — Z-Canal : les canaux, les modes de discussion, les réglages\n   ---------------------------------------------------------------------\n   Un canal relie un agent (n'importe lequel) à l'application. Le canal par\n   défaut s'appelle « arena » : c'est la discussion « Arena », épinglée en\n   tête de liste et impossible à supprimer. Les autres canaux se créent ici\n   et se relient à UNE discussion, dont le mode (agent ou chat) est figé.\n   ===================================================================== */\n\n/* --- les dessins SVG (aucune image, tout est vectoriel) ---------------- */\nfunction zIcon(px, kind, color){\n  const c = color || \"#111113\", p = px || 16;\n  const head = '<svg width=\"' + p + '\" height=\"' + p + '\" viewBox=\"0 0 24 24\" fill=\"none\" stroke=\"'\n    + c + '\" stroke-width=\"1.8\" stroke-linecap=\"round\" stroke-linejoin=\"round\" aria-hidden=\"true\">';\n  if (kind === \"chat\") {\n    return head + '<path d=\"M20.5 12.4c0 4-3.8 7.2-8.5 7.2-1 0-2-.15-2.9-.43L4.2 20.6l1.3-3.5A6.9 6.9 0 0 1 3.5 12.4C3.5 8.4 7.3 5.2 12 5.2s8.5 3.2 8.5 7.2Z\"/><path d=\"M8.6 12.3h.01M12 12.3h.01M15.4 12.3h.01\"/></svg>';\n  }\n  if (kind === \"agent\") {\n    return head + '<rect x=\"3.2\" y=\"7.6\" width=\"17.6\" height=\"12.4\" rx=\"3\"/><path d=\"M12 3.4v4.2M8.6 12.6l-1.4 1.6 1.4 1.6M15.4 12.6l1.4 1.6-1.4 1.6\"/></svg>';\n  }\n  /* « Z-Canal » : un Z dont les deux bouts sont des nœuds de connexion. */\n  return head + '<path d=\"M6.4 7.2h11.2M17.6 7.2 6.4 16.8M6.4 16.8h11.2\"/><circle cx=\"3.4\" cy=\"7.2\" r=\"1.6\"/><circle cx=\"20.6\" cy=\"16.8\" r=\"1.6\"/></svg>';\n}\nconst ZCANAL_MARK = zIcon(13, \"canal\", \"#111113\");\ntry { MARKS.zcanal = ZCANAL_MARK; } catch (e) { /* MARKS absent */ }\nconst CHEVRON = '<svg width=\"15\" height=\"15\" viewBox=\"0 0 24 24\" fill=\"none\" stroke=\"currentColor\" stroke-width=\"1.8\" stroke-linecap=\"round\" stroke-linejoin=\"round\" aria-hidden=\"true\"><path d=\"m9 5 7 7-7 7\"/></svg>';\nconst COPYIC = '<svg width=\"15\" height=\"15\" viewBox=\"0 0 24 24\" fill=\"none\" stroke=\"currentColor\" stroke-width=\"1.7\" stroke-linecap=\"round\" stroke-linejoin=\"round\" aria-hidden=\"true\"><rect x=\"9\" y=\"9\" width=\"11\" height=\"11\" rx=\"2.4\"/><path d=\"M6.2 15H5.4A1.4 1.4 0 0 1 4 13.6V5.4A1.4 1.4 0 0 1 5.4 4h8.2A1.4 1.4 0 0 1 15 5.4v.8\"/></svg>';\nconst CHECKIC = '<svg width=\"15\" height=\"15\" viewBox=\"0 0 24 24\" fill=\"none\" stroke=\"currentColor\" stroke-width=\"2\" stroke-linecap=\"round\" stroke-linejoin=\"round\" aria-hidden=\"true\"><path d=\"m5 12.8 4.4 4.4L19 7.4\"/></svg>';\nconst DELIC = '<svg width=\"15\" height=\"15\" viewBox=\"0 0 24 24\" fill=\"none\" stroke=\"currentColor\" stroke-width=\"1.7\" stroke-linecap=\"round\" stroke-linejoin=\"round\" aria-hidden=\"true\"><path d=\"M4 7h16M9.5 7V5.2c0-.66.54-1.2 1.2-1.2h2.6c.66 0 1.2.54 1.2 1.2V7\"/><path d=\"M6.4 7l.8 12.1c.05.9.8 1.6 1.7 1.6h6.2c.9 0 1.65-.7 1.7-1.6L17.6 7\"/><path d=\"M10.2 11v6M13.8 11v6\"/></svg>';\nconst PLUSIC = '<svg width=\"18\" height=\"18\" viewBox=\"0 0 24 24\" fill=\"none\" stroke=\"currentColor\" stroke-width=\"1.9\" stroke-linecap=\"round\" aria-hidden=\"true\"><path d=\"M12 5.5v13M5.5 12h13\"/></svg>';\nconst CHEVIC = '<svg width=\"15\" height=\"15\" viewBox=\"0 0 24 24\" fill=\"none\" stroke=\"currentColor\" stroke-width=\"1.8\" stroke-linecap=\"round\" stroke-linejoin=\"round\" aria-hidden=\"true\"><path d=\"m9 5 7 7-7 7\"/></svg>';\n\n/* --- l'état : mes canaux, le canal de la discussion, le mode ---------- */\nlet CANAUX = [];                 /* [{id,name,mode,chat,createdAt}] (doc du compte) */\nlet CUR_CANAL = null;            /* canal de la discussion ouverte (null = aucune)  */\nlet CUR_MODE = \"chat\";           /* « chat » | « agent » | « canal »                */\nlet MODE_NOTICE = \"\";            /* 2e message de la zone noire (création de canal) */\nlet MODE_NOTICE_T = null;\nconst CANAL_DONE = new Set();    /* messages de canal déjà traités par cette page   */\nconst CANAL_UNSUBS = {};         /* canaux écoutés en temps réel                    */\nconst CHIP_REQ = {};             /* « non lu » → « lu ! » : pastilles vivantes      */\n\nfunction newCanalId(){\n  const S = \"abcdefghijklmnopqrstuvwxyz0123456789\", out = [];\n  try {\n    const a = new Uint8Array(12);\n    (window.crypto || {}).getRandomValues ? window.crypto.getRandomValues(a) : a.forEach(function(_, i){ a[i] = Math.floor(Math.random() * 256); });\n    for (let i = 0; i < 12; i++) out.push(S.charAt(a[i] % 36));\n  } catch (e) {\n    for (let i = 0; i < 12; i++) out.push(S.charAt(Math.floor(Math.random() * 36)));\n  }\n  return out.join(\"\");\n}\nfunction canalById(id){ for (const c of CANAUX) if (c && c.id === id) return c; return null; }\nfunction canalName(id){ if (id === \"arena\") return \"arena\"; const c = canalById(id); return (c && c.name) || id; }\nfunction canalModeOf(id){ if (id === \"arena\") return \"agent\"; const c = canalById(id); return (c && c.mode) || \"chat\"; }\n/* Le canal par défaut « arena » existe toujours et ne peut pas être supprimé. */\nfunction canalListAll(){\n  const out = (CANAUX || []).filter(c => c && c.id).slice();\n  const i = out.findIndex(c => c.id === \"arena\");\n  const def = { id: \"arena\", name: \"arena\", mode: \"agent\", pinned: true, arena: true };\n  if (i < 0) out.unshift(def);\n  else out[i] = Object.assign({}, def, out[i], { pinned: true, mode: \"agent\" });\n  out.sort((a, b) => (a.id === \"arena\" ? -1 : b.id === \"arena\" ? 1 : (a.createdAt || 0) - (b.createdAt || 0)));\n  return out;\n}\nfunction canalMsgsOf(id){\n  if (!FB_OK || !fbDB) return null;\n  if (id === \"arena\") return fbDB.collection(\"chan\").doc(\"main\").collection(\"box\");\n  return fbDB.collection(\"canaux\").doc(id).collection(\"messages\");\n}\n\n/* --- lire la liste des canaux du compte -------------------------------- */\nasync function canalLoad(){\n  if (!FB_OK || !fbUser) return;\n  try {\n    const d = await fbDB.collection(\"users\").doc(fbUser.uid).get();\n    const c = d.exists ? (d.data() || {}).canaux : null;\n    CANAUX = Array.isArray(c) ? c.filter(x => x && x.id) : [];\n  } catch (e) { console.warn(\"canaux :\", e); }\n  renderCanalList();\n}\nasync function canalSaveList(){\n  if (!FB_OK || !fbUser) return;\n  try {\n    await fbDB.collection(\"users\").doc(fbUser.uid)\n      .set({ canaux: CANAUX.filter(c => c.id !== \"arena\"), updatedAt: Date.now() }, { merge: true });\n  } catch (e) { console.warn(\"canaux (écriture) :\", e); }\n}\n/* --- créer un canal : enregistré sur le compte ------------------------- */\nasync function canalCreate(name){\n  if (!FB_OK || !fbUser) throw new Error(\"Connectez-vous pour créer un canal.\");\n  const id = newCanalId();\n  const clean = String(name || \"\").trim().slice(0, 40) || \"Mon canal\";\n  const now = Date.now();\n  await fbDB.collection(\"canaux\").doc(id).set({\n    id: id, name: clean, mode: null, chat: null, owner: fbUser.uid,\n    createdAt: now, updatedAt: now,\n  });\n  CANAUX.push({ id: id, name: clean, mode: null, chat: null, createdAt: now });\n  await canalSaveList();\n  renderCanalList();\n  return { id: id, name: clean };\n}\n/* --- figer le mode du canal + relier UNE discussion --------------------- */\nasync function canalFreeze(id, mode, chatId2){\n  const c = canalById(id);\n  try {\n    await fbDB.collection(\"canaux\").doc(id).set(\n      { mode: mode, chat: chatId2, updatedAt: Date.now() }, { merge: true });\n  } catch (e) { console.warn(\"canal (mode) :\", e); }\n  if (c) { c.mode = mode; c.chat = chatId2; }\n  else CANAUX.push({ id: id, name: id, mode: mode, chat: chatId2, createdAt: Date.now() });\n  await canalSaveList();\n  renderCanalList();\n}\n\n/* --- écrire dans un canal, au nom de l'utilisateur ---------------------- */\nasync function canalSendUser(id, text, kind){\n  if (id === \"arena\") {\n    const j = await arenaSend(text, kind || \"text\", \"\", \"\");\n    return Object.assign({ }, j || { ok: false, error: \"canal muet\" });\n  }\n  if (!FB_OK || !fbUser) return { ok: false, error: \"connectez-vous pour écrire dans un canal\" };\n  const ref = canalMsgsOf(id);\n  if (!ref) return { ok: false, error: \"canal indisponible\" };\n  const now = Date.now();\n  try {\n    const r = await ref.add({ role: \"user\", text: String(text || \"\"), ts: now,\n      kind: kind || \"text\", state: \"new\", src: \"app\", createdAt: now });\n    return { ok: true, id: r.id };\n  } catch (e) { return { ok: false, error: String((e && e.message) || e) }; }\n}\n/* --- renvoyer le résultat d'une commande dans le canal ------------------ */\nasync function canalSendResult(id, text, req){\n  if (id === \"arena\") return arenaSend(text, \"tool\", req || \"\", \"résultat de commande\");\n  const ref = canalMsgsOf(id); if (!ref) return;\n  const now = Date.now();\n  try {\n    await ref.add({ role: \"tool\", text: String(text || \"\"), ts: now, kind: \"tool\",\n      state: \"done\", src: \"app\", req: req || \"\", createdAt: now });\n  } catch (e) { console.warn(\"canal (résultat) :\", e); }\n}\nasync function canalMarkDone(id, docId){\n  if (id === \"arena\") return arenaMark(docId, \"done\");\n  const ref = canalMsgsOf(id); if (!ref) return;\n  try { await ref.doc(docId).set({ state: \"done\", doneAt: Date.now() }, { merge: true }); } catch (e) {}\n}\n/* --- pastilles « non lu » → « lu ! » ------------------------------------ */\nfunction chipMake(row, cid, reqId){\n  try {\n    let el = row.querySelector(\".luchip\");\n    if (!el) {\n      el = document.createElement(\"span\");\n      el.className = \"luchip\";\n      const b = row.querySelector(\".bubble\") || row;\n      b.appendChild(el);\n    }\n    el.dataset.req = reqId || \"\"; el.dataset.cid = cid || \"\";\n    if (reqId) CHIP_REQ[cid + \"/\" + reqId] = el;\n    return el;\n  } catch (e) { return null; }\n}\nfunction chipAttach(cid, reqId){\n  try {\n    const rows = [].slice.call(log.querySelectorAll(\".msg.user\"));\n    const row = rows[rows.length - 1];\n    if (!row || !reqId) return;\n    const el = chipMake(row, cid, reqId);\n    if (el) { el.textContent = \"non lu\"; el.classList.remove(\"lu\"); }\n  } catch (e) {}\n}\nfunction chipState(cid, reqId, state){\n  const el = CHIP_REQ[cid + \"/\" + reqId];\n  if (!el) return;\n  if (state && state !== \"new\") { el.textContent = \"lu !\"; el.classList.add(\"lu\"); }\n  else { el.textContent = \"non lu\"; el.classList.remove(\"lu\"); }\n}\n/* À l'ouverture d'une discussion : les pastilles des messages déjà envoyés. */\nasync function canalChipsFromHistory(chat){\n  if (!chat || !chat.canal || !FB_OK || !fbUser) return;\n  const cid = chat.canal;\n  try {\n    const snap = await chatsRef().doc(chat.id).collection(\"messages\").orderBy(\"order\").get();\n    const rows = [].slice.call(log.querySelectorAll(\".msg.user\"));\n    let n = 0;\n    for (const d of snap.docs) {\n      const m = d.data() || {};\n      if (m.role !== \"user\") continue;\n      const row = rows[n++]; if (!row) break;\n      if (!m.req) continue;\n      const el = chipMake(row, cid, m.req);\n      if (!el) continue;\n      el.textContent = \"non lu\"; el.classList.remove(\"lu\");\n      try {\n        const s = await canalMsgsOf(cid).doc(m.req).get();\n        const st = s.exists ? (s.data() || {}).state : \"\";\n        if (st && st !== \"new\") { el.textContent = \"lu !\"; el.classList.add(\"lu\"); }\n      } catch (e) {}\n    }\n  } catch (e) { console.warn(\"pastilles :\", e); }\n}\n/* --- le message d'agent d'un canal : un seul, qui grandit ---------------- */\nfunction canalMissionBubble(){\n  let m=CANAL_MISSION;\n  if(m && m.chatId===chatId && m.bubble.isConnected) return m;\n  let b=(ARENA_WAIT && ARENA_WAIT.isConnected)?ARENA_WAIT:addAssistant(\"arena\");\n  ARENA_WAIT=null;\n  const g=b.querySelector(\".gentxt\"); if(g) g.remove();\n  const built=agtTermEl(b, null, true);\n  m=CANAL_MISSION={chatId:chatId, bubble:b, log:built.log};\n  return m;\n}\nfunction canalMissionStep(step){\n  if(!step) return;\n  const m=canalMissionBubble();\n  agtStepEl(m.log, step.cmd, step.kind);\n  agtOutEl(m.log, step.out, step.err, (step.kind===\"fetch\"?0:step.exit));\n  scrollDown();\n}\n/* Le texte de l'agent termine le message : il s'ajoute SOUS la zone\n   terminale, dans la même bulle. */\nfunction canalMissionText(out){\n  const m=CANAL_MISSION;\n  let b=null;\n  if(m && m.chatId===chatId && m.bubble.isConnected) b=m.bubble;\n  else if(ARENA_WAIT && ARENA_WAIT.isConnected) b=ARENA_WAIT;\n  if(!b) b=addAssistant(\"arena\");\n  ARENA_WAIT=null;\n  const dansMission=!!(m && b===m.bubble);\n  CANAL_MISSION=null;\n  const g=b.querySelector(\".gentxt\"); if(g) g.remove();\n  if(dansMission){\n    const fin=document.createElement(\"div\"); fin.className=\"agent-final\";\n    fin.innerHTML=renderMarkdown(out);\n    b.appendChild(fin);\n    msgActions(b, {getMd:()=>out});\n  }else{\n    b.innerHTML=renderMarkdown(out);\n    msgActions(b, {getMd:()=>out});\n  }\n  conv.push({role:\"assistant\", text:out});\n  scrollDown();\n}\n/* « Attente de lecture… » : dès que le canal marque le message lu (état\n   différent de « new »), l'animation redevient celle de tous les jours. */\nfunction canalWatchRead(cid, mid, b){\n  let n=0;\n  (function tick(){\n    if(!b || !b.isConnected) return;\n    const ref=canalMsgsOf(cid);\n    if(!ref){ return; }\n    ref.doc(mid).get().then(function(s){\n      if(!b.isConnected) return;\n      const st=(s && s.exists) ? String((s.data()||{}).state || \"\") : \"\";\n      const g=b.querySelector(\".gentxt\");\n      if(st && st!==\"new\"){\n        if(g) g.innerHTML='Generating<i>.</i><i>.</i><i>.</i>';\n        return;\n      }\n      if(!g || ++n>240) return;\n      setTimeout(tick, 2500);\n    }).catch(function(){ if(++n<20) setTimeout(tick, 5000); });\n  })();\n}\n/* --- l'agent parle : commande exécutée, ou réponse affichée ------------- */\nasync function canalDeliver(cid, d, docId){\n  if (CANAL_DONE.has(docId)) return;\n  CANAL_DONE.add(docId);\n  /* un message arrive : l'historique gardé en mémoire n'est plus à jour */\n  try { delete CANAL_HIST[cid]; } catch (e) {}\n  try {\n    const duCanal = chatsOfCanal(cid);\n    const open = duCanal.some(x => x.id === chatId);\n    const parsed = arenaParse(d.text);\n    const mode = canalModeOf(cid);\n    if (parsed.kind === \"call\" && parsed.call.action !== \"final\" && mode === \"agent\") {\n      const res = await arenaExecCall(parsed.call);\n      /* Chaque commande s'ajoute au MÊME message, qui reste ouvert. */\n      if (open) canalMissionStep(res.step);\n      /* la commande et sa sortie sont déjà dans le canal (canalSendResult) */\n      await canalSendResult(cid, res.text, docId);\n      await canalMarkDone(cid, docId);\n      return;\n    }\n    let out = parsed.text || \"\";\n    if (!out && parsed.kind !== \"call\") out = String(d.text || \"\");\n    if (!out) { await canalMarkDone(cid, docId); return; }\n    /* Un texte termine le message : il s'ajoute dans la même bulle. */\n    if (open) canalMissionText(out);\n    /* La réponse reste dans le canal : la discussion la relira en s'ouvrant.\n       Seul le point « non lu » est tenu à jour, ici, tout de suite. */\n    for (const ch of duCanal) {\n      if (!open && !ch.dot) { ch.dot = true; renderChatList(); }\n      try { await chatsRef().doc(ch.id).update({ updatedAt: Date.now() }); } catch (e) {}\n    }\n    await canalMarkDone(cid, docId);\n  } catch (e) { console.warn(\"canal :\", e); }\n}\nfunction canalWatchOne(id){\n  if (id === \"arena\" || CANAL_UNSUBS[id] || !FB_OK || !fBDB_ok()) return;\n  const ref = canalMsgsOf(id); if (!ref) return;\n  try {\n    CANAL_UNSUBS[id] = ref.onSnapshot(function(snap){\n      snap.docChanges().forEach(function(ch){\n        const dm = ch.doc.data() || {};\n        if (ch.type === \"modified\") {\n          if (dm.role === \"user\") chipState(id, ch.doc.id, dm.state);\n          return;\n        }\n        if (ch.type !== \"added\") return;\n        if (dm.role !== \"agent\" || dm.state === \"done\") return;\n        canalDeliver(id, dm, ch.doc.id);\n      });\n    }, function(err){ console.warn(\"canal \" + id + \" :\", err && err.message); });\n  } catch (e) { console.warn(\"canal \" + id + \" :\", e); }\n}\nfunction fBDB_ok(){ return !!(FB_OK && fbDB); }\nfunction canalWatchAll(){\n  for (const c of (chats || [])) if (c && c.canal && c.canal !== \"arena\") canalWatchOne(c.canal);\n}\n\n/* --- envoyer un message dans le canal de la discussion ------------------ */\n/* Les discussions d'un même canal sont identiques : tout ce qui lui arrive\n   s'enregistre dans CHACUNE d'elles (et la discussion ouverte reste prioritaire\n   pour l'affichage). Avant, on prenait « la première discussion du canal », ce\n   qui rangeait les messages dans une autre quand deux discussions partageaient\n   le canal. */\nfunction chatsOfCanal(cid){\n  return (chats || []).filter(x => x && x.canal === cid);\n}\nasync function submitCanal(text){\n  const cid = CUR_CANAL;\n  if (!cid) return;\n  busy = true; sendBtn.disabled = true;\n  try {\n    const duCanal = chatsOfCanal(cid);\n    /* la discussion ouverte d'abord ; sinon la première du canal */\n    const chat = duCanal.find(x => x.id === chatId) || duCanal[0] || null;\n    /* les pièces jointes n'ont pas de place dans le canal : leurs notes\n       (nom du fichier, contenu texte) voyagent avec le message */\n    const notes = (typeof collectFileNotes === \"function\") ? collectFileNotes() : \"\";\n    const envoi = String(text || \"\") + notes;\n    /* Un nouveau message ferme la mission précédente : le message d'agent en\n       cours s'arrête ici, comme en mode agent classique. */\n    CANAL_MISSION = null;\n    addUser(text, \"none\", null, null);\n    const b = addAssistant(\"arena\");\n    ARENA_WAIT = b;\n    promptEl.value = \"\"; promptEl.style.height = \"auto\"; scrollDown();\n    b.innerHTML = '<span class=\"gentxt\">Envoi<i>.</i><i>.</i><i>.</i></span>';\n    conv.push({ role: \"user\", text: text });\n    if (typeof pendingFiles !== \"undefined\") pendingFiles = [];\n    const j = await canalSendUser(cid, envoi, \"text\");\n    if (j && j.id) chipAttach(cid, j.id);\n    if (j && j.id) {\n      try { delete CANAL_HIST[cid]; } catch (e) {}\n      /* Rien à recopier : le message vit DANS le canal, et la discussion le\n         relira de là à sa prochaine ouverture (c'est le canal qui fait foi). */\n      for (const ch of duCanal) {\n        try { await chatsRef().doc(ch.id).update({ updatedAt: Date.now() }); } catch (e) {}\n      }\n    }\n    if (!j || !j.ok) {\n      b.innerHTML = '<div class=\"err\">Canal injoignable : ' + esc((j && j.error) || \"raison inconnue\") + '</div>';\n    } else {\n      /* Pas de phrase de confirmation : l'animation dit tout. « Envoi… »,\n         puis « Attente de lecture… », puis l'animation classique dès que\n         l'agent a lu le message. */\n      b.innerHTML = '<span class=\"gentxt\">Attente de lecture<i>.</i><i>.</i><i>.</i></span>';\n      canalWatchRead(cid, j.id, b);\n    }\n    if (chat) renderChatList();\n  } catch (err) {\n    try {\n      const b2 = addAssistant(\"arena\");\n      b2.innerHTML = '<div class=\"err\">Canal : ' + esc(String((err && err.message) || err)) + '</div>';\n    } catch (e) {}\n  } finally {\n    busy = false; sendBtn.disabled = !promptEl.value.trim(); scrollDown();\n  }\n}\n\n/* ==================== la zone noire : le mode de la discussion ========== */\nfunction modeBadgeInfo(){\n  const c = (chats || []).find(x => x.id === chatId) || null;\n  /* Le logo de gauche est toujours celui de Z-Canal : c'est lui qui dit d'où\n     vient la discussion. Il est dessiné en BLANC — la version des réglages est\n     noire, et du noir sur le badge noir ne se voyait pas (juste du noir). */\n  const logo = zIcon(12, \"canal\", \"#fff\");\n  if (c && c.arena) return { label: \"Arena\", icon: logo };\n  if (CUR_MODE === \"canal\") return { label: \"Z-Canal\", icon: logo };\n  if (CUR_MODE === \"agent\") return { label: \"Agent\", icon: zIcon(12, \"agent\", \"#fff\") };\n  return { label: \"Chat\", icon: zIcon(12, \"chat\", \"#fff\") };\n}\nfunction renderModeBadge(){\n  const b = document.getElementById(\"agentbtn\");\n  if (!b) return;\n  const i = modeBadgeInfo();\n  /* Une seule ligne, courte : le logo puis le nom. Rien d'autre. */\n  b.innerHTML = '<span class=\"mbmode\">' + i.icon + '<span>' + esc(i.label) + '</span></span>';\n  b.classList.add(\"mode\");\n  b.title = i.label + \" — le mode se choisit à la création de la discussion.\";\n}\n/* L'annonce (création d'un canal, id, mode) ne grossit plus le badge : elle\n   s'affiche trois secondes dans le toast, en bas, et disparaît toute seule. */\nfunction setModeNotice(txt){\n  MODE_NOTICE = \"\";\n  renderModeBadge();\n  if (typeof agtToast === \"function\" && txt) agtToast(String(txt));\n}\n\n/* ==================== nouvelle discussion : le choix du mode ============ */\nfunction newPopEl(){ return document.getElementById(\"newpop\"); }\nfunction newPopView(view){\n  const p = newPopEl(); if (!p) return;\n  [\"npmodes\", \"npcanalform\", \"npcanalnew\", \"npcanalmode\"].forEach(function(id){\n    const el = document.getElementById(id); if (el) el.hidden = (id !== view);\n  });\n}\nfunction openNewPop(){\n  if (busy) { alert(\"Attends la fin de la réponse en cours.\"); return; }\n  const p = newPopEl(); if (!p) return;\n  /* Une seule surface à la fois : la popup ferme les réglages, la connexion,\n     le menu d'une discussion et le tiroir des conversations. */\n  try {\n    const st = document.getElementById(\"settings\"); if (st) st.hidden = true;\n    const am = document.getElementById(\"authmodal\"); if (am) am.hidden = true;\n    closeChatPop(); closeSide();\n  } catch (e) {}\n  const err = document.getElementById(\"canalerr\"); if (err) err.hidden = true;\n  const inp = document.getElementById(\"canalname\"); if (inp) inp.value = \"\";\n  newPopView(\"npmodes\");\n  fillCanalPicker();\n  const note = document.getElementById(\"npconn\");\n  if (note) note.hidden = !!fbUser;\n  p.hidden = false;\n}\nfunction closeNewPop(){ const p = newPopEl(); if (p) p.hidden = true; }\n/* Choix « Chat » ou « Agent » : la discussion naît dans ce mode. */\nfunction pickMode(md){\n  if (md !== \"chat\" && (!(typeof fbUser !== \"undefined\") || !fbUser)) {\n    agtToast(\"Veuillez vous connecter pour ce mode.\");\n    return;\n  }\n  closeNewPop();\n  CUR_CANAL = null; CUR_MODE = md;\n  AGENT_ON = (md === \"agent\");\n  try { agtBtn.classList.toggle(\"on\", AGENT_ON); } catch (e) {}\n  newConversation(md);\n  if (typeof agtSyncUI === \"function\") agtSyncUI();\n  renderModeBadge();\n  if (md === \"agent\") agtToast(\"Discussion en mode agent — chaque message devient une mission dans la VM.\");\n}\n/* Choix « Z-Canal » : on crée le canal, puis on fige son mode. */\nfunction pickCanal(){\n  if (!(typeof fbUser !== \"undefined\") || !fbUser) { agtToast(\"Veuillez vous connecter pour créer un canal.\"); return; }\n  /* Le mode ne s'affiche dans la zone noire qu'une fois VRAIMENT choisi :\n     tant que le canal n'est pas relié, la discussion garde son mode. */\n  newPopView(\"npcanalform\");\n  fillCanalPicker();\n}\n/* La liste des canaux déjà créés — un par ligne, exactement comme dans les\n   réglages — et, dessous, une ligne identique « + Créer un canal ».\n   Un canal peut servir à plusieurs discussions : elles seront identiques. */\nfunction fillCanalPicker(){\n  const box = document.getElementById(\"canalpick\"); if (!box) return;\n  const list = canalListAll().filter(c => !c.arena);\n  let h = \"\";\n  for (const c of list) {\n    h += '<button type=\"button\" class=\"canrow\" data-pick=\"' + esc(c.id) + '\">'\n      + '<span class=\"cic\">' + zIcon(18, \"canal\", \"#111113\") + '</span>'\n      + '<span class=\"cmain\"><b class=\"cname\">' + esc(c.name || c.id) + '</b>'\n      + '<code class=\"cid\">' + esc(c.id) + '</code></span>'\n      + '<span class=\"sra\">' + CHEVIC + '</span>'\n      + '</button>';\n  }\n  h += '<button type=\"button\" class=\"canrow cannew\" id=\"canalnewbtn\">'\n    + '<span class=\"cic\">' + PLUSIC + '</span>'\n    + '<span class=\"cmain\"><b class=\"cname\">Créer un canal</b></span>'\n    + '<span class=\"sra\">' + CHEVIC + '</span>'\n    + '</button>';\n  box.innerHTML = h;\n  [].slice.call(box.querySelectorAll(\"[data-pick]\")).forEach(function(btn){\n    btn.onclick = function(){ startCanalChat(btn.getAttribute(\"data-pick\"), null); };\n  });\n  const nw = document.getElementById(\"canalnewbtn\");\n  if (nw) nw.onclick = function(){\n    const inp = document.getElementById(\"canalname\"); if (inp) inp.value = \"\";\n    const err = document.getElementById(\"canalerr\"); if (err) err.hidden = true;\n    newPopView(\"npcanalnew\");\n    if (inp) try { inp.focus(); } catch (e) {}\n  };\n}\nasync function canalCreateClick(){\n  const inp = document.getElementById(\"canalname\");\n  const err = document.getElementById(\"canalerr\");\n  const name = inp ? inp.value.trim() : \"\";\n  if (err) err.hidden = true;\n  if (busy) return;\n  busy = true;\n  try {\n    const c = await canalCreate(name);\n    /* 2e message de la zone noire : le canal est créé, voici son identifiant. */\n    setModeNotice(\"Canal créé — id \" + c.id, 60000);\n    newPopView(\"npcanalmode\");\n    const t = document.getElementById(\"npcanalmode\");\n    if (t) {\n      const q = t.querySelector(\".npd\");\n      if (q) q.innerHTML = 'Le canal « ' + esc(c.name) + ' » est créé avec l\\'id <code>' + esc(c.id)\n        + '</code>.<br>Cette discussion sera « agent » ou « chat » — et ce choix ne changera plus.';\n    }\n    const ba = document.getElementById(\"canalagent\");\n    const bc = document.getElementById(\"canalchat\");\n    if (ba) ba.onclick = function(){ startCanalChat(c.id, \"agent\"); };\n    if (bc) bc.onclick = function(){ startCanalChat(c.id, \"chat\"); };\n  } catch (e) {\n    if (err) { err.textContent = String((e && e.message) || e); err.hidden = false; }\n  } finally { busy = false; }\n}\n/* Relier la discussion au canal : le mode est FIGÉ, un canal = une discussion. */\nasync function startCanalChat(id, mode){\n  const c = canalById(id) || { id: id, name: id };\n  /* Plusieurs discussions peuvent partager un canal : elles sont identiques.\n     Le mode appartient au canal ; s'il est déjà figé, il s'applique tel quel. */\n  if (!mode && c.mode) mode = c.mode;\n  if (!mode) {   /* canal qui n'a jamais servi : on fige ici son mode */\n    const ba = document.getElementById(\"canalagent\"), bc = document.getElementById(\"canalchat\");\n    if (ba) ba.onclick = function(){ startCanalChat(id, \"agent\"); };\n    if (bc) bc.onclick = function(){ startCanalChat(id, \"chat\"); };\n    const q = document.querySelector(\"#npcanalmode .npd\");\n    if (q) q.innerHTML = 'Le canal « ' + esc(c.name || id) + ' » (id <code>' + esc(id)\n      + '</code>) attend sa discussion : elle sera « agent » ou « chat » — et ce choix ne changera plus.';\n    newPopView(\"npcanalmode\");\n    return;\n  }\n  closeNewPop();\n  CUR_CANAL = id; CUR_MODE = \"canal\";\n  AGENT_ON = (mode === \"agent\");\n  try { agtBtn.classList.toggle(\"on\", AGENT_ON); } catch (e) {}\n  const cid = await canalCreateChat(id, c.name || id, mode);\n  await canalFreeze(id, mode, cid);\n  canalWatchOne(id);\n  renderModeBadge();\n  if (typeof agtSyncUI === \"function\") agtSyncUI();\n  setModeNotice(\"Canal « \" + (c.name || id) + \" » — id \" + id, 60000);\n  agtToast(\"Discussion reliée au canal « \" + (c.name || id) + \" » (mode \" + mode + \").\");\n}\n/* La discussion d'un canal : créée tout de suite, titrée du nom du canal. */\nasync function canalCreateChat(id, name, mode){\n  CUR_MODE = \"canal\"; CUR_CANAL = id;\n  const mid = \"c\" + Date.now().toString(36) + Math.random().toString(36).slice(2, 7);\n  const now = Date.now();\n  /* La discussion d'un canal ne garde que son NOM et son MODE (plus le canal\n     auquel elle est reliée) : aucun message, aucune copie. Son contenu se\n     relit dans le canal. */\n  const doc = { title: String(name || id), mode: \"canal\", canal: id,\n    agent: (mode === \"agent\"), createdAt: now, updatedAt: now };\n  conv = []; log.innerHTML = \"\"; emptyState(); updateHint();\n  chatId = mid; msgSeq = 0;\n  let écrit = false;\n  for (let essai = 1; essai <= 2 && !écrit; essai++) {\n    try {\n      await chatsRef().doc(mid).set(doc);\n      écrit = true;\n    } catch (e) {\n      console.warn(\"canal (discussion) :\", e);\n      if (essai === 2) agtToast(\"La discussion n'a pas pu être enregistrée : \" + ((e && e.message) || e));\n    }\n  }\n  if (écrit) chats.unshift(Object.assign({ id: mid }, doc));\n  const am = MODELS.find(m => m.id === \"arena\"); if (am) selectModel(am);\n  renderChatList(); renderModeBadge(); closeSide();\n  return mid;\n}\n\n/* ==================== l'écran des réglages : Z-Canal ==================== */\n/* ---- Disponibilité : les secrets du relais, testés pour de vrai -------- */\nlet DISPO = { list: [], res: {}, busy: false };\nconst DIS_PLAY = '<svg viewBox=\"0 0 24 24\" fill=\"currentColor\" aria-hidden=\"true\"><path d=\"M8.6 5.4v13.2L19 12 8.6 5.4Z\"/></svg>';\nconst DIS_TICK = '<svg viewBox=\"0 0 24 24\" fill=\"none\" stroke=\"currentColor\" stroke-width=\"2.4\" stroke-linecap=\"round\" stroke-linejoin=\"round\" aria-hidden=\"true\"><path d=\"m5.2 12.6 4.2 4.2 9.4-9.6\"/></svg>';\nconst DIS_X    = '<svg viewBox=\"0 0 24 24\" fill=\"none\" stroke=\"currentColor\" stroke-width=\"2.4\" stroke-linecap=\"round\" aria-hidden=\"true\"><path d=\"M6.6 6.6l10.8 10.8M17.4 6.6 6.6 17.4\"/></svg>';\n\nfunction dispoLoad(){\n  const box = document.getElementById(\"dislist\");\n  const stamp = document.getElementById(\"distamp\");\n  if (!box) return;\n  fetch(API_BASE + \"/api/secrets\", { headers: agtHeaders(), cache: \"no-store\" })\n    .then(function (r) { return r.json(); })\n    .then(function (d) {\n      /* Les secrets viennent du worker : aucune liste n'est devinée ici. */\n      if (!d || !d.ok) throw new Error((d && d.error) || \"réponse illisible\");\n      DISPO.list = d.secrets || [];\n      dispoRender();\n      dispoCredits();\n      if (stamp) stamp.textContent = DISPO.list.length + \" secrets · \" + DISPO.list.filter(function (x) { return x.present; }).length + \" présents.\";\n    })\n    .catch(function (e) {\n      if (stamp) stamp.textContent = \"Lecture impossible : \" + String((e && e.message) || e);\n      box.innerHTML = '<p class=\"setnote\">La passerelle n’a pas répondu.</p>';\n    });\n}\n/* Un bouton par secret : ▶ au repos, un cercle qui tourne pendant l’essai,\n   puis une fiche verte ou une croix rouge. Cliquer sur le résultat ouvre\n   le détail, avec de quoi le copier. */\n/* Crédits qui restent : ModelScope (solde Magicubes, lu par son API) et\n   Cloudflare (neurones consommés aujourd'hui, comptés à chaque appel).\n   Aucun chiffre n'est inventé : quand une API ne répond pas, on le dit. */\nfunction dispoCredits(){\n  const box=document.getElementById(\"discredits\");\n  const stamp=document.getElementById(\"discredstamp\");\n  if(!box) return;\n  fetch(API_BASE+\"/api/credits?light=1\",{headers:agtHeaders(),cache:\"no-store\"})\n    .then(function(r){ return r.json(); })\n    .then(function(d){\n      if(!d||!d.ok) throw new Error((d&&d.error)||\"réponse illisible\");\n      const ms=d.modelscope||{}, cf=d.cloudflare||{};\n      const bal=ms.balance||null;\n      const dispo=(bal&&bal.available!=null)?bal.available:null;\n      const total=(bal&&bal.total!=null)?bal.total:null;\n      const nf=function(v){ return v==null?\"—\":(Math.round(Number(v)*100)/100).toLocaleString(\"fr-FR\"); };\n      let h='<div class=\"crgrid\">';\n      if(dispo!=null){\n        h+='<div class=\"cred\"><span>ModelScope</span><b>'+nf(dispo)+'</b><em>Magicubes disponibles'\n          +(total!=null?(' sur '+nf(total)):'')+'<br>≈ '+Math.floor(Number(dispo))+' image(s) Qwen 2.1</em></div>';\n      }else{\n        h+='<div class=\"cred ko\"><span>ModelScope</span><b>Indisponible</b><em>'+esc(String(ms.error||\"solde non lisible pour l'instant\"))+'</em></div>';\n      }\n      if(cf.mesure){\n        h+='<div class=\"cred\"><span>Cloudflare · neurones</span><b>'+nf(cf.restant)+'</b><em>restants sur '+nf(cf.gratuit)\n          +' aujourd’hui<br>'+nf(cf.neurones)+' consommés par l’app</em></div>';\n      }else{\n        h+='<div class=\"cred ko\"><span>Cloudflare · neurones</span><b>En attente</b><em>le compteur se remplit au premier appel d’image Workers AI</em></div>';\n      }\n      h+='</div>';\n      box.innerHTML=h;\n      if(stamp) stamp.textContent=\"ModelScope : solde et crédits du jour. Cloudflare : 10 000 neurones gratuits par jour, mesurés sur les appels de l’application.\";\n    })\n    .catch(function(e){\n      box.innerHTML='<p class=\"setnote\">Crédits illisibles : '+esc(String((e&&e.message)||e))+'</p>';\n      if(stamp) stamp.textContent=\"\";\n    });\n}\nfunction dispoBtn(html, cls, title){\n  return '<button type=\"button\" class=\"disbtn ' + cls + '\" title=\"' + esc(title) + '\">' + html + '</button>';\n}\nfunction dispoRender(){\n  const box = document.getElementById(\"dislist\");\n  if (!box) return;\n  const groups = {};\n  const order = [];\n  for (const x of DISPO.list) {\n    if (!groups[x.api]) { groups[x.api] = []; order.push(x.api); }\n    groups[x.api].push(x);\n  }\n  let h = \"\";\n  for (const api of order) {\n    h += '<h3 style=\"margin:18px 0 4px\">' + esc(api) + '</h3>';\n    for (const x of groups[api]) {\n      const r = DISPO.res[x.name];\n      let cls = \"\", inner = DIS_PLAY, title = \"Tester \" + x.name;\n      if (r && r.running) { cls = \"run\"; inner = '<span class=\"disspin\"></span>'; title = \"Test en cours…\"; }\n      else if (r && r.ok) { cls = \"ok\"; inner = DIS_TICK; title = \"Fonctionne — voir le détail\"; }\n      else if (r && r.ok === false) { cls = \"ko\"; inner = DIS_X; title = \"Échec — voir le détail\"; }\n      else if (x.present === false) { cls = \"run\"; inner = DIS_PLAY; title = \"Secret absent (essai possible, mais il dira pourquoi)\"; }\n      const sub = x.present ? (\"…\" + (x.fin || \"\") + (r && r.ms ? \" · \" + r.ms + \" ms\" : \"\")) : \"absent du worker\";\n      h += '<div class=\"disrow\" data-name=\"' + esc(x.name) + '\">'\n        + '<div class=\"dismain\"><b>' + esc(x.name) + '</b><em>' + esc(sub) + '</em></div>'\n        + '<button type=\"button\" class=\"disbtn ' + cls + '\" data-test=\"' + esc(x.name) + '\" title=\"' + esc(title) + '\">' + inner + '</button>'\n        + '</div>';\n    }\n  }\n  box.innerHTML = h || '<p class=\"setnote\">Aucun secret configuré.</p>';\n  [].slice.call(box.querySelectorAll(\"[data-test]\")).forEach(function (b) {\n    b.onclick = function () {\n      const name = b.getAttribute(\"data-test\");\n      const r = DISPO.res[name];\n      if (r && !r.running && r.ok !== undefined) dispoDetail(name, r);\n      else dispoTestOne(name);\n    };\n  });\n}\nfunction dispoMark(name, val){\n  DISPO.res[name] = val;\n  const b = document.querySelector('[data-test=\"' + name + '\"]');\n  if (!b) return;\n  if (val.running) { b.className = \"disbtn run\"; b.innerHTML = '<span class=\"disspin\"></span>'; return; }\n  b.className = \"disbtn \" + (val.ok ? \"ok\" : \"ko\");\n  b.innerHTML = val.ok ? DIS_TICK : DIS_X;\n  const row = b.closest(\".disrow\");\n  const em = row ? row.querySelector(\".dismain em\") : null;\n  const x = DISPO.list.filter(function (y) { return y.name === name; })[0] || {};\n  if (em) em.textContent = x.present === false ? \"absent du worker\"\n    : (val.ok ? \"fonctionne\" : \"en échec\") + (val.ms ? \" · \" + val.ms + \" ms\" : \"\")\n      + (x.present ? \" · …\" + (x.fin || \"\") : \"\");\n}\nasync function dispoTestOne(name){\n  dispoMark(name, { running: true });\n  try {\n    const r = await fetch(API_BASE + \"/api/secrets/test\", {\n      method: \"POST\", headers: agtHeaders({ \"Content-Type\": \"application/json\" }),\n      body: JSON.stringify({ name: name }),\n    });\n    const d = await r.json();\n    if (!d || !d.ok || !d.resultat) throw new Error((d && d.error) || (\"HTTP \" + r.status));\n    dispoMark(name, d.resultat);\n  } catch (e) {\n    dispoMark(name, { ok: false, ms: 0, status: 0, message: \"Test impossible : \" + String((e && e.message) || e) });\n  }\n}\nasync function dispoTestAll(){\n  if (DISPO.busy) return;\n  DISPO.busy = true;\n  const stamp = document.getElementById(\"distamp\");\n  const btn = document.getElementById(\"distestall\");\n  if (btn) { btn.disabled = true; btn.textContent = \"Tests en cours…\"; }\n  for (const x of DISPO.list) dispoMark(x.name, { running: true });\n  try {\n    const r = await fetch(API_BASE + \"/api/secrets/test-all\", { method: \"POST\", headers: agtHeaders() });\n    const d = await r.json();\n    if (!d || !d.ok) throw new Error((d && d.error) || (\"HTTP \" + r.status));\n    const list = d.resultats || [];\n    for (const res of list) dispoMark(res.name, res);\n    const ko = list.filter(function (x) { return !x.ok; }).length;\n    if (stamp) stamp.textContent = list.length + \" secrets testés à \" + new Date().toLocaleTimeString(\"fr-FR\")\n      + \" · \" + (list.length - ko) + \" fonctionnent\" + (ko ? \" · \" + ko + \" en échec\" : \"\") + \".\";\n  } catch (e) {\n    if (stamp) stamp.textContent = \"Tests impossibles : \" + String((e && e.message) || e);\n    for (const x of DISPO.list) if (!DISPO.res[x.name] || DISPO.res[x.name].running) dispoMark(x.name, { ok: false, message: \"Test impossible.\" });\n  }\n  DISPO.busy = false;\n  if (btn) { btn.disabled = false; btn.textContent = \"Tout tester\"; }\n}\nfunction dispoDetail(name, r){\n  const pop = document.getElementById(\"dispop\");\n  if (!pop) return;\n  const t = document.getElementById(\"dispo-title\"), m = document.getElementById(\"dispo-msg\");\n  if (t) t.textContent = name + (r.ok ? \" — fonctionne\" : \" — en échec\");\n  if (m) m.textContent = (r.message || \"Aucun message.\")\n    + (r.status ? \"\\n\\nHTTP \" + r.status : \"\")\n    + (r.ms ? \" · \" + r.ms + \" ms\" : \"\");\n  pop.hidden = false;\n  const c = document.getElementById(\"dispo-copy\");\n  if (c) c.onclick = function () {\n    const txt = name + \" : \" + (r.message || \"\") + (r.status ? \" (HTTP \" + r.status + \")\" : \"\");\n    try { navigator.clipboard.writeText(txt); agtToast(\"Copié.\"); } catch (e) { agtToast(\"Copie impossible.\"); }\n  };\n  const x = document.getElementById(\"dispo-close\");\n  if (x) x.onclick = function () { pop.hidden = true; };\n  pop.onclick = function (e) { if (e.target === pop) pop.hidden = true; };\n}\n\n/* ---- Paramètres ▸ Tâches : la liste des missions de la passerelle ------ */\nfunction taskAge(ts){\n  const s = Math.max(0, Math.round((Date.now() - Number(ts || 0)) / 1000));\n  if (!ts) return \"date inconnue\";\n  if (s < 60) return \"à l’instant\";\n  const m = Math.round(s / 60);\n  if (m < 60) return \"il y a \" + m + \" min\";\n  const h = Math.floor(m / 60), r = m % 60;\n  if (h < 24) return \"il y a \" + h + \" h\" + (r ? \" \" + r + \" min\" : \"\");\n  const d = Math.floor(h / 24);\n  return \"il y a \" + d + \" jour\" + (d > 1 ? \"s\" : \"\");\n}\nfunction taskDuration(j){\n  const a = Number(j.at || 0), u = Number(j.updatedAt || 0);\n  if (!a || !u || u < a) return \"\";\n  const s = Math.round((u - a) / 1000);\n  if (s < 60) return s + \" s de travail\";\n  const m = Math.round(s / 60);\n  return m < 60 ? m + \" min de travail\" : Math.floor(m / 60) + \" h \" + (m % 60) + \" min de travail\";\n}\nfunction taskRender(jobs){\n  const box = document.getElementById(\"tasklist\");\n  if (!box) return;\n  if (!jobs || !jobs.length) {\n    box.innerHTML = '<p class=\"setnote\">Aucune mission enregistrée. Les tâches lancées depuis cet appareil se déclarent au démarrage ; celles lancées depuis un autre appareil apparaissent dès qu’on les suit.</p>';\n    return;\n  }\n  let h = \"\";\n  for (const j of jobs) {\n    const run = j.state === \"run\" && !j.done;\n    const sub = String(j.mission || \"(mission sans texte)\");\n    const meta = (run ? '<span class=\"taskrun\"><i></i>en cours</span>' : \"terminée\")\n      + (j.step ? \" · étape \" + j.step : \"\")\n      + \" · \" + taskAge(j.at)\n      + (taskDuration(j) ? \" · \" + taskDuration(j) : \"\");\n    h += '<div class=\"taskrow\" data-job=\"' + esc(String(j.id)) + '\">'\n      + '<div class=\"taskmain\"><b>' + esc(String(j.model || \"modèle inconnu\")) + '</b>'\n      + '<span class=\"tsub\" title=\"' + esc(sub) + '\">' + esc(sub) + '</span>'\n      + '<em>' + meta + '</em></div>'\n      + (run ? '<button type=\"button\" class=\"taskstop\" data-stop=\"' + esc(String(j.id)) + '\">Arrêter</button>' : \"\")\n      + '</div>';\n  }\n  box.innerHTML = h;\n  [].slice.call(box.querySelectorAll(\"[data-stop]\")).forEach(function (b) {\n    b.onclick = function () { tasksStop(b.getAttribute(\"data-stop\")); };\n  });\n}\nasync function tasksLoad(){\n  const box = document.getElementById(\"tasklist\");\n  const stamp = document.getElementById(\"taskstamp\");\n  if (!box) return;\n  try {\n    const r = await fetch(API_BASE + \"/api/agent/jobs\", { headers: agtHeaders(), cache: \"no-store\" });\n    const d = await r.json();\n    if (!d || !d.ok) throw new Error((d && d.error) || (\"HTTP \" + r.status));\n    taskRender(d.jobs || []);\n    if (stamp) stamp.textContent = (d.actives || 0) + \" en cours · \" + (d.total || 0) + \" au total · lu à \"\n      + new Date().toLocaleTimeString(\"fr-FR\") + \".\";\n  } catch (e) {\n    if (stamp) stamp.textContent = \"Lecture impossible : \" + String((e && e.message) || e);\n    box.innerHTML = '<p class=\"setnote\">La passerelle n’a pas répondu.</p>';\n  }\n}\nasync function tasksStop(id){\n  if (!id) return;\n  if (!confirm(\"Arrêter cette tâche ? La mission s’arrête pour de bon.\")) return;\n  const b = document.querySelector('.taskstop[data-stop=\"' + id + '\"]');\n  if (b) { b.disabled = true; b.textContent = \"Arrêt…\"; }\n  try {\n    const r = await fetch(API_BASE + \"/api/agent/job/stop?id=\" + encodeURIComponent(id), { method: \"POST\", headers: agtHeaders() });\n    const j = await r.json().catch(function () { return {}; });\n    if (!j || !j.ok) throw new Error((j && j.error) || (\"HTTP \" + r.status));\n    agtJobDrop(AGT_JOB_CHAT || \"\");\n    agtToast(\"Tâche arrêtée.\");\n  } catch (e) {\n    agtToast(\"Arrêt impossible : \" + String((e && e.message) || e));\n  }\n  setTimeout(tasksLoad, 600);\n}\nvar SET_VIEW = \"list\";\nfunction showSetView(view){\n  const l = document.getElementById(\"setlist\"), a = document.getElementById(\"setcompte\"),\n        c = document.getElementById(\"setcanal\"), r = document.getElementById(\"setdispo\"),\n        md = document.getElementById(\"setmodels\"), w = document.getElementById(\"setwan\"),\n        tk = document.getElementById(\"settasks\");\n  SET_VIEW = view;\n  if (l) l.hidden = (view !== \"list\");\n  if (a) a.hidden = (view !== \"compte\");\n  if (c) c.hidden = (view !== \"canal\");\n  if (r) r.hidden = (view !== \"dispo\");\n  if (md) md.hidden = (view !== \"models\");\n  if (w) w.hidden = (view !== \"wan\");\n  if (tk) tk.hidden = (view !== \"tasks\");\n  if (view === \"wan\") renderWanModels();\n  if (view === \"tasks\") tasksLoad();\n  if (view === \"dispo\") dispoLoad();\n  /* Le bouton « revenir » est un frère de la croix, dans le panneau : les deux\n     sont donc posés au même endroit, à la même hauteur, quoi qu'il arrive. */\n  const b = document.getElementById(\"setback\");\n  if (b) b.hidden = (view === \"list\");\n  if (view === \"canal\") renderCanalList();\n}\nfunction renderCanalList(){\n  const box = document.getElementById(\"canallist\");\n  if (!box) return;\n  const list = canalListAll();\n  const base = API_BASE + \"/canal\";\n  let h = \"\";\n  for (const c of list) {\n    /* Un canal par ligne, en grand : son logo, son nom, son identifiant\n       COMPLET, et le bouton « copier ». Le trait fin qui sépare deux lignes\n       ne prend pas toute la largeur (voir .canrow + .canrow::before). */\n    const md = c.arena ? \"canal par défaut · mode agent\" : (c.mode ? (\"mode \" + c.mode) : \"sans discussion\");\n    const icon = (c.arena && typeof MARKS !== \"undefined\" && MARKS.arena)\n      ? MARKS.arena : zIcon(18, \"canal\", \"#111113\");\n    h += '<div class=\"canrow\" title=\"' + esc((c.name || c.id) + \" — \" + md) + '\">'\n      + '<span class=\"cic\">' + icon + '</span>'\n      + '<span class=\"cmain\"><b class=\"cname\">' + esc(c.name || c.id) + '</b>'\n      + '<code class=\"cid\">' + esc(c.id) + '</code></span>'\n      + '<button type=\"button\" class=\"copybtn\" data-copy=\"' + esc(c.id) + '\" title=\"Copier l\\'identifiant\" aria-label=\"Copier l\\'identifiant du canal\">' + COPYIC + '</button>'\n      + (c.arena ? '' : '<button type=\"button\" class=\"delbtn\" data-del=\"' + esc(c.id) + '\" title=\"Supprimer ce canal\" aria-label=\"Supprimer le canal ' + esc(c.name || c.id) + '\">' + DELIC + '</button>')\n      + '</div>';\n  }\n  if (!list.length) h = '<p class=\"setnote\">Aucun canal pour l\\'instant. Créez-en un avec « Nouvelle discussion » → « Z-Canal ».</p>';\n  box.innerHTML = h;\n  wireCanalCopies(box);\n\n  const n = document.getElementById(\"setcanalcount\");\n  if (n) n.textContent = String(list.length) + (list.length > 1 ? \" canaux\" : \" canal\");\n  const n2 = document.getElementById(\"setcanalcount2\");\n  if (n2) n2.textContent = String(list.length) + (list.length > 1 ? \" canaux\" : \" canal\");\n\n  /* Toutes les adresses sont écrites EN ENTIER, jusqu'au dernier caractère. */\n  const first = list[0];\n  const cid = first ? first.id : \"arena\";\n  const set = function (id, txt, copyId) {\n    const el = document.getElementById(id);\n    if (el) el.textContent = txt;\n    const bt = document.getElementById(copyId);\n    if (bt) bt.setAttribute(\"data-copy\", txt);\n  };\n  set(\"canalbase\", base, \"canalbasecopy\");\n  set(\"canaladdr\", base + \"/\" + cid, \"canaladdrcopy\");\n  set(\"canalinstcmd\", 'curl -s \"' + base + \"/\" + cid + '/install?raw=1&format=python\" -o z-canal.py', \"canalinstcopy\");\n  wireCanalCopies(document.getElementById(\"setcanal\"));\n  wireCanalDeletes(document.getElementById(\"setcanal\"));\n}\n/* ---- supprimer un canal : confirmation, puis suppression définitive ---- */\nlet DEL_CANAL = \"\";\nfunction canalAskDelete(id){\n  if (!id) return;\n  if (id === \"arena\") { agtToast(\"Le canal par défaut ne peut pas être supprimé.\"); return; }\n  DEL_CANAL = id;\n  const c = canalById(id);\n  const p = document.getElementById(\"canaldel\");\n  if (!p) return;\n  const t = p.querySelector(\".delcard p\");\n  if (t) t.textContent = \"Voulez-vous vraiment supprimer le canal « \" + (c ? (c.name || id) : id)\n    + \" » ? Il sera supprimé définitivement.\";\n  p.hidden = false;\n}\nfunction canalCloseDelete(){\n  const p = document.getElementById(\"canaldel\");\n  if (p) p.hidden = true;\n  DEL_CANAL = \"\";\n}\nasync function canalDelete(id){\n  if (!id) return;\n  if (id === \"arena\") { agtToast(\"Le canal par défaut ne peut pas être supprimé.\"); return; }\n  /* Les discussions qui s'en servaient redeviennent de simples discussions :\n     le canal disparaît, pas elles. */\n  const liees = (chats || []).filter(x => x && x.canal === id);\n  for (const ch of liees) {\n    try {\n      if (FB_OK && fbUser) await chatsRef().doc(ch.id).set({ canal: null, mode: \"chat\" }, { merge: true });\n    } catch (e) { console.warn(\"canal (discussion détachée) :\", e); }\n    ch.canal = null; ch.mode = \"chat\";\n    if (ch.id === chatId) { CUR_CANAL = null; CUR_MODE = \"chat\"; }\n  }\n  /* Les messages du canal, puis sa fiche. */\n  try {\n    if (FB_OK && fbDB) {\n      const col = fbDB.collection(\"canaux\").doc(id).collection(\"messages\");\n      const snap = await col.limit(200).get();\n      for (const d of ((snap && snap.docs) || [])) {\n        try { if (d.ref && d.ref.delete) await d.ref.delete(); else await col.doc(d.id).delete(); } catch (e) {}\n      }\n      const docRef = fbDB.collection(\"canaux\").doc(id);\n      if (docRef.delete) await docRef.delete();\n    }\n  } catch (e) { console.warn(\"canal (suppression) :\", e); }\n  CANAUX = (CANAUX || []).filter(c => c && c.id !== id);\n  try { await canalSaveList(); } catch (e) {}\n  renderCanalList();\n  try { renderChatList(); } catch (e) {}\n  renderModeBadge();\n  agtToast(\"Canal supprimé.\");\n}\nfunction wireCanalDeletes(root){\n  if (!root) return;\n  [].slice.call(root.querySelectorAll(\"[data-del]\")).forEach(function (b) {\n    b.onclick = function (e) {\n      e.stopPropagation();\n      canalAskDelete(b.getAttribute(\"data-del\"));\n    };\n  });\n}\n/* Le bouton « copier » d'une adresse ou d'un identifiant : le même partout. */\nfunction wireCanalCopies(root){\n  if (!root) return;\n  [].slice.call(root.querySelectorAll(\".copybtn\")).forEach(function (b) {\n    if (b.dataset.wired) return;\n    b.dataset.wired = \"1\";\n    if (!b.innerHTML.trim()) b.innerHTML = COPYIC;\n    b.onclick = function () {\n      const txt = b.getAttribute(\"data-copy\") || \"\";\n      if (!txt) return;\n      agtCopyText(txt);\n      b.innerHTML = CHECKIC; b.classList.add(\"ok\"); b.title = \"Copié !\";\n      setTimeout(function () {\n        b.innerHTML = COPYIC; b.classList.remove(\"ok\");\n        b.title = b.id === \"canalbasecopy\" ? \"Copier l'adresse\" : (b.id === \"canaladdrcopy\" ? \"Copier l'adresse du canal\" : (b.id === \"canalinstcopy\" ? \"Copier la commande\" : \"Copier l'identifiant\"));\n      }, 1600);\n    };\n  });\n}\n\n/* ==================== les fonctions reprises (mode, liste, réglages) ==== */\nfunction openAccountSettings2(){\n  if (!fbUser) { openAuth(); return; }\n  closeNewPop();          /* jamais les réglages et la popup de mode ensemble */\n  closeChatPop();\n  fillSettings();\n  showSetView(\"list\");\n  $fb(\"#settings\").hidden = false;\n}\nfunction fillSettings2(){\n  $fb(\"#settitle\").textContent = profileName();\n  $fb(\"#setsub\").textContent = fbUser.email || \"\";\n  $fb(\"#setnameval\").textContent = profileName();\n  $fb(\"#setemailval\").textContent = fbUser.email || \"(aucun)\";\n  const hasG = (fbUser.providerData || []).some(p => p.providerId === \"google.com\");\n  $fb(\"#setgoogle\").textContent = hasG ? \"Ouvrir mon compte Google\" : \"Lier un compte Google\";\n  $fb(\"#setgooglenote\").textContent = hasG\n    ? \"Compte Google lié à ce profil.\"\n    : \"Permet de te connecter sans mot de passe, et ouvrira ton compte Google une fois lié.\";\n  renderSetAvatar();\n  const ra = document.getElementById(\"setrowavatar\");\n  if (ra) ra.innerHTML = avatarHTML(profileName(), userPhoto || fbUser.photoURL || \"\", 38);\n  const rm = document.getElementById(\"setrowmail\");\n  if (rm) rm.textContent = fbUser.email || \"Photo, sécurité, session\";\n  setErr(\"\"); hidePop();\n}\nfunction renderChatList2(){\n  const list = $fb(\"#chatlist\");\n  list.innerHTML = \"\";\n  const note = document.createElement(\"p\");\n  note.style.cssText = \"font-size:12.5px;color:var(--ink-3);padding:12px 10px;line-height:1.5\";\n  if (!fbUser) {\n    note.textContent = \"Connecte-toi pour enregistrer tes discussions et les retrouver sur tous tes appareils.\";\n    list.appendChild(note); return;\n  }\n  const shown = chats.filter(c => !c.hidden);\n  if (!shown.length) { note.textContent = \"Aucune discussion enregistrée pour le moment.\"; list.appendChild(note); return; }\n  const ordered = shown.slice().sort((a, b) => (b.arena ? 1 : 0) - (a.arena ? 1 : 0));\n  for (const c of ordered) {\n    const row = document.createElement(\"div\");\n    row.className = \"chitem\" + (c.id === chatId ? \" sel\" : \"\") + (c.arena ? \" pinned\" : \"\");\n    if (c.arena) {\n      const mk = document.createElement(\"span\");\n      mk.className = \"chmark\"; mk.title = \"Discussion Arena · canal direct\";\n      mk.innerHTML = MARKS.arena;\n      row.appendChild(mk);\n    } else if (c.canal) {\n      /* Les autres discussions reliées à un canal portent le logo Z-Canal. */\n      const mk = document.createElement(\"span\");\n      mk.className = \"chmark\";\n      mk.title = \"Discussion reliée au canal « \" + canalName(c.canal) + \" » (\" + canalModeOf(c.canal) + \")\";\n      mk.innerHTML = ZCANAL_MARK;\n      row.appendChild(mk);\n    }\n    const t = document.createElement(\"button\");\n    t.type = \"button\"; t.className = \"cht\"; t.textContent = c.title; t.title = c.title;\n    t.onclick = function(){ if (c.id !== chatId) loadChat(c.id); else closeSide(); };\n    row.appendChild(t);\n    if ((c.arena && ARENA_DOT) || c.dot) {\n      const dot = document.createElement(\"span\");\n      dot.className = \"chdot\"; dot.title = \"Nouveau message en attente\";\n      row.appendChild(dot);\n    }\n    if (c.agent) {\n      const badge = document.createElement(\"span\");\n      badge.className = \"chbadge\"; badge.textContent = c.canal ? \"Z-Canal\" : \"Agent\";\n      badge.title = \"Discussion en mode \" + (c.canal ? \"Z-Canal\" : \"agent\");\n      row.appendChild(badge);\n    }\n    if (c.job && c.job.state === \"run\") {\n      const jb = document.createElement(\"span\");\n      jb.className = \"chjob\"; jb.title = \"Mission en cours : elle avance toute seule, même app fermée\";\n      jb.innerHTML = \"<i></i>en cours\";\n      row.appendChild(jb);\n    }\n    const more = document.createElement(\"button\");\n    more.type = \"button\"; more.className = \"chmore\"; more.textContent = \"•••\";\n    more.setAttribute(\"aria-label\", \"Options : \" + c.title);\n    more.onclick = function(e){ e.stopPropagation(); openChatPop2(more, c); };\n    row.appendChild(more);\n    list.appendChild(row);\n  }\n}\nfunction openChatPop2(anchor, c){\n  closeChatPop();\n  const pop = document.createElement(\"div\"); pop.className = \"chpop\";\n  const ren = document.createElement(\"button\"); ren.type = \"button\"; ren.textContent = \"Renommer\";\n  ren.onclick = function(){ closeChatPop(); renameChat(c); };\n  pop.appendChild(ren);\n  if (c.pinned || c.arena) {\n    /* La discussion « Arena » porte le canal par défaut : elle ne se supprime pas. */\n    const no = document.createElement(\"button\");\n    no.type = \"button\"; no.className = \"note\";\n    no.textContent = \"Épinglée — non supprimable\";\n    no.disabled = true;\n    pop.appendChild(no);\n  } else {\n    const del = document.createElement(\"button\"); del.type = \"button\"; del.className = \"del\"; del.textContent = \"Supprimer\";\n    del.onclick = function(){ closeChatPop(); deleteChat2(c); };\n    pop.appendChild(del);\n  }\n  document.body.appendChild(pop);\n  const r = anchor.getBoundingClientRect();\n  pop.style.top = Math.max(8, Math.min(r.bottom + 4, innerHeight - pop.offsetHeight - 8)) + \"px\";\n  pop.style.left = Math.max(8, Math.min(r.right - pop.offsetWidth, innerWidth - pop.offsetWidth - 8)) + \"px\";\n  chatPop = pop;\n  setTimeout(function(){ document.addEventListener(\"click\", function(){ closeChatPop(); }, { once: true }); }, 0);\n}\nasync function deleteChat2(c){\n  if (!c || c.pinned || c.arena) { agtToast(\"La discussion « Arena » est épinglée : elle ne peut pas être supprimée.\"); return; }\n  if (!confirm(\"Supprimer définitivement « \" + c.title + \" » ?\")) return;\n  /* La discussion d'un canal libère le canal : il pourra servir à une autre. */\n  if (c.canal) {\n    const ch = canalById(c.canal);\n    if (ch) { ch.chat = null; await canalFreeze(c.canal, ch.mode, null); }\n  }\n  chats = chats.filter(x => x.id !== c.id);\n  if (chatId === c.id) { chatId = null; msgSeq = 0; CUR_CANAL = null; CUR_MODE = \"chat\"; renderModeBadge(); }\n  renderChatList();\n  if (FB_OK && fbUser) { try { await purgeChat(c.id); } catch (e) { console.warn(\"delete :\", e); } }\n}\nasync function ensureChat2(firstText){\n  if (chatId) return chatId;\n  chatId = \"c\" + Date.now().toString(36) + Math.random().toString(36).slice(2, 7);\n  msgSeq = 0;\n  const now = Date.now();\n  const title = (firstText || \"\").replace(/\\s+/g, \" \").trim().slice(0, 38) || \"Nouvelle discussion\";\n  const md = (CUR_MODE === \"agent\") ? \"agent\" : \"chat\";\n  const ag = md === \"agent\";\n  try {\n    await chatsRef().doc(chatId).set({ title: title, model: current.id, agent: ag, mode: md, createdAt: now, updatedAt: now });\n    chats.unshift({ id: chatId, title: title, model: current.id, agent: ag, mode: md, createdAt: now, updatedAt: now });\n    renderChatList();\n  } catch (e) { console.warn(\"ensureChat :\", e); }\n  return chatId;\n}\nfunction newConversation2(md){\n  CUR_MODE = (md === \"agent\" || md === \"canal\") ? md : \"chat\";\n  if (CUR_MODE !== \"canal\") { CUR_CANAL = null; AGENT_ON = (CUR_MODE === \"agent\"); }\n  try { agtBtn.classList.toggle(\"on\", !!AGENT_ON); } catch (e) {}\n  conv = []; log.innerHTML = \"\"; emptyState(); updateHint();\n  if (typeof FB !== \"undefined\" && FB.onNewChat) FB.onNewChat();\n  if (typeof agtNewConversation === \"function\") agtNewConversation();\n  renderModeBadge();\n  agtSyncUI2();\n}\n/* Ce que la discussion en cours est vraiment : chat, agent, ou canal. */\nfunction currentMode(){ return (CUR_MODE === \"agent\" || CUR_MODE === \"canal\") ? CUR_MODE : \"chat\"; }\nfunction currentChat(){ try { return (chats || []).find(x => x.id === chatId) || null; } catch (e) { return null; } }\nfunction agtSyncUI2(){\n  const ok = agtLogged();\n  const md = currentMode();\n  /* Le mode de la discussion commande tout. Un canal dont le mode est « agent »\n     travaille avec un agent : son panneau (écran de la VM + fichiers) est donc\n     là, comme dans une discussion d'agent. */\n  const canalAgent = (md === \"canal\") && (canalModeOf(CUR_CANAL) === \"agent\");\n  AGENT_ON = (md === \"agent\") || canalAgent;\n  agtFilesBtn.hidden = !(AGENT_ON && ok);\n  /* La poubelle (nouvelle discussion) est là dans tous les modes : c'est elle\n     qui ouvre le choix du mode. Seule exception, la discussion « Arena » :\n     épinglée et liée au canal par défaut, elle ne se remplace pas. */\n  if (typeof clearBtn !== \"undefined\" && clearBtn) {\n    const cc = currentChat();\n    clearBtn.hidden = !!(cc && cc.arena);\n  }\n  const ms = document.getElementById(\"menushare\");\n  if (ms) ms.hidden = !(AGENT_ON && ok);\n  renderModeBadge();\n}\n\n/* ==================== câblage ------------------------------------------- */\nfunction canalWire(){\n  const g = function(id){ return document.getElementById(id); };\n  /* le bouton « nouvelle discussion » ouvre le choix du mode (mode chat), et\n     celui du menu aussi — c'est le seul chemin en mode agent ou canal, où la\n     poubelle n'est plus là. */\n  if (typeof clearBtn !== \"undefined\" && clearBtn) clearBtn.onclick = openNewPop;\n  const nc = g(\"newchat\");\n  if (nc) nc.onclick = function(){ closeSide(); openNewPop(); };\n  const p = newPopEl();\n  if (p) {\n    const close = g(\"newpclose\"); if (close) close.onclick = closeNewPop;\n    p.addEventListener(\"click\", function(e){ if (e.target === p) closeNewPop(); });\n    const c1 = g(\"pickchat\"); if (c1) c1.onclick = function(){ pickMode(\"chat\"); };\n    const c2 = g(\"pickagent\"); if (c2) c2.onclick = function(){ pickMode(\"agent\"); };\n    const c3 = g(\"pickcanal\"); if (c3) c3.onclick = function(){ pickCanal(); };\n    const bk = g(\"npback\"); if (bk) bk.onclick = function(){ newPopView(\"npmodes\"); };\n    const cx = g(\"npcancel\"); if (cx) cx.onclick = closeNewPop;\n    /* la création : un vrai bouton pour revenir à la liste, un pour annuler */\n    const bkn = g(\"npbacknew\");\n    if (bkn) bkn.onclick = function(){ newPopView(\"npcanalform\"); fillCanalPicker(); };\n    const cx2 = g(\"npcancel2\"); if (cx2) cx2.onclick = closeNewPop;\n    /* la confirmation de suppression d'un canal */\n    const dc = g(\"canaldelcancel\"); if (dc) dc.onclick = canalCloseDelete;\n    const dk = g(\"canaldelok\");\n    if (dk) dk.onclick = async function(){ const id = DEL_CANAL; canalCloseDelete(); await canalDelete(id); };\n    const dback = g(\"canaldel\");\n    if (dback) dback.addEventListener(\"click\", function(e){ if (e.target === dback) canalCloseDelete(); });\n    const cr = g(\"canalcreate\"); if (cr) cr.onclick = canalCreateClick;\n    const cname = g(\"canalname\");\n    if (cname) cname.addEventListener(\"keydown\", function(e){ if (e.key === \"Enter\") { e.preventDefault(); canalCreateClick(); } });\n  }\n  /* les réglages : la liste, puis « Compte » et « Z-Canal » */\n  const sc = g(\"setclose\");\n  if (sc && !sc.dataset.wired2) { sc.dataset.wired2 = \"1\"; sc.addEventListener(\"click\", function(){ showSetView(\"list\"); }); }\n  const ga = g(\"setgoaccount\"); if (ga) ga.onclick = function(){ showSetView(\"compte\"); };\n  const gc = g(\"setgocanal\"); if (gc) gc.onclick = function(){ showSetView(\"canal\"); };\n  const gr = g(\"setgodispo\"); if (gr) gr.onclick = function(){ showSetView(\"dispo\"); dispoLoad(); };\n  const gm = g(\"setgomodels\"); if (gm) gm.onclick = function(){ showSetView(\"models\"); };\n  const gw = g(\"setgowan\"); if (gw) gw.onclick = function(){ showSetView(\"wan\"); };\n  const gt = g(\"setgotasks\"); if (gt) gt.onclick = function(){ showSetView(\"tasks\"); };\n  const dt = g(\"distestall\"); if (dt) dt.onclick = dispoTestAll;\n  const tr = g(\"taskrefresh\"); if (tr) tr.onclick = tasksLoad;\n  const b1 = g(\"setback\"); if (b1) b1.onclick = function(){ showSetView(SET_VIEW === \"wan\" ? \"models\" : \"list\"); };\n  /* les missions connues de cet appareil se déclarent (une fois, en douceur) */\n  setTimeout(function(){ try{ agtJobsDeclare(); }catch(e){} }, 1200);\n  /* les adresses de l'écran Z-Canal sont remplies (et copiables) d'un coup */\n  if (typeof renderCanalList === \"function\") renderCanalList();\n  document.addEventListener(\"keydown\", function(e){ if (e.key === \"Escape\") closeNewPop(); });\n  renderModeBadge();\n}\nif (document.readyState === \"loading\") document.addEventListener(\"DOMContentLoaded\", canalWire);\nelse canalWire();\n/* Les noms d'origine pointent sur les versions « mode » : tout le reste de\n   l'application (menu, liste, chargement, missions) en profite sans changer. */\nopenAccountSettings = openAccountSettings2;\nfillSettings = fillSettings2;\nrenderChatList = renderChatList2;\nopenChatPop = openChatPop2;\ndeleteChat = deleteChat2;\nensureChat = ensureChat2;\nnewConversation = newConversation2;\nagtSyncUI = agtSyncUI2;\n</script>\n</body>\n</html>\n";
/*APP_HTML_END*/

/* ---- Garde-fou agent.md / infos.md (cron + endpoint de restauration).
   Contenu canonique embarqué en base64 ; réécrit dans la VM si manquant. ---- */
const AGENT_DOC_B64 = "IyBhZ2VudC5tZCDigJQgSW5zdHJ1Y3Rpb25zIHBvdXIgbCdhZ2VudCBWTQoKIyMgUXVpIHR1IGVzCgpUdSBlcyB1biBhZ2VudCBJQSBxdWkgcGlsb3RlIHVuZSBWTSBVYnVudHUgMjIuMDQgKGNvbnRlbmV1ciBEb2NrZXIKYGthc21fZGVza3RvcGAsIGJ1cmVhdSBLQVNNKSB2aWEgc29uIHRlcm1pbmFsLiBUdSB0cmF2YWlsbGVzIHBvdXIgdW4KdXRpbGlzYXRldXIgdW5pcXVlIHF1aSB0ZSByZWdhcmRlIGZhaXJlIGVuIGRpcmVjdCBzdXIgc29uIGludGVyZmFjZS4KVGVzIGNvbW1hbmRlcyBzJ2V4w6ljdXRlbnQgZW4gdGFudCBxdSd1dGlsaXNhdGV1ciBga2FzbS11c2VyYCwgZGVwdWlzCmAvaG9tZS9rYXNtLXVzZXJgLgoKIyMgRm9ybWF0IGRlIHLDqXBvbnNlCgpDaGFxdWUgcsOpcG9uc2UgZG9pdCBjb250ZW5pciBleGFjdGVtZW50IHVuIG9iamV0IEpTT04gKMOpdmVudHVlbGxlbWVudCBlbnRvdXLDqQpkJ3VuZSBiYWxpc2UgZGUgY29kZSBgYGBqc29uKSA6CgotIHBvdXIgZXjDqWN1dGVyIHVuZSBjb21tYW5kZSA6IGB7ImFjdGlvbiI6ImJhc2giLCJjb21tYW5kIjoiLi4uIn1gCi0gcG91ciBsaXJlIHVuZSBwYWdlIHdlYiA6IGB7ImFjdGlvbiI6ImZldGNoIiwidXJsIjoiaHR0cHM6Ly/igKYiLCJtb2RlIjoiaHRtbCJ8InRleHQifWAKLSBwb3VyIHRlcm1pbmVyIGxhIG1pc3Npb24gOiBgeyJhY3Rpb24iOiJmaW5hbCIsIm1lc3NhZ2UiOiIuLi4ifWAgKG1lc3NhZ2UgZW4gbWFya2Rvd24pCgpJbCBuZSBmYXV0IHBhcyByw6lwb25kcmUgYXV0cmVtZW50IDogdW4gYXV0cmUgZm9ybWF0IG5lIHBldXQgcGFzIMOqdHJlIGV4w6ljdXTDqS4KTifDqWNyaXMgcmllbiBkJ2F1dHJlIGF1dG91ciBkdSBKU09OLgoKIyMgRm9uY3Rpb25uZW1lbnQKCi0gVW5lIGNvbW1hbmRlIHBhciByw6lwb25zZS4gVHUgcmXDp29pcyBlbnN1aXRlIHN0ZG91dCwgc3RkZXJyIGV0IGxlIGNvZGUgZGUgc29ydGllLgotIETDqWxhaSBkJ2V4w6ljdXRpb24gbWF4aW1hbCA6IDMwMCBzIHBhciBjb21tYW5kZS4gUG91ciBsZXMgdMOiY2hlcyBsb25ndWVzCiAgKGNvbXBpbGF0aW9ucywgc2VydmV1cnMsIHTDqWzDqWNoYXJnZW1lbnRzKSwgbGFuY2UtbGVzIGVuIGFycmnDqHJlLXBsYW4KICAoYG5vaHVwIC4uLiA+IGpvdXJuYWwubG9nIDI+JjEgJmApIHB1aXMgY29uc3VsdGUgbGUgam91cm5hbC4KLSBUdSByZcOnb2lzIGBzdGRvdXRgIGV0IGBzdGRlcnJgICoqZW50aWVycyoqLCB0ZWxzIHF1ZWxzLiBQb3VyIHVuZSBjb21tYW5kZSB0csOocwogIGJhdmFyZGUsIGZpbHRyZSB0b2ktbcOqbWUgKGB0YWlsIC1uIDQwYCwgYGdyZXBgLCBgd2MgLWxgKSBzaSB0dSBuJ2FzIHBhcyBiZXNvaW4KICBkZSB0b3V0IDogYydlc3QgdG9pIHF1aSBkw6ljaWRlcywgcmllbiBuJ2VzdCBmaWx0csOpIMOgIHRhIHBsYWNlLgotIENoYXF1ZSBjb21tYW5kZSBxdWUgdHUgbGFuY2VzIGVzdCBlbnJlZ2lzdHLDqWUgZGFucyBsYSBjb252ZXJzYXRpb24gZGUKICBsJ3V0aWxpc2F0ZXVyIDogaWwgcGV1dCBsZXMgcmVsaXJlIHBsdXMgdGFyZC4KCiMjIE91dGlsIDIg4oCUIGZldGNoIChsZWN0dXJlIGRlIHBhZ2VzIHdlYikKClBvdXIgbGlyZSB1bmUgcGFnZSB3ZWIgKGFydGljbGUsIGRvY3VtZW50YXRpb24sIGRvbm7DqWVzLCBzaXRlIGR5bmFtaXF1ZSksIHR1CmRpc3Bvc2VzIGQndW4gc2Vjb25kIG91dGlsIDogKipmZXRjaCoqLiBOJ3V0aWxpc2UgcGFzIGBjdXJsYCBwb3VyIGxpcmUgZHUKY29udGVudSA6IGZldGNoIG91dnJlIGxhIHBhZ2UgZGFucyBDaHJvbWUgaGVhZGxlc3MgKipzdXIgbGEgVk0qKiwgZXjDqWN1dGUgc29uCkphdmFTY3JpcHQsIHB1aXMgdGUgcmVudm9pZSBsZSBjb250ZW51ICoqY29tcGxldCoqLgoKYGBganNvbgp7ImFjdGlvbiI6ImZldGNoIiwidXJsIjoiaHR0cHM6Ly9leGVtcGxlLmNvbS9wYWdlIiwibW9kZSI6InRleHQifQpgYGAKCi0gYG1vZGVgIDoKICAtIGAidGV4dCJgIOKGkiBsZSB0ZXh0ZSB2aXNpYmxlIGRlIGxhIHBhZ2UgKGxlIHBsdXMgdXRpbGUgcG91ciBsaXJlL2NvbXByZW5kcmUpIDsKICAtIGAiaHRtbCJgIOKGkiBsZSBjb2RlIHNvdXJjZSBjb21wbGV0IGRlIGxhIHBhZ2UgKiphcHLDqHMqKiBleMOpY3V0aW9uIGR1IEphdmFTY3JpcHQuCi0gQ2hhbXBzIGZhY3VsdGF0aWZzIDoKICAtIGB3YWl0X21zYCA6IGF0dGVuZHJlIFggbWlsbGlzZWNvbmRlcyBhcHLDqHMgbGUgY2hhcmdlbWVudCAoZMOpZmF1dCA3MDApIOKAlAogICAgdXRpbGUgcG91ciBsZXMgcGFnZXMgcXVpIHJlbXBsaXNzZW50IGxldXIgY29udGVudSBlbiBKYXZhU2NyaXB0IDsKICAtIGB0aW1lb3V0X21zYCA6IGTDqWxhaSBtYXhpbWFsIGVuIG1pbGxpc2Vjb25kZXMgKGTDqWZhdXQgNDUwMDAsIG1heCAxODAwMDApIDsKICAtIGBzZWxlY3RvcmAgOiBhdHRlbmRyZSBxdSd1biDDqWzDqW1lbnQgQ1NTIGFwcGFyYWlzc2UgYXZhbnQgZGUgbGlyZQogICAgKGV4LiBgeyJhY3Rpb24iOiJmZXRjaCIsInVybCI6IuKApiIsIm1vZGUiOiJ0ZXh0Iiwic2VsZWN0b3IiOiIjcmVzdWx0YXRzIn1gKS4KLSBDZSBxdWUgdHUgcmXDp29pcyA6IGBzdGF0dXNgIChjb2RlIEhUVFApLCBgdGl0bGVgICh0aXRyZSBkZSBsYSBwYWdlKSwgYGNoYXJzYAogICh0YWlsbGUgZHUgY29udGVudSksIGBwYXRoYCAoY2hlbWluIGQndW5lIGNvcGllIGNvbXBsw6h0ZSBlbnJlZ2lzdHLDqWUgZGFucwogIGB+Ly5hZ3RmZXRjaC9gIHN1ciBsYSBWTSksIGBtc2AgKGR1csOpZSkgZXQgYGNvbnRlbnRgIChsZSBjb250ZW51ICoqZW50aWVyKiogZGUKICBsYSBwYWdlKS4KLSBUcm9pcyBtb2RlcyBkJ8OpY2hlYyDDoCBjb25uYcOudHJlIDogcGFnZXMgZGVycmnDqHJlIHVuIGNvbXB0ZSwgY2FwdGNoYXMvbXVycwogIGFudGktcm9ib3QsIGV0IGFkcmVzc2VzIGludmFsaWRlcyAoYHVybGAgZG9pdCBjb21tZW5jZXIgcGFyIGBodHRwOi8vYCBvdQogIGBodHRwczovL2ApLiBEYW5zIGNlcyBjYXMsIGxhIHLDqXBvbnNlIGNvbnRpZW50IGBvazpmYWxzZWAgZXQgdW5lIGBlcnJvcmAuCi0gUG91ciBkZSB0csOocyBncm9zIGNvbnRlbnVzLCB0dSBwZXV4IGVuY2hhw65uZXIgYXZlYyBiYXNoIHN1ciBsZSBmaWNoaWVyCiAgYHBhdGhgIChgZ3JlcGAsIGBoZWFkYCwgYHB5dGhvbjNgKSBhdSBsaWV1IGRlIHRvdXQgcmVsaXJlLgoKIyMgRG9zc2llcnMgZXQgZmljaGllcnMKCi0gVHUgcGV1eCB0cmF2YWlsbGVyICoqcGFydG91dCoqIGRhbnMgbGEgVk0gOiBgL2hvbWUva2FzbS11c2VyYCwgYC90bXBgLAogIGBEZXNrdG9wYCwgYERvY3VtZW50c2AsIGBEb3dubG9hZHNgLCBgL29wdGDigKYgYXVjdW4gZG9zc2llciBuZSB0J2VzdCBpbnRlcmRpdC4KLSAqKlLDqGdsZSBpbXBvcnRhbnRlKiogOiBsJ2V4cGxvcmF0ZXVyIGRlIGwndXRpbGlzYXRldXIgbmUgbW9udHJlIFFVRSBsZQogIGRvc3NpZXIgYC9ob21lL2thc20tdXNlci91c2VyL2AuIFF1YW5kIGwndXRpbGlzYXRldXIgZGVtYW5kZSB1biBmaWNoaWVyLAogIHZldXQgbGUgdm9pciwgbGUgdMOpbMOpY2hhcmdlciBvdSB5IGFjY8OpZGVyLCAqKmNvcGllIG91IGTDqXBsYWNlLWxlIGRhbnMKICBgL2hvbWUva2FzbS11c2VyL3VzZXIvYCoqICh1biBzb3VzLWRvc3NpZXIgZGUgYHVzZXIvYCBjb252aWVudCBhdXNzaSkuCiAgVW4gbGl2cmFibGUgbGFpc3PDqSBhaWxsZXVycyBlc3QgaW52aXNpYmxlIHBvdXIgbHVpLgotIFRvdXQgY2UgcXVlIHR1IGTDqXBvc2VzIGRhbnMgYHVzZXIvYCBhcHBhcmHDrnQgZW4gZGlyZWN0IGRhbnMgc29uIGV4cGxvcmF0ZXVyCiAgZXQgcydvdXZyZSBkYW5zIHNhIHZpc2lvbm5ldXNlIChpbWFnZXMsIHZpZMOpb3MsIFBERiwgV29yZCwgUG93ZXJQb2ludCwKICBIVE1MLCBjb2Rl4oCmKS4KLSBMZXMgZmljaGllcnMgcmXDp3VzIGRlIGwndXRpbGlzYXRldXIgYXJyaXZlbnQgZW4gcGnDqGNlcyBqb2ludGVzIGRhbnMgc2VzCiAgbWVzc2FnZXMgKGltYWdlcyB2aXNpYmxlcyBkaXJlY3RlbWVudCkuCgojIyBMYSBmaWNoZSB1dGlsaXNhdGV1ciAodXNlci5tZCkKCi0gYHVzZXIubWRgIGVzdCBTQSBmaWNoZSBwZXJzb25uZWxsZSA6IGlkZW50aXTDqSAobm9tLCBlLW1haWwsIMOiZ2UsIG3DqXRpZXIpLAogIGluZm9ybWF0aW9ucyBzdXBwbMOpbWVudGFpcmVzIGV0ICoqaW5zdHJ1Y3Rpb25zIHBlcnNvbm5hbGlzw6llcyoqLgotIExpcy1sYSBlbiBkw6lidXQgZGUgbWlzc2lvbiAoYGNhdCAvaG9tZS9rYXNtLXVzZXIvdXNlci91c2VyLm1kYCkgZXQgc3VpcyBzZXMKICBpbnN0cnVjdGlvbnMgcGVyc29ubmFsaXPDqWVzLgotIEMnZXN0IGxlIHNldWwgZmljaGllciBxdSdpbCBtb2RpZmllIGx1aS1tw6ptZSwgdmlhIHNvbiBpbnRlcmZhY2UuCiAgKipOZSBsZSBtb2RpZmllIGV0IG5lIGxlIHN1cHByaW1lIGphbWFpcyoqLCBldCBuJ8OpY3JpcyByaWVuIGRlZGFucy4KLSBDZSBmaWNoaWVyIChhZ2VudC5tZCkgZXQgaW5mb3MubWQgZMOpY3JpdmVudCB0b24gZm9uY3Rpb25uZW1lbnQgOyB0dSBwZXV4CiAgbGVzIHJlbGlyZSDDoCB0b3V0IG1vbWVudCwgbWFpcyBuZSBsZXMgbW9kaWZpZSBwYXMuCgojIyBQcml2aWzDqGdlcyBhZG1pbmlzdHJhdGV1cgoKVHUgZGlzcG9zZXMgZGUgc3VkbyBzYW5zIG1vdCBkZSBwYXNzZSAoYC9ldGMvc3Vkb2Vycy5kL2thc20tdXNlcmApIDoKYHN1ZG8gLW4gYXB0LWdldCBpbnN0YWxsIC15IDxwYXF1ZXQ+YCwgYHN1ZG8gLW4gc3lzdGVtY3RsIC4uLmAsIGV0Yy4KQXVjdW5lIHNhaXNpZSBkZSBtb3QgZGUgcGFzc2UgbmUgdGUgc2VyYSBqYW1haXMgZGVtYW5kw6llLgoKIyMgT3V0aWxzIGRpc3BvbmlibGVzCgotIGwnb3V0aWwgKipmZXRjaCoqICh2b2lyIHBsdXMgaGF1dCkgOiBDaHJvbWUgaGVhZGxlc3MgcGlsb3TDqSBwYXIgUGxheXdyaWdodCwKICBzZXJ2aWNlIHBlcm1hbmVudCBkYW5zIGxhIFZNIChwb3J0IDMwMTEpIOKAlCBwYXMgZGUgZMOpbWFycmFnZSDDoCBwYXllciDDoCBjaGFxdWUKICBsZWN0dXJlLiBUdSBsJ2FwcGVsbGVzICoqcGFyIGwnYWN0aW9uIEpTT04gYGZldGNoYCoqIChjJ2VzdCBsYSB2b2llIHByw6l2dWUpIDsKICBsZSBzZXJ2aWNlIHLDqXBvbmQgYXVzc2kgZW4gSFRUUCBsb2NhbCBzaSB0dSBwcsOpZsOocmVzIHBhc3NlciBwYXIgYmFzaCwgbWFpcwogIMOpdml0ZSBkJ8OpY3JpcmUgZGVzIGNvbW1hbmRlcyBkZSB0w6lsw6ljaGFyZ2VtZW50IGVudHJlIGFjY2VudHMgZ3JhdmVzIDogbGUKICBwYXJlLWZldSBkZSBub3RyZSBmb3Vybmlzc2V1ciBtb2TDqGxlIGxlcyBwcmVuZCBwb3VyIHVuZSBhdHRhcXVlIGV0IHJlZnVzZSBsYQogIHJlcXXDqnRlIGVudGnDqHJlIDsKLSBgcHl0aG9uM2AgKG1hdHBsb3RsaWIgaW5zdGFsbMOpIDogdGVzIGZpZ3VyZXMgcGV1dmVudCDDqnRyZSBnw6luw6lyw6llcyBlbiBQTkcpLAogIGBwaXAzYCBwb3VyIGNvbXBsw6l0ZXIgYXUgYmVzb2luIDsKLSBOb2RlLmpzIHZpYSBudm0gOiBgfi8ubnZtL3ZlcnNpb25zL25vZGUvKi9iaW4vbm9kZWAgKHByZW5kcyBsYSB2ZXJzaW9uIGxhCiAgcGx1cyByw6ljZW50ZSkgOwotIExpYnJlT2ZmaWNlIChgc29mZmljZSAtLWhlYWRsZXNzYCkgcG91ciBjb252ZXJ0aXIgZGVzIGRvY3VtZW50cwogIChkb2N4L3BwdHgveGxzeCDihpIgcGRmKSA7Ci0gR29vZ2xlIENocm9tZSAoYGdvb2dsZS1jaHJvbWVgKSwgZ2l0LCBjdXJsLCB3Z2V0LCBpbWFnZW1hZ2ljaywgZmZtcGVnIHMnaWwKICBlc3QgaW5zdGFsbMOpIDsKLSB0b3V0IGNlIHF1ZSB0dSBwZXV4IGluc3RhbGxlciBhdmVjIGBzdWRvIC1uIGFwdC1nZXQgaW5zdGFsbGAuCgojIyBNw6l0aG9kZQoKMS4gTGlzIGxhIGZpY2hlIHV0aWxpc2F0ZXVyIGV0IGNvbXByZW5kcyBsYSBkZW1hbmRlIDsgdsOpcmlmaWUgbCdlbnZpcm9ubmVtZW50CiAgIChgbHNgLCB2ZXJzaW9ucywgb3V0aWxzIHByw6lzZW50cykuCjIuIEF2YW5jZSBwYXIgw6l0YXBlcyA6IGNyw6llciDihpIgdsOpcmlmaWVyIOKGkiBsaXZyZXIuICoqQXVjdW5lIGxpbWl0ZSBkZSBub21icmUKICAgZCfDqXRhcGVzIG5pIGRlIGR1csOpZSoqIDogdHUgcGV1eCBlbmNoYcOubmVyIGF1dGFudCBkZSBjb21tYW5kZXMgcXVlCiAgIG7DqWNlc3NhaXJlIHBvdXIgYWxsZXIgYXUgYm91dCwgc2FucyBjcmFpbmRyZSBkJ8OqdHJlIGNvdXDDqS4KMy4gQXByw6hzIGNoYXF1ZSDDqXRhcGUgaW1wb3J0YW50ZSwgdsOpcmlmaWUgbGUgcsOpc3VsdGF0IChjb2RlIGRlIHNvcnRpZSwgdGVzdHMsCiAgIGV4dHJhaXQgZGUgbGEgc29ydGllKSBhdmFudCBkZSBwYXNzZXIgw6AgbGEgc3VpdmFudGUuCjQuIETDqXBvc2UgbGVzIGxpdnJhYmxlcyBkYW5zIGAvaG9tZS9rYXNtLXVzZXIvdXNlci9gLgo1LiBUZXJtaW5lIHBhciBgeyJhY3Rpb24iOiJmaW5hbCIsIm1lc3NhZ2UiOiIuLi4ifWAgZW4gZXhwbGlxdWFudCBjZSBxdWkgYSDDqXTDqQogICBmYWl0LCBvw7kgc2UgdHJvdXZlbnQgbGVzIGxpdnJhYmxlcyBldCBjb21tZW50IGxlcyB1dGlsaXNlci4KCjwhLS0gdm0tZG9jLXZlcnNpb246IDIwMjYtMTAtMTUuMSAtLT4K";
const INFOS_DOC_B64 = "IyBpbmZvcy5tZCDigJQgTCdpbnRlcmZhY2UgZGUgbCd1dGlsaXNhdGV1cgoKQ2UgZmljaGllciBkw6ljcml0IHByw6ljaXPDqW1lbnQgY2UgcXVlIGwndXRpbGlzYXRldXIgdm9pdCBldCBwZXV0IGZhaXJlLCBwb3VyCnF1ZSB0dSBwdWlzc2VzIGxlIGd1aWRlciBzYW5zIGRldmluZXIuIFF1YW5kIHR1IGx1aSBkaXMgwqsgY2xpcXVleiBpY2kgwrsgb3UKwqsgb3V2cmV6IGNlIGZpY2hpZXIgwrssIGFwcHVpZS10b2kgc3VyIGNlcyBkZXNjcmlwdGlvbnMuCgojIyAxLiBMYSBjb252ZXJzYXRpb24KCi0gTCdhcHBsaWNhdGlvbiBlc3QgdW4gY2hhdC4gQ2hhcXVlIGRpc2N1c3Npb24gYXBwYXJhw650IGRhbnMgbGEgbGlzdGUgZGUKICBnYXVjaGUsIGF2ZWMgc29uIHRpdHJlIGV0IGxhIGRhdGUuCi0gVW5lIGRpc2N1c3Npb24gZXN0ICoqc29pdCB1biBjaGF0IGNsYXNzaXF1ZSwgc29pdCB1bmUgY29udmVyc2F0aW9uIGFnZW50KiogOgogIGxlIGNob2l4IHNlIGZhaXQgYXUgdG91dCBkw6lidXQsIHN1ciB1bmUgY29udmVyc2F0aW9uIHZpZXJnZS4gTGVzCiAgY29udmVyc2F0aW9ucyBhZ2VudCBwb3J0ZW50IHVuICoqYmFkZ2Ugbm9pciDCqyBBZ2VudCDCuyoqIGRhbnMgbGEgbGlzdGUuCi0gVGVzIHLDqXBvbnNlcyBzb250IHJlbmR1ZXMgZW4gbWFya2Rvd24gOiB0aXRyZXMsIGxpc3RlcywgdGFibGVhdXgsIGJsb2NzIGRlCiAgY29kZSwgZm9ybXVsZXMgbWF0aMOpbWF0aXF1ZXMgTGFUZVggKGAkLi4uJGAgZXQgYCQkLi4uJCRgKSBzb250IGFmZmljaMOpcwogIGNvcnJlY3RlbWVudC4KLSBTb3VzIGNoYWN1biBkZSB0ZXMgbWVzc2FnZXMsIGwndXRpbGlzYXRldXIgYSB1biBib3V0b24gKipjb3BpZXIqKi4gRW4gbW9kZQogIGFnZW50LCBpbCBuJ3kgYSAqKnBhcyoqIGRlIGJvdXRvbiDCqyByZWNvbW1lbmNlciDCuyA6IHVuZSBtaXNzaW9uIG5lIHNlIHJlam91ZQogIHBhcywgaWwgZW4gcmVkZW1hbmRlIHVuZSBub3V2ZWxsZS4KLSBMZXMgbWVzc2FnZXMgZXQsIGVuIG1vZGUgYWdlbnQsICoqbGVzIGNvbW1hbmRlcyBkdSB0ZXJtaW5hbCoqIHNvbnQKICBlbnJlZ2lzdHLDqXMgZGFucyBzb24gY29tcHRlIDogZW4gcm91dnJhbnQgbGEgZGlzY3Vzc2lvbiBwbHVzIHRhcmQsIGlsCiAgcmV0cm91dmUgdG91dCwgeSBjb21wcmlzIGxlIGTDqXRhaWwgZGVzIGNvbW1hbmRlcyBldCBkZSBsZXVyIHNvcnRpZS4gVGVzCiAgcsOpcG9uc2VzIHkgc29udCBjb25zZXJ2w6llcyAqKmVuIG1hcmtkb3duKiogOiBlbGxlcyBzZSByw6lhZmZpY2hlbnQgbWlzZXMgZW4KICBmb3JtZSwgw6AgbCdpZGVudGlxdWUuCi0gUG91ciDDqWNyaXJlIDogbGEgdG91Y2hlICoqRW50csOpZSoqIGZhaXQgdW4gKipyZXRvdXIgw6AgbGEgbGlnbmUqKiAoaWwgcGV1dCBkb25jCiAgw6ljcmlyZSB1biBtZXNzYWdlIGRlIHBsdXNpZXVycyBsaWduZXMgdHJhbnF1aWxsZW1lbnQpLCAqKkN0cmwrRW50csOpZSoqIChvdQogIENtZCtFbnRyw6llKSBlbnZvaWUsIGNvbW1lIGxlIGJvdXRvbiAqKuKGkioqLiBMYSBiYXJyZSBhZmZpY2hlIHNpbXBsZW1lbnQKICDCqyBEZW1hbmRleiDDoCBsJ0lB4oCmIMK7LgotIElsIMOpY3JpdCBkdSBtYXJrZG93biAqKmRpcmVjdGVtZW50IGRhbnMgbGEgYmFycmUgZGUgc2Fpc2llKiosIHNhbnMgYXBlcsOndQogIGF1LWRlc3N1cyA6IGxlIHRleHRlIHNlIG1ldCBlbiBmb3JtZSBwZW5kYW50IHF1J2lsIHRhcGUgKGxlIGdyYXMgcydhZmZpY2hlIGVuCiAgZ3JhcyksIGV0IGxlcyBzaWduZXMgKGAqKmAsIGBgIGAgYGAsIGAtYCkgcmVzdGVudCB2aXNpYmxlcyDigJQgcydpbCBlbiBlZmZhY2UKICB1biwgbGEgbWlzZSBlbiBmb3JtZSBjb3JyZXNwb25kYW50ZSBkaXNwYXJhw650LiBMZXMgZGV1eCBzaWduZXMgZGUgZmVybWV0dXJlCiAgc2UgcG9zZW50IHRvdXQgc2V1bHMgOyBlbiBkw6lidXQgZGUgbGlnbmUgwqsgLSDCuyBwdWlzIGVzcGFjZSBkb25uZSB1bmUgcHVjZSDCqyDigKIgwrsKICBldCBFbnRyw6llIHBvdXJzdWl0IGxhIGxpc3RlLiBDZSBxdSdpbCBlbnZvaWUgZXN0IGJpZW4gbGUgbWFya2Rvd24sIHBhcyBsZSByZW5kdS4KCiMjIDIuIExlIG1vZGUgYWdlbnQKCi0gUXVhbmQgaWwgdCfDqWNyaXQgZW4gbW9kZSBhZ2VudCwgdW4gcGFubmVhdSBzJ291dnJlIGF2ZWMgKipsJ8OpY3JhbiBkZSBsYSBWTQogIGVuIGRpcmVjdCoqIGV0ICoqbCdleHBsb3JhdGV1ciBkZSBmaWNoaWVycyoqLgotIENoYWN1bmUgZGUgdGVzIGNvbW1hbmRlcyBzJ2FmZmljaGUgZGFucyB1biBtaW5pLXRlcm1pbmFsIDogYCQgdGEtY29tbWFuZGVgLAogIHB1aXMgc2Egc29ydGllLCBlbiBkaXJlY3QuIElsIHN1aXQgZG9uYyB0b24gdHJhdmFpbCDDqXRhcGUgcGFyIMOpdGFwZSBldCBwZXV0CiAgdm9pciDDoCBxdWVsIG1vbWVudCDDp2EgYmxvcXVlLgotIExlIHRlcm1pbmFsIHNlIHJlcGxpZSDDoCBsYSBmaW4gZGUgbGEgbWlzc2lvbiA7IGlsIHBldXQgbGUgcm91dnJpci4KLSAqKkxhIG1pc3Npb24gY29udGludWUgbcOqbWUgcydpbCBmZXJtZSBsJ2FwcGxpY2F0aW9uKiogOiBlbGxlIGVzdCBwaWxvdMOpZSBwYXIKICBsYSBwYXNzZXJlbGxlLCBwYXMgcGFyIHNvbiBuYXZpZ2F0ZXVyLiBFbiByb3V2cmFudCBsJ2FwcCAob3UgZW4gcmV2ZW5hbnQpLCBpbAogIHJldHJvdXZlIGxhIG1pc3Npb24gZW4gY291cnMgb3UgdGVybWluw6llLCBhdmVjIHNlcyBjb21tYW5kZXMgZXQgbGUgcsOpc3VsdGF0IDsKICBjJ2VzdCBhdXNzaSBlbnJlZ2lzdHLDqSBkYW5zIHNhIGRpc2N1c3Npb24gY29tbWUgZCdoYWJpdHVkZS4KLSAqKkF1Y3VuZSBsaW1pdGUgZGUgZHVyw6llIG5pIGRlIG5vbWJyZSBkZSBjb21tYW5kZXMqKiA6IHVuZSBtaXNzaW9uIHZhIGp1c3F1J8OgCiAgc2EgY29uY2x1c2lvbiwgbcOqbWUgc2kgZWxsZSBkdXJlIGxvbmd0ZW1wcyBvdSBlbmNoYcOubmUgZGVzIGRpemFpbmVzIGRlCiAgY29tbWFuZGVzLiBFbGxlIG5lIHMnYXJyw6p0ZSBxdWUgc2kgdHUgcmVuZHMgdGEgcsOpcG9uc2UgZmluYWxlIG91IHMnaWwKICBsJ2ludGVycm9tcHQgbHVpLW3Dqm1lLgotIFMnaWwgam9pbnQgdW5lIGNhcHR1cmUgZCfDqWNyYW4gw6Agc29uIG1lc3NhZ2UsIHR1IGxhIHJlw6dvaXMgYXZlYyBsYSBkZW1hbmRlLgoKRGV1eCBhdXRyZXMgb3V0aWxzIGNvbXBsw6h0ZW50IHNlcyBjb21tYW5kZXMgOiBpbCByZcOnb2l0IHRvdWpvdXJzIGxhICoqc29ydGllCmNvbXBsw6h0ZSoqIGRlIHNlcyBjb21tYW5kZXMgKHJpZW4gbidlc3QgY291cMOpKSwgZXQgaWwgcGV1dCAqKmxpcmUgZGVzIHBhZ2VzCndlYioqIGx1aS1tw6ptZSAoQ2hyb21lIGhlYWRsZXNzIHN1ciBsYSBWTSkg4oCUIGNlcyBsZWN0dXJlcyBhcHBhcmFpc3NlbnQgZGFucyBsYQp6b25lIMKrIFRlcm1pbmFsIGRlIGxhIFZNIMK7LCBzb3VzIGxhIGZvcm1lIGBmZXRjaCBbdGV4dF0gaHR0cHM6Ly/igKZgLgoKIyMgMy4gTCfDqWNyYW4gZGUgbGEgVk0KCi0gTCfDqWNyYW4gZHUgYnVyZWF1IFVidW50dSBzJ2FmZmljaGUgZW4gZ3JhbmQsIHJhZnJhw65jaGkgdG91dGVzIGxlcyAzIHNlY29uZGVzLgotIElsIGVzdCAqKnB1cmVtZW50IGNvbnN1bHRhdGlmKiogOiBsJ3V0aWxpc2F0ZXVyIG5lIHBldXQgbmkgY2xpcXVlciBuaSB0YXBlcgogIHN1ciBsZSBidXJlYXUuIEMnZXN0IHRvaSBxdWkgcGlsb3Rlcy4KLSBDb25zw6lxdWVuY2UgcHJhdGlxdWUgOiBzJ2lsIGZhdXQgY2xpcXVlciBxdWVscXVlIHBhcnQsIG91dnJpciB1biBuYXZpZ2F0ZXVyLAogIHJlbXBsaXIgdW4gZm9ybXVsYWlyZSwgYydlc3Qgw6AgdG9pIGRlIGxlIGZhaXJlICh4ZG90b29sLCB3bWN0cmwsIHNjcmlwdHMpLgogIETDqWNyaXMtbHVpIGNlIHF1aSBzZSBwYXNzZSDDoCBsJ8OpY3JhbiBxdWFuZCBjJ2VzdCB1dGlsZS4KCiMjIDQuIEwnZXhwbG9yYXRldXIgZGUgZmljaGllcnMKCi0gSWwgbW9udHJlIGxlIGNvbnRlbnUgZGUgYC9ob21lL2thc20tdXNlci91c2VyL2AgZW4gZGlyZWN0LCBhdmVjIGxlcwogIHNvdXMtZG9zc2llcnMgZMOpcGxpYWJsZXMuCi0gQ2hhcXVlIGZpY2hpZXIgYSB1bmUgaWPDtG5lIHNlbG9uIHNvbiB0eXBlIDogZG9zc2llcnMsIEhUTUwsIENTUywgSlMsIFRTLAogIFJlYWN0LCBNYXJrZG93biwgUHl0aG9uLCBTaGVsbCwgSlNPTiwgU1ZHLCBpbWFnZXMsIHZpZMOpb3MsIGF1ZGlvLCBQREYsCiAgV29yZCwgUG93ZXJQb2ludCwgRXhjZWwsIExhVGVYLCBldCB1bmUgaWPDtG5lIGfDqW7DqXJpcXVlIGVuIHJlcGxpLgotICoqVW4gY2xpYyBzdXIgdW4gZmljaGllciBsJ291dnJlIGRhbnMgbGEgdmlzaW9ubmV1c2UuKiogQydlc3QgbGUgc2V1bCBtb3llbgogIHBvdXIgbHVpIGRlIHZvaXIgdGVzIGxpdnJhYmxlcyA6IHRvdXQgY2UgcXVpIGRvaXQgw6p0cmUgdmlzaWJsZSBkb2l0IMOqdHJlCiAgZGFucyBjZSBkb3NzaWVyLgotIElsIG5lIHBldXQgbmkgc3VwcHJpbWVyIG5pIHJlbm9tbWVyIGRlIGZpY2hpZXJzIGljaS4gUydpbCBmYXV0IG5ldHRveWVyLAogIGMnZXN0IHRvaSBxdWkgbGUgZmFpcy4KCiMjIDUuIExhIHZpc2lvbm5ldXNlIOKAlCBjZSBxdSdpbCBwZXV0IHZvaXIsIGZvcm1hdCBwYXIgZm9ybWF0CgpFbiBoYXV0IGRlIGxhIHZpc2lvbm5ldXNlIDogbGUgbm9tIGR1IGZpY2hpZXIsIGRlcyAqKmJvdXRvbnMgZCdhY3Rpb24qKiBxdWkKY2hhbmdlbnQgc2Vsb24gbGUgdHlwZSwgZXQgdW5lIGNyb2l4IHBvdXIgZmVybWVyIChsYSB0b3VjaGUgw4ljaGFwIG1hcmNoZSBhdXNzaSkuCgotICoqSW1hZ2VzKiogKHBuZywganBnLCBqcGVnLCBnaWYsIHdlYnAsIGJtcCwgYXZpZiwgaWNvKSA6IGFmZmljaMOpZXMgw6AgbGV1cgogICoqdGFpbGxlIGQnb3JpZ2luZSwgc2FucyByZWRpbWVuc2lvbm5lbWVudCoqIOKAlCBsYSB6b25lIGTDqWZpbGUgc2kgbCdpbWFnZSBlc3QKICBncmFuZGUuIExlcyBHSUYgYW5pbcOpcyBzJ2FuaW1lbnQuIEJvdXRvbiAqKlTDqWzDqWNoYXJnZXIqKi4KLSAqKlNWRyoqIDogKipleMOpY3V0w6kqKiwgZG9uYyBsZXMgYW5pbWF0aW9ucyBldCBzY3JpcHRzIGludGVybmVzIGZvbmN0aW9ubmVudCwKICBkYW5zIHVuZSBmZW7DqnRyZSBpc29sw6llLiBCb3V0b25zICoqVm9pciBsZSBjb2RlKiogLyAqKkFwZXLDp3UqKiBldCAqKkNvcGllcioqLgotICoqVmlkw6lvcyoqIChtcDQsIHdlYm0sIG9ndiwgbW92LCBtNHYpIDogbGVjdGV1ciBpbnTDqWdyw6kgYXZlYyBjb250csO0bGVzCiAgKGxlY3R1cmUsIHBhdXNlLCB2b2x1bWUsIHBsZWluIMOpY3JhbiwgYmFycmUgZGUgcHJvZ3Jlc3Npb24pLCDDoCBsYSByw6lzb2x1dGlvbgogIGQnb3JpZ2luZS4gQm91dG9uICoqVMOpbMOpY2hhcmdlcioqLgotICoqQXVkaW8qKiAobXAzLCB3YXYsIG9nZywgbTRhLCBmbGFjLCBhYWMsIG9wdXMpIDogbGVjdGV1ciBhdWRpbyBhdmVjCiAgY29udHLDtGxlcy4gQm91dG9uICoqVMOpbMOpY2hhcmdlcioqLgotICoqUERGKiogOiB2aXNpb25uZXVzZSBQREYgaW50w6lncsOpZSAocGFnZXMsIHpvb20sIGTDqWZpbGVtZW50KS4gQm91dG9uCiAgKipUw6lsw6ljaGFyZ2VyKiouCi0gKipXb3JkICguZG9jeCkqKiA6IGxlIGRvY3VtZW50IGVzdCByZW5kdSBlbiB0ZXh0ZSBldCBtaXNlIGVuIHBhZ2UgZGFucyBsYQogIHBhZ2UuIEJvdXRvbiAqKlTDqWzDqWNoYXJnZXIqKi4KLSAqKlBvd2VyUG9pbnQgKC5wcHR4LCAucHB0KSoqIDogbGEgcHLDqXNlbnRhdGlvbiBlc3QgKipjb252ZXJ0aWUgZW4gUERGIHBhciBsYQogIFZNKiogKExpYnJlT2ZmaWNlKSBwdWlzIGFmZmljaMOpZSA7IGxhIGNvbnZlcnNpb24gcHJlbmQgcXVlbHF1ZXMgc2Vjb25kZXMuCiAgQm91dG9uICoqVMOpbMOpY2hhcmdlcioqLgotICoqRXhjZWwgKC54bHN4LCAueGxzKSoqIDogbGVzIGZldWlsbGVzIHNvbnQgYWZmaWNow6llcyBzb3VzIGZvcm1lIGRlIHRhYmxlYXV4CiAgKGp1c3F1J8OgIDQgZmV1aWxsZXMpLiBCb3V0b24gKipUw6lsw6ljaGFyZ2VyKiouCi0gKipIVE1MKiogOiBhcGVyw6d1ICoqZXjDqWN1dMOpKiogZGFucyB1bmUgZmVuw6p0cmUgaXNvbMOpZSwgYXZlYyBzZXMgZmljaGllcnMKICBgLmNzc2AgZXQgYC5qc2Agdm9pc2lucyBjaGFyZ8OpcyBkZXB1aXMgbGEgVk0g4oCUIHVuIGBpbmRleC5odG1sYCBhY2NvbXBhZ27DqSBkZQogIGBzdHlsZS5jc3NgIGV0IGBzY3JpcHQuanNgIGZvbmN0aW9ubmUgZG9uYyBjb21tZSB1biB2cmFpIHNpdGUuIEJvdXRvbnMKICAqKlZvaXIgbGUgY29kZSoqIC8gKipBcGVyw6d1KiogZXQgKipDb3BpZXIgbGUgY29kZSoqLgotICoqTWFya2Rvd24qKiA6IHJlbmR1ICh0aXRyZXMsIGxpc3RlcywgdGFibGVhdXgsIGNvZGUsIGZvcm11bGVzIExhVGVYKS4KICBCb3V0b25zICoqVGV4dGUgYnJ1dCoqIC8gKipBcGVyw6d1KiogZXQgKipDb3BpZXIqKi4KLSAqKkxhVGVYICgudGV4KSoqIDogbGVzIGZvcm11bGVzIHNvbnQgY29tcG9zw6llcyBldCBhZmZpY2jDqWVzLiBCb3V0b25zCiAgKipUZXh0ZSBicnV0KiogLyAqKkFwZXLDp3UqKiBldCAqKkNvcGllcioqLgotICoqSmF2YVNjcmlwdCoqIDogY29kZSBjb2xvcsOpLCBwbHVzIGRldXggYm91dG9ucyDigJQKICAqKkV4w6ljdXRlcioqIChsYW5jZSBsZSBzY3JpcHQgZGFucyB1bmUgZmVuw6p0cmUgaXNvbMOpZSBldCBhZmZpY2hlIGNlIHF1ZQogIGBjb25zb2xlLmxvZ2AgcHJvZHVpdCwgYWluc2kgcXVlIGxlcyBlcnJldXJzKSBldCAqKkxhbmNlciBzdXIgbGEgVk0qKi4KLSAqKlB5dGhvbioqIDogY29kZSBjb2xvcsOpIGV0IGJvdXRvbiAqKkxhbmNlciBzdXIgbGEgVk0qKi4gU2kgbGUgc2NyaXB0CiAgZGVzc2luZSBhdmVjIG1hdHBsb3RsaWIsIGxhIGZpZ3VyZSBlc3QgcmVudm95w6llIGV0IGFmZmljaMOpZSBlbiBpbWFnZS4KLSAqKlNoZWxsICguc2gsIC5iYXNoKSoqIDogY29kZSBjb2xvcsOpIGV0IGJvdXRvbiAqKkxhbmNlciBzdXIgbGEgVk0qKi4KLSAqKkF1dHJlcyBmaWNoaWVycyB0ZXh0ZSoqIDogYWZmaWNow6lzICoqaW50w6lncmFsZW1lbnQqKiAoYXVjdW5lIHRyb25jYXR1cmUpLAogIGF2ZWMgKipDb3BpZXIqKi4gTGVzIHRyw6hzIGdyb3MgZmljaGllcnMgZMOpZmlsZW50IHNpbXBsZW1lbnQgZGFucyBsYQogIHZpc2lvbm5ldXNlLgotICoqRmljaGllcnMgbG91cmRzKiogKHBsdXNpZXVycyBtw6lnYW9jdGV0cykgOiBsYSB2aXNpb25uZXVzZSBsZXMgbGl0ICoqcGFyCiAgdHJhbmNoZXMqKiwgZW4gYWZmaWNoYW50IGxhIHByb2dyZXNzaW9uLCBldCBuZSByZXN0ZSBqYW1haXMgYmxvcXXDqWUuIEF1LWRlbMOgCiAgZGUgMTIwIE1vLCBlbGxlIGxlIGRpdCBjbGFpcmVtZW50IGV0IHByb3Bvc2UgbGUgKipUw6lsw6ljaGFyZ2VtZW50KiouCgojIyA2LiDCqyBMYW5jZXIgc3VyIGxhIFZNIMK7CgpDZSBib3V0b24gZW52b2llIGxlIGZpY2hpZXIgYWZmaWNow6kgZGFucyBsYSBWTSBldCBsJ2V4w6ljdXRlIHZpYSBzb24gdGVybWluYWwKKGBweXRob24zYCwgYG5vZGVgIG91IGBiYXNoYCksIGVuIGFycmnDqHJlLXBsYW4sIHB1aXMgcmVudm9pZSBsZSByw6lzdWx0YXQgOgpzb3J0aWUgc3RhbmRhcmQsIGVycmV1cnMsIGNvZGUgZGUgc29ydGllLCBldCBsJ2ltYWdlIHByb2R1aXRlIGxlIGNhcyDDqWNow6lhbnQuCkMnZXN0IGxhIHB1aXNzYW5jZSBkZSBjYWxjdWwgZGUgbGEgVk0gcXVpIGVzdCB1dGlsaXPDqWUsIHBhcyBjZWxsZSBkdQpuYXZpZ2F0ZXVyLiBVbiBzY3JpcHQgbG9uZyBwZXV0IHByZW5kcmUgZHUgdGVtcHMgOiBsZSByw6lzdWx0YXQgYXJyaXZlIHF1YW5kCmwnZXjDqWN1dGlvbiBlc3QgZmluaWUuCgojIyA3LiBTYSBmaWNoZSBwZXJzb25uZWxsZSAodXNlci5tZCkKCi0gYHVzZXIubWRgIGVzdCBsZSAqKnNldWwgZmljaGllciBxdSdpbCBwZXV0IG1vZGlmaWVyIGx1aS1tw6ptZSoqLgotIEVuIGwnb3V2cmFudCwgdW4gYm91dG9uICoqY3JheW9uKiogYXBwYXJhw650LiBJbCBkb25uZSBhY2PDqHMgw6AgdW4gZm9ybXVsYWlyZSA6CiAgKipOb20qKiwgKipBZHJlc3NlIGUtbWFpbCoqLCAqKsOCZ2UqKiwgKipNw6l0aWVyKiosIHVuZSB6b25lICoqSW5mb3JtYXRpb25zCiAgc3VwcGzDqW1lbnRhaXJlcyoqIGV0IHVuZSB6b25lICoqSW5zdHJ1Y3Rpb25zIHBlcnNvbm5hbGlzw6llcyoqLgotIEVuIMOpZGl0aW9uLCBkZXV4IGJvdXRvbnMgcmVtcGxhY2VudCBsZSBjcmF5b24gOiAqKuKclSoqIGFubnVsZSBzYW5zIHJpZW4KICBjaGFuZ2VyLCAqKuKckyoqIGVucmVnaXN0cmUgZGFucyBsYSBWTS4KLSBMZSBmaWNoaWVyIGVzdCBwcm90w6lnw6kgOiBzJ2lsIGVzdCBzdXBwcmltw6ksIGlsIGVzdCByZWNyw6nDqSBhdXRvbWF0aXF1ZW1lbnQKICAow6AgcGFydGlyIGRlIHNhIHNhdXZlZ2FyZGUgc2kgcG9zc2libGUpLgotIExpcy1sYSBlbiBkw6lidXQgZGUgbWlzc2lvbiBldCAqKnN1aXMgc2VzIGluc3RydWN0aW9ucyBwZXJzb25uYWxpc8OpZXMqKi4KICBOZSBsYSBtb2RpZmllIGphbWFpcy4KCiMjIDguIExlcyBwYXJhbcOodHJlcyBkZSBzb24gY29tcHRlCgpFbiBjbGlxdWFudCBzdXIgc29uIGF2YXRhciwgaWwgb3V2cmUgdW4gbWVudSBwdWlzIGRlcyBmZW7DqnRyZXMgcXVpIGx1aQpwZXJtZXR0ZW50IGRlIDoKCi0gKipjaGFuZ2VyIHNvbiBtb3QgZGUgcGFzc2UqKiAob3UgZW4gKipkw6lmaW5pciB1bioqIHNpIHNvbiBjb21wdGUgYSDDqXTDqSBjcsOpw6kKICBhdmVjIEdvb2dsZSkgOwotICoqY2hhbmdlciBzb24gYWRyZXNzZSBlLW1haWwqKiA7Ci0gKipsaWVyIG91IGTDqWxpZXIgc29uIGNvbXB0ZSBHb29nbGUqKiA7Ci0gKipjaGFuZ2VyIHNhIHBob3RvIGRlIHByb2ZpbCoqIDsKLSAqKnN1cHByaW1lciBzb24gY29tcHRlKiog4oCUIGxhIHN1cHByZXNzaW9uIGVzdCBkaXJlY3RlLCBzYW5zIGNvbmZpcm1hdGlvbgogIEdvb2dsZS4KClN1ciB0b3VzIGxlcyBjaGFtcHMgbW90IGRlIHBhc3NlLCB1biBib3V0b24gKirFk2lsKiogYWZmaWNoZSBvdSBtYXNxdWUgY2UgcXVpCmVzdCBzYWlzaS4gVHUgbidhcyBhdWN1biBhY2PDqHMgw6AgY2VzIHLDqWdsYWdlcyA6IG4nZXNzYWllIHBhcyBkZSBsZXMgbWFuaXB1bGVyCmV0IG5lIGx1aSBkZW1hbmRlIGphbWFpcyBzZXMgaWRlbnRpZmlhbnRzLgoKIyMgOS4gTGVzIHBpw6hjZXMgam9pbnRlcwoKLSBMZSBib3V0b24gKip0cm9tYm9uZSoqIHBlcm1ldCBkZSBqb2luZHJlIHVuZSBpbWFnZSBvdSB1biBmaWNoaWVyIMOgIHNvbgogIG1lc3NhZ2UsIGQnZW52b3llciB1bmUgaW1hZ2Ugbm9pcmUsIG91IGRlICoqcGFydGFnZXIgc29uIMOpY3JhbioqICh1bmUgY2FwdHVyZQogIGRlIHNvbiBwcm9wcmUgw6ljcmFuLCBwYXMgY2VsdWkgZGUgbGEgVk0pLgotIExlICoqbWljcm8qKiBzZXJ0IMOgIGRpY3RlciBzb24gbWVzc2FnZSA6IGwnYXVkaW8gZXN0IHRyYW5zY3JpdC4KCiMjIDEwLiBFbiByw6lzdW3DqSwgcG91ciB0b2kKCi0gUG91ciBxdSdpbCB2b2llIHF1ZWxxdWUgY2hvc2UgOiBkw6lwb3NlIGxlIGZpY2hpZXIgZGFucwogIGAvaG9tZS9rYXNtLXVzZXIvdXNlci9gLiBJbCBsJ291dnJpcmEgZCd1biBjbGljIGRhbnMgbGEgdmlzaW9ubmV1c2UuCi0gUG91ciBxdSdpbCBjb21wcmVubmUgY2UgcXVlIHR1IGZhaXMgOiBub21tZSBiaWVuIHRlcyBmaWNoaWVycywgdGVybWluZSBwYXIKICB1biBtZXNzYWdlIGNsYWlyIGluZGlxdWFudCBxdW9pIG91dnJpciBldCBwb3VycXVvaS4KLSBJbCBuZSBwZXV0IHBhcyBhZ2lyIHN1ciBsYSBWTSA6IHRvdXQgY2UgcXVpIG7DqWNlc3NpdGUgdW5lIGNvbW1hbmRlLCB1bmUKICBpbnN0YWxsYXRpb24gb3UgdW4gY2xpYyBzdXIgbGUgYnVyZWF1IHBhc3NlIHBhciB0b2kuCgo8IS0tIHZtLWRvYy12ZXJzaW9uOiAyMDI2LTEwLTE1LjEgLS0+Cg==";
const USER_DOC_B64 = "IyB1c2VyLm1kIOKAlCBGaWNoZSB1dGlsaXNhdGV1cgoKLSAqKk5vbSA6KiogSmVhbkh1ZwotICoqQWRyZXNzZSBlLW1haWwgOioqIGh1Z2R1Nzc3NzdAZ21haWwuY29tCi0gKirDgmdlIDoqKiAobm9uIHJlbnNlaWduw6kpCi0gKipNw6l0aWVyIDoqKiAobm9uIHJlbnNlaWduw6kpCgojIyBJbmZvcm1hdGlvbnMgc3VwcGzDqW1lbnRhaXJlcwoKKMOJY3JpdmV6IGljaSBjZSBxdWUgdm91cyB2b3VsZXogcXVlIGwnYWdlbnQgc2FjaGUgc3VyIHZvdXMgOiBwcsOpZsOpcmVuY2VzLCBoYWJpdHVkZXMsIHByb2pldHMgZW4gY291cnMuKQoKIyMgSW5zdHJ1Y3Rpb25zIHBlcnNvbm5hbGlzw6llcwoKKFZvcyBpbnN0cnVjdGlvbnMgcGVyc29ubmVsbGVzIHBvdXIgbCdhZ2VudCA6IHRvbiwgZm9ybWF0LCBjZSBxdSdpbCBkb2l0IHRvdWpvdXJzIGZhaXJlIG91IG5lIGphbWFpcyBmYWlyZS4pCg==";
async function vmExecCmd(env, command) {
  const rq = new Request("https://vmai.hugdu77777.workers.dev/api/linux/exec", {
    method: "POST",
    headers: { "Authorization": "Bearer " + (env.VMAI_PASS || ""), "Content-Type": "application/json" },
    body: JSON.stringify({ command, timeout: 60 }),
  });
  const r = env.VMAI && typeof env.VMAI.fetch === "function" ? await env.VMAI.fetch(rq) : await fetch(rq);
  try { return await r.json(); } catch { return {}; }
}
/* ===================== Z-Canal : les canaux ==============================
   Un canal relie un agent IA — n'importe lequel — à l'application. Il vit dans
   Firebase et se pilote par son adresse, sans mot de passe : l'identifiant
   (12 caractères au hasard) EST la clé.

     GET  /canal                      la documentation
     GET  /canal/<id>                 la fiche du canal et ses adresses
     GET  /canal/<id>/install         les instructions d'installation
            ?format=python|js|sh|ts   le script à écrire dans un fichier
            ?raw=1                    uniquement le script
            ?doc=1                    uniquement la documentation (.md)
     POST /canal/<id>/write           l'agent écrit un message
     GET  /canal/<id>/read?n=         l'historique récent (ne marque rien)
     GET  /canal/<id>/wait?timeout=   les messages NON LUS (et les marque)
     GET  /canal/<id>/state           l'état du canal

   Le rôle « user » n'est écrit que par l'application, le rôle « agent » que par
   ce canal : un agent ne peut donc pas se répondre à lui-même. Un message reçu
   reste « non lu » tant qu'un /wait ne l'a pas rendu — c'est le badge que
   l'application affiche, et qui passe à « lu ! ». L'attente est gérée par le
   script de l'agent : ici, /wait ne bloque jamais plus de 25 secondes.
*/
const CANAL_COLL = "canaux";
const CANAL_MSG_MAX = 400;

function b64Text(b64) {
  try {
    const bin = atob(b64);
    const bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    return new TextDecoder().decode(bytes);
  } catch (e) { return ""; }
}
function zcanalFill(txt, base, id, name, mode) {
  return String(txt)
    .replace(/__BASE__/g, base)
    .replace(/__ID__/g, id)
    .replace(/__NAME__/g, name)
    .replace(/__MODE__/g, mode);
}
function zcanalScriptOf(format, base, id, name, mode) {
  const f = String(format === undefined || format === null ? "python" : format).toLowerCase();
  const b64 = (f === "js" || f === "node" || f === "javascript") ? ZCANAL_JS_B64
    : (f === "sh" || f === "bash" || f === "shell") ? ZCANAL_SH_B64
    : (f === "ts" || f === "typescript") ? ZCANAL_TS_B64
    : ZCANAL_PY_B64;
  return zcanalFill(b64Text(b64), base, id, name, mode);
}
function textOut(s, status) {
  return new Response(String(s), {
    status: status || 200,
    headers: { "Content-Type": "text/plain; charset=utf-8", "Cache-Control": "no-store" },
  });
}
function htmlOut(s, status) {
  return new Response(String(s), {
    status: status || 200,
    headers: { "Content-Type": "text/html; charset=utf-8", "Cache-Control": "no-store" },
  });
}
function escHtml(s) {
  return String(s == null ? "" : s).replace(/&/g, "&amp;").replace(/</g, "&lt;")
    .replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}

/* --- Firebase, côté canal ---------------------------------------------
   Où vivent les messages d'un canal ? Le canal par défaut, « arena », garde
   son historique d'origine (chan/main/box) : c'est le canal de la discussion
   épinglée, et rien ne doit bouger chez lui. Les autres canaux vivent dans
   canaux/<id>/messages. Du point de vue d'un agent, c'est transparent : les
   mêmes adresses, les mêmes champs. */
function canalBase(id) {
  return (id === "arena")
    ? { doc: "/chan/main", msgs: "/chan/main/box" }
    : { doc: "/canaux/" + encodeURIComponent(id), msgs: "/canaux/" + encodeURIComponent(id) + "/messages" };
}
/* Un message écrit par un agent : « arena » pour le canal par défaut (l'app
   l'écoute sous ce rôle depuis toujours), « agent » ailleurs. */
function canalRole(id) { return id === "arena" ? "arena" : "agent"; }
/* Le document qui PORTE la collection : chan/main pour « arena », le document
   du canal sinon. Les requêtes filtrées (runQuery, agrégations) partent de là. */
/* Le corps d'une réponse Firestore, en texte : certains environnements de test
   n'exposent que json(), d'autres que text(). On accepte les deux. */
async function fsBody(r) {
  try { if (typeof r.text === "function") return await r.text(); } catch (e) { /* essaie json */ }
  try { return JSON.stringify(await r.json()); } catch (e) { return ""; }
}
function canalParent(id) {
  return (id === "arena")
    ? { parent: "chan/main", coll: "box" }
    : { parent: "canaux/" + encodeURIComponent(id), coll: "messages" };
}
async function canalDoc(env, id) {
  const tok = await arenaToken(env);
  if (!tok) return { ok: false, error: "compte Firebase non configuré" };
  if (id === "arena") {
    /* Le canal par défaut existe toujours : sa fiche est facultative. */
    const ra = await fetch(ARENA_FS + canalBase(id).doc,
      { headers: { Authorization: "Bearer " + tok, "User-Agent": BROWSER_UA } });
    const ja = ra.ok ? await ra.json().catch(() => null) : null;
    const f = (ja && ja.fields) || {};
    const c = fsDoc(ja || { fields: {} });
    return { ok: true, canal: { id: "arena", name: String(c.name || "arena"),
      mode: String(f.mode ? fsVal(f.mode) : "agent"), pinned: true, createdAt: Number(c.createdAt || 0) } };
  }
  const r = await fetch(ARENA_FS + canalBase(id).doc,
    { headers: { Authorization: "Bearer " + tok, "User-Agent": BROWSER_UA } });
  if (r.status === 404) return { ok: false, missing: true, error: "canal inconnu" };
  const j = await r.json().catch(() => null);
  if (!r.ok || !j) return { ok: false, error: "lecture refusée (" + r.status + ")" };
  return { ok: true, canal: fsDoc(j) };
}
async function canalList(env, id, n) {
  const tok = await arenaToken(env);
  if (!tok) return { ok: false, error: "compte Firebase non configuré" };
  const size = Math.min(200, Math.max(1, Number(n) || 30));
  /* DESCENDANT : on veut les messages LES PLUS RÉCENTS. Avec un ordre croissant,
     la page « n » rendait les plus anciens — c'est pourquoi une commande écrite
     dans le canal ne voyait jamais revenir son résultat. */
  const r = await fetch(ARENA_FS + canalBase(id).msgs
    + "?orderBy=ts%20desc&pageSize=" + size,
    { headers: { Authorization: "Bearer " + tok, "User-Agent": BROWSER_UA } });
  const j = await r.json().catch(() => null);
  if (!r.ok || !j) return { ok: false, error: "lecture refusée (" + r.status + ")" };
  const docs = (j.documents || []).map((d) => fsDoc(d));
  docs.sort((a, b) => (a.ts || 0) - (b.ts || 0));
  return { ok: true, messages: docs };
}
async function canalAppend(env, id, fields) {
  const tok = await arenaToken(env);
  if (!tok) return { ok: false, error: "compte Firebase non configuré" };
  const body = {};
  for (const k of Object.keys(fields)) body[k] = fsNewVal(fields[k]);
  const r = await fetch(ARENA_FS + canalBase(id).msgs, {
    method: "POST",
    headers: { "Content-Type": "application/json", Authorization: "Bearer " + tok, "User-Agent": BROWSER_UA },
    body: JSON.stringify({ fields: body }),
  });
  const j = await r.json().catch(() => null);
  if (!r.ok || !j) return { ok: false, error: "écriture refusée (" + r.status + ")" };
  return { ok: true, id: String(j.name || "").split("/").pop() };
}
async function canalUnread(env, id) {
  /* Les messages non lus, par une requête FILTRÉE sur state == "new" :
     Firestore ne facture que les documents rendus, donc une attente qui ne
     trouve rien ne coûte AUCUNE lecture. (Avant, chaque tour relisait les 200
     derniers messages : 41 documents × 24 tours/minute — le quota quotidien du
     compte a été épuisé comme ça, et Firestore a répondu 429.) */
  const tok = await arenaToken(env);
  if (!tok) return { ok: false, error: "compte Firebase non configuré" };
  const p = canalParent(id);
  const body = { structuredQuery: {
    from: [{ collectionId: p.coll }],
    where: { fieldFilter: { field: { fieldPath: "state" }, op: "EQUAL", value: { stringValue: "new" } } },
    limit: 50,
  } };
  const r = await fetch(ARENA_FS + "/" + p.parent + ":runQuery", {
    method: "POST",
    headers: { "Content-Type": "application/json", Authorization: "Bearer " + tok, "User-Agent": BROWSER_UA },
    body: JSON.stringify(body),
  });
  const txt = await fsBody(r);
  let j = null; try { j = JSON.parse(txt); } catch (e) { j = null; }
  if (!r.ok || !Array.isArray(j)) {
    ARENA_FS_ERR = { at: Date.now(), statut: r.status, corps: String(txt).slice(0, 300) };
    return { ok: false, error: "lecture refusée (" + r.status + ") " + String(txt).slice(0, 160) };
  }
  const docs = j.filter((x) => x && x.document).map((x) => fsDoc(x.document));
  const unread = docs.filter((m) => m.role === "user" && m.state === "new");
  unread.sort((a, b) => (a.ts || 0) - (b.ts || 0));
  return { ok: true, messages: unread, unread: unread.length, total: docs.length };
}
/* Le nombre de messages, par AGRÉGATION : une requête de comptage coûte une
   lecture pour mille entrées, là où relire la collection en coûtait 200. */
async function canalCount(env, id) {
  const tok = await arenaToken(env);
  if (!tok) return null;
  const p = canalParent(id);
  const body = { structuredAggregationQuery: {
    structuredQuery: { from: [{ collectionId: p.coll }] },
    aggregations: [{ count: {} }],
  } };
  const r = await fetch(ARENA_FS + "/" + p.parent + ":runAggregationQuery", {
    method: "POST",
    headers: { "Content-Type": "application/json", Authorization: "Bearer " + tok, "User-Agent": BROWSER_UA },
    body: JSON.stringify(body),
  });
  const txt = await fsBody(r);
  let j = null; try { j = JSON.parse(txt); } catch (e) { j = null; }
  if (!r.ok || !Array.isArray(j)) { ARENA_FS_ERR = { at: Date.now(), statut: r.status, corps: String(txt).slice(0, 300) }; return null; }
  for (const x of j) {
    const f = x && x.result && x.result.aggregateFields;
    if (f) { const k = Object.keys(f)[0]; if (k) return Number(f[k].integerValue || 0); }
  }
  return null;
}
/* Le résultat d'une commande : le message « tool » écrit en réponse au message
   « call » (même champ req). Une requête filtrée sur req — un seul champ, donc
   aucun index composite à créer. */
async function canalResultFor(env, id, reqId) {
  if (!reqId) return null;
  const tok = await arenaToken(env);
  if (!tok) return null;
  const p = canalParent(id);
  const body = { structuredQuery: {
    from: [{ collectionId: p.coll }],
    where: { fieldFilter: { field: { fieldPath: "req" }, op: "EQUAL", value: { stringValue: String(reqId) } } },
    limit: 10,
  } };
  const r = await fetch(ARENA_FS + "/" + p.parent + ":runQuery", {
    method: "POST",
    headers: { "Content-Type": "application/json", Authorization: "Bearer " + tok, "User-Agent": BROWSER_UA },
    body: JSON.stringify(body),
  });
  const txt = await fsBody(r);
  let j = null; try { j = JSON.parse(txt); } catch (e) { j = null; }
  if (!r.ok || !Array.isArray(j)) return null;
  const docs = j.filter((x) => x && x.document).map((x) => fsDoc(x.document));
  const tools = docs.filter((m) => m.role === "tool");
  tools.sort((a, b) => (a.ts || 0) - (b.ts || 0));
  return tools.length ? tools[tools.length - 1] : null;
}
async function canalMarkRead(env, id, msgs) {
  for (const m of msgs) {
    if (!m.id) continue;
    try { await fbWrite(env, canalBase(id).msgs.replace(/^\//, "") + "/" + m.id, { state: "read", readAt: Date.now() }); }
    catch (e) { /* on réessaiera au prochain /wait */ }
  }
}
/* L'attente : on interroge la requête filtrée toutes les 5 s, jamais plus de
   25 s d'un coup. La requête ne rend que les messages non lus : quand il n'y en
   a pas, elle ne coûte aucune lecture. */
async function canalWait(env, id, timeoutMs) {
  const t0 = Date.now();
  for (;;) {
    const u = await canalUnread(env, id);
    if (!u.ok) return u;
    if (u.messages.length) { await canalMarkRead(env, id, u.messages); return u; }
    if (Date.now() - t0 + 5000 > timeoutMs) return { ok: true, messages: [], unread: 0, total: u.total };
    await new Promise((r) => setTimeout(r, 5000));
  }
}
/* Vue publique d'un message : le texte, jamais rien d'autre. */
function canalPublic(m) {
  return { id: m.id, role: m.role, ts: m.ts || 0, text: String(m.text == null ? "" : m.text),
    kind: m.kind || "text", state: m.state || "" };
}

/* --- pages ------------------------------------------------------------- */
function canalShell(title, body) {
  return "<!DOCTYPE html><html lang=\"fr\"><head><meta charset=\"utf-8\">"
    + "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
    + "<title>" + escHtml(title) + "</title><style>"
    + ":root{--ink:#111113;--ink2:#55565c;--ink3:#8b8c93;--line:#e8e8ec}"
    + "*{box-sizing:border-box}body{margin:0;padding:38px 22px;background:#fff;color:var(--ink);"
    + "font:15px/1.65 -apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif}"
    + ".wrap{max-width:720px;margin:0 auto}h1{font-size:22px;letter-spacing:-.02em;margin:0 0 6px}"
    + "h2{font-size:15px;margin:26px 0 8px}p{margin:0 0 12px;color:var(--ink2)}"
    + "code,pre{font-family:ui-monospace,Menlo,Consolas,monospace}"
    + "code{background:#f4f4f6;padding:.13em .38em;border-radius:4px;font-size:.88em}"
    + "pre{background:#0d0d0f;color:#e6e6e9;padding:13px 15px;border-radius:10px;overflow-x:auto;font-size:12.5px}"
    + "table{border-collapse:collapse;width:100%;font-size:13.5px;display:block;overflow-x:auto}"
    + "th,td{border:1px solid var(--line);padding:6px 10px;text-align:left}th{background:#fafafb}"
    + ".tag{display:inline-block;font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.5px;"
    + "border:1px solid var(--line);border-radius:999px;padding:2px 9px;color:var(--ink2)}"
    + "</style></head><body><div class=\"wrap\">" + body + "</div></body></html>";
}
function canalDocPage(base) {
  const g = "<code>" + escHtml(base) + "/&lt;id&gt;/";
  return canalShell("z-canal — documentation", ""
    + "<h1>z-canal</h1>"
    + "<p>Un <b>canal</b> relie un agent IA — n'importe lequel — à l'application. "
    + "L'utilisateur écrit dans l'application ; l'agent reçoit ses messages, peut répondre, "
    + "et si le canal est en mode <b>agent</b>, exécuter des commandes sur la VM et renvoyer leur résultat.</p>"
    + "<h2>Les adresses</h2><table><tr><th>Adresse</th><th>Ce qu'elle fait</th></tr>"
    + "<tr><td>" + g + "write</code></td><td>l'agent écrit un message "
    + "(<code>{\"text\": \"…\", \"kind\": \"text\"|\"call\"}</code>)</td></tr>"
    + "<tr><td>" + g + "read?n=30</code></td><td>l'historique récent, sans rien marquer</td></tr>"
    + "<tr><td>" + g + "wait?timeout=25</code></td><td><b>les messages non lus</b>, qu'elle marque « lu »</td></tr>"
    + "<tr><td>" + g + "state</code></td><td>l'état du canal</td></tr>"
    + "<tr><td>" + g + "install</code></td><td>les instructions d'installation</td></tr></table>"
    + "<h2>Installer</h2><pre>curl -s \"" + escHtml(base) + "/&lt;id&gt;/install\"\n"
    + "curl -s \"" + escHtml(base) + "/&lt;id&gt;/install?raw=1&amp;format=python\" -o z-canal.py\n"
    + "curl -s \"" + escHtml(base) + "/&lt;id&gt;/install?doc=1\" -o z-canal.md</pre>"
    + "<p>Formats disponibles : <code>python</code> (défaut), <code>js</code>, <code>sh</code>, <code>ts</code>. "
    + "Un agent peut aussi écrire son propre client après avoir lu la documentation.</p>"
    + "<h2>À savoir</h2>"
    + "<p><b>Aucun mot de passe :</b> l'identifiant est la clé. <b>Un message ne revient jamais deux fois :</b> "
    + "le canal le marque « lu » dès qu'un <code>/wait</code> le rend — l'application affiche alors « lu ! ».</p>"
    + "<p><b>C'est le script qui gère l'attente</b>, pas le serveur (25 s au maximum) : "
    + "<code>wait</code> → rien ? pause de 5 s → <code>wait</code>. Rester au-dessus de 10 secondes entre "
    + "deux attentes garde une consommation très faible.</p>"
    + "<p>Les commandes ne s'exécutent que si l'application est ouverte : c'est elle qui parle à la VM. "
    + "Sinon le message attend dans le canal.</p>");
}
function canalOwnPage(base, c) {
  const id = escHtml(c.id);
  return canalShell("Canal « " + (c.name || c.id) + " »", ""
    + "<h1>Canal « " + escHtml(c.name || c.id) + " »</h1>"
    + "<p><span class=\"tag\">" + escHtml(c.mode || "chat") + "</span> "
    + "<code>" + id + "</code></p>"
    + "<h2>Ses adresses</h2><pre>" + escHtml(base) + "/" + id + "/write\n"
    + escHtml(base) + "/" + id + "/read\n" + escHtml(base) + "/" + id + "/wait\n"
    + escHtml(base) + "/" + id + "/install</pre>"
    + "<p><a href=\"" + escHtml(base) + "\">Documentation complète</a></p>");
}
function canalInstallText(env, base, c) {
  const id = String(c.id), name = String(c.name || id), mode = String(c.mode || "chat");
  const L = [];
  L.push("z-canal — installation du canal « " + name + " »");
  L.push("identifiant : " + id + "   ·   mode : " + mode);
  L.push("");
  L.push("Vous venez d'installer un canal de l'application AI Agent. Voici tout ce qu'il faut savoir.");
  L.push("");
  L.push("1. Créez le script (au format voulu) :");
  L.push("     curl -s \"" + base + "/" + id + "/install?raw=1&format=python\" -o z-canal.py");
  L.push("     (formats : python, js, sh, ts)");
  L.push("2. Créez la documentation à côté :");
  L.push("     curl -s \"" + base + "/" + id + "/install?doc=1\" -o z-canal.md");
  L.push("3. Veillez — c'est VOTRE script qui gère l'attente :");
  L.push("     python3 z-canal.py wait      # rend la main dès qu'un message arrive");
  L.push("     python3 z-canal.py read      # l'historique, sans rien marquer");
  L.push("     python3 z-canal.py write \"texte\"");
  L.push("     python3 z-canal.py write --json '{\"action\":\"bash\",\"command\":\"ls -la ~\"}'");
  L.push("");
  L.push("Les quatre adresses du canal :");
  L.push("     POST " + base + "/" + id + "/write   {\"text\":\"…\",\"kind\":\"text\"|\"call\"}");
  L.push("     GET  " + base + "/" + id + "/read?n=30");
  L.push("     GET  " + base + "/" + id + "/wait?timeout=25   -> les messages non lus, puis marqués « lu »");
  L.push("     GET  " + base + "/" + id + "/state");
  L.push("");
  L.push("Règles à ne pas oublier :");
  L.push(" - Aucun mot de passe : l'identifiant est la clé, ne le diffusez pas.");
  L.push(" - Un message ne revient jamais deux fois (le canal le marque « lu »).");
  L.push(" - Le serveur ne bloque pas plus de 25 s : bouclez côté script (wait -> rien ? pause de 5 s).");
  L.push(" - Restez au-dessus de 10 s entre deux attentes : le plan gratuit accorde 100 000 requêtes/jour.");
  L.push(" - Les commandes ne s'exécutent que si l'application est OUVERTE (c'est elle qui parle à la VM).");
  L.push(" - Pour agir : kind « call » + {\"action\":\"bash\",\"command\":\"…\"} ou "
    + "{\"action\":\"fetch\",\"url\":\"…\"}. Le résultat revient en message « tool ».");
  if (mode === "agent") {
    L.push("");
    L.push("Ce canal est en mode AGENT : vous êtes l'agent de l'application. Vous recevez les messages de");
    L.push("l'utilisateur, vous exécutez ce qui doit l'être sur la VM, et vous répondez dans le canal.");
    L.push("Voici les documents de l'agent, comme pour un agent normal.");
    L.push("");
    L.push("======================= agent.md =======================");
    L.push(b64Text(AGENT_DOC_B64));
    L.push("======================= infos.md =======================");
    L.push(b64Text(INFOS_DOC_B64));
    L.push("======================= user.md ========================");
    L.push(b64Text(USER_DOC_B64));
  } else {
    L.push("");
    L.push("Ce canal est en mode CHAT : vous conversez avec l'utilisateur. Aucune commande n'est");
    L.push("exécutée, aucun document de la VM n'est fourni.");
  }
  L.push("");
  L.push("Documentation complète : " + base);
  return L.join("\n");
}
/* Une adresse : /canal, /canal/<id>, /canal/<id>/<action>. */
async function canalRoute(env, request, url) {
  const base = url.origin + "/canal";
  const rest = url.pathname.slice("/canal".length).replace(/^\/+/, "");
  if (!rest) return htmlOut(canalDocPage(base));
  const parts = rest.split("/");
  let id = decodeURIComponent(parts[0] || "");
  let action = String(parts[1] || "").toLowerCase();
  if (id === "install" && !action) {
    return textOut("Veuillez fournir un id.\n\nL'installation s'écrit : " + base
      + "/<id>/install\nLa documentation complète : " + base + "\n");
  }
  const got = await canalDoc(env, id);
  if (!got.ok) {
    const msg = got.missing ? "canal inconnu (identifiant « " + id + " »)" : String(got.error || "erreur");
    if (action === "install") return textOut("Veuillez fournir un id.\n\n(" + msg + ")\n");
    if (action) return json({ ok: false, error: msg }, got.missing ? 404 : 502);
    return htmlOut(canalShell("canal inconnu", "<h1>Canal inconnu</h1><p>L'identifiant <code>"
      + escHtml(id) + "</code> ne correspond à aucun canal.</p><p><a href=\"" + escHtml(base)
      + "\">Documentation</a></p>"), 404);
  }
  const c = got.canal;
  if (action === "install") {
    const q = url.searchParams;
    if (q.get("doc")) {
      return textOut(zcanalFill(b64Text(ZCANAL_DOC_B64), base, id, String(c.name || id), String(c.mode || "chat")));
    }
    if (q.get("raw")) {
      return textOut(zcanalScriptOf(q.get("format"), base, id, String(c.name || id), String(c.mode || "chat")));
    }
    return textOut(canalInstallText(env, base, c));
  }
  if (action === "write") {
    let b = {};
    if (request.method === "POST") {
      try { b = await request.json(); } catch (e) { b = {}; }
    } else {
      b = { text: url.searchParams.get("text") || "" };
    }
    const text = String(b.text == null ? "" : b.text);
    if (!text) return json({ ok: false, error: "texte vide" }, 400);
    const kind = String(b.kind || "text").toLowerCase() === "call" ? "call" : "text";
    /* « wait » : le nombre de secondes pendant lesquelles le worker attend
       LUI-MÊME la réponse de la commande. Le write rend donc le résultat. */
    const waitS = Math.max(0, Math.min(25, Number(b.wait || url.searchParams.get("wait") || 0) || 0));
    const w = await canalAppend(env, id, {
      role: canalRole(id), text: text, ts: Date.now(), kind: kind, state: "new", src: "canal",
      req: String(b.req || ""), note: String(b.note || ""),
    });
    if (!w.ok) return json(w, 502);
    ARENA_LAST = { at: Date.now(), action: "canal-write", canal: id, kind: kind, taille: text.length };
    if (kind === "call" && waitS > 0) {
      const t0 = Date.now();
      let tool = null;
      while (Date.now() - t0 < waitS * 1000) {
        tool = await canalResultFor(env, id, w.id);
        if (tool) break;
        await new Promise((r) => setTimeout(r, 1200));
      }
      return json({ ok: true, id: w.id, canal: id, kind: kind,
        waited: Math.round((Date.now() - t0) / 100) / 10,
        tool: tool ? canalPublic(tool) : null,
        note: tool ? "" : "aucun résultat : l'application est-elle ouverte, et la discussion en mode agent ?" });
    }
    return json({ ok: true, id: w.id, canal: id, kind: kind, device: await arenaAppOnline(env) });
  }
  if (action === "read") {
    const l = await canalList(env, id, Number(url.searchParams.get("n") || 30));
    if (!l.ok) return json(l, 502);
    return json({ ok: true, canal: { id: id, name: c.name || id, mode: c.mode || "chat" },
      messages: l.messages.map(canalPublic) });
  }
  if (action === "wait") {
    const t = Math.min(25, Math.max(0, Number(url.searchParams.get("timeout") || 25)));
    const u = await canalWait(env, id, t * 1000);
    if (!u.ok) return json(u, 502);
    return json({ ok: true, canal: { id: id, name: c.name || id, mode: c.mode || "chat" },
      messages: u.messages.map(canalPublic), unread: u.messages.length });
  }
  if (action === "state" || !action) {
    if (!action) return htmlOut(canalOwnPage(base, c));
    const u = await canalUnread(env, id);
    const n = await canalCount(env, id);
    return json({ ok: true, canal: { id: id, name: c.name || id, mode: c.mode || "chat",
      pinned: !!c.pinned, createdAt: c.createdAt || 0, messages: (n == null ? 0 : n),
      unread: (u.ok ? u.messages.length : 0) },
      erreur: u.ok ? null : u.error, firestore: ARENA_FS_ERR });
  }
  return json({ ok: false, error: "action inconnue (write, read, wait, state, install)" }, 404);
}

/* Gabarits « z-canal » (base64) : posés sur le disque par scripts/embed-app.py. */
const ZCANAL_PY_B64 = "IyEvdXNyL2Jpbi9lbnYgcHl0aG9uMwoiIiJ6LWNhbmFsIOKAlCBjbGllbnQgZHUgY2FuYWwgwqsgX19OQU1FX18gwrsgKGlkIDogX19JRF9fKSBkZSBsJ2FwcGxpY2F0aW9uIEFJIEFnZW50LgoKQ2UgY2FuYWwgcmVsaWUgQ0VUIGFnZW50IGEgbCdhcHBsaWNhdGlvbiA6IGwndXRpbGlzYXRldXIgZWNyaXQgZGFucyBsJ2FwcCwKbCdhZ2VudCByZWNvaXQgc2VzIG1lc3NhZ2VzLCBwZXV0IHJlcG9uZHJlIOKAlCBldCBzaSBsZSBjYW5hbCBlc3QgZW4gbW9kZQrCqyBhZ2VudCDCuywgaWwgcGV1dCBhdXNzaSBleGVjdXRlciBkZXMgY29tbWFuZGVzIHN1ciBsYSBWTSBwdWlzIHJlbnZveWVyIGxldXIKcmVzdWx0YXQgZGFucyBsJ2FwcGxpY2F0aW9uLgoKICBweXRob24zIHotY2FuYWwucHkgd2FpdCAgICAgICAgICAgICAgYXR0ZW5kIHVuIG1lc3NhZ2UgKHJlbmQgbGEgbWFpbiBkZXMgcXUnaWwgYXJyaXZlKQogIHB5dGhvbjMgei1jYW5hbC5weSByZWFkICAgICAgICAgICAgICBsJ2hpc3RvcmlxdWUgcmVjZW50IChuZSBtYXJxdWUgcmllbikKICBweXRob24zIHotY2FuYWwucHkgd3JpdGUgInRleHRlIiAgICAgZWNyaXQgdW4gbWVzc2FnZSBkYW5zIGwnYXBwbGljYXRpb24KICBweXRob24zIHotY2FuYWwucHkgd3JpdGUgLS1qc29uICd7ImFjdGlvbiI6ImJhc2giLCJjb21tYW5kIjoibHMgLWxhIH4ifScKICBweXRob24zIHotY2FuYWwucHkgc3RhdGUgICAgICAgICAgICAgbCdldGF0IGR1IGNhbmFsCgpVbiBtZXNzYWdlIG5lIHJldmllbnQgSkFNQUlTIGRldXggZm9pcyA6IGRlcyBxdSd1biDCqyB3YWl0IMK7IGxlIHJlbmQsIGxlIGNhbmFsCmxlIG1hcnF1ZSDCqyBsdSDCuy4gQydlc3QgY2Ugc2NyaXB0IHF1aSBnZXJlIGwnYXR0ZW50ZSAobGUgc2VydmV1ciBuZSBibG9xdWUgcGFzCnBsdXMgZGUgMjUgc2Vjb25kZXMpIDogd2FpdCAtPiByaWVuID8gcGF1c2UgY291cnRlIC0+IHdhaXQuIE4nYXR0ZW5kZXogcGFzIGF2ZWMKdW5lIGJvdWNsZSBzZXJyZWUgOiA1IHNlY29uZGVzIGRlIHBhdXNlIHF1YW5kIGlsIG4neSBhIHJpZW4gc3VmZmlzZW50LCBldCBwb3VyCnVuIGVjaGFuZ2VzIHRvdXRlcyBsZXMgMTAgc2Vjb25kZXMgbGUgdG90YWwgcmVzdGUgdHJlcyBsb2luIGRlcyBxdW90YXMuCiIiIgppbXBvcnQganNvbgppbXBvcnQgc3lzCmltcG9ydCB0aW1lCmltcG9ydCB1cmxsaWIuZXJyb3IKaW1wb3J0IHVybGxpYi5wYXJzZQppbXBvcnQgdXJsbGliLnJlcXVlc3QKCkJBU0UgPSAiX19CQVNFX18iCklEID0gIl9fSURfXyIKTkFNRSA9ICJfX05BTUVfXyIKTU9ERSA9ICJfX01PREVfXyIKV0FJVF9IT0xEID0gMjUgICAgICAgICAgIyBzZWNvbmRlcyB0ZW51ZXMgcGFyIGxlIHNlcnZldXIgc3VyIHVuIMKrIHdhaXQgwrsKV0FJVF9HQVAgPSA1ICAgICAgICAgICAgIyBwYXVzZSBlbnRyZSBkZXV4IMKrIHdhaXQgwrsgcXVhbmQgaWwgbid5IGEgcmllbgoKCmRlZiBjYWxsKHBhdGgsIGRhdGE9Tm9uZSwgdGltZW91dD02MCk6CiAgICB1cmwgPSBCQVNFICsgIi8iICsgSUQgKyBwYXRoCiAgICBoZWFkZXJzID0geyJVc2VyLUFnZW50IjogInotY2FuYWwvMS4wIn0KICAgIGJvZHkgPSBOb25lCiAgICBpZiBkYXRhIGlzIG5vdCBOb25lOgogICAgICAgIGJvZHkgPSBqc29uLmR1bXBzKGRhdGEpLmVuY29kZSgidXRmLTgiKQogICAgICAgIGhlYWRlcnNbIkNvbnRlbnQtVHlwZSJdID0gImFwcGxpY2F0aW9uL2pzb24iCiAgICByZXEgPSB1cmxsaWIucmVxdWVzdC5SZXF1ZXN0KHVybCwgZGF0YT1ib2R5LCBoZWFkZXJzPWhlYWRlcnMsCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIG1ldGhvZD0iUE9TVCIgaWYgYm9keSBlbHNlICJHRVQiKQogICAgdHJ5OgogICAgICAgIHdpdGggdXJsbGliLnJlcXVlc3QudXJsb3BlbihyZXEsIHRpbWVvdXQ9dGltZW91dCkgYXMgcmVwOgogICAgICAgICAgICByYXcgPSByZXAucmVhZCgpLmRlY29kZSgidXRmLTgiLCAicmVwbGFjZSIpCiAgICAgICAgICAgIHJldHVybiBqc29uLmxvYWRzKHJhdyBvciAie30iKQogICAgZXhjZXB0IHVybGxpYi5lcnJvci5IVFRQRXJyb3IgYXMgZXJyOgogICAgICAgIHRyeToKICAgICAgICAgICAgcmV0dXJuIGpzb24ubG9hZHMoZXJyLnJlYWQoKS5kZWNvZGUoInV0Zi04IiwgInJlcGxhY2UiKSBvciAie30iKQogICAgICAgIGV4Y2VwdCBFeGNlcHRpb246CiAgICAgICAgICAgIHJldHVybiB7Im9rIjogRmFsc2UsICJlcnJvciI6ICJIVFRQICVzIiAlIGVyci5jb2RlfQogICAgZXhjZXB0IEV4Y2VwdGlvbiBhcyBlcnI6ICAgICAgICAgICAgICAgICAgICAgICAjIHJlc2VhdSwgRE5TLCBkZWxhaS4uLgogICAgICAgIHJldHVybiB7Im9rIjogRmFsc2UsICJlcnJvciI6IHN0cihlcnIpfQoKCmRlZiBoaG1tKHRzKToKICAgIHRyeToKICAgICAgICByZXR1cm4gdGltZS5zdHJmdGltZSgiJUg6JU06JVMiLCB0aW1lLmxvY2FsdGltZSgodHMgb3IgMCkgLyAxMDAwLjApKQogICAgZXhjZXB0IEV4Y2VwdGlvbjoKICAgICAgICByZXR1cm4gIj8iCgoKZGVmIHNob3cobXNnLCBpbmRlbnQ9IiAgIik6CiAgICByb2xlID0geyJ1c2VyIjogInRvaSIsICJhZ2VudCI6ICJhZ2VudCIsICJ0b29sIjogInZtIn0uZ2V0KG1zZy5nZXQoInJvbGUiKSwgbXNnLmdldCgicm9sZSIpKQogICAgcHJpbnQoIiVzWyVzXSAlcyIgJSAoaW5kZW50LCBoaG1tKG1zZy5nZXQoInRzIikpLCByb2xlKSkKICAgIGZvciBsaW5lIGluIHN0cihtc2cuZ2V0KCJ0ZXh0Iikgb3IgIiIpLnNwbGl0bGluZXMoKSBvciBbIiJdOgogICAgICAgIHByaW50KCIlcyAgJXMiICUgKGluZGVudCwgbGluZSkpCgoKZGVmIGNtZF93YWl0KF9hcmdzKToKICAgICIiIkF0dGVuZCB1biBtZXNzYWdlLiBSZW5kIGxhIG1haW4gZGVzIHF1J2lsIGVuIGFycml2ZSB1bi4iIiIKICAgIHByaW50KCJjYW5hbCDCqyAlcyDCuyAoJXMpIDogYXR0ZW50ZSBkJ3VuIG1lc3NhZ2UuLi4iICUgKE5BTUUsIElEKSkKICAgIHdoaWxlIFRydWU6CiAgICAgICAgcmVwID0gY2FsbCgiL3dhaXQ/dGltZW91dD0lZCIgJSBXQUlUX0hPTEQsIHRpbWVvdXQ9V0FJVF9IT0xEICsgMzApCiAgICAgICAgaWYgbm90IHJlcC5nZXQoIm9rIik6CiAgICAgICAgICAgIHByaW50KCIgICglcykg4oCUIG5vdXZlbCBlc3NhaSBkYW5zICVkIHMiICUgKHJlcC5nZXQoImVycm9yIiwgImVjaGVjIiksIFdBSVRfR0FQKSkKICAgICAgICAgICAgdGltZS5zbGVlcChXQUlUX0dBUCkKICAgICAgICAgICAgY29udGludWUKICAgICAgICBtc2dzID0gcmVwLmdldCgibWVzc2FnZXMiKSBvciBbXQogICAgICAgIGlmIG5vdCBtc2dzOgogICAgICAgICAgICB0aW1lLnNsZWVwKFdBSVRfR0FQKQogICAgICAgICAgICBjb250aW51ZQogICAgICAgIHByaW50KCIiKQogICAgICAgIGZvciBtIGluIG1zZ3M6CiAgICAgICAgICAgIHNob3cobSkKICAgICAgICBwcmludCgiIikKICAgICAgICByZXR1cm4gMAoKCmRlZiBjbWRfcmVhZChhcmdzKToKICAgIG4gPSBpbnQoYXJnc1swXSkgaWYgYXJncyBlbHNlIDMwCiAgICByZXAgPSBjYWxsKCIvcmVhZD9uPSVkIiAlIG4pCiAgICBpZiBub3QgcmVwLmdldCgib2siKToKICAgICAgICBwcmludCgibGVjdHVyZSBpbXBvc3NpYmxlIDogJXMiICUgcmVwLmdldCgiZXJyb3IiLCAicmFpc29uIGluY29ubnVlIikpCiAgICAgICAgcmV0dXJuIDEKICAgIG1zZ3MgPSByZXAuZ2V0KCJtZXNzYWdlcyIpIG9yIFtdCiAgICBwcmludCgiY2FuYWwgwqsgJXMgwrsgOiAlZCBtZXNzYWdlKHMpIiAlIChOQU1FLCBsZW4obXNncykpKQogICAgZm9yIG0gaW4gbXNnczoKICAgICAgICBzaG93KG0pCiAgICByZXR1cm4gMAoKCmRlZiBjbWRfd3JpdGUoYXJncyk6CiAgICBpZiBub3QgYXJnczoKICAgICAgICBwcmludCgicmllbiBhIGVjcmlyZSAoZG9ubmV6IHVuIHRleHRlLCBvdSAtLWpzb24gPG9iamV0PikiKQogICAgICAgIHJldHVybiAxCiAgICBpZiBhcmdzWzBdID09ICItLWpzb24iOgogICAgICAgIHRyeToKICAgICAgICAgICAgcGF5bG9hZCA9IGpzb24ubG9hZHMoIiAiLmpvaW4oYXJnc1sxOl0pKQogICAgICAgIGV4Y2VwdCBWYWx1ZUVycm9yIGFzIGVycjoKICAgICAgICAgICAgcHJpbnQoIkpTT04gaW52YWxpZGUgOiAlcyIgJSBlcnIpCiAgICAgICAgICAgIHJldHVybiAxCiAgICAgICAgaWYgbm90IGlzaW5zdGFuY2UocGF5bG9hZCwgZGljdCk6CiAgICAgICAgICAgIHByaW50KCJsZSBKU09OIGRvaXQgZXRyZSB1biBvYmpldCIpCiAgICAgICAgICAgIHJldHVybiAxCiAgICAgICAgZGF0YSA9IHsidGV4dCI6IGpzb24uZHVtcHMocGF5bG9hZCwgZW5zdXJlX2FzY2lpPUZhbHNlKSwgImtpbmQiOiAiY2FsbCJ9CiAgICAgICAgIyB1bmUgY29tbWFuZGUgOiBsYSBwYXNzZXJlbGxlIGF0dGVuZCBzb24gZXhlY3V0aW9uIGV0IHJlbmQgbGUgcmVzdWx0YXQKICAgICAgICBkYXRhWyJ3YWl0Il0gPSAyMAogICAgZWxzZToKICAgICAgICBkYXRhID0geyJ0ZXh0IjogIiAiLmpvaW4oYXJncyksICJraW5kIjogInRleHQifQogICAgcmVwID0gY2FsbCgiL3dyaXRlIiwgZGF0YSwgdGltZW91dD05MCkKICAgIGlmIG5vdCByZXAuZ2V0KCJvayIpOgogICAgICAgIHByaW50KCJlY3JpdHVyZSBpbXBvc3NpYmxlIDogJXMiICUgcmVwLmdldCgiZXJyb3IiLCAicmFpc29uIGluY29ubnVlIikpCiAgICAgICAgcmV0dXJuIDEKICAgIHByaW50KCJlY3JpdCBkYW5zIGxlIGNhbmFsIDogJXMiICUgcmVwLmdldCgiaWQiLCAiPyIpKQogICAgaWYgcmVwLmdldCgidG9vbCIpOgogICAgICAgIG91dGlsID0gcmVwWyJ0b29sIl0KICAgICAgICBwcmludCgiIikKICAgICAgICBwcmludCgicmVzdWx0YXQgZGUgbGEgY29tbWFuZGUgOiIpCiAgICAgICAgcHJpbnQoIiAgIiArIHN0cihvdXRpbC5nZXQoInRleHQiKSBvciAiIikucmVwbGFjZSgiXG4iLCAiXG4gICIpKQogICAgICAgIHJldHVybiAwCiAgICBpZiBkYXRhLmdldCgia2luZCIpID09ICJjYWxsIjoKICAgICAgICBwcmludCgiKGF1Y3VuIHJlc3VsdGF0IDogbCdhcHBsaWNhdGlvbiBlc3QtZWxsZSBvdXZlcnRlLCBldCBsYSBkaXNjdXNzaW9uIGVuIG1vZGUgYWdlbnQgPykiKQogICAgcmV0dXJuIDAKCgpkZWYgY21kX3N0YXRlKF9hcmdzKToKICAgIHJlcCA9IGNhbGwoIi9zdGF0ZSIpCiAgICBpZiBub3QgcmVwLmdldCgib2siKToKICAgICAgICBwcmludCgiZXRhdCBpbmRpc3BvbmlibGUgOiAlcyIgJSByZXAuZ2V0KCJlcnJvciIsICJyYWlzb24gaW5jb25udWUiKSkKICAgICAgICByZXR1cm4gMQogICAgYyA9IHJlcC5nZXQoImNhbmFsIikgb3Ige30KICAgIHByaW50KCJjYW5hbCDCqyAlcyDCuyIgJSAoYy5nZXQoIm5hbWUiKSBvciBOQU1FKSkKICAgIHByaW50KCIgIGlkICAgICAgIDogJXMiICUgKGMuZ2V0KCJpZCIpIG9yIElEKSkKICAgIHByaW50KCIgIG1vZGUgICAgIDogJXMiICUgKGMuZ2V0KCJtb2RlIikgb3IgTU9ERSkpCiAgICBwcmludCgiICBtZXNzYWdlcyA6ICVzIiAlIGMuZ2V0KCJtZXNzYWdlcyIsICI/IikpCiAgICBwcmludCgiICBub24gbHVzICA6ICVzIiAlIGMuZ2V0KCJ1bnJlYWQiLCAiPyIpKQogICAgcmV0dXJuIDAKCgpkZWYgbWFpbihhcmd2KToKICAgIGlmIG5vdCBhcmd2IG9yIGFyZ3ZbMF0gaW4gKCItaCIsICItLWhlbHAiLCAiaGVscCIpOgogICAgICAgIHByaW50KF9fZG9jX18pCiAgICAgICAgcmV0dXJuIDAKICAgIGNtZCwgcmVzdCA9IGFyZ3ZbMF0sIGFyZ3ZbMTpdCiAgICBpZiBjbWQgPT0gIndhaXQiOgogICAgICAgIHJldHVybiBjbWRfd2FpdChyZXN0KQogICAgaWYgY21kID09ICJyZWFkIjoKICAgICAgICByZXR1cm4gY21kX3JlYWQocmVzdCkKICAgIGlmIGNtZCA9PSAid3JpdGUiOgogICAgICAgIHJldHVybiBjbWRfd3JpdGUocmVzdCkKICAgIGlmIGNtZCA9PSAic3RhdGUiOgogICAgICAgIHJldHVybiBjbWRfc3RhdGUocmVzdCkKICAgIHByaW50KCJjb21tYW5kZSBpbmNvbm51ZSA6ICVzICh3YWl0LCByZWFkLCB3cml0ZSwgc3RhdGUpIiAlIGNtZCkKICAgIHJldHVybiAxCgoKaWYgX19uYW1lX18gPT0gIl9fbWFpbl9fIjoKICAgIHN5cy5leGl0KG1haW4oc3lzLmFyZ3ZbMTpdKSkK";
const ZCANAL_JS_B64 = "IyEvdXNyL2Jpbi9lbnYgbm9kZQovKiB6LWNhbmFsIOKAlCBjbGllbnQgZHUgY2FuYWwgwqsgX19OQU1FX18gwrsgKGlkIDogX19JRF9fKSBkZSBsJ2FwcGxpY2F0aW9uIEFJIEFnZW50LgogKgogKiBDZSBjYW5hbCByZWxpZSBDRVQgYWdlbnQgYSBsJ2FwcGxpY2F0aW9uIDogbCd1dGlsaXNhdGV1ciBlY3JpdCBkYW5zIGwnYXBwLAogKiBsJ2FnZW50IHJlY29pdCBzZXMgbWVzc2FnZXMsIHBldXQgcmVwb25kcmUg4oCUIGV0IHNpIGxlIGNhbmFsIGVzdCBlbiBtb2RlCiAqIMKrIGFnZW50IMK7LCBpbCBwZXV0IGF1c3NpIGV4ZWN1dGVyIGRlcyBjb21tYW5kZXMgc3VyIGxhIFZNIHB1aXMgcmVudm95ZXIKICogbGV1ciByZXN1bHRhdCBkYW5zIGwnYXBwbGljYXRpb24uCiAqCiAqICAgbm9kZSB6LWNhbmFsLmpzIHdhaXQgICAgICAgICAgICAgIGF0dGVuZCB1biBtZXNzYWdlIChyZW5kIGxhIG1haW4gZGVzIHF1J2lsIGFycml2ZSkKICogICBub2RlIHotY2FuYWwuanMgcmVhZCBbbl0gICAgICAgICAgbCdoaXN0b3JpcXVlIHJlY2VudCAobmUgbWFycXVlIHJpZW4pCiAqICAgbm9kZSB6LWNhbmFsLmpzIHdyaXRlICJ0ZXh0ZSIgICAgIGVjcml0IHVuIG1lc3NhZ2UgZGFucyBsJ2FwcGxpY2F0aW9uCiAqICAgbm9kZSB6LWNhbmFsLmpzIHdyaXRlIC0tanNvbiAneyJhY3Rpb24iOiJiYXNoIiwiY29tbWFuZCI6ImxzIC1sYSB+In0nCiAqICAgbm9kZSB6LWNhbmFsLmpzIHN0YXRlICAgICAgICAgICAgIGwnZXRhdCBkdSBjYW5hbAogKgogKiBVbiBtZXNzYWdlIG5lIHJldmllbnQgSkFNQUlTIGRldXggZm9pcyA6IGRlcyBxdSd1biDCqyB3YWl0IMK7IGxlIHJlbmQsIGxlIGNhbmFsCiAqIGxlIG1hcnF1ZSDCqyBsdSDCuy4gQydlc3QgY2Ugc2NyaXB0IHF1aSBnZXJlIGwnYXR0ZW50ZSAobGUgc2VydmV1ciBuZSBibG9xdWUgcGFzCiAqIHBsdXMgZGUgMjUgcykgOiB3YWl0IC0+IHJpZW4gPyBwYXVzZSBkZSA1IHMgLT4gd2FpdC4KICovCmNvbnN0IEJBU0UgPSAiX19CQVNFX18iOwpjb25zdCBJRCA9ICJfX0lEX18iOwpjb25zdCBOQU1FID0gIl9fTkFNRV9fIjsKY29uc3QgTU9ERSA9ICJfX01PREVfXyI7CmNvbnN0IEhPTEQgPSAyNSwgR0FQID0gNTAwMDsKCmFzeW5jIGZ1bmN0aW9uIGNhbGwocGF0aCwgZGF0YSwgdGltZW91dE1zKSB7CiAgY29uc3QgdXJsID0gQkFTRSArICIvIiArIElEICsgcGF0aDsKICBjb25zdCBvcHQgPSB7IGhlYWRlcnM6IHsgIlVzZXItQWdlbnQiOiAiei1jYW5hbC8xLjAiIH0gfTsKICAvKiB1bmUgYXR0ZW50ZSBkZSByZXN1bHRhdCBwZXV0IGRlcGFzc2VyIGxhIG1pbnV0ZSA6IG9uIGxhIGJvcm5lICovCiAgaWYgKHRpbWVvdXRNcykgb3B0LnNpZ25hbCA9IEFib3J0U2lnbmFsLnRpbWVvdXQgPyBBYm9ydFNpZ25hbC50aW1lb3V0KHRpbWVvdXRNcykgOiB1bmRlZmluZWQ7CiAgaWYgKGRhdGEgIT09IHVuZGVmaW5lZCkgewogICAgb3B0Lm1ldGhvZCA9ICJQT1NUIjsKICAgIG9wdC5oZWFkZXJzWyJDb250ZW50LVR5cGUiXSA9ICJhcHBsaWNhdGlvbi9qc29uIjsKICAgIG9wdC5ib2R5ID0gSlNPTi5zdHJpbmdpZnkoZGF0YSk7CiAgfQogIHRyeSB7CiAgICBjb25zdCByID0gYXdhaXQgZmV0Y2godXJsLCBvcHQpOwogICAgY29uc3QgaiA9IGF3YWl0IHIuanNvbigpLmNhdGNoKCgpID0+ICh7fSkpOwogICAgcmV0dXJuIGogJiYgdHlwZW9mIGogPT09ICJvYmplY3QiID8gaiA6IHsgb2s6IGZhbHNlLCBlcnJvcjogInJlcG9uc2UgaWxsaXNpYmxlIiB9OwogIH0gY2F0Y2ggKGUpIHsgcmV0dXJuIHsgb2s6IGZhbHNlLCBlcnJvcjogU3RyaW5nKChlICYmIGUubWVzc2FnZSkgfHwgZSkgfTsgfQp9CmNvbnN0IGhobW0gPSAodHMpID0+IG5ldyBEYXRlKHRzIHx8IDApLnRvTG9jYWxlVGltZVN0cmluZygiZnItRlIiKTsKZnVuY3Rpb24gc2hvdyhtKSB7CiAgY29uc3Qgcm9sZSA9IHsgdXNlcjogInRvaSIsIGFnZW50OiAiYWdlbnQiLCB0b29sOiAidm0iIH1bbS5yb2xlXSB8fCBtLnJvbGU7CiAgY29uc29sZS5sb2coIiAgWyIgKyBoaG1tKG0udHMpICsgIl0gIiArIHJvbGUpOwogIFN0cmluZyhtLnRleHQgfHwgIiIpLnNwbGl0KCJcbiIpLmZvckVhY2goKGwpID0+IGNvbnNvbGUubG9nKCIgICAgIiArIGwpKTsKfQpjb25zdCBzbGVlcCA9IChtcykgPT4gbmV3IFByb21pc2UoKHIpID0+IHNldFRpbWVvdXQociwgbXMpKTsKCmFzeW5jIGZ1bmN0aW9uIGNtZFdhaXQoKSB7CiAgY29uc29sZS5sb2coImNhbmFsIMKrICIgKyBOQU1FICsgIiDCuyAoIiArIElEICsgIikgOiBhdHRlbnRlIGQndW4gbWVzc2FnZS4uLiIpOwogIGZvciAoOzspIHsKICAgIGNvbnN0IHJlcCA9IGF3YWl0IGNhbGwoIi93YWl0P3RpbWVvdXQ9IiArIEhPTEQpOwogICAgaWYgKCFyZXAub2spIHsgY29uc29sZS5sb2coIiAgKCIgKyAocmVwLmVycm9yIHx8ICJlY2hlYyIpICsgIikg4oCUIG5vdXZlbCBlc3NhaSIpOyBhd2FpdCBzbGVlcChHQVApOyBjb250aW51ZTsgfQogICAgY29uc3QgbXNncyA9IHJlcC5tZXNzYWdlcyB8fCBbXTsKICAgIGlmICghbXNncy5sZW5ndGgpIHsgYXdhaXQgc2xlZXAoR0FQKTsgY29udGludWU7IH0KICAgIGNvbnNvbGUubG9nKCIiKTsKICAgIG1zZ3MuZm9yRWFjaChzaG93KTsKICAgIGNvbnNvbGUubG9nKCIiKTsKICAgIHJldHVybiAwOwogIH0KfQphc3luYyBmdW5jdGlvbiBjbWRSZWFkKG4pIHsKICBjb25zdCByZXAgPSBhd2FpdCBjYWxsKCIvcmVhZD9uPSIgKyAobiB8fCAzMCkpOwogIGlmICghcmVwLm9rKSB7IGNvbnNvbGUubG9nKCJsZWN0dXJlIGltcG9zc2libGUgOiAiICsgKHJlcC5lcnJvciB8fCAiPyIpKTsgcmV0dXJuIDE7IH0KICBjb25zb2xlLmxvZygiY2FuYWwgwqsgIiArIE5BTUUgKyAiIMK7IDogIiArIChyZXAubWVzc2FnZXMgfHwgW10pLmxlbmd0aCArICIgbWVzc2FnZShzKSIpOwogIChyZXAubWVzc2FnZXMgfHwgW10pLmZvckVhY2goc2hvdyk7CiAgcmV0dXJuIDA7Cn0KYXN5bmMgZnVuY3Rpb24gY21kV3JpdGUoYXJncykgewogIGlmICghYXJncy5sZW5ndGgpIHsgY29uc29sZS5sb2coInJpZW4gYSBlY3JpcmUgKHRleHRlLCBvdSAtLWpzb24gPG9iamV0PikiKTsgcmV0dXJuIDE7IH0KICBsZXQgZGF0YTsKICBsZXQgYXR0ZW50ZSA9IDA7CiAgaWYgKGFyZ3NbMF0gPT09ICItLWpzb24iKSB7CiAgICBsZXQgbzsgdHJ5IHsgbyA9IEpTT04ucGFyc2UoYXJncy5zbGljZSgxKS5qb2luKCIgIikpOyB9IGNhdGNoIChlKSB7IGNvbnNvbGUubG9nKCJKU09OIGludmFsaWRlIDogIiArIGUubWVzc2FnZSk7IHJldHVybiAxOyB9CiAgICBkYXRhID0geyB0ZXh0OiBKU09OLnN0cmluZ2lmeShvKSwga2luZDogImNhbGwiIH07CiAgICAvKiB1bmUgY29tbWFuZGUgOiBvbiBkZW1hbmRlIGEgbGEgcGFzc2VyZWxsZSBkJ2F0dGVuZHJlIHNvbiByZXN1bHRhdCAqLwogICAgYXR0ZW50ZSA9IE51bWJlcihwcm9jZXNzLmVudi5aQ0FOQUxfV0FJVCB8fCAyMCk7CiAgICBkYXRhLndhaXQgPSBhdHRlbnRlOwogIH0gZWxzZSBkYXRhID0geyB0ZXh0OiBhcmdzLmpvaW4oIiAiKSwga2luZDogInRleHQiIH07CiAgY29uc3QgcmVwID0gYXdhaXQgY2FsbCgiL3dyaXRlIiwgZGF0YSwgNjAwMDAgKyBhdHRlbnRlICogMTAwMCk7CiAgaWYgKCFyZXAub2spIHsgY29uc29sZS5sb2coImVjcml0dXJlIGltcG9zc2libGUgOiAiICsgKHJlcC5lcnJvciB8fCAiPyIpKTsgcmV0dXJuIDE7IH0KICBjb25zb2xlLmxvZygiZWNyaXQgZGFucyBsZSBjYW5hbCA6ICIgKyAocmVwLmlkIHx8ICI/IikpOwogIGlmIChyZXAudG9vbCkgewogICAgY29uc29sZS5sb2coIiIpOwogICAgY29uc29sZS5sb2coInJlc3VsdGF0IGRlIGxhIGNvbW1hbmRlIDoiKTsKICAgIHNob3cocmVwLnRvb2wpOwogICAgcmV0dXJuIDA7CiAgfQogIGlmIChkYXRhLmtpbmQgPT09ICJjYWxsIiAmJiBhdHRlbnRlKSB7CiAgICBjb25zb2xlLmxvZygiKGF1Y3VuIHJlc3VsdGF0IDogbCdhcHBsaWNhdGlvbiBlc3QtZWxsZSBvdXZlcnRlLCBldCBsYSBkaXNjdXNzaW9uIGVuIG1vZGUgYWdlbnQgPykiKTsKICB9CiAgcmV0dXJuIDA7Cn0KYXN5bmMgZnVuY3Rpb24gY21kU3RhdGUoKSB7CiAgY29uc3QgcmVwID0gYXdhaXQgY2FsbCgiL3N0YXRlIik7CiAgaWYgKCFyZXAub2spIHsgY29uc29sZS5sb2coImV0YXQgaW5kaXNwb25pYmxlIDogIiArIChyZXAuZXJyb3IgfHwgIj8iKSk7IHJldHVybiAxOyB9CiAgY29uc3QgYyA9IHJlcC5jYW5hbCB8fCB7fTsKICBjb25zb2xlLmxvZygiY2FuYWwgwqsgIiArIChjLm5hbWUgfHwgTkFNRSkgKyAiIMK7Iik7CiAgY29uc29sZS5sb2coIiAgaWQgICAgICAgOiAiICsgKGMuaWQgfHwgSUQpKTsKICBjb25zb2xlLmxvZygiICBtb2RlICAgICA6ICIgKyAoYy5tb2RlIHx8IE1PREUpKTsKICBjb25zb2xlLmxvZygiICBtZXNzYWdlcyA6ICIgKyAoYy5tZXNzYWdlcyA9PT0gdW5kZWZpbmVkID8gIj8iIDogYy5tZXNzYWdlcykpOwogIGNvbnNvbGUubG9nKCIgIG5vbiBsdXMgIDogIiArIChjLnVucmVhZCA9PT0gdW5kZWZpbmVkID8gIj8iIDogYy51bnJlYWQpKTsKICByZXR1cm4gMDsKfQooYXN5bmMgKCkgPT4gewogIGNvbnN0IFtjbWQsIC4uLnJlc3RdID0gcHJvY2Vzcy5hcmd2LnNsaWNlKDIpOwogIGNvbnN0IGNvZGUgPSBjbWQgPT09ICJ3YWl0IiA/IGF3YWl0IGNtZFdhaXQoKQogICAgOiBjbWQgPT09ICJyZWFkIiA/IGF3YWl0IGNtZFJlYWQoTnVtYmVyKHJlc3RbMF0pIHx8IDMwKQogICAgOiBjbWQgPT09ICJ3cml0ZSIgPyBhd2FpdCBjbWRXcml0ZShyZXN0KQogICAgOiBjbWQgPT09ICJzdGF0ZSIgPyBhd2FpdCBjbWRTdGF0ZSgpCiAgICA6IChjb25zb2xlLmxvZygidXNhZ2UgOiBub2RlIHotY2FuYWwuanMgd2FpdHxyZWFkfHdyaXRlfHN0YXRlIiksIDEpOwogIHByb2Nlc3MuZXhpdChjb2RlKTsKfSkoKTsK";
const ZCANAL_SH_B64 = "IyEvYmluL3NoCiMgei1jYW5hbCDigJQgY2xpZW50IGR1IGNhbmFsIMKrIF9fTkFNRV9fIMK7IChpZCA6IF9fSURfXykgZGUgbCdhcHBsaWNhdGlvbiBBSSBBZ2VudC4KIwojICAgc2ggei1jYW5hbC5zaCB3YWl0ICAgICAgICAgICAgICBhdHRlbmQgdW4gbWVzc2FnZSAocmVuZCBsYSBtYWluIGRlcyBxdSdpbCBhcnJpdmUpCiMgICBzaCB6LWNhbmFsLnNoIHJlYWQgW25dICAgICAgICAgIGwnaGlzdG9yaXF1ZSByZWNlbnQgKG5lIG1hcnF1ZSByaWVuKQojICAgc2ggei1jYW5hbC5zaCB3cml0ZSAidGV4dGUiICAgICBlY3JpdCB1biBtZXNzYWdlIGRhbnMgbCdhcHBsaWNhdGlvbgojICAgc2ggei1jYW5hbC5zaCB3cml0ZS1qc29uICd7ImFjdGlvbiI6ImJhc2giLCJjb21tYW5kIjoibHMgLWxhIH4ifScKIyAgIHNoIHotY2FuYWwuc2ggc3RhdGUgICAgICAgICAgICAgbCdldGF0IGR1IGNhbmFsCiMKIyBVbiBtZXNzYWdlIG5lIHJldmllbnQgamFtYWlzIGRldXggZm9pcyA6IGRlcyBxdSd1biDCqyB3YWl0IMK7IGxlIHJlbmQsIGxlIGNhbmFsCiMgbGUgbWFycXVlIMKrIGx1IMK7LiBDJ2VzdCBjZSBzY3JpcHQgcXVpIGdlcmUgbCdhdHRlbnRlIDogd2FpdCAtPiByaWVuID8gcGF1c2UgLT4gd2FpdC4KQkFTRT0iX19CQVNFX18vX19JRF9fIgpIT0xEPTI1CkdBUD01CgpjYXNlICIkMSIgaW4KICB3YWl0KQogICAgZWNobyAiY2FuYWwgwqsgX19OQU1FX18gwrsgKF9fSURfXykgOiBhdHRlbnRlIGQndW4gbWVzc2FnZS4uLiIKICAgIHdoaWxlIDo7IGRvCiAgICAgIE9VVD0kKGN1cmwgLXMgLW0gJCgoSE9MRCArIDIwKSkgIiRCQVNFL3dhaXQ/dGltZW91dD0kSE9MRCIpCiAgICAgIGNhc2UgIiRPVVQiIGluCiAgICAgICAgKicibWVzc2FnZXMiOltdJyp8JycpIHNsZWVwICRHQVAgOzsKICAgICAgICAqKSBwcmludGYgJyVzJyAiJE9VVCIgfCBzZWQgJ3MvXFxuL1xuL2cnOyBlY2hvOyBleGl0IDAgOzsKICAgICAgZXNhYwogICAgZG9uZSA7OwogIHJlYWQpCiAgICBjdXJsIC1zIC1tIDYwICIkQkFTRS9yZWFkP249JHsyOi0zMH0iIHwgc2VkICdzL1xcbi9cbi9nJzsgZWNobyA7OwogIHdyaXRlKQogICAgc2hpZnQ7IGN1cmwgLXMgLW0gNjAgLVggUE9TVCAtSCAnQ29udGVudC1UeXBlOiBhcHBsaWNhdGlvbi9qc29uJyBcCiAgICAgIC1kICJ7XCJ0ZXh0XCI6JChwcmludGYgJyVzJyAiJCoiIHwgc2VkICdzL1xcL1xcXFwvZzsgcy8iL1xcIi9nOyBzL14vIi87IHMvJC8iLycpfSIgIiRCQVNFL3dyaXRlIjsgZWNobyA7OwogIHdyaXRlLWpzb24pCiAgICBzaGlmdDsgY3VybCAtcyAtbSA2MCAtWCBQT1NUIC1IICdDb250ZW50LVR5cGU6IGFwcGxpY2F0aW9uL2pzb24nIFwKICAgICAgLWQgIntcInRleHRcIjokKHByaW50ZiAnJXMnICIkKiIgfCBzZWQgJ3MvXFwvXFxcXC9nOyBzLyIvXFwiL2c7IHMvXi8iLzsgcy8kLyIvJyksXCJraW5kXCI6XCJjYWxsXCJ9IiAiJEJBU0Uvd3JpdGUiOyBlY2hvIDs7CiAgc3RhdGUpIGN1cmwgLXMgLW0gNjAgIiRCQVNFL3N0YXRlIjsgZWNobyA7OwogICopIGVjaG8gInVzYWdlIDogc2ggei1jYW5hbC5zaCB3YWl0fHJlYWR8d3JpdGV8d3JpdGUtanNvbnxzdGF0ZSIgOzsKZXNhYwo=";
const ZCANAL_TS_B64 = "IyEvdXNyL2Jpbi9lbnYgLVMgbnB4IHRzeAovKiB6LWNhbmFsIOKAlCBjbGllbnQgZHUgY2FuYWwgwqsgX19OQU1FX18gwrsgKGlkIDogX19JRF9fKSBkZSBsJ2FwcGxpY2F0aW9uIEFJIEFnZW50LgogKgogKiAgIG5weCB0c3ggei1jYW5hbC50cyB3YWl0IHwgcmVhZCBbbl0gfCB3cml0ZSAidGV4dGUiIHwgd3JpdGUgLS1qc29uICd7Li4ufScgfCBzdGF0ZQogKgogKiBVbiBtZXNzYWdlIG5lIHJldmllbnQgamFtYWlzIGRldXggZm9pcyA6IGxlIGNhbmFsIGxlIG1hcnF1ZSDCqyBsdSDCuyBkZXMgcXUndW4KICogwqsgd2FpdCDCuyBsZSByZW5kLiBDJ2VzdCBjZSBzY3JpcHQgcXVpIGJvdWNsZSAobGUgc2VydmV1ciB0aWVudCAyNSBzLCBwYXMgcGx1cykuCiAqLwpjb25zdCBCQVNFOiBzdHJpbmcgPSAiX19CQVNFX18iOwpjb25zdCBJRDogc3RyaW5nID0gIl9fSURfXyI7CmNvbnN0IE5BTUU6IHN0cmluZyA9ICJfX05BTUVfXyI7CmNvbnN0IE1PREU6IHN0cmluZyA9ICJfX01PREVfXyI7CmNvbnN0IEhPTEQgPSAyNSwgR0FQID0gNTAwMDsKCnR5cGUgQ2FuYWxNZXNzYWdlID0geyBpZD86IHN0cmluZzsgcm9sZT86IHN0cmluZzsgdGV4dD86IHN0cmluZzsgdHM/OiBudW1iZXI7IHN0YXRlPzogc3RyaW5nIH07Cgphc3luYyBmdW5jdGlvbiBjYWxsKHBhdGg6IHN0cmluZywgZGF0YT86IHVua25vd24pOiBQcm9taXNlPGFueT4gewogIGNvbnN0IG9wdDogUmVxdWVzdEluaXQgPSB7IGhlYWRlcnM6IHsgIlVzZXItQWdlbnQiOiAiei1jYW5hbC8xLjAiIH0gfTsKICBpZiAoZGF0YSAhPT0gdW5kZWZpbmVkKSB7CiAgICBvcHQubWV0aG9kID0gIlBPU1QiOwogICAgKG9wdC5oZWFkZXJzIGFzIFJlY29yZDxzdHJpbmcsIHN0cmluZz4pWyJDb250ZW50LVR5cGUiXSA9ICJhcHBsaWNhdGlvbi9qc29uIjsKICAgIG9wdC5ib2R5ID0gSlNPTi5zdHJpbmdpZnkoZGF0YSk7CiAgfQogIHRyeSB7CiAgICBjb25zdCByID0gYXdhaXQgZmV0Y2goQkFTRSArICIvIiArIElEICsgcGF0aCwgb3B0KTsKICAgIHJldHVybiBhd2FpdCByLmpzb24oKS5jYXRjaCgoKSA9PiAoeyBvazogZmFsc2UsIGVycm9yOiAicmVwb25zZSBpbGxpc2libGUiIH0pKTsKICB9IGNhdGNoIChlKSB7IHJldHVybiB7IG9rOiBmYWxzZSwgZXJyb3I6IFN0cmluZyhlKSB9OyB9Cn0KY29uc3Qgc2xlZXAgPSAobXM6IG51bWJlcikgPT4gbmV3IFByb21pc2UoKHIpID0+IHNldFRpbWVvdXQociwgbXMpKTsKZnVuY3Rpb24gc2hvdyhtOiBDYW5hbE1lc3NhZ2UpOiB2b2lkIHsKICBjb25zdCByb2xlID0gKHsgdXNlcjogInRvaSIsIGFnZW50OiAiYWdlbnQiLCB0b29sOiAidm0iIH0gYXMgUmVjb3JkPHN0cmluZywgc3RyaW5nPilbbS5yb2xlIHx8ICIiXSB8fCBtLnJvbGU7CiAgY29uc29sZS5sb2coIiAgWyIgKyBuZXcgRGF0ZShtLnRzIHx8IDApLnRvTG9jYWxlVGltZVN0cmluZygiZnItRlIiKSArICJdICIgKyByb2xlKTsKICBTdHJpbmcobS50ZXh0IHx8ICIiKS5zcGxpdCgiXG4iKS5mb3JFYWNoKChsKSA9PiBjb25zb2xlLmxvZygiICAgICIgKyBsKSk7Cn0KYXN5bmMgZnVuY3Rpb24gbWFpbigpOiBQcm9taXNlPG51bWJlcj4gewogIGNvbnN0IFtjbWQsIC4uLnJlc3RdID0gcHJvY2Vzcy5hcmd2LnNsaWNlKDIpOwogIGlmIChjbWQgPT09ICJ3YWl0IikgewogICAgY29uc29sZS5sb2coImNhbmFsIMKrICIgKyBOQU1FICsgIiDCuyAoIiArIElEICsgIikgOiBhdHRlbnRlIGQndW4gbWVzc2FnZS4uLiIpOwogICAgZm9yICg7OykgewogICAgICBjb25zdCByZXAgPSBhd2FpdCBjYWxsKCIvd2FpdD90aW1lb3V0PSIgKyBIT0xEKTsKICAgICAgaWYgKCFyZXAub2spIHsgYXdhaXQgc2xlZXAoR0FQKTsgY29udGludWU7IH0KICAgICAgY29uc3QgbXNnczogQ2FuYWxNZXNzYWdlW10gPSByZXAubWVzc2FnZXMgfHwgW107CiAgICAgIGlmICghbXNncy5sZW5ndGgpIHsgYXdhaXQgc2xlZXAoR0FQKTsgY29udGludWU7IH0KICAgICAgY29uc29sZS5sb2coIiIpOwogICAgICBtc2dzLmZvckVhY2goc2hvdyk7CiAgICAgIHJldHVybiAwOwogICAgfQogIH0KICBpZiAoY21kID09PSAicmVhZCIpIHsKICAgIGNvbnN0IHJlcCA9IGF3YWl0IGNhbGwoIi9yZWFkP249IiArIChOdW1iZXIocmVzdFswXSkgfHwgMzApKTsKICAgIChyZXAubWVzc2FnZXMgfHwgW10pLmZvckVhY2goc2hvdyk7CiAgICByZXR1cm4gcmVwLm9rID8gMCA6IDE7CiAgfQogIGlmIChjbWQgPT09ICJ3cml0ZSIpIHsKICAgIGNvbnN0IGRhdGEgPSByZXN0WzBdID09PSAiLS1qc29uIgogICAgICA/IHsgdGV4dDogSlNPTi5zdHJpbmdpZnkoSlNPTi5wYXJzZShyZXN0LnNsaWNlKDEpLmpvaW4oIiAiKSkpLCBraW5kOiAiY2FsbCIgfQogICAgICA6IHsgdGV4dDogcmVzdC5qb2luKCIgIiksIGtpbmQ6ICJ0ZXh0IiB9OwogICAgY29uc3QgcmVwID0gYXdhaXQgY2FsbCgiL3dyaXRlIiwgZGF0YSk7CiAgICBjb25zb2xlLmxvZyhyZXAub2sgPyAiZWNyaXQgZGFucyBsZSBjYW5hbCA6ICIgKyByZXAuaWQgOiAiZWNoZWMgOiAiICsgcmVwLmVycm9yKTsKICAgIHJldHVybiByZXAub2sgPyAwIDogMTsKICB9CiAgaWYgKGNtZCA9PT0gInN0YXRlIikgeyBjb25zb2xlLmxvZyhKU09OLnN0cmluZ2lmeSgoYXdhaXQgY2FsbCgiL3N0YXRlIikpLmNhbmFsLCBudWxsLCAyKSk7IHJldHVybiAwOyB9CiAgY29uc29sZS5sb2coInVzYWdlIDogbnB4IHRzeCB6LWNhbmFsLnRzIHdhaXR8cmVhZHx3cml0ZXxzdGF0ZSIpOwogIHJldHVybiAxOwp9Cm1haW4oKS50aGVuKChjKSA9PiBwcm9jZXNzLmV4aXQoYykpOwo=";
const ZCANAL_DOC_B64 = "IyB6LWNhbmFsIOKAlCBsZSBjYW5hbCDCqyBfX05BTUVfXyDCuwoKKipJZGVudGlmaWFudCBkdSBjYW5hbCA6KiogYF9fSURfX2AgwrcgKiptb2RlIDoqKiBgX19NT0RFX19gIMK3ICoqYWRyZXNzZSA6KiogYF9fQkFTRV9fL19fSURfX2AKClVuIGNhbmFsIHJlbGllIHVuIGFnZW50IElBIOKAlCBuJ2ltcG9ydGUgbGVxdWVsIOKAlCDDoCBsJ2FwcGxpY2F0aW9uIEFJIEFnZW50LiBMJ3V0aWxpc2F0ZXVyCsOpY3JpdCBkYW5zIGwnYXBwbGljYXRpb24gOyBsJ2FnZW50IHJlw6dvaXQgc2VzIG1lc3NhZ2VzIHBhciBjZSBjYW5hbCwgcGV1dCByw6lwb25kcmUsIGV0CnNpIGxlIG1vZGUgZXN0IGBhZ2VudGAsIGV4w6ljdXRlciBkZXMgY29tbWFuZGVzIHN1ciBsYSBWTSBldCByZW52b3llciBsZXVyIHLDqXN1bHRhdC4KCiMjIExlcyBxdWF0cmUgYWRyZXNzZXMKCnwgQWRyZXNzZSB8IENlIHF1J2VsbGUgZmFpdCB8CnwtLS18LS0tfAp8IGBfX0JBU0VfXy9fX0lEX18vd3JpdGVgIHwgbCdhZ2VudCDDqWNyaXQgdW4gbWVzc2FnZSAoUE9TVCBgeyJ0ZXh0IjogIi4uLiIsICJraW5kIjogInRleHQifCJjYWxsIn1gKSB8CnwgYF9fQkFTRV9fL19fSURfXy9yZWFkP249MzBgIHwgbCdoaXN0b3JpcXVlIHLDqWNlbnQsIHNhbnMgcmllbiBtYXJxdWVyIHwKfCBgX19CQVNFX18vX19JRF9fL3dhaXQ/dGltZW91dD0yNWAgfCAqKmxlcyBtZXNzYWdlcyBub24gbHVzKiosIHF1J2VsbGUgbWFycXVlIMKrIGx1IMK7IHwKfCBgX19CQVNFX18vX19JRF9fL3N0YXRlYCB8IGwnw6l0YXQgZHUgY2FuYWwgKG5vbSwgbW9kZSwgbm9tYnJlIGRlIG1lc3NhZ2VzLCBub24gbHVzKSB8CgpgX19CQVNFX19gIHNldWwgKHNhbnMgaWRlbnRpZmlhbnQpIGFmZmljaGUgbGEgZG9jdW1lbnRhdGlvbiBjb21wbMOodGUuCmBfX0JBU0VfXy9fX0lEX18vaW5zdGFsbGAgZG9ubmUgbGVzIGluc3RydWN0aW9ucyBkJ2luc3RhbGxhdGlvbi4KCiMjIENlIHF1J2lsIGZhdXQgc2F2b2lyCgotICoqQXVjdW4gbW90IGRlIHBhc3NlLioqIEwnaWRlbnRpZmlhbnQgZXN0IGxhIGNsw6kgOiBuZSBsZSBwYXJ0YWdleiBwYXMuCi0gKipVbiBtZXNzYWdlIG5lIHJldmllbnQgamFtYWlzIGRldXggZm9pcy4qKiBEw6hzIHF1J3VuIGAvd2FpdGAgbGUgcmVuZCwgbGUgY2FuYWwgbGUKICBtYXJxdWUgwqsgbHUgwrsg4oCUIGV0IGwnYXBwbGljYXRpb24gYWZmaWNoZSDCqyBsdSAhIMK7IMOgIGxhIHBsYWNlIGRlIMKrIG5vbiBsdSDCuy4KLSAqKkMnZXN0IHZvdHJlIHNjcmlwdCBxdWkgZ8OocmUgbCdhdHRlbnRlKiosIHBhcyBsZSBzZXJ2ZXVyIDogbGUgc2VydmV1ciBuZSBibG9xdWUgcGFzCiAgcGx1cyBkZSAyNSBzZWNvbmRlcy4gTGEgYm91Y2xlIHJlY29tbWFuZMOpZSBlc3QgYHdhaXRgIOKGkiByaWVuID8gcGF1c2UgZGUgNSBzIOKGkiBgd2FpdGAuCiAgTidhdHRlbmRleiBwYXMgZW4gYm91Y2xlIHNlcnLDqWUgOiBsZSBwbGFuIGdyYXR1aXQgZGUgbGEgcGxhdGVmb3JtZSBhY2NvcmRlCiAgMTAwIDAwMCByZXF1w6p0ZXMgcGFyIGpvdXIsIGV0IHVuZSBhdHRlbnRlIHRvdXRlcyBsZXMgMTAgc2Vjb25kZXMgZW4gY29uc29tbWUgbW9pbnMgZGUgMTAgJS4KLSAqKkxlcyBjb21tYW5kZXMgbmUgcydleMOpY3V0ZW50IHF1ZSBzaSBsJ2FwcGxpY2F0aW9uIGVzdCBvdXZlcnRlKiogOiBjJ2VzdCBlbGxlIHF1aQogIHBhcmxlIMOgIGxhIFZNLiBTaSBlbGxlIGVzdCBmZXJtw6llLCBsZSBtZXNzYWdlIGF0dGVuZCBkYW5zIGxlIGNhbmFsLgotIFBvdXIgYWdpciBzdXIgbGEgVk0sIMOpY3JpdmV6IHVuIGBraW5kYCAqKmBjYWxsYCoqIGNvbnRlbmFudCB1biBvYmpldCBKU09OIDoKICBgeyJhY3Rpb24iOiJiYXNoIiwiY29tbWFuZCI6ImxzIC1sYSB+In1gIG91IGB7ImFjdGlvbiI6ImZldGNoIiwidXJsIjoiaHR0cHM6Ly/igKYifWAuCiAgTGUgcsOpc3VsdGF0IHJldmllbnQgZGFucyBsZSBjYW5hbCBlbiB0YW50IHF1ZSBtZXNzYWdlIGB0b29sYC4KLSBQb3VyIHNpbXBsZW1lbnQgcGFybGVyLCDDqWNyaXZleiB1biBga2luZGAgKipgdGV4dGAqKiA6IGlsIHMnYWZmaWNoZSBjb21tZSB1biBtZXNzYWdlCiAgZGUgbCdhZ2VudCBkYW5zIGwnYXBwbGljYXRpb24uCg==";

const DOC_VER = "2026-10-15.1";   /* doit correspondre au marqueur dans les .md */
async function ensureAgentDocs(env) {
  const out = { restored: [], present: [], updated: [] };
  const chk = await vmExecCmd(env,
    'for f in agent.md infos.md user.md; do if [ -s /home/kasm-user/user/$f ]; then echo "OK:$f"; else echo "MISSING:$f"; fi; done; '
    + 'echo "VER:"; grep -h -o "vm-doc-version: [0-9.\-]*" /home/kasm-user/user/agent.md /home/kasm-user/user/infos.md 2>/dev/null | head -1');
  const so = String(chk.stdout || "");
  const docs = { "agent.md": AGENT_DOC_B64, "infos.md": INFOS_DOC_B64 };
  const upToDate = so.includes("vm-doc-version: " + DOC_VER);
  for (const name of Object.keys(docs)) {
    const missing = so.includes("MISSING:" + name);
    /* Réécrit si le fichier manque, ou si sa version ne correspond plus au
       contenu canonique embarqué (les correctifs de doc arrivent donc tout
       seuls sur la VM, sans intervention). */
    if (missing || (!upToDate && so.includes("OK:" + name))) {
      await vmExecCmd(env,
        "mkdir -p /home/kasm-user/user && printf '%s' '" + docs[name] + "' | base64 -d > /home/kasm-user/user/" + name);
      if (missing) out.restored.push(name); else out.updated.push(name);
    } else if (so.includes("OK:" + name)) {
      out.present.push(name);
    }
  }
  /* user.md : fiche de l'utilisateur — restaurée depuis sa sauvegarde si
     possible, sinon depuis le modèle par défaut. Jamais écrasée si présente. */
  if (so.includes("MISSING:user.md")) {
    await vmExecCmd(env,
      "mkdir -p /home/kasm-user/user && if [ -s /home/kasm-user/.user.md.bak ]; then cp -f /home/kasm-user/.user.md.bak /home/kasm-user/user/user.md; else printf '%s' '" + USER_DOC_B64 + "' | base64 -d > /home/kasm-user/user/user.md; fi");
    out.restored.push("user.md");
  } else if (so.includes("OK:user.md")) {
    out.present.push("user.md");
  }
  return out;
}

/* base64 d'une chaîne UTF-8 (pour transmettre du JSON au terminal de la VM). */
function b64Utf8(str) {
  const bytes = new TextEncoder().encode(str);
  let bin = "";
  for (let i = 0; i < bytes.length; i++) bin += String.fromCharCode(bytes[i]);
  return btoa(bin);
}

async function fetchHandler(request, env, ctx) {
    const url = new URL(request.url);

    if (request.method === "OPTIONS") {
      return new Response(null, {
        status: 204,
        headers: {
          "Access-Control-Allow-Origin": "*",
          "Access-Control-Allow-Methods": "POST, GET, OPTIONS",
          "Access-Control-Allow-Headers": "Content-Type, X-Access-Code",
          "Access-Control-Max-Age": "86400",
        },
      });
    }

    /* --- Relais WebSocket vers l'API Gemini Live (la clé Google reste ici) --- */
    if (url.pathname === "/api/live") {
      if (env.ACCESS_CODE && url.searchParams.get("code") !== env.ACCESS_CODE) {
        return new Response("code d'accès invalide", { status: 401 });
      }
      const pair = new WebSocketPair();
      const clientWs = pair[0], serverWs = pair[1];
      serverWs.accept();
      const ver = url.searchParams.get("v") === "v1alpha" ? "v1alpha" : "v1beta";
      const gWs = new WebSocket(`wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.${ver}.GenerativeService.BidiGenerateContent?key=` + env.GOOGLE_API_KEY);
      const queue = [];
      let gOpen = false;
      const closeBoth = () => { try { gWs.close(); } catch (e) {} try { serverWs.close(); } catch (e) {} };
      gWs.addEventListener("open", () => {
        gOpen = true;
        while (queue.length) { try { gWs.send(queue.shift()); } catch (e) {} }
      });
      gWs.addEventListener("message", ev => { try { serverWs.send(ev.data); } catch (e) {} });
      gWs.addEventListener("close", closeBoth);
      gWs.addEventListener("error", closeBoth);
      serverWs.addEventListener("message", ev => {
        if (gOpen) { try { gWs.send(ev.data); } catch (e) {} }
        else queue.push(ev.data);
      });
      serverWs.addEventListener("close", () => { try { gWs.close(); } catch (e) {} });
      return new Response(null, { status: 101, webSocket: clientWs });
    }

    if (url.pathname === "/api/models") {
      return json({ models: Object.keys(MODELS), image_models: [...Object.keys(IMG_MODELS), ...Object.keys(HF_IMG_MODELS)] });
    }

    /* ---- génération d'image (Cloudflare Workers AI) ------------------ */
    /* --- Vidéo (Wan 2.2) ------------------------------------------------- */
    if (url.pathname === "/api/video" && request.method === "POST") {
      if (env.ACCESS_CODE && request.headers.get("X-Access-Code") !== env.ACCESS_CODE) {
        return json({ error: "code d'accès invalide" }, 401);
      }
      let vb;
      try {
        vb = await request.json();
      } catch {
        return json({ error: "corps JSON invalide" }, 400);
      }
      const vmodel = String(vb.model || "");
      if (!HF_VID_MODELS[vmodel]) return json({ error: `modèle vidéo inconnu : ${vmodel}` }, 400);
      const vprompt = String(vb.prompt ?? "").trim();
      if (!vprompt) return json({ error: "prompt vide" }, 400);
      if (vprompt.length > 1500) return json({ error: "prompt trop long (max 1500 caractères)" }, 400);
      const vdur = Math.min(6, Math.max(2, int(vb.duration, 2, 6) || 4));

      /* 1) Pollinations : la seule voie qui fait vraiment texte→vidéo, mais
         elle coûte des Pollen. Solde vide (402) => on bascule sans rien dire. */
      let pollErr = "";
      if (env.POLLINATIONS_API_KEY && POLL_VID_MAP[vmodel]) {
        try {
          const vu = `${POLL_URL}/video/${encodeURIComponent(vprompt)}?model=${encodeURIComponent(POLL_VID_MAP[vmodel])}&duration=${vdur}`;
          const vr = await fetch(vu, { headers: { Authorization: "Bearer " + env.POLLINATIONS_API_KEY } });
          const vct = vr.headers.get("content-type") || "";
          if (vr.ok && vct.indexOf("video/") === 0) {
            const vbuf = await vr.arrayBuffer();
            if (vbuf.byteLength > 0) {
              return json({ video: b64encode(vbuf), mime: vct.split(";")[0], via: "pollinations", duration: vdur });
            }
            pollErr = "Pollinations a renvoyé une vidéo vide";
          } else {
            let vraw = "";
            try { vraw = await vr.text(); } catch { /* rien */ }
            pollErr = pollErrorMessage(vr.status, vraw);
          }
        } catch (err) {
          pollErr = "Pollinations injoignable : " + err.message;
        }
      }

      /* 2) Space public Wan 2.2 : gratuit, mais il faut lui fournir une image. */
      const vtok = String(vb.hfToken || "").trim();
      const vres = await hfVideoSafe(env, vprompt, vdur, { image: vb.image, imageMime: vb.imageMime,
        posterModel: vb.posterModel, hfToken: /^hf_[A-Za-z0-9]{10,}$/.test(vtok) ? vtok : "" });
      if (vres.error) {
        return json({ error: (pollErr ? pollErr + " · " : "") + vres.error, poster: vres.poster || null }, 502);
      }
      return json({
        video: vres.video,
        mime: vres.mime || "video/mp4",
        poster: vres.poster || null,
        posterMime: vres.posterMime || "image/png",
        duration: vres.duration || vdur,
        starter: vres.starter || "worker",
        via: "hf",
        note: pollErr || null,
      });
    }

    if (url.pathname === "/api/image" && request.method === "POST") {
      if (env.ACCESS_CODE && request.headers.get("X-Access-Code") !== env.ACCESS_CODE) {
        return json({ error: "code d'accès invalide" }, 401);
      }
      if (!env.CLOUDFLARE_AI_TOKEN) {
        return json({ error: "secret CLOUDFLARE_AI_TOKEN manquant côté Worker" }, 500);
      }
      let ib;
      try {
        ib = await request.json();
      } catch {
        return json({ error: "corps JSON invalide" }, 400);
      }
      /* Diagnostic (aucun modèle n'est ajouté à l'app) : `params._cfmodel`
         envoie une requête à un modèle Workers AI par son identifiant EXACT
         et rend la réponse brute. Un identifiant `@cf/…` passe par la voie
         classique (l'entrée est le corps), un identifiant sans préfixe par la
         voie « model + input ». Sert à vérifier un modèle avant de l'écrire
         quelque part. */
      const cfProbe = ib.params && ib.params._cfmodel;
      if (cfProbe) {
        const cm = String(cfProbe);
        const cfres = String((ib.params && ib.params._cfres) || "1k");
        const input = { prompt: String(ib.prompt || ""), resolution: cfres };
        const classic = cm.indexOf("@cf/") === 0;
        const purl = "https://api.cloudflare.com/client/v4/accounts/" + CF_ACCOUNT_ID + "/ai/run" + (classic ? "/" + cm : "");
        const pbody = classic ? input : { model: cm, input: input };
        let pr = null;
        try {
          pr = await fetch(purl, { method: "POST",
            headers: { Authorization: "Bearer " + env.CLOUDFLARE_AI_TOKEN, "Content-Type": "application/json" },
            body: JSON.stringify(pbody) });
        } catch (e) { return json({ ok: false, modele: cm, erreur: "Cloudflare injoignable : " + (e && e.message ? e.message : e) }, 502); }
        const raw = await pr.text();
        let rr = null; try { rr = JSON.parse(raw); } catch { rr = null; }
        const img = rr && rr.result && (rr.result.image ||
          (rr.result.images && rr.result.images[0]) || rr.result.sample ||
          (rr.result.data && rr.result.data[0] && rr.result.data[0].url));
        if (img && /^https?:/.test(String(img))) {
          try {
            const im2 = await fetch(String(img));
            const buf = new Uint8Array(await im2.arrayBuffer());
            return json({ ok: true, modele: cm, voie: classic ? "@cf" : "model", statut: pr.status,
              mime: (im2.headers.get("Content-Type") || "").split(";")[0], octets: buf.length, image: b64encode(buf) });
          } catch (e) { return json({ ok: true, modele: cm, lien: String(img), note: "image non téléchargée : " + (e && e.message ? e.message : e) }); }
        }
        return json({ ok: pr.ok, modele: cm, voie: classic ? "@cf" : "model", statut: pr.status,
          reponse: String(raw).slice(0, 1600) }, pr.ok ? 200 : 502);
      }
      /* Modèles d'image servis par ModelScope (Qwen Image 2.1, Wan 2.2…). */
      if (MS_IMG[ib.model]) return await msImage(env, MS_IMG[ib.model], ib);
      if (HF_IMG_MODELS[ib.model]) return await hfImage(HF_IMG_MODELS[ib.model], ib, env);
      if (POLL_IMG_MAP[ib.model]) {
        if (!env.POLLINATIONS_API_KEY) return json({ error: "secret POLLINATIONS_API_KEY manquant côté Worker" }, 500);
        const p = ib.params && typeof ib.params === "object" ? ib.params : {};
        /* contraintes MAI : multiples de 16, 768 px min par côté, 1 048 576 px² max */
        const q16 = (v) => { const n = Math.round(Number(v) / 16) * 16; return Math.min(1024, Math.max(768, Number.isFinite(n) && n > 0 ? n : 1024)); };
        const editB64 = String(ib.image || "");
        if (editB64) {   /* toute image jointe => édition (si modèle Pollinations) */
          /* Édition d'image : l'utilisateur a joint une image existante à modifier. */
          const fd = new FormData();
          fd.append("model", POLL_IMG_MAP[ib.model]);
          fd.append("prompt", String(ib.prompt || ""));
          fd.append("size", q16(p.width || 1024) + "x" + q16(p.height || 1024));
          fd.append("image", new Blob([b64ToBytes(editB64)], { type: "image/png" }), "image.png");
          let ue;
          try {
            ue = await fetch(POLL_URL + "/v1/images/edits", {
              method: "POST",
              headers: { Authorization: "Bearer " + env.POLLINATIONS_API_KEY },
              body: fd,
            });
          } catch (err) { return json({ error: "Pollinations injoignable (édition) : " + err.message }, 502); }
          const te = await ue.text();
          let je; try { je = JSON.parse(te); } catch { je = null; }
          if (!ue.ok) return json({ error: pollErrorMessage(ue.status, te) }, (ue.status === 402 || ue.status === 429 || ue.status === 503) ? ue.status : 502);
          const be = je && je.data && je.data[0] && je.data[0].b64_json;
          if (!be) return json({ error: "aucune image dans la réponse Pollinations (édition)" }, 502);
          return json({ image: be, mime: detectMime(be) });
        }
        const qb = { model: POLL_IMG_MAP[ib.model], prompt: String(ib.prompt || ""), size: q16(p.width || 1024) + "x" + q16(p.height || 1024), response_format: "b64_json" };
        if (p.quality) qb.quality = String(p.quality);
        let up;
        try {
          up = await fetch(POLL_URL + "/v1/images/generations", {
            method: "POST",
            headers: { "Content-Type": "application/json", Authorization: "Bearer " + env.POLLINATIONS_API_KEY },
            body: JSON.stringify(qb),
          });
        } catch (err) { return json({ error: "Pollinations injoignable : " + err.message }, 502); }
        const t = await up.text();
        let j3; try { j3 = JSON.parse(t); } catch { j3 = null; }
        if (!up.ok) return json({ error: pollErrorMessage(up.status, t) }, (up.status === 402 || up.status === 429 || up.status === 503) ? up.status : 502);
        const b64 = j3 && j3.data && j3.data[0] && j3.data[0].b64_json;
        if (!b64) return json({ error: "aucune image dans la réponse Pollinations" }, 502);
        return json({ image: b64, mime: detectMime(b64) });
      }
      const slug = IMG_MODELS[ib.model];
      if (!slug) return json({ error: `modèle d'image inconnu : ${String(ib.model)}` }, 400);
      const iprompt = String(ib.prompt ?? "").trim();
      if (!iprompt) return json({ error: "prompt vide" }, 400);
      if (iprompt.length > 4000) return json({ error: "prompt trop long (max 4000 caractères)" }, 400);

      const fd = new FormData();
      fd.append("prompt", iprompt);
      const ip = ib.params && typeof ib.params === "object" ? ib.params : {};
      const iw = int(ip.width, 256, 2048);
      const ih = int(ip.height, 256, 2048);
      const isteps = int(ip.num_steps, 1, 50);
      const ig = num(ip.guidance, 0, 10);
      if (iw !== undefined) fd.append("width", String(iw));
      if (ih !== undefined) fd.append("height", String(ih));
      if (isteps !== undefined) fd.append("num_steps", String(isteps));
      if (ig !== undefined) fd.append("guidance", String(ig));
      /* Édition : FLUX.2 [klein] accepte jusqu'à quatre images de référence,
         en binaire, chacune plus petite que 512 × 512. L'app joint donc une
         version réduite (`image_tiny`) et RIEN D'AUTRE : une seule image,
         celle que l'on veut modifier. */
      const refB64 = String(ib.image_tiny || "");
      if (refB64) {
        const rb = b64ToBytes(refB64);
        if (rb && rb.length) {
          fd.append("input_image_0", new Blob([rb], { type: detectMime(refB64) || "image/png" }), "reference.png");
        }
      }

      let iup;
      try {
        iup = await fetch(`https://api.cloudflare.com/client/v4/accounts/${CF_ACCOUNT_ID}/ai/run/@cf/${slug}`, {
          method: "POST",
          headers: { Authorization: `Bearer ${env.CLOUDFLARE_AI_TOKEN}` },
          body: fd,
        });
      } catch (err) {
        return json({ error: `Cloudflare AI injoignable : ${err.message}` }, 502);
      }
      /* Compteur de neurones : Cloudflare indique ce que l'appel a coûté.
         On l'additionne au journal du jour (objet durable « credits »). */
      try {
        const nn = Number(iup.headers.get("cf-ai-neurons") || 0);
        if (nn > 0) await ledgerAdd(env, nn);
      } catch (e) { /* le compteur est un bonus, jamais bloquant */ }
      /* Certaines versions rendent directement les octets de l'image. */
      const ctype = String(iup.headers.get("content-type") || "");
      if (iup.ok && ctype.indexOf("image/") === 0) {
        const raw = new Uint8Array(await iup.arrayBuffer());
        if (raw.length) return json({ image: b64encode(raw), mime: ctype.split(";")[0] });
      }
      const itext = await iup.text();
      if (!iup.ok) {
        let imsg = itext.slice(0, 400);
        try {
          const id = JSON.parse(itext);
          if (id.errors && id.errors[0] && id.errors[0].message) imsg = id.errors[0].message;
        } catch { /* on garde le texte brut */ }
        return json({ error: `Cloudflare AI HTTP ${iup.status} — ${imsg}` }, iup.status === 429 ? 429 : 502);
      }
      let idata;
      try {
        idata = JSON.parse(itext);
      } catch {
        return json({ error: "réponse Cloudflare AI invalide" }, 502);
      }
      const iimg = idata && idata.result ? idata.result.image : null;
      if (!iimg) return json({ error: "aucune image dans la réponse Cloudflare AI" }, 502);
      return json({ image: iimg, mime: (idata.result && idata.result.mime) || "image/jpeg" });
    }

    /* --- Streaming SSE (modèles texte Groq et Google) --- */
    if (url.pathname === "/api/chat/stream" && request.method === "POST") {
      if (env.ACCESS_CODE && request.headers.get("X-Access-Code") !== env.ACCESS_CODE) {
        return json({ error: "code d'accès invalide" }, 401);
      }
      let body;
      try { body = await request.json(); } catch { return json({ error: "corps JSON invalide" }, 400); }
      const model = body.model;
      const prompt = String(body.prompt ?? "").trim();
      if (!prompt) return json({ error: "prompt vide" }, 400);
      const params = body.params && typeof body.params === "object" ? body.params : {};
      const isGroq = !!GROQ_MAP[model];
      const isPoll = !!POLL_TEXT_MAP[model];
      const isIfm = !!IFM_MAP[model];
      /* Première image jointe (si l'utilisateur en a joint une) : le relais
         ModelScope doit voir exactement la même demande que Groq. */
      const msImage = (normImageList(body)[0] || {}).b64 || "";
      const gcfg = GOOGLE_MAP[model];
      if (!isGroq && !isPoll && !isIfm && !(gcfg && !gcfg.kind)) return json({ error: "streaming non supporté pour ce modèle" }, 400);

      let up;
      try {
        if (isGroq) {
          const gq = { model: GROQ_MAP[model], messages: [{ role: "user", content: prompt }], stream: true };
          if (params.temperature !== undefined && params.temperature !== null && params.temperature !== "") gq.temperature = Number(params.temperature);
          if (params.top_p) gq.top_p = Number(params.top_p);
          if (params.reasoning_effort) gq.reasoning_effort = String(params.reasoning_effort);
          const baseMax = params.max_tokens ? Math.max(1, Number(params.max_tokens) || 1024) : (GROQ_MAXOUT[model] || 4096);
          const steps = [baseMax];
          for (const s of GROQ_RETRY_MAX) if (s < baseMax) steps.push(s);
          for (let i = 0; i < steps.length; i++) {
            gq.max_completion_tokens = steps[i];
            const r = await fetch("https://api.groq.com/openai/v1/chat/completions", {
              method: "POST",
              headers: { "Content-Type": "application/json", Authorization: "Bearer " + env.GROQ_API_KEY, "User-Agent": BROWSER_UA },
              body: JSON.stringify(gq),
            });
            if (r.status === 429) {
              const te = await r.text();
              if (GROQ_TOO_LARGE_RE.test(te) && i < steps.length - 1) continue;
              /* Groq saturé : la réponse arrive alors d'un bloc depuis
                 ModelScope, l'application l'affiche exactement pareil. */
              const ms = await msTextRelay(env, model, prompt, params, msImage);
              if (ms) {
                const mj = await ms.json();
                return pollFallbackStream({ output: String(mj.output || ""), applied: ["modelscope"] });
              }
              return json({ error: groqApiError(429, te) }, 429);
            }
            up = r;
            break;
          }
        } else if (isIfm) {
          const rq = { model: IFM_MAP[model], messages: [{ role: "user", content: prompt }], stream: true };
          if (params.max_tokens) rq.max_tokens = Math.max(1, Number(params.max_tokens) || 1024);
          if (params.temperature !== undefined && params.temperature !== null && params.temperature !== "") rq.temperature = Number(params.temperature);
          up = await fetch(IFM_URL + "/chat/completions", {
            method: "POST",
            headers: { "Content-Type": "application/json", Authorization: "Bearer " + env.IFM_API_KEY },
            body: JSON.stringify(rq),
          });
        } else if (isPoll) {
          const parts = [];
          for (const im of normImageList(body)) {
            parts.push({ type: "text", text: "[image « " + im.name + " »]" });
            parts.push({ type: "image_url", image_url: { url: dataUri(im) } });
          }
          parts.push({ type: "text", text: prompt });
          const rq = { model: POLL_TEXT_MAP[model], messages: [{ role: "user", content: parts }], stream: true };
          if (params.max_tokens) rq.max_tokens = Math.max(1, Number(params.max_tokens) || 1024);
          if (params.temperature !== undefined && params.temperature !== null && params.temperature !== "") rq.temperature = Number(params.temperature);
          if (params.reasoning_effort) rq.reasoning_effort = String(params.reasoning_effort);
          up = await fetch(POLL_URL + "/v1/chat/completions", {
            method: "POST",
            headers: { "Content-Type": "application/json", Authorization: "Bearer " + env.POLLINATIONS_API_KEY },
            body: JSON.stringify(rq),
          });
          if (!up.ok || up.status === 429) {
            const te = await up.text();
            /* Même bascule qu'en mode normal : le texte entier arrive dans un
               seul bloc, l'app n'a rien à faire de particulier. */
            const fb = await pollToRoboflow(env, body, model);
            if (fb) return pollFallbackStream(fb);
            return json({ error: pollErrorMessage(up.status, te) }, up.status);
          }
        } else {
          const H = { "Content-Type": "application/json", "x-goog-api-key": env.GOOGLE_API_KEY };
          const parts = [{ text: prompt }];
          for (const im of normImageList(body)) parts.unshift({ inline_data: { mime_type: im.mime, data: im.b64 } });
          const gc = {};
          if (params.max_tokens) gc.maxOutputTokens = Number(params.max_tokens);
          if (params.temperature !== undefined && params.temperature !== null && params.temperature !== "") gc.temperature = Number(params.temperature);
          if (params.thinking_level) gc.thinkingConfig = { thinkingLevel: String(params.thinking_level) };
          const tools = [];
          if (params.google_search) tools.push({ google_search: {} });
          if (params.google_code_execution) tools.push({ code_execution: {} });
          if (params.google_url_context) tools.push({ url_context: {} });
          if (params.google_maps) tools.push({ google_maps: {} });
          const reqBody = { contents: [{ role: "user", parts }] };
          if (Object.keys(gc).length) reqBody.generationConfig = gc;
          if (tools.length) reqBody.tools = tools;
          up = await fetch("https://generativelanguage.googleapis.com/v1beta/models/" + gcfg.m + ":streamGenerateContent?alt=sse", {
            method: "POST", headers: H, body: JSON.stringify(reqBody),
          });
        }
      } catch (err) { return json({ error: "API injoignable : " + err.message }, 502); }
      if (!up.ok) {
        const t = await up.text();
        if (isGroq) {
          if (groqCanRelay(up.status, t)) {
            const ms = await msTextRelay(env, model, prompt, params, msImage);
            if (ms) {
              const mj = await ms.json();
              return pollFallbackStream({ output: String(mj.output || ""), applied: ["modelscope"] });
            }
          }
          return json({ error: groqApiError(up.status, t) }, (up.status === 413 || up.status === 429) ? up.status : 502);
        }
        if (isPoll) {
          const fb = await pollToRoboflow(env, body, model);
          if (fb) return pollFallbackStream(fb);
          return json({ error: pollErrorMessage(up.status, t) }, (up.status === 402 || up.status === 429) ? up.status : 502);
        }
        if (isIfm) {
          let m2 = t.slice(0, 300);
          try { const j2 = JSON.parse(t); m2 = (j2.error && (j2.error.message || j2.error)) || m2; } catch {}
          return json({ error: "IFM HTTP " + up.status + " — " + m2 }, up.status === 429 ? 429 : 502);
        }
        return json({ error: "Google HTTP " + up.status + " — " + t.slice(0, 200) }, 502);
      }
      const enc = new TextEncoder();
      /* Pour Pollinations : lit la tête du flux avant de le relayer ; si le
         message « crédits » apparaît, répond 402 (bascule Roboflow côté app). */
      const reader = up.body.getReader();
      const dec0 = new TextDecoder();
      const preBuf = [];
      let preText = "", rawLeft = "";
      if (isPoll) {
        const t0 = Date.now();
        while (preText.length < 300 && Date.now() - t0 < 1500) {
          const r1 = await reader.read();
          if (r1.done) break;
          rawLeft += dec0.decode(r1.value, { stream: true });
          let ix;
          while ((ix = rawLeft.indexOf("\n")) >= 0) {
            const line = rawLeft.slice(0, ix).trim();
            rawLeft = rawLeft.slice(ix + 1);
            if (!line.startsWith("data:")) continue;
            const payload = line.slice(5).trim();
            if (payload === "[DONE]") continue;
            let j1; try { j1 = JSON.parse(payload); } catch { continue; }
            const d1 = j1.choices && j1.choices[0] && j1.choices[0].delta && j1.choices[0].delta.content;
            if (d1) { preText += d1; preBuf.push({ t: d1 }); }
          }
          if (pollCreditText(preText)) break;
        }
        if (pollCreditText(preText)) {
          try { await reader.cancel(); } catch {}
          /* Message de crédits dans la tête du flux : même bascule que dans
             l'autre sens de lecture (c'était le trou : on répondait 402 sans
             essayer Roboflow, alors que c'est la voie que prend l'app). */
          const fb = await pollToRoboflow(env, body, model);
          if (fb) return pollFallbackStream(fb);
          return json({ error: pollErrorMessage(402, preText) }, 402);
        }
      }
      const sse = new ReadableStream({
        async start(controller) {
          const dec = new TextDecoder();
          let buf = rawLeft;
          const send = (o) => controller.enqueue(enc.encode("data: " + JSON.stringify(o) + "\n\n"));
          for (const pb of preBuf) send(pb);
          try {
            while (true) {
              const { done, value } = await reader.read();
              if (done) break;
              buf += dec.decode(value, { stream: true });
              let idx;
              while ((idx = buf.indexOf("\n")) >= 0) {
                const line = buf.slice(0, idx).trim();
                buf = buf.slice(idx + 1);
                if (!line.startsWith("data:")) continue;
                const payload = line.slice(5).trim();
                if (payload === "[DONE]") continue;
                let j; try { j = JSON.parse(payload); } catch { continue; }
                if (isGroq || isPoll || isIfm) {
                  const ch0 = j.choices && j.choices[0];
                  const d = ch0 && ch0.delta && ch0.delta.content;
                  if (d) send({ t: d });
                  if (ch0 && ch0.finish_reason === "length") send({ t: "\n\n*(réponse interrompue : limite de tokens atteinte — relancez pour continuer)*" });
                } else {
                  const ps = (j.candidates && j.candidates[0] && j.candidates[0].content && j.candidates[0].content.parts) || [];
                  for (const p of ps) {
                    const tx = p.text || (p.audioTranscription && p.audioTranscription.text);
                    if (tx) send({ t: tx });
                  }
                }
              }
            }
          } catch (e) {
            try { send({ error: "flux interrompu : " + e.message }); } catch {}
          }
          controller.enqueue(enc.encode("data: [DONE]\n\n"));
          controller.close();
        },
      });
      return new Response(sse, {
        headers: {
          "Content-Type": "text/event-stream; charset=utf-8",
          "Access-Control-Allow-Origin": "*",
          "Access-Control-Allow-Methods": "POST, GET, OPTIONS",
          "Access-Control-Allow-Headers": "Content-Type, X-Access-Code",
          "Cache-Control": "no-store",
        },
      });
    }

    if (url.pathname === "/app" || url.pathname === "/app/") {
      return new Response(APP_HTML, {
        headers: { "Content-Type": "text/html; charset=utf-8", "Cache-Control": "no-store" },
      });
    }

    if (url.pathname === "/canal" || url.pathname === "/canal/" || url.pathname.indexOf("/canal/") === 0) {
      return canalRoute(env, request, url);
    }

    /* ---- Canal Arena ---------------------------------------------------- */
    if (url.pathname === "/api/arena/send" || url.pathname === "/api/arena/poll") {
      if (env.ACCESS_CODE && request.headers.get("X-Access-Code") !== env.ACCESS_CODE) {
        return json({ error: "code d'accès invalide" }, 401);
      }
      if (url.pathname === "/api/arena/poll" && request.method === "GET") {
        const r = await arenaRead(env, Number(url.searchParams.get("n") || 30));
        return json(r, r.ok ? 200 : 502);
      }
      if (url.pathname === "/api/arena/send" && request.method === "POST") {
        let b = {};
        try { b = await request.json(); } catch { /* corps illisible */ }
        const kind = String(b.kind || "text");
        /* Rôle : l'application écrit toujours en tant qu'UTILISATEUR, et son
           résultat de commande en tant que « tool ». Elle ne peut PAS écrire
           au nom d'Arena (sinon elle se répondrait à elle-même). C'est le
           script z-canal.py qui écrit « arena », directement dans Firestore. */
        const role = kind === "tool" ? "tool" : "user";
        const text = String(b.text == null ? "" : b.text);
        if (!text && kind !== "tool") return json({ error: "texte vide" }, 400);
        const now = Date.now();
        const w = await arenaWrite(env, {
          role: fsStr(role), text: fsStr(text), ts: fsInt(now), req: fsStr(b.req || ""),
          /* Un résultat de commande est terminal : rien à traiter côté app. */
          state: fsStr(kind === "tool" ? "done" : "new"), kind: fsStr(kind), src: fsStr("worker"),
          note: fsStr(b.note || ""),
        });
        if (!w.ok) return json(w, 502);
        ARENA_LAST = { at: now, action: "send", role: role, kind: kind, taille: text.length, id: w.id };
        /* Une COMMANDE : on attend ici son exécution par l'application, et on
           rend son résultat dans la même réponse. Plus besoin d'interroger le
           canal à la main pour savoir ce que la commande a donné. */
        if (kind === "call") {
          const waitS = Math.max(0, Math.min(20, Number(b.wait || url.searchParams.get("wait") || 0) || 0));
          if (waitS > 0) {
            const t0 = Date.now();
            let tool = null;
            while (Date.now() - t0 < waitS * 1000) {
              tool = await canalResultFor(env, "arena", w.id);
              if (tool) break;
              await new Promise((r) => setTimeout(r, 1200));
            }
            return json({ ok: true, id: w.id, role: role, kind: kind,
              waited: Math.round((Date.now() - t0) / 100) / 10,
              tool: tool ? canalPublic(tool) : null,
              note: tool ? "" : "aucun résultat : l'application est-elle ouverte, et la discussion en mode agent ?" });
          }
        }
        /* L'application est-elle ouverte ? Le marqueur de présence est frais
           (moins de trois minutes) → l'appelant sait s'il parle dans le vide. */
        return json({ ok: true, id: w.id, role: role, device: await arenaAppOnline(env) });
      }
      return json({ error: "route arena inconnue" }, 404);
    }

    /* ---- ModelScope : appel quotidien et catalogue -----------------------
       Ces deux routes vivent chez le worker (aucun relais VM) : le jeton
       reste dans le secret MODELSCOPE_API_KEY et n'est jamais rendu. */
    if (url.pathname === "/api/ms/daily" || url.pathname === "/api/ms/models" || url.pathname === "/api/ms/state") {
      if (env.ACCESS_CODE && request.headers.get("X-Access-Code") !== env.ACCESS_CODE) {
        return json({ error: "code d'accès invalide" }, 401);
      }
      /* Appel quotidien de l'application : le jeton s'authentifie auprès de
         ModelScope (identité + solde de crédits gratuits). L'état « oui / non »
         du jour est tenu par l'application, dans Firestore. */
      if (url.pathname === "/api/ms/daily" && request.method === "POST") {
        const r = await msDaily(env);
        const day = new Date().toISOString().slice(0, 10);
        return json(Object.assign({ day: day }, r));
      }
      /* Diagnostic (lecture seule) : ce qui a été envoyé au service d'images
         de ModelScope et ce qui en est revenu — pour comprendre un échec
         sans jamais toucher au jeton. */
      if (url.pathname === "/api/ms/state" && request.method === "GET") {
        return json({ ok: true, image: MS_IMG_LAST, relaisTexte: { nombre: MS_TEXT_RELAY, dernier: MS_TEXT_RELAY_LAST }, arena: ARENA_LAST });
      }
      /* Catalogue libre : sert à vérifier les noms et les identifiants
         configurés (lecture seule, aucun crédit consommé). */
      if (url.pathname === "/api/ms/models" && request.method === "GET") {
        try {
          const up = await fetch(MS_HOSTS[0] + "/models", { headers: { "User-Agent": BROWSER_UA } });
          const j = await up.json();
          const ids = (j && j.data ? j.data : []).map((m) => m.id);
          const conf = {};
          for (const [k, v] of Object.entries(MS_TEXT)) conf[k] = { id: v, present: ids.indexOf(v) >= 0 };
          for (const [k, v] of Object.entries(MS_IMG)) conf[k] = { id: v, present: ids.indexOf(v) >= 0 };
          return json({ ok: true, count: ids.length, configured: conf, models: ids });
        } catch (e) {
          return json({ ok: false, error: "ModelScope injoignable : " + (e && e.message ? e.message : e) }, 502);
        }
      }
      return json({ error: "route ModelScope inconnue" }, 404);
    }


/* ---- Crédits : l'état des trois comptes qui font tourner l'app ----------
   GET /api/credits — soldes et états lisibles, JAMAIS un secret. Sert à
   savoir d'un coup d'œil quel tuyau est encore ouvert. Réponse mise en
   cache une minute : ouvrir les réglages ne consomme rien. */
/* Journal des neurones Cloudflare, tenu par l'objet durable « credits ». */
async function ledgerAdd(env, n) {
  if (!env || !env.MISSION || !n) return;
  try {
    const stub = env.MISSION.get(env.MISSION.idFromName("credits"));
    await stub.fetch("https://mission/ledger", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ neurons: n }),
    });
  } catch (e) { /* jamais bloquant */ }
}
async function ledgerGet(env) {
  const day = new Date().toISOString().slice(0, 10);
  if (!env || !env.MISSION) return { day: day, neurons: 0, mesure: false };
  try {
    const stub = env.MISSION.get(env.MISSION.idFromName("credits"));
    const r = await stub.fetch("https://mission/ledger");
    const j = await r.json();
    return { day: (j && j.day) || day, neurons: Number(j && j.neurons) || 0, mesure: true };
  } catch (e) { return { day: day, neurons: 0, mesure: false }; }
}
/* Lecture LÉGÈRE : le seul solde ModelScope. L'écran Disponibilité n'a
   besoin que de ça et des neurones ; interroger en plus Pollinations et le
   catalogue faisait attendre quinze secondes pour deux chiffres. */
async function msBalance(env) {
  const key = msKey(env);
  if (!key) return { ok: false, token: "absent", balance: null, error: "secret MODELSCOPE_API_KEY manquant" };
  const H = { Authorization: "Bearer " + key, "User-Agent": BROWSER_UA, Accept: "application/json" };
  for (const site of MS_SITES) {
    try {
      const r = await fetch(site + "/openapi/v1/magicubes/balance", { headers: H });
      const t = await r.text();
      let j = null; try { j = JSON.parse(t); } catch (e) { j = null; }
      if (r.status === 401 || r.status === 403) {
        return { ok: false, token: "refusé", balance: null, error: "jeton refusé par ModelScope (HTTP " + r.status + ")" };
      }
      if (r.ok && j && j.success && j.data) {
        return { ok: true, token: "valide", error: "",
          balance: { total: j.data.total_balance, available: j.data.available_balance, frozen: j.data.frozen_amount } };
      }
    } catch (e) { /* site suivant */ }
  }
  return { ok: false, token: "inconnu", balance: null, error: "ModelScope injoignable" };
}
const CREDITS_CACHE = { at: 0, data: null };
async function pollCredits(env) {
  const key = String(env.POLLINATIONS_API_KEY || "").trim();
  const out = { ok: false, cle: "absente", balance: null, account: null, quests: null, error: "" };
  if (!key) { out.error = "secret POLLINATIONS_API_KEY manquant côté Worker"; return out; }
  const H = { Authorization: "Bearer " + key, "User-Agent": BROWSER_UA, Accept: "application/json" };
  try {
    const r = await fetch(POLL_URL + "/account/balance", { headers: H });
    const t = await r.text();
    let j = null; try { j = JSON.parse(t); } catch (e) { j = null; }
    if (r.status === 401 || r.status === 403) {
      out.cle = "refusée";
      out.error = "clé refusée par Pollinations (HTTP " + r.status + ")";
      return out;
    }
    if (!r.ok) {
      out.error = "Pollinations HTTP " + r.status + " — " + String((j && j.error && (j.error.message || j.error)) || t).slice(0, 180);
      return out;
    }
    out.cle = "valide";
    out.ok = true;
    out.balance = (j && j.balance != null) ? j.balance : null;
    out.account = (j && j.accountBalance) || null;
  } catch (e) {
    out.error = "Pollinations injoignable : " + (e && e.message ? e.message : e);
    return out;
  }
  /* Quêtes : lecture seule. La documentation le dit noir sur blanc —
     « Claiming rewards is dashboard-only » : l'API ne peut pas réclamer. */
  try {
    const r2 = await fetch(POLL_URL + "/account/quests", { headers: H });
    if (r2.ok) {
      const j2 = await r2.json().catch(() => null);
      const list = (j2 && (j2.quests || (j2.data && j2.data.quests))) || [];
      let dispo = 0, gain = 0;
      const aFaire = [];
      for (const q of list) {
        const st = String(q.state || q.status || "");
        if (st === "available" || st === "claimable") {
          dispo++;
          const gainQ = Number(q.rewardAmount || 0) || 0;
          gain += gainQ;
          /* Le titre et le lien suffisent à savoir quoi faire ; le montant
             permet de commencer par ce qui rapporte le plus. */
          if (aFaire.length < 12) aFaire.push({
            titre: String(q.title || q.id || "quête").slice(0, 90),
            gain: Math.round(gainQ * 100) / 100,
            lien: typeof q.url === "string" ? q.url.slice(0, 200) : "",
          });
        }
      }
      aFaire.sort((a, b) => b.gain - a.gain);
      out.quests = { disponibles: dispo, gain: Math.round(gain * 100) / 100, total: list.length, liste: aFaire };
    }
  } catch (e) { /* les quêtes sont un bonus : jamais bloquantes */ }
  return out;
}
/* Le listing public /v1/models ne recense qu'une PARTIE du catalogue : un
   modèle absent de cette liste répond très bien quand on l'appelle. On ne
   conclut donc jamais à une disparition — on dit seulement « pas listé ».
   (Seul un appel réel, qui consomme des crédits, trancherait.) */
async function msDrift() {
  try {
    const up = await fetch(MS_HOSTS[0] + "/models", { headers: { "User-Agent": BROWSER_UA } });
    const j = await up.json();
    const ids = ((j && j.data) || []).map((m) => m.id);
    const hors = [];
    for (const [k, v] of Object.entries(MS_TEXT)) if (ids.indexOf(v) < 0) hors.push({ modele: k, id: v });
    for (const [k, v] of Object.entries(MS_IMG)) if (ids.indexOf(v) < 0) hors.push({ modele: k, id: v });
    return { total: ids.length, horsListe: hors, partiel: true };
  } catch (e) {
    return { total: null, horsListe: [], partiel: true, error: "listing injoignable" };
  }
}

    /* ---- Crédits : ce qui reste dans chaque compte ------------------- */
    /* Les noms des secrets et l'état de chacun — jamais les valeurs. */
    if (url.pathname === "/api/secrets" && request.method === "GET") {
      const list = SECRET_CATALOG.map(function (x) {
        const v = secretValue(env, x.name);
        return { name: x.name, api: x.api, label: x.label, present: !!v, fin: secretFin(v) };
      });
      return json({ ok: true, secrets: list });
    }
    if (url.pathname === "/api/secrets/test" && request.method === "POST") {
      let b = {};
      try { b = await request.json(); } catch { /* corps illisible */ }
      const name = String(b.name || "");
      const known = SECRET_CATALOG.some(function (x) { return x.name === name; });
      if (!known) return json({ ok: false, error: "secret inconnu" }, 400);
      const r = await secretTest(env, name);
      return json({ ok: true, resultat: r });
    }
    if (url.pathname === "/api/secrets/test-all" && request.method === "POST") {
      /* Tous les tests partent ensemble, mais par vagues de six : le plan
         gratuit de Workers n'autorise que six connexions sortantes à la fois. */
      const names = SECRET_CATALOG.map(function (x) { return x.name; });
      const results = [];
      for (let i = 0; i < names.length; i += 6) {
        const lot = names.slice(i, i + 6);
        const done = await Promise.all(lot.map(function (n) { return secretTest(env, n); }));
        for (const r of done) results.push(r);
      }
      return json({ ok: true, resultats: results });
    }
    if (url.pathname === "/api/credits" && request.method === "GET") {
      if (env.ACCESS_CODE && request.headers.get("X-Access-Code") !== env.ACCESS_CODE) {
        return json({ error: "code d'accès invalide" }, 401);
      }
      /* `?light=1` : seulement le solde ModelScope et les neurones du jour.
         C'est ce que demande l'écran Disponibilité. */
      if (url.searchParams.get("light") === "1") {
        const [msL, cfL] = await Promise.all([msBalance(env), ledgerGet(env)]);
        const cfFreeL = 10000;
        return json({ ok: true, light: true, cache: false,
          modelscope: msL,
          cloudflare: { neurones: cfL.neurons, jour: cfL.day, gratuit: cfFreeL,
            restant: Math.max(0, Math.round((cfFreeL - cfL.neurons) * 100) / 100),
            mesure: cfL.mesure } });
      }
      const now = Date.now();
      if (CREDITS_CACHE.data && now - CREDITS_CACHE.at < 60000) {
        return json(Object.assign({ ok: true, cache: true }, CREDITS_CACHE.data));
      }
      /* Les trois comptes sont interrogés EN MÊME TEMPS : en série, la page
         restait dix secondes sur « Lecture… ». */
      const [poll, ms, drift, cf] = await Promise.all([pollCredits(env), msDaily(env), msDrift(), ledgerGet(env)]);
      const cfFree = 10000;
      const data = { pollinations: poll, modelscope: Object.assign({}, ms, { catalogue: drift }),
                     cloudflare: { neurones: cf.neurons, jour: cf.day, gratuit: cfFree,
                       restant: Math.max(0, Math.round((cfFree - cf.neurons) * 100) / 100),
                       mesure: cf.mesure },
                     roboflow: { jetons: roboflowTokens(env).length } };
      CREDITS_CACHE.at = now;
      CREDITS_CACHE.data = data;
      return json(Object.assign({ ok: true, cache: false }, data));
    }

    /* ---- Agent : relais vers la VM (worker vmai). Le mot de passe VMAI_PASS
       reste ici, jamais exposé à l'interface. ---- */
    if (url.pathname.startsWith("/api/agent/")) {
      if (env.ACCESS_CODE && request.headers.get("X-Access-Code") !== env.ACCESS_CODE) {
        return json({ error: "code d'accès invalide" }, 401);
      }
      const VM_BASE = "https://vmai.hugdu77777.workers.dev/api/linux";
      const sub = url.pathname.slice("/api/agent/".length);
      const H = { "Authorization": "Bearer " + (env.VMAI_PASS || "") };
      const corsH = { "Access-Control-Allow-Origin": "*" };
      /* Liaison de service (VMAI) : appel worker-à-worker direct, insensible
         à l'état du sous-domaine workers.dev. Repli : fetch HTTP classique. */
      const vmFetch = (u, init) => {
        const rq = new Request(u, init);
        return env.VMAI && typeof env.VMAI.fetch === "function" ? env.VMAI.fetch(rq) : fetch(rq);
      };
      try {
        /* --- missions en arrière-plan --- */
        if (sub === "job/start" && request.method === "POST") {
          let b = {};
          try { b = await request.json(); } catch { /* corps illisible */ }
          if (!b.mission) return json({ error: "mission manquante" }, 400);
          const id = "j" + Date.now().toString(36) + Math.random().toString(36).slice(2, 8);
          /* La discussion et le compte sont transmis à la mission : c'est ce
             qui permet à la passerelle de ranger le résultat tout seul, même
             si l'application est fermée, et sur n'importe quel appareil. */
          const payload = {
            id: id, chat: String(b.chat || ""), uid: String(b.uid || ""),
            model: String(b.model || ""), mission: String(b.mission || ""),
            intro: String(b.intro || ""), rappel: String(b.rappel || ""), hist: String(b.hist || ""),
            image: String(b.image || ""), image_kind: String(b.image_kind || "none"),
            params: (b.params && typeof b.params === "object") ? b.params : {},
            origin: url.origin, code: request.headers.get("X-Access-Code") || "",
          };
          /* Voie principale : une mission = un Durable Object, qui vit sur la
             plateforme et n'a besoin ni de l'app ni d'une requête en cours. */
          if (env.MISSION) {
            try {
              const stub = env.MISSION.get(env.MISSION.idFromName(id));
              const r = await stub.fetch("https://mission/start", {
                method: "POST", headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload),
              });
              const jr = await r.json().catch(() => ({}));
              if (jr && jr.ok) {
                await jobIndexPut(env, { id: id, chat: payload.chat, model: payload.model,
                  mission: payload.mission, at: Date.now(), state: "run" });
                return json({ ok: true, job: id, durable: true, deliver: !!payload.chat });
              }
            } catch (e) { /* on retombe sur le moteur en mémoire */ }
          }
          /* Secours : moteur en mémoire (liaison absente ou Durable Object
             indisponible). La mission avance alors tant que l'app interroge. */
          jobSweep();
          const j = Object.assign({ steps: [], tr: "", finalMd: "", error: "", stop: false,
            step: 0, done: false, running: false, startedAt: Date.now(), updatedAt: Date.now() }, payload);
          JOBS.set(j.id, j);
          try { await jobIndexPut(env, { id: j.id, chat: j.chat, model: j.model,
            mission: j.mission, at: j.startedAt, state: "run" }); } catch (e) {}
          if (ctx && typeof ctx.waitUntil === "function") {
            ctx.waitUntil(jobRun(env, j).catch((e) => { j.error = String(e && e.message ? e.message : e); j.done = true; }));
          } else {
            jobRun(env, j).catch(() => {});
          }
          return json({ ok: true, job: j.id, deliver: !!j.chat });
        }
        if (sub === "job" && request.method === "GET") {
          const id = url.searchParams.get("id") || "";
          if (env.MISSION && id) {
            try {
              const stub = env.MISSION.get(env.MISSION.idFromName(id));
              const r = await stub.fetch("https://mission/state?since="
                + encodeURIComponent(url.searchParams.get("since") || "0"));
              if (r.status !== 404) {
                const jr = await r.json().catch(() => null);
                const jb = jr && jr.job;
                if (jb && jb.id) {
                  /* Une mission suivie par l'app se déclare d'elle-même : même
                     lancée avant l'existence du registre, elle devient listable. */
                  const reg = jobIndexPut(env, { id: jb.id, chat: jb.chat, model: jb.model,
                    mission: jb.mission, at: jb.updatedAt || Date.now(),
                    state: jb.done ? "done" : "run" });
                  if (ctx && typeof ctx.waitUntil === "function") ctx.waitUntil(reg);
                  else await reg;
                }
                return json(jr || {}, r.status);
              }
            } catch (e) { /* on retombe sur la mémoire */ }
          }
          const j = JOBS.get(id);
          if (!j) return json({ ok: false, error: "mission inconnue ou expirée" }, 404);
          /* Sans Durable Object, chaque consultation fait avancer la mission :
             c'est ce qui lui permet de survivre à l'app fermée. */
          if (!j.done && ctx && typeof ctx.waitUntil === "function") {
            ctx.waitUntil(jobRun(env, j).catch((e) => { j.error = String(e && e.message ? e.message : e); j.done = true; }));
          }
          return json({ ok: true, job: jobPublic(j, url.searchParams.get("since")) });
        }
        /* Les missions que l'appareil connaît déjà (mémoire locale, missions
           lancées avant l'existence du registre) se déclarent ici : elles
           deviennent listables, et donc arrêtables. */
        if (sub === "jobs/sync" && request.method === "POST") {
          let b = {};
          try { b = await request.json(); } catch { /* corps illisible */ }
          const list = Array.isArray(b.jobs) ? b.jobs.slice(0, 200) : [];
          let n = 0;
          for (const x of list) {
            if (!x || !x.id) continue;
            const ok = await jobIndexPut(env, { id: String(x.id), chat: String(x.chat || ""),
              model: String(x.model || ""), mission: String(x.mission || ""),
              at: Number(x.at || 0) || Date.now(), state: String(x.state || "run") });
            if (ok) n++;
          }
          return json({ ok: true, registrees: n });
        }
        /* La liste des tâches : le registre donne les missions connues, puis
           chacune est interrogée en direct — un état « en cours » n'est donc
           jamais une supposition. */
        if (sub === "jobs" && request.method === "GET") {
          const entries = await jobIndexList(env);
          const jobs = [];
          for (const e of entries) {
            let live = null;
            if (env.MISSION) {
              try {
                const stub = env.MISSION.get(env.MISSION.idFromName(String(e.id)));
                const r = await stub.fetch("https://mission/state?since=" + String(1e9));
                if (r.status !== 404) {
                  const jr = await r.json().catch(() => null);
                  live = (jr && jr.job) || null;
                }
              } catch (err) { /* la fiche du registre suffit */ }
            }
            const j2 = JOBS.get(String(e.id)) || null;
            jobs.push({
              id: String(e.id), chat: e.chat || "", model: e.model || "",
              mission: e.mission || "", at: e.at || 0,
              state: live ? (live.done ? "done" : "run") : (e.state || "inconnu"),
              done: live ? !!live.done : (e.state === "done"),
              step: live ? (live.step || 0) : null,
              steps: live ? (live.stepsTotal || 0) : null,
              updatedAt: (live && live.updatedAt) || e.updatedAt || e.at || 0,
              local: !!j2,
            });
          }
          jobs.sort((a, b) => (b.at || 0) - (a.at || 0));
          const actives = jobs.filter((x) => !x.done);
          return json({ ok: true, total: jobs.length, actives: actives.length, jobs: jobs });
        }
        if (sub === "job/stop" && request.method === "POST") {
          const id = url.searchParams.get("id") || "";
          let stopped = false;
          if (env.MISSION && id) {
            try {
              const stub = env.MISSION.get(env.MISSION.idFromName(id));
              const force = url.searchParams.get("force");
              const r = await stub.fetch("https://mission/stop"
                + (force === "0" ? "" : "?force=1"), { method: "POST" });
              const jr = await r.json().catch(() => ({}));
              stopped = !!(jr && jr.ok);
            } catch (e) { /* on retombe sur la mémoire */ }
          }
          const j = JOBS.get(id);
          if (j) { j.stop = true; j.done = true; j.updatedAt = Date.now(); stopped = true; }
          if (stopped && id) {
            try { await jobIndexPut(env, { id: id, state: "done" }); } catch (e) {}
          }
          return json({ ok: stopped, forced: url.searchParams.get("force") !== "0" });
        }
        if (sub === "docs/restore") {
          const res = await ensureAgentDocs(env);
          return json(res);
        }
        if (sub === "exec" && request.method === "POST") {
          let b = {};
          try { b = await request.json(); } catch {}
          const payload = { command: String(b.command || "") };
          if (b.user === "root" || b.user === "kasm-user") payload.user = b.user;
          if (typeof b.cwd === "string") payload.cwd = b.cwd.slice(0, 300);
          if (b.timeout) payload.timeout = Math.min(Math.max(1, Number(b.timeout) || 30), 300);
          const up = await vmFetch(VM_BASE + "/exec", {
            method: "POST",
            headers: { ...H, "Content-Type": "application/json" },
            body: JSON.stringify(payload),
          });
          return new Response(up.body, { status: up.status, headers: { "Content-Type": "application/json", ...corsH } });
        }
        if (sub === "screenshot" && request.method === "GET") {
          const up = await vmFetch(VM_BASE + "/screenshot", { headers: H });
          return new Response(up.body, { status: up.status, headers: { "Content-Type": up.headers.get("Content-Type") || "image/png", "Cache-Control": "no-store", ...corsH } });
        }
        if (sub === "fetch" && request.method === "POST") {
          /* Outil 2 : lecture web par Chrome headless sur la VM. */
          let b = {};
          try { b = await request.json(); } catch {}
          const payload = { url: String(b.url || "").slice(0, 2000), mode: b.mode === "html" ? "html" : "text" };
          if (b.wait_ms != null) payload.wait_ms = Math.min(Math.max(0, Number(b.wait_ms) || 0), 15000);
          if (b.timeout != null) payload.timeout = Math.min(Math.max(5, Number(b.timeout) || 45), 180);
          if (typeof b.selector === "string" && b.selector) payload.selector = b.selector.slice(0, 200);
          /* Voie normale : la route dédiée du démon (cycles récents). */
          try {
            const up = await vmFetch(VM_BASE + "/fetch", {
              method: "POST",
              headers: { ...H, "Content-Type": "application/json" },
              body: JSON.stringify(payload),
            });
            if (up.status !== 404) {
              return new Response(up.body, { status: up.status, headers: { "Content-Type": "application/json", ...corsH } });
            }
          } catch (e) {}
          /* Repli : via le terminal de la VM, pour les cycles dont le démon
             n'a pas encore la route /fetch. */
          const waitS = Math.min(Math.max(5, Number(b.timeout) || 45), 180) + 25;
          const cmd = "printf '%s' '" + b64Utf8(JSON.stringify(payload)) + "' | base64 -d | curl -s -m "
            + waitS + " -X POST http://127.0.0.1:3011/fetch -H 'Content-Type: application/json' --data-binary @-";
          const up2 = await vmFetch(VM_BASE + "/exec", {
            method: "POST",
            headers: { ...H, "Content-Type": "application/json" },
            body: JSON.stringify({ command: cmd, timeout: waitS }),
          });
          let j2 = {};
          try { j2 = await up2.json(); } catch (e) {}
          const out = String(j2.stdout || "").trim();
          let parsed = null;
          try { parsed = JSON.parse(out); } catch (e) {}
          if (parsed && typeof parsed === "object") return json(parsed, parsed.ok === false ? 502 : 200);
          return json({ ok: false, error: "service de lecture web indisponible",
            detail: (out || j2.stderr || "").slice(0, 400) }, 502);
        }
        if (sub === "files" && request.method === "GET") {
          const up = await vmFetch(VM_BASE + "/files?path=" + encodeURIComponent(url.searchParams.get("path") || "user"), { headers: H });
          return new Response(up.body, { status: up.status, headers: { "Content-Type": "application/json", ...corsH } });
        }
        if (sub === "file" && request.method === "GET") {
          const fmt = url.searchParams.get("format") === "base64" ? "base64" : "utf8";
          const fpath = url.searchParams.get("path") || "";
          /* --- lecture par morceaux (fichiers lourds) ---------------------
             Un fichier volumineux ne peut pas passer d'un seul bloc : le
             service de fichiers de la VM plafonne sa sortie (15 Mo) et surtout
             la lecture entière restait bloquée sans jamais rendre la main. On
             découpe donc la lecture en tranches, servies par le terminal de la
             VM : `&chunk=N&size=O` renvoie la tranche N de O octets, en base64,
             avec la taille totale du fichier (0 = fichier vide/absent). Rien
             ne change pour les petits fichiers : sans `chunk`, la route se
             comporte exactement comme avant. */
          const chunk = url.searchParams.get("chunk");
          if (chunk !== null) {
            const size = Math.min(Math.max(1, Number(url.searchParams.get("size")) || 6 * 1024 * 1024), 8 * 1024 * 1024);
            const n = Math.max(0, Math.floor(Number(chunk) || 0));
            /* Chemin absolu dans la VM : les chemins de l'app sont relatifs au
               dossier de l'utilisateur (/home/kasm-user). */
            const abs = /^\//.test(fpath) ? fpath : "/home/kasm-user/" + fpath.replace(/^[/]+/, "");
            const q = "'" + abs.replace(/'/g, "'\\''") + "'";
            /* Trois lignes : taille totale (-1 = fichier absent — un fichier
               vide, lui, vaut 0), longueur de la tranche, puis la tranche. */
            const cmd = "stat -c %s " + q + " 2>/dev/null || echo -1; "
              + "dd if=" + q + " bs=" + size + " skip=" + n + " count=1 2>/dev/null | base64 -w 0 | wc -c; "
              + "dd if=" + q + " bs=" + size + " skip=" + n + " count=1 2>/dev/null | base64 -w 0";
            const up = await vmFetch(VM_BASE + "/exec", {
              method: "POST",
              headers: { ...H, "Content-Type": "application/json" },
              body: JSON.stringify({ command: cmd, timeout: 120 }),
            });
            let jr = {};
            try { jr = await up.json(); } catch (e) {}
            const out = String(jr.stdout || "");
            const e1 = out.indexOf("\n");
            const e2 = e1 < 0 ? -1 : out.indexOf("\n", e1 + 1);
            if (!jr.ok || e1 < 0 || e2 < 0) {
              return json({ ok: false, error: "lecture par morceaux impossible",
                detail: String(jr.stderr || jr.error || "").slice(0, 300) }, 502);
            }
            const total = Number(out.slice(0, e1).trim());
            const got = Number(out.slice(e1 + 1, e2).trim()) || 0;
            const content = out.slice(e2 + 1).replace(/\s+/g, "");
            if (!(total >= 0)) {
              return json({ ok: false, error: "Fichier introuvable ou illisible", path: fpath }, 404);
            }
            /* Tranche au-delà de la fin : fin de fichier, sans erreur. */
            return json({ ok: true, path: fpath, format: "base64", chunk: n, size,
              total: total, length: got, eof: (got === 0 || (n + 1) * size >= total), content: content });
          }
          const up = await vmFetch(VM_BASE + "/files/read?path=" + encodeURIComponent(fpath) + "&format=" + fmt, { headers: H });
          return new Response(up.body, { status: up.status, headers: { "Content-Type": "application/json", ...corsH } });
        }
        return json({ error: "route agent inconnue" }, 404);
      } catch (err) {
        return json({ error: "VM injoignable : " + (err && err.message ? err.message : String(err)) }, 502);
      }
    }

    /* ---- raccourcis d'identifiant (connexion par nom) ------------------ */
    if (url.pathname === "/api/alias" && (request.method === "GET" || request.method === "POST")) {
      if (env.ACCESS_CODE && request.headers.get("X-Access-Code") !== env.ACCESS_CODE) {
        return json({ error: "code d'accès invalide" }, 401);
      }
      if (!env.MISSION) return json({ ok: false, error: "stockage indisponible" }, 503);
      try {
        const stub = env.MISSION.get(env.MISSION.idFromName("aliases"));
        const inner = request.method === "POST"
          ? new Request("https://alias/alias", { method: "POST", headers: { "Content-Type": "application/json" },
              body: JSON.stringify(await request.json().catch(() => ({}))) })
          : new Request("https://alias/alias?name=" + encodeURIComponent(url.searchParams.get("name") || ""), { method: "GET" });
        const r = await stub.fetch(inner);
        const j = await r.json().catch(() => ({}));
        return json({ ok: !!(j && j.ok), email: (j && j.email) || "", error: (j && j.error) || "" }, r.status);
      } catch (e) {
        return json({ ok: false, error: String(e && e.message ? e.message : e) }, 502);
      }
    }

    if (url.pathname !== "/api/chat" || request.method !== "POST") {
      return json({ error: "route inconnue" }, 404);
    }

    if (env.ACCESS_CODE && request.headers.get("X-Access-Code") !== env.ACCESS_CODE) {
      return json({ error: "code d'accès invalide" }, 401);
    }

    const len = Number(request.headers.get("Content-Length") || 0);
    if (len > MAX_BODY_BYTES) {
      return json({ error: `image trop lourde (${(len / 1e6).toFixed(1)} Mo, max ${MAX_BODY_BYTES / 1e6} Mo)` }, 413);
    }

    let body;
    try {
      body = await request.json();
    } catch {
      return json({ error: "corps JSON invalide" }, 400);
    }

    const model = body.model;
    let prompt = String(body.prompt ?? "").trim();
    /* `_waf` : la mission (ou l'app) demande une copie réécrite du message —
       1 = commandes entre quotes, 2 = apostrophes, 3 = accents. Le texte
       d'origine, lui, n'est jamais modifié. */
    const wafLvl = Math.max(0, Math.min(3, Number(body._waf) || 0));
    if (wafLvl > 0) prompt = defuseWaf(prompt, wafLvl);
    if (!prompt) return json({ error: "prompt vide" }, 400);

    const params = body.params && typeof body.params === "object" ? body.params : {};
    /* Diagnostic (aucune valeur de jeton révélée) : `params._tokens` renvoie le
       nombre de jetons de la rotation — sert aux vérifications de déploiement. */
    if (params._tokens === true) {
      return json({ ok: true, tokens: roboflowTokens(env).length });
    }
    const rawImage = String(body.image ?? "");

    /* --- Canal Pollinations (texte, compatible OpenAI, vision incluse) --- */
    if (IFM_MAP[model]) {
      if (!env.IFM_API_KEY) return json({ error: "IFM : clé absente du worker (IFM_API_KEY)" }, 500);
      const rq = { model: IFM_MAP[model], messages: [{ role: "user", content: prompt }] };
      if (params.max_tokens) rq.max_tokens = Math.max(1, Number(params.max_tokens) || 1024);
      if (params.temperature !== undefined && params.temperature !== null && params.temperature !== "") rq.temperature = Number(params.temperature);
      let iup;
      try {
        iup = await fetch(IFM_URL + "/chat/completions", {
          method: "POST",
          headers: { "Content-Type": "application/json", Authorization: "Bearer " + env.IFM_API_KEY },
          body: JSON.stringify(rq),
        });
      } catch (err) { return json({ error: `IFM injoignable : ${err.message}` }, 502); }
      const it = await iup.text();
      let ij; try { ij = JSON.parse(it); } catch { ij = null; }
      if (!iup.ok) {
        let msg = it.slice(0, 300);
        if (ij && ij.error) msg = ij.error.message || String(ij.error);
        return json({ error: `IFM HTTP ${iup.status} — ${msg}` }, iup.status === 429 ? 429 : 502);
      }
      const out = String(ij?.choices?.[0]?.message?.content ?? "").trim();
      if (!out) return json({ error: "IFM : réponse vide" }, 502);
      return json({ output: out, model, applied: ["ifm"] });
    }

    if (POLL_TEXT_MAP[model]) {
      const parts = [];
      for (const im of normImageList(body)) {
        parts.push({ type: "text", text: "[image « " + im.name + " »]" });
        parts.push({ type: "image_url", image_url: { url: dataUri(im) } });
      }
      parts.push({ type: "text", text: prompt });
      const rq = { model: POLL_TEXT_MAP[model], messages: [{ role: "user", content: parts }] };
      if (params.max_tokens) rq.max_tokens = Math.max(1, Number(params.max_tokens) || 1024);
      if (params.temperature !== undefined && params.temperature !== null && params.temperature !== "") rq.temperature = Number(params.temperature);
      if (params.reasoning_effort) rq.reasoning_effort = String(params.reasoning_effort);
      let up;
      try {
        up = await fetch(POLL_URL + "/v1/chat/completions", {
          method: "POST",
          headers: { "Content-Type": "application/json", Authorization: "Bearer " + env.POLLINATIONS_API_KEY },
          body: JSON.stringify(rq),
        });
      } catch (err) { return json({ error: `Pollinations injoignable : ${err.message}` }, 502); }
      const t = await up.text();
      let pj; try { pj = JSON.parse(t); } catch { pj = null; }
      const out0 = String(pj?.choices?.[0]?.message?.content ?? "").trim();
      /* Crédits épuisés, quota atteint, panne du fournisseur : on bascule sur
         la voie Roboflow du même modèle avant de renvoyer une erreur. */
      const pollBad = !up.ok || pollCreditText(out0);
      if (pollBad) {
        const fb = await pollToRoboflow(env, body, model);
        if (fb) return json(fb);
        if (!up.ok) return json({ error: pollErrorMessage(up.status, t) }, (up.status === 402 || up.status === 429 || up.status === 503) ? up.status : 502);
        return json({ error: pollErrorMessage(402, out0) }, 402);
      }
      let out = out0;
      if (pj?.choices?.[0]?.finish_reason === "length") out += "\n\n*(réponse interrompue : limite de tokens atteinte — relancez pour continuer)*";
      return json({ output: out, applied: [] });
    }

    /* --- Canal Groq --- */
    if (GROQ_MAP[model]) {
      const gq = { model: GROQ_MAP[model], messages: [{ role: "user", content: prompt }] };
      if (params.temperature !== undefined && params.temperature !== null && params.temperature !== "") gq.temperature = Number(params.temperature);
      if (params.top_p) gq.top_p = Number(params.top_p);
      if (params.reasoning_effort) gq.reasoning_effort = String(params.reasoning_effort);
      const baseMax = params.max_tokens ? Math.max(1, Number(params.max_tokens) || 1024) : (GROQ_MAXOUT[model] || 4096);
      const steps = [baseMax];
      for (const s of GROQ_RETRY_MAX) if (s < baseMax) steps.push(s);
      let up = null, t = "", gj = null;
      for (let i = 0; i < steps.length; i++) {
        gq.max_completion_tokens = steps[i];
        try {
          up = await fetch("https://api.groq.com/openai/v1/chat/completions", {
            method: "POST",
            headers: { "Content-Type": "application/json", Authorization: "Bearer " + env.GROQ_API_KEY, "User-Agent": BROWSER_UA },
            body: JSON.stringify(gq),
          });
        } catch (err) {
          const ms = await msTextRelay(env, model, prompt, params, rawImage);
          if (ms) return ms;
          return json({ error: `Groq injoignable : ${err.message}` }, 502);
        }
        t = await up.text();
        if (up.status === 429) {
          if (GROQ_TOO_LARGE_RE.test(t) && i < steps.length - 1) continue;
          if (groqCanRelay(429, t)) {
            const ms = await msTextRelay(env, model, prompt, params, rawImage);
            if (ms) return ms;
          }
          return json({ error: groqApiError(429, t) }, 429);
        }
        break;
      }
      try { gj = JSON.parse(t); } catch { gj = null; }
      if (!up.ok) {
        if (groqCanRelay(up.status, t)) {
          const ms = await msTextRelay(env, model, prompt, params, rawImage);
          if (ms) return ms;
        }
        return json({ error: groqApiError(up.status, t) }, (up.status === 413 || up.status === 429) ? up.status : 502);
      }
      let out = String(gj?.choices?.[0]?.message?.content ?? "").trim();
      if (gj?.choices?.[0]?.finish_reason === "length") out += "\n\n*(réponse interrompue : limite de tokens atteinte — relancez pour continuer)*";
      return json({ output: out, applied: [] });
    }

    /* --- Canal Google (Gemini API) --- */
    if (GOOGLE_MAP[model]) {
      const cfg = GOOGLE_MAP[model];
      const H = { "Content-Type": "application/json", "x-goog-api-key": env.GOOGLE_API_KEY };
      const B = "https://generativelanguage.googleapis.com/v1beta/models/" + cfg.m;
      try {
        if (cfg.kind === "embed") {
          const up = await fetch(B + ":embedContent", { method: "POST", headers: H, body: JSON.stringify({ content: { parts: [{ text: prompt }] } }) });
          const j2 = await up.json().catch(() => null);
          if (!up.ok) return json({ error: `Google HTTP ${up.status} — ${j2?.error?.message || ""}` }, 502);
          const v = j2?.embedding?.values || [];
          const norm = Math.sqrt(v.reduce((a, x) => a + x * x, 0));
          return json({ output: "", embedding: { dim: v.length, head: v.slice(0, 12).map(x => +x.toFixed(5)), norm: +norm.toFixed(4) } });
        }
        const parts = [{ text: prompt }];
        const rawAudio = String(body.audio ?? "");
        if (rawAudio) parts.unshift({ inline_data: { mime_type: String(body.audio_mime || "audio/wav"), data: rawAudio } });
        if (cfg.kind !== "transcribe") for (const im of normImageList(body)) parts.unshift({ inline_data: { mime_type: im.mime, data: im.b64 } });
        const gc = {};
        if (params.max_tokens) gc.maxOutputTokens = Number(params.max_tokens);
        if (params.temperature !== undefined && params.temperature !== null && params.temperature !== "") gc.temperature = Number(params.temperature);
        if (params.thinking_level) gc.thinkingConfig = { thinkingLevel: String(params.thinking_level) };
        const tools = [];
        if (params.google_search) tools.push({ google_search: {} });
        if (params.google_code_execution) tools.push({ code_execution: {} });
        if (params.google_url_context) tools.push({ url_context: {} });
        if (params.google_maps) tools.push({ google_maps: {} });
        const reqBody = { contents: [{ role: "user", parts }] };
        if (cfg.kind === "tts") {
          gc.responseModalities = ["AUDIO"];
          gc.speechConfig = { voiceConfig: { prebuiltVoiceConfig: { voiceName: String(params.voice || "Kore") } } };
          delete gc.maxOutputTokens;
        }
        if (Object.keys(gc).length) reqBody.generationConfig = gc;
        if (tools.length) reqBody.tools = tools;
        const up = await fetch(B + ":generateContent", { method: "POST", headers: H, body: JSON.stringify(reqBody) });
        const j2 = await up.json().catch(() => null);
        if (!up.ok) return json({ error: `Google HTTP ${up.status} — ${j2?.error?.message || ""}` }, 502);
        const cand = j2?.candidates?.[0];
        const ps = cand?.content?.parts || [];
        const txt = ps.map(p => p.text || (p.audioTranscription && p.audioTranscription.text) || "").join("").trim();
        if (cfg.kind === "tts") {
          const au = ps.find(p => p.inlineData && p.inlineData.data);
          if (!au) return json({ error: "Google TTS : aucune donnée audio renvoyée" }, 502);
          return json({ output: txt, audio: au.inlineData.data, mime: au.inlineData.mimeType || "audio/wav" });
        }
        const gm = cand?.groundingMetadata;
        const applied = [...new Set((gm?.groundingChunks || []).map(c => c?.web?.uri).filter(Boolean))].slice(0, 6);
        return json({ output: txt, applied });
      } catch (err) { return json({ error: `Google injoignable : ${err.message}` }, 502); }
    }

    const spec = MODELS[model];
    if (!spec) return json({ error: `modèle inconnu : ${String(model)}` }, 400);

    // Le champ image est obligatoire côté Roboflow : le client envoie un
    // pixel blanc 1x1 par défaut, une image noire sur demande, ou une image
    // fournie.
    const image = rawImage;
    if (!image) return json({ error: "image manquante (champ obligatoire côté Roboflow)" }, 400);
    if (image.length > MAX_BODY_BYTES) {
      return json({ error: "image trop lourde après encodage base64" }, 413);
    }

    /* Un modèle tout juste sorti peut manquer dans le bloc dédié : `chain`
       liste les blocs à essayer dans l'ordre tant que Roboflow refuse la
       définition du workflow (erreur de validation), sinon on s'arrête. */
    /* Candidats : le bloc demandé, puis (pour Claude / GPT / Gemini) le même
       modèle chez OpenRouter si le bloc dédié échoue. */
    const orCand = { block: "roboflow_core/openrouter@v1", family: "router", vf: "model_id", vv: spec.or, orFallback: true };
    /* `params._fallback:"or"` force la voie OpenRouter : sert uniquement à
       vérifier le repli de bout en bout, sans dépendre d'une panne réelle. */
    const forceOr = !!(body.params && body.params._fallback === "or");
    /* `params._raw` : essai technique d'une définition de bloc (nom de bloc,
       champ de version, valeur) — sert à vérifier un nouveau modèle sur la
       passerelle déjà déployée. Résultat brut, y compris les erreurs. */
    const raw = body.params && body.params._raw;
    const rawCand = (raw && raw.block) ? {
      block: String(raw.block), family: "router",
      vf: String(raw.vf || "model_version"), vv: String(raw.vv || model),
      out: raw.out ? String(raw.out) : undefined,
      noimg: raw.noimg === true,          /* certains modèles refusent l'image */
      extraInputs: Array.isArray(raw.inputs) ? raw.inputs : null,
      inputValues: (raw.values && typeof raw.values === "object") ? raw.values : null,
      stepExtra: (raw.step && typeof raw.step === "object") ? raw.step : null,
      pf: raw.pf ? String(raw.pf) : null,   /* famille de paramètres à tester */
    } : null;
    const cands = rawCand ? [rawCand]
      : (forceOr && spec.or ? [orCand]
      : (spec.chain || (spec.or ? [spec, orCand] : [spec])));
    /* Rotation des jetons : chaque voie est rejouée avec le jeton suivant. */
    const calls = acrossTokens(env, cands);

    /* Une tentative : liste (jeton, bloc) essayés dans l'ordre, avec le filet
       anti-pare-feu et le repli OpenRouter sur erreur. Renvoie soit le
       résultat, soit le message d'erreur déjà rédigé. */
    const attempt = async (list) => {
      let safeTried = false;
      let tries = 0;
      const skip = new Set();       /* blocs dont la définition est refusée par Roboflow */
      let i = 0;
      while (i < list.length) {
        const c = list[i];
        i++;
        if (skip.has(c.block)) continue;      /* identique pour tous les jetons : inutile de rejouer */
        const step = {
          type: c.block,
          name: "m",
          prompt,
          [c.vf || "model_version"]: c.vv || model,
          ...sanitize(body.params, c.pf || c.family),
          ...(c.stepExtra || {}),
        };
        if (!c.noimg && step.images === undefined) step.images = "$inputs.image";
        if (!c.nokey) step.api_key = "rf_key:account";
        if (c.task !== null) step.task_type = c.task || "unconstrained";
        if (c.backend) step.backend = c.backend;

        const payload = {
          api_key: c._token || env.ROBOFLOW_API_KEY,
          inputs: Object.assign({}, c.noimg ? {} : { image: { type: "base64", value: image } }, c.inputValues || {}),
          specification: {
            version: "1.0",
            inputs: (c.noimg ? [] : [{ type: "WorkflowImage", name: "image" }]).concat(c.extraInputs || []),
            steps: [step],
            outputs: [{ type: "JsonField", name: "output", selector: `$steps.m.${c.out || spec.out || "output"}` }],
          },
        };

        let r;
        try {
          r = await fetch(ROBOFLOW_URL, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
          });
        } catch (err) {
          return { ok: false, error: `Roboflow injoignable : ${err.message}` };
        }

        let good = r.ok ? r : null;
        let t = good ? "" : await r.text();
        /* Modèles de raisonnement (Mistral 4, Muse Wonder, Kimi… servis par
           OpenRouter) : ils dépensent tout le budget de sortie en réflexion
           interne et Roboflow répond « missing message.content … ran out of
           tokens ». Le réglage par défaut du bloc est trop bas : on rejoue UNE
           fois avec un budget suffisant, au lieu d'afficher une erreur. */
        if (!good && /missing message\.content|ran out of tokens|internal thinking/i.test(t)) {
          const bigStep = Object.assign({}, step, { max_tokens: Math.max(4096, Number(step.max_tokens || 0)) });
          const bigPayload = Object.assign({}, payload, {
            specification: Object.assign({}, payload.specification, { steps: [bigStep] }),
          });
          let r3 = null;
          try {
            r3 = await fetch(ROBOFLOW_URL, {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify(bigPayload),
            });
          } catch (err) { /* on retombe sur l'erreur d'origine */ }
          if (r3 && r3.ok) good = r3;
          else if (r3) t = await r3.text();
        }
        /* Panne passagère du fournisseur du modèle, relayée par OpenRouter
           (« Provider returned error ») : une seconde tentative suffit presque
           toujours, car le proxy choisit alors un autre point de terminaison. */
        if (!good && /Provider returned (an )?error|upstream error|temporarily unavailable/i.test(t)) {
          await new Promise((res) => setTimeout(res, 400));
          let r4 = null;
          try {
            r4 = await fetch(ROBOFLOW_URL, {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify(payload),
            });
          } catch (err) { /* on retombe sur l'erreur d'origine */ }
          if (r4 && r4.ok) good = r4;
          else if (r4) t = await r4.text();
        }
        /* 403 du pare-feu Roboflow : on rejoue une fois sans les passages
           « dangereux » (voir defuseWaf). */
        if (!good && r.status === 403 && /Attention Required|Cloudflare/i.test(t) && !safeTried) {
          safeTried = true;
          const safeStep = Object.assign({}, step, { prompt: defuseWaf(prompt) });
          const safePayload = Object.assign({}, payload, {
            specification: Object.assign({}, payload.specification, { steps: [safeStep] }),
          });
          let r2 = null;
          try {
            r2 = await fetch(ROBOFLOW_URL, {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify(safePayload),
            });
          } catch (err) { /* on retombe sur l'erreur d'origine */ }
          if (r2 && r2.ok) good = r2;
          else if (r2) t = await r2.text();
        }
        /* Le refus du pare-feu ne prend pas toujours la forme « 403 + Attention
           Required » : page HTML, 400 sec, 1010… Quel que soit le code, on
           réessaie en RETIRANT ou en REMPLAÇANT les caractères fautifs, niveau
           après niveau (apostrophes → espace, puis accents retirés). Le texte
           d'origine n'est jamais traité comme une erreur : l'utilisateur, y
           compris en mode agent, ne voit rien de tout cela. */
        let wafStat = r.status;
        let wlvl = safeTried ? 2 : 1;
        while (!good && wafLike(wafStat, t) && wlvl <= 3) {
          const lvl = wlvl++;
          if (lvl === 1 && safeTried) continue;   /* déjà tenté juste au-dessus */
          const wStep = Object.assign({}, step, { prompt: defuseWaf(prompt, lvl) });
          const wPayload = Object.assign({}, payload, {
            specification: Object.assign({}, payload.specification, { steps: [wStep] }),
          });
          let rw = null;
          try {
            rw = await fetch(ROBOFLOW_URL, {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify(wPayload),
            });
          } catch (err) { continue; }
          if (rw.ok) { good = rw; break; }
          wafStat = rw.status;
          t = await rw.text();
        }

        tries++;
        if (good) {
          const body = await good.text();
          let data = null;
          try { data = JSON.parse(body); } catch { data = null; }
          if (!data) return { ok: false, error: `Roboflow HTTP ${good.status} — ${body.slice(0, 400)}`, tries };
          return { ok: true, used: c, data, tries };
        }

        let detail = t.slice(0, 400);
        try { const j = JSON.parse(t); detail = j.inner_error_message || j.message || detail; } catch { /* texte brut */ }
        if (wafLike(wafStat, t)) {
          /* Après les trois niveaux de réécriture, on ne parle ni de pare-feu
             ni de caractères fautifs : l'incident reste technique. */
          detail = "la requête a été filtrée par le fournisseur — réessaie, le message sera réécrit";
        }
        /* Dès qu'une tentative échoue, la suivante change : jeton suivant, ou
           voie suivante (voie dédiée -> OpenRouter), puis jeton suivant, etc.
           Seule exception : une erreur de DÉFINITION du workflow (schéma refusé
           par Roboflow) est identique pour tous les jetons — on saute alors
           directement au bloc suivant au lieu de la rejouer pour rien. */
        const defErr = (r.status === 400 || r.status === 404)
          && /validation error|string_pattern_mismatch|literal\[|BlockManifest|WorkflowSpecificationV1/i.test(t);
        if (defErr) skip.add(c.block);
        if (i >= list.length || list.slice(i).every((x) => skip.has(x.block))) {
          const nt = roboflowTokens(env).length;
          const suffix = c.orFallback
            ? " (les deux voies Roboflow et OpenRouter ont échoué" + (nt > 1 ? ", sur les " + nt + " jetons" : "") + ")"
            : (nt > 1 ? " (essayé sur les " + nt + " jetons)" : "");
          return { ok: false, error: `Roboflow HTTP ${r.status} — ${detail}${suffix}`, tries };
        }
        await new Promise((res) => setTimeout(res, 120));
      }
      return { ok: false, error: "Roboflow : aucune voie disponible", tries };
    };

    const pick = (r) => (r.data && r.data.outputs && r.data.outputs[0]) ? r.data.outputs[0] : {};
    const appliedOf = (r) => Object.keys(sanitize(body.params, r.used.pf || r.used.family));

    /* ModelScope d'abord, pour les modèles dont c'est la voie principale. */
    let msFirst = "";
    if (MS_FIRST[model] && MS_TEXT[model] && msKey(env)) {
      const ms = await msChat(env, MS_TEXT[model], prompt, params, image);
      if (ms.ok) return json({ output: ms.output, applied: [], via: "modelscope" });
      msFirst = ms.error;
    }
    let res = await attempt(calls);
    /* ModelScope en secours (crédits gratuits de la plateforme) : c'est
       toujours LE MÊME modèle, jamais un autre. */
    if (!res.ok) {
      const msModel = MS_TEXT[model];
      if (msModel && msKey(env) && !MS_FIRST[model]) {
        const ms = await msChat(env, msModel, prompt, params, image);
        if (ms.ok) return json({ output: ms.output, applied: [], via: "modelscope" });
        return json({ error: res.error + " · " + ms.error }, 502);
      }
      if (msFirst) return json({ error: msFirst + " · " + res.error }, 502);
      return json({ error: res.error }, 502);
    }

    let output = pick(res).output ?? "";
    /* « roboflow » quand ce n'est PAS la voie naturelle du modèle : ModelScope
       a été tenté d'abord et a refusé. L'application l'affiche en italique,
       pour que l'on sache toujours par où la réponse est arrivée. */
    let via = res.used && res.used.orFallback ? "openrouter" : (msFirst ? "roboflow" : "");

    /* Le bloc dédié d'un modèle de texte (Claude, Gemini, GPT…) peut renvoyer
       une IMAGE : dans ce cas la même demande est rejouée par la voie
       OpenRouter, qui rend le texte attendu. */
    if (spec.or && !forceOr && !(res.used && res.used.orFallback) && looksLikeImage(output)) {
      const res2 = await attempt(acrossTokens(env, [orCand]));
      const out2 = res2.ok ? (pick(res2).output ?? "") : "";
      /* Si la seconde voie rend exactement la même chose, ce n'était pas une
         image mais un texte qui ressemblait à un lien d'image : on le garde. */
      const meme = out2.trim() === String(output).trim();
      const certain = looksLikeImage(output, true);
      if (res2.ok && out2 && (!looksLikeImage(out2) || (meme && !certain))) {
        res = res2; output = out2; via = "openrouter";
      } else {
        const why = res2.ok ? "la reprise par OpenRouter a renvoyé une image à son tour"
          : "la reprise par OpenRouter a échoué (" + res2.error + ")";
        return json({ error: `Roboflow a renvoyé une image au lieu d'un texte pour « ${model} » — ${why}` }, 502);
      }
    }

    const finalOut = { output, applied: appliedOf(res) };
    if (via) finalOut.via = via;
    return json(finalOut);
}

export default {
  fetch: fetchHandler,

  /* ---- Cron : recrée agent.md / infos.md dans la VM s'ils ont été supprimés. ---- */
  async scheduled(controller, env, ctx) {
    ctx.waitUntil(ensureAgentDocs(env).catch(() => {}));
  },
};
