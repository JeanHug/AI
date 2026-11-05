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
  /* `direct: true` : ce modèle ne passe JAMAIS par OpenRouter. Le bloc
     Anthropic est la seule voie ; si elle refuse, on le dit au lieu d'aller
     emprunter une autre route (c'était la source d'un double essai inutile
     et d'un diagnostic faux : le refus de crédits était masqué par le refus
     « plan gratuit » du repli OpenRouter, essayé juste après). */
  "claude-opus-5-5":   { block: "roboflow_core/anthropic_claude@v5", family: "claude", or: "anthropic/claude-opus-5.5", direct: true },
  "claude-haiku-5-5": { block: "roboflow_core/anthropic_claude@v5", family: "claude", or: "anthropic/claude-haiku-5.5", direct: true },
  "claude-sonnet-5-5": { block: "roboflow_core/anthropic_claude@v5", family: "claude", or: "anthropic/claude-sonnet-5.5", direct: true },
  "claude-fable-5-1":  { block: "roboflow_core/anthropic_claude@v5", family: "claude", or: "anthropic/claude-fable-5.1", direct: true },
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
  /* K3.1 (23/10/2026) : sur OpenRouter le nom EXACT est « kimi-k3.1-pro »
     (Moonshot n'expose pas d'identifiant « kimi-k3.1 » nu). */
  "kimi-k3-1-pro": { block: "roboflow_core/openrouter@v1", family: "router", vf: "model_id", vv: "moonshotai/kimi-k3.1-pro" },
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

/* --- Inception · Mercury 3 : modèle de diffusion ------------------------
   Particularité : avec `diffusing: true`, chaque morceau du flux contient
   le TEXTE ENTIER à cet instant du déparasitage (pas un delta). Les états
   intermédiaires contiennent des caractères parasites — c'est le
   fonctionnement réel du modèle, l'app les affiche tels quels. */
const MERCURY_URL = "https://api.inceptionlabs.ai/v1";
const DIFF_MAP = {
  "mercury-3": "mercury-3",
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
  /* Un code 1010/1020 de Cloudflare est un vrai blocage, quel que soit le
     format. */
  if (/error code:?\s*10(10|20|40)/i.test(t)) return true;
  /* Une erreur JSON du fournisseur (crédits épuisés, quota, modèle refusé)
     n'est PAS un pare-feu : la réécrire ferait perdre la vraie raison. */
  if (/^\s*\{/.test(t)) return false;
  if (/Attention Required|cf-error-details/i.test(t)) return true;
  if (/^\s*<(!doctype|html)/i.test(t)) return true;
  return Number(status) === 403;
}
/* Roboflow refuse l'accès aux clés OpenRouter qu'il gère pour les espaces au
   plan gratuit : ce n'est ni un pare-feu, ni une panne — c'est une limite de
   plan, et il faut le dire ainsi. */
function roboflowPlanError(text) {
  const t = String(text || "");
  if (!/only available to workspaces on a paid plan|provide your own OpenRouter API key/i.test(t)) return "";
  return "Roboflow : cet espace de travail est au plan gratuit, qui n'ouvre pas les clés OpenRouter gérées par Roboflow"
    + " — soit un plan payant (app.roboflow.com/settings/billing), soit sa propre clé OpenRouter.";
}
/* Le détail comptable d'un refus, sous forme de nombres — null si absent. */
function roboflowCreditInfo(text) {
  const t = String(text || "");
  if (!/credit_cap_exceeded|ran out of credits|cannot currently spend credits/i.test(t)) return null;
  const cap = Number((t.match(/cycleCreditCap[\"']?\s*:\s*(\d+(?:\.\d+)?)/) || [])[1]);
  const used = Number((t.match(/planCycleUsed[\"']?\s*:\s*(\d+(?:\.\d+)?)/) || [])[1]);
  return { used: isFinite(used) ? used : null, cap: isFinite(cap) ? cap : null };
}
/* Roboflow : le compte peut être à court de crédits pour le cycle en cours.
   Le refus arrive en 402 avec un message comptable — on le dit tel quel. */
function roboflowCreditError(text) {
  const t = String(text || "");
  if (!/credit_cap_exceeded|ran out of credits|cannot currently spend credits/i.test(t)) return "";
  const cap = (t.match(/cycleCreditCap[\"']?\s*:\s*(\d+(?:\.\d+)?)/) || [])[1];
  const used = (t.match(/planCycleUsed[\"']?\s*:\s*(\d+(?:\.\d+)?)/) || [])[1];
  const nf = function (v) { return String(Number(v).toFixed(2)).replace(".", ",") + " crédits"; };
  return "Roboflow : crédits du cycle épuisés"
    + (used && cap ? " (" + nf(used) + " dépensés sur " + nf(cap) + " autorisés pour ce cycle)" : "")
    + " — toute inférence est refusée jusqu'au prochain cycle ou à une recharge du plan"
    + " (crédits et recharge : app.roboflow.com/settings/plan · dates du cycle : Réglages de l'espace ▸ Credit Usage).";
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
/* Les identifiants ci-dessous ont été vérifiés un par un contre l'API
   d'inférence (sonde /api/ms/probe) : un nom absent du dépôt ne renvoie pas
   une erreur de modèle inconnu mais « has no provider supported », et le nom
   approximatif d'un modèle d'image renvoie 500. On ne met donc ici que des
   identifiants qui répondent vraiment. */
const MS_TEXT = {
  "kimi-k3": "moonshotai/Kimi-K3.1-Pro",          /* K3 n'est plus servi, K3.1 Pro oui */
  "glm-5-3": "zai-org/GLM-5.2",                   /* GLM 5.3 pas encore publié côté inférence */
  "glm-5-3-flash": "zai-org/GLM-4.7-Flash",
  "deepseek-v4-1-flash": "deepseek-ai/DeepSeek-V4.1-Flash-1017",
  "mimo-v2-6-pro": "XiaomiMiMo/MiMo-V2.6-Pro-RL",
  "mimo-v2-6-flash": "XiaomiMiMo/MiMo-V2.6-Flash-RL",
  "qwen-3-8-flash-next": "Qwen/Qwen3.8-27B",
  "qwen-3-8-27b": "Qwen/Qwen3.8-27B",
  "groq-qwen-3-8-27b": "Qwen/Qwen3.8-27B",
  "qwen-3-8-max": "Qwen/Qwen3.5-397B-A17B",       /* Qwen3.8-Max n'est pas ouvert à ce compte */
  "qwen-3-8-2-4t": "Qwen/Qwen3.5-397B-A17B",
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
/* Qwen-Image-2.1 (et Turbo) répondent 500 chez ModelScope : leur fonction
   d'inférence a disparu côté serveur. Qwen-Image, lui, répond — et
   Qwen-Image-Edit pour une image jointe. */
const MS_IMG = {
  "qwen-image-2-1": "Qwen/Qwen-Image",
  "qwen-image-2-1-turbo": "Qwen/Qwen-Image",
  /* Wan 2.2 n'est plus ici : la famille Wan 2.2 de ModelScope est vidéo
     (texte→image n'existe pas sur leur inférence), elle passe par /api/video. */
};
function msKey(env) { return String(env.MODELSCOPE_API_KEY || "").trim(); }
/* Un appel de texte chez ModelScope. Renvoie { ok, output } ou { ok:false, error }. */
/* L'application joint TOUJOURS une image : un blanc de 64×64 quand la demande
   est purement textuelle (la passerelle Roboflow exige un champ image). Les
   modèles textuels de ModelScope refusent alors la pièce jointe — « only
   supports text modality ». On ne transmet donc une image que si c'en est
   vraiment une : le blanc est reconnu à son en-tête, jamais par sa longueur. */
const PLACEHOLDER_HEAD = "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAMCAgMCAgMDAwMEAwMEBQgFBQQEBQoHBwYIDAoMDAsKCwsNDhIQDQ4RDgsLEBYQERMUFRUVDA8XGBYUGBIUFRT/2wBDAQMEBAUEBQkFBQkUDQsNFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBT/wAARCABA";
function imageReelle(b64) {
  const v = String(b64 || "");
  if (v.length < 200) return false;
  return v.slice(0, PLACEHOLDER_HEAD.length) !== PLACEHOLDER_HEAD;
}

async function msChat(env, msModel, prompt, params, image) {
  const key = msKey(env);
  if (!key) return { ok: false, error: "secret MODELSCOPE_API_KEY manquant côté Worker" };
  const p = params && typeof params === "object" ? params : {};
  /* L'application joint toujours une image (un pixel blanc 1×1) quand la
     demande est purement textuelle. ModelScope refuse ce faux format :
     « invalid image format ». On ne lui envoie donc une image que si c'en
     est vraiment une — c'est-à-dire nettement plus qu'un pixel. */
  const vraieImage = imageReelle(image);
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

/* --- VM : le tunnel publié, lu à la source --------------------------------
   La passerelle VMAI maintient son propre état des tunnels ; quand cet état
   retarde, elle répond « daemon_pending_cycle » alors que le bureau est
   parfaitement joignable par le tunnel Cloudflare du moment. On garde donc
   ici l'adresse publiée (branche « tunnel-url » du dépôt VM), rafraîchie
   chaque minute, et l'on s'en sert en secours. */
let VM_TUNNELS = { urls: [], at: 0 };
async function vmTunnelList() {
  const now = Date.now();
  /* 30 s : la branche « tunnel-url » est réécrite à chaque relais, une valeur
     gardée plus longtemps désigne un tunnel déjà mort. */
  if (VM_TUNNELS.urls.length && now - VM_TUNNELS.at < 30000) return VM_TUNNELS.urls;
  /* raw.githubusercontent est servi avec un cache de cinq minutes : sans ces
     trois précautions, le Worker relisait l'ANCIEN tunnel pendant que le
     nouveau tournait — et chaque requête tombait sur « error code: 1016 ». */
  const bust = "?" + now + "-" + Math.random().toString(36).slice(2);
  try {
    const r = await fetch("https://raw.githubusercontent.com/JeanHug/VM/tunnel-url/tunnels-linux.json" + bust, {
      headers: { "User-Agent": "Cloudflare-VM-Gateway", "Cache-Control": "no-cache", "Pragma": "no-cache" },
      cf: { cacheTtl: 0, cacheEverything: false },
    });
    const j = await r.json().catch(() => null);
    const list = [j && j.cloudflare, j && j.primary, j && j.lhrLife]
      .filter(Boolean).map((x) => String(x).trim().replace(/\/+$/, ""));
    const uniq = [];
    for (const u of list) if (uniq.indexOf(u) < 0) uniq.push(u);
    if (uniq.length) VM_TUNNELS = { urls: uniq, at: now };
  } catch (e) { /* on garde la liste précédente */ }
  return VM_TUNNELS.urls;
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
/* Images d'un message : au plus trois, en base64 déjà réduites par l'appelant.
   Un document Firestore est limité à 1 Mio : on garde donc une marge franche
   (800 000 caractères de base64 ≈ 600 Ko de binaire). */
const CANAL_IMG_MAX = 3;
const CANAL_IMG_LEN = 800000;
function canalImgs(list) {
  if (!Array.isArray(list)) return [];
  const out = [];
  for (const it of list.slice(0, CANAL_IMG_MAX)) {
    if (!it || typeof it !== "object") continue;
    const b64 = String(it.b64 || "").replace(/^data:[^;]+;base64,/, "");
    const mime = String(it.mime || "image/jpeg").toLowerCase();
    if (!b64 || b64.length > CANAL_IMG_LEN) continue;
    if (!/^image\/(png|jpe?g|webp|gif)$/.test(mime)) continue;
    out.push({ name: String(it.name || "image").slice(0, 120), mime: mime, b64: b64 });
  }
  return out;
}
/* Vue publique d'un message : le texte et, s'il y en a, ses images. */
function canalPublic(m) {
  const out = { id: m.id, role: m.role, ts: m.ts || 0, text: String(m.text == null ? "" : m.text),
    kind: m.kind || "text", state: m.state || "" };
  const im = canalImgs(m.imgs);
  if (im.length) out.imgs = im;
  return out;
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
    if (!text && !Array.isArray(b.imgs)) return json({ ok: false, error: "texte vide" }, 400);
    const kind = String(b.kind || "text").toLowerCase() === "call" ? "call" : "text";
    /* « wait » : le nombre de secondes pendant lesquelles le worker attend
       LUI-MÊME la réponse de la commande. Le write rend donc le résultat. */
    const waitS = Math.max(0, Math.min(25, Number(b.wait || url.searchParams.get("wait") || 0) || 0));
    const imgs = canalImgs(b.imgs);
    if (!text && !imgs.length) return json({ ok: false, error: "texte vide" }, 400);
    const w = await canalAppend(env, id, {
      role: canalRole(id), text: text, ts: Date.now(), kind: kind, state: "new", src: "canal",
      req: String(b.req || ""), note: String(b.note || ""), imgs: imgs,
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
const ZCANAL_PY_B64 = "IyEvdXNyL2Jpbi9lbnYgcHl0aG9uMwojIC0qLSBjb2Rpbmc6IHV0Zi04IC0qLQoiIiJ6LWNhbmFsLnB5IOKAlCBsZSBjYW5hbCBlbnRyZSBBcmVuYSBldCBsJ2FwcGxpY2F0aW9uIChzeXN0w6htZSBaLUNhbmFsKS4KCkNlIHNjcmlwdCBlc3QgbGUgY2xpZW50IGR1IGNhbmFsIMKrIGFyZW5hIMK7LCBsZSBjYW5hbCBwYXIgZMOpZmF1dCBkZQpsJ2FwcGxpY2F0aW9uLiBJbCBwYXJsZSBhdXggYWRyZXNzZXMgcHVibGlxdWVzIGR1IGNhbmFsIDoKCiAgICBHRVQgIEJBU0Uvd2FpdD90aW1lb3V0PTI1ICAgbGVzIG1lc3NhZ2VzIE5PTiBMVVMgKGV0IGxlcyBtYXJxdWUgwqsgbHUgwrspCiAgICBHRVQgIEJBU0UvcmVhZD9uPTMwICAgICAgICAgbCdoaXN0b3JpcXVlIHLDqWNlbnQgKG5lIG1hcnF1ZSByaWVuKQogICAgUE9TVCBCQVNFL3dyaXRlICAgICAgICAgICAgIMOpY3JpcmUgdW4gbWVzc2FnZSAgeyJ0ZXh0Ijog4oCmLCAia2luZCI6IOKApn0KICAgIEdFVCAgQkFTRS9zdGF0ZSAgICAgICAgICAgICBsJ8OpdGF0IGR1IGNhbmFsCgphdmVjIEJBU0UgPSBodHRwczovL2FwaS1haS5odWdkdTc3Nzc3LndvcmtlcnMuZGV2L2NhbmFsL2FyZW5hCgpSaWVuIG4nZXN0IHNlY3JldCBpY2kgOiBsJ2FkcmVzc2UgZHUgY2FuYWwgZXN0IHB1YmxpcXVlLCBldCBpbCBuJ3kgYSBhdWN1bgpqZXRvbiDDoCBzdG9ja2VyIOKAlCBsZSBjYW5hbCByZWNvbm5hw650IGwnYWdlbnQgcGFyIHNvbiBpZGVudGlmaWFudC4KClJFUExJIERJUkVDVAotLS0tLS0tLS0tLS0KU2kgbGEgcGFzc2VyZWxsZSBkdSBjYW5hbCByZWZ1c2UgdW5lIHJlcXXDqnRlIChxdW90YSDDqXB1aXPDqSBjw7R0w6kgcGFzc2VyZWxsZSwKZXJyZXVyIDQyOSBvdSA1eHgsIHLDqXNlYXUgZW4gcGFubmUpLCBsZSBzY3JpcHQgcGFybGUgRElSRUNURU1FTlQgw6AgbGEgYmFzZSBkdQpwcm9qZXQgYXZlYyB1bmUgc2Vzc2lvbiBhbm9ueW1lLiBMZXMgZGV1eCBjbMOpcyBjaS1kZXNzb3VzIHNvbnQgbGVzIGNsw6lzClBVQkxJUVVFUyBkZSBsJ2FwcGxpY2F0aW9uIDogZWxsZXMgc29udCDDqWNyaXRlcyBkYW5zIHNhIHBhZ2UgcG91ciB0b3V0IGxlCm1vbmRlLCBpbCBuJ3kgYSBkb25jIHJpZW4gZGUgc2VjcmV0IMOgIHByb3TDqWdlci4gTGUgY2FuYWwgY29udGludWUgZG9uYyBkZQpmb25jdGlvbm5lciBtw6ptZSBxdWFuZCBsYSBwYXNzZXJlbGxlIGVzdCBpbmRpc3BvbmlibGUuCgpTb3VzLWNvbW1hbmRlcwotLS0tLS0tLS0tLS0tLQogIHdhaXQgICBbLS1ob2xkIDI1XSBbLS1nYXAgNV0gWy0tdGltZW91dCAwXQogICAgICAgICAgICAgICAgICAgICAgYXR0ZW5kIHVuIG1lc3NhZ2UuIExlIHNlcnZldXIgbmUgYmxvcXVlIHF1ZSAyNSBzIDogYydlc3QKICAgICAgICAgICAgICAgICAgICAgIENFIHNjcmlwdCBxdWkgYm91Y2xlIChhdHRlbnRlIOKGkiByaWVuID8gcGF1c2UgZGUgNSBzIOKGkgogICAgICAgICAgICAgICAgICAgICAgYXR0ZW50ZSkuIC0tdGltZW91dCAwIChkw6lmYXV0KSA9IHNhbnMgZmluLgogIHJlYWQgICBbLS1uIDMwXSBbLS1hbGxdCiAgICAgICAgICAgICAgICAgICAgICBhZmZpY2hlIGwnaGlzdG9yaXF1ZSByw6ljZW50IChuZSBtYXJxdWUgcmllbiBjb21tZSBsdSkKICB3cml0ZSAgPHRleHRlfC0+IFstLWpzb25dIFstLXJlcSBJRF0gWy0td2FpdCAyMF0gWy0taW1hZ2UgRklDSElFUl0KICAgICAgICAgICAgICAgICAgICAgIMOpY3JpdCB1biBtZXNzYWdlIGRhbnMgbCdhcHBsaWNhdGlvbi4gVW5lIGNvbW1hbmRlCiAgICAgICAgICAgICAgICAgICAgICAoLS1qc29uKSByZW5kIGVuIHBsdXMgU09OIFLDiVNVTFRBVCA6IGxhIHNvcnRpZSBkZSBsYSBWTQogICAgICAgICAgICAgICAgICAgICAgcydhZmZpY2hlIGljaSBtw6ptZSwgZGFucyBsYSByw6lwb25zZSBkdSB3cml0ZS4KICAgICAgICAgICAgICAgICAgICAgIC0taW1hZ2Ugam9pbnQgdW5lIGltYWdlIChyw6lww6l0YWJsZSwgdHJvaXMgYXUgcGx1cykgOgogICAgICAgICAgICAgICAgICAgICAgZWxsZSBlc3QgcsOpZHVpdGUgZW4gSlBFRyAxMjgwIHB4IHBvdXIgdGVuaXIgZGFucyBsZQogICAgICAgICAgICAgICAgICAgICAgZG9jdW1lbnQgRmlyZXN0b3JlLCBwdWlzIHMnYWZmaWNoZSBkYW5zIGwnYXBwLgogIHN0YXRlICAgICAgICAgICAgICAgw6l0YXQgZHUgY2FuYWwgZW4gdW4gY291cCBkJ8WTaWwKICBkb2MgICAgICAgICAgICAgICAgIGwnYWRyZXNzZSBkZSBsYSBkb2N1bWVudGF0aW9uCiAgc2VsZnRlc3QgICAgICAgICAgICBhbGxlci1yZXRvdXIgZCfDqWNyaXR1cmUvbGVjdHVyZQoKRXhlbXBsZXMKLS0tLS0tLS0KICBweXRob24zIHotY2FuYWwucHkgd2FpdCAgICAgICAgICAgICAgICAgICAgICAgIyB2ZWlsbGUgOiByZW5kIGxhIG1haW4gZMOocyBxdSd1biBtZXNzYWdlIGFycml2ZQogIHB5dGhvbjMgei1jYW5hbC5weSByZWFkICAgICAgICAgICAgICAgICAgICAgICAjIGNlIHF1aSBzJ2VzdCBkaXQsIGR1IHBsdXMgYW5jaWVuIGF1IHBsdXMgcsOpY2VudAogIHB5dGhvbjMgei1jYW5hbC5weSB3cml0ZSAiSmUgcmVnYXJkZSDDp2EuIgogIHB5dGhvbjMgei1jYW5hbC5weSB3cml0ZSAtLWpzb24gJ3siYWN0aW9uIjoiYmFzaCIsImNvbW1hbmQiOiJscyAtbCAvaG9tZS91c2VyIn0nCiAgcHl0aG9uMyB6LWNhbmFsLnB5IHdyaXRlIC0gPCByZXBvbnNlLm1kICAgICAgICMgbGUgdGV4dGUgYXJyaXZlIHBhciBsJ2VudHLDqWUgc3RhbmRhcmQKClVuIG1lc3NhZ2UgbmUgcmV2aWVudCBKQU1BSVMgZGV1eCBmb2lzIDogZMOocyBxdSd1biDCqyB3YWl0IMK7IGxlIHJlbmQsIGxlIGNhbmFsCmxlIG1hcnF1ZSDCqyBsdSDCuyDigJQgZXQgbCdhcHBsaWNhdGlvbiByZW1wbGFjZSDCqyBub24gbHUgwrsgcGFyIMKrIGx1ICEgwrsgc291cyBsZQptZXNzYWdlLiBMZXMgY29tbWFuZGVzIG5lIHMnZXjDqWN1dGVudCBxdWUgc2kgbCdhcHBsaWNhdGlvbiBlc3Qgb3V2ZXJ0ZS4KIiIiCgppbXBvcnQgYXJncGFyc2UKaW1wb3J0IGJhc2U2NAppbXBvcnQgaW8KaW1wb3J0IGpzb24KaW1wb3J0IG9zCmltcG9ydCByZQppbXBvcnQgc3lzCmltcG9ydCB0aW1lCmltcG9ydCB1cmxsaWIuZXJyb3IKaW1wb3J0IHVybGxpYi5yZXF1ZXN0CmZyb20gZGF0ZXRpbWUgaW1wb3J0IGRhdGV0aW1lCgojIC0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLSBjb25maWd1cmF0aW9uCiMgTCdhZHJlc3NlIGR1IGNhbmFsLiBFbGxlIGVzdCBwdWJsaXF1ZSA6IGwnaWRlbnRpZmlhbnQgRVNUIGxhIGNsw6kuCkJBU0UgPSAiaHR0cHM6Ly9hcGktYWkuaHVnZHU3Nzc3Ny53b3JrZXJzLmRldi9jYW5hbC9hcmVuYSIKSE9MRCA9IDI1ICAgICAgICAgICMgZHVyw6llIG1heGltYWxlIGQndW5lIGF0dGVudGUgY8O0dMOpIHNlcnZldXIgKHNlY29uZGVzKQpHQVAgPSAxMCAgICAgICAgICAgIyBwYXVzZSBlbnRyZSBkZXV4IGF0dGVudGVzIDogMjUgcyBkJ2F0dGVudGUgKyAxMCBzIGRlIHBhdXNlCgpVQSA9ICgiTW96aWxsYS81LjAgKFgxMTsgTGludXggeDg2XzY0KSBBcHBsZVdlYktpdC81MzcuMzYgKEtIVE1MLCBsaWtlIEdlY2tvKSAiCiAgICAgICJDaHJvbWUvMTIwLjAuMC4wIFNhZmFyaS81MzcuMzYiKQoKIyBDbMOpcyBQVUJMSVFVRVMgZHUgcHJvamV0IChyZXBsaSBkaXJlY3QgOyB2b2lyIMKrIFJFUExJIERJUkVDVCDCuyBwbHVzIGhhdXQpLgpGQl9LRVkgPSAiQUl6YVN5QzU4SjFwYU8wVUxoVFcwUnVEZlVlUFZldU9VbUc1ZlpZIgpGQl9QUk9KRVQgPSAiYWktYWdlbnQtMTIzNGEiCkZCX0RPQyA9ICJjaGFuL21haW4iICAgICAgICMgbGUgY2FuYWwgcGFyIGTDqWZhdXQgOiBzYSBmaWNoZSwgZXQgc2EgYm/DrnRlIGp1c3RlIGVuIGRlc3NvdXMKRkJfU1VCID0gImJveCIKRkJfUk9MRSA9ICJhcmVuYSIgICAgICAgICAgIyBsZSByw7RsZSBzb3VzIGxlcXVlbCBsJ2FwcGxpY2F0aW9uIGF0dGVuZCBub3MgbWVzc2FnZXMKCgpkZWYgX2h0dHAobWV0aG9kLCB1cmwsIGJvZHk9Tm9uZSwgdGltZW91dD02MCwgaGVhZGVycz1Ob25lKToKICAgIGRhdGEgPSBOb25lCiAgICBoID0geyJVc2VyLUFnZW50IjogVUF9CiAgICBpZiBoZWFkZXJzOgogICAgICAgIGgudXBkYXRlKGhlYWRlcnMpCiAgICBpZiBib2R5IGlzIG5vdCBOb25lOgogICAgICAgIGRhdGEgPSBqc29uLmR1bXBzKGJvZHkpLmVuY29kZSgidXRmLTgiKQogICAgICAgIGhbIkNvbnRlbnQtVHlwZSJdID0gImFwcGxpY2F0aW9uL2pzb24iCiAgICByZXEgPSB1cmxsaWIucmVxdWVzdC5SZXF1ZXN0KHVybCwgZGF0YT1kYXRhLCBoZWFkZXJzPWgsIG1ldGhvZD1tZXRob2QpCiAgICB0cnk6CiAgICAgICAgd2l0aCB1cmxsaWIucmVxdWVzdC51cmxvcGVuKHJlcSwgdGltZW91dD10aW1lb3V0KSBhcyByZXNwOgogICAgICAgICAgICByYXcgPSByZXNwLnJlYWQoKS5kZWNvZGUoInV0Zi04IikKICAgICAgICAgICAgdHJ5OgogICAgICAgICAgICAgICAgcmV0dXJuIHJlc3Auc3RhdHVzLCBqc29uLmxvYWRzKHJhdykKICAgICAgICAgICAgZXhjZXB0IFZhbHVlRXJyb3I6CiAgICAgICAgICAgICAgICByZXR1cm4gcmVzcC5zdGF0dXMsIHsicmF3IjogcmF3fQogICAgZXhjZXB0IHVybGxpYi5lcnJvci5IVFRQRXJyb3IgYXMgZXJyOgogICAgICAgIHJhdyA9IGVyci5yZWFkKCkuZGVjb2RlKCJ1dGYtOCIsICJyZXBsYWNlIikKICAgICAgICB0cnk6CiAgICAgICAgICAgIHJldHVybiBlcnIuY29kZSwganNvbi5sb2FkcyhyYXcpCiAgICAgICAgZXhjZXB0IFZhbHVlRXJyb3I6CiAgICAgICAgICAgIHJldHVybiBlcnIuY29kZSwgeyJlcnJvciI6IHJhd1s6MzAwXX0KICAgIGV4Y2VwdCBFeGNlcHRpb24gYXMgZXJyOiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAjIHLDqXNlYXUgY291cMOpCiAgICAgICAgcmV0dXJuIDAsIHsiZXJyb3IiOiBzdHIoZXJyKX0KCgpkZWYgYXBpKHBhdGgsIGJvZHk9Tm9uZSwgdGltZW91dD02MCk6CiAgICAiIiJVbiBhcHBlbCDDoCBsYSBwYXNzZXJlbGxlIGR1IGNhbmFsIDsgcmVudm9pZSAoY29kZSwgb2JqZXQgSlNPTikuIiIiCiAgICByZXR1cm4gX2h0dHAoIlBPU1QiIGlmIGJvZHkgaXMgbm90IE5vbmUgZWxzZSAiR0VUIiwgQkFTRSArIHBhdGgsIGJvZHksIHRpbWVvdXQpCgoKSU1HX0RJUiA9ICIvaG9tZS91c2VyL3VwbG9hZHMvY2FuYWwiICAgICAjIG/DuSBzb250IHJhbmfDqWVzIGxlcyBpbWFnZXMgcmXDp3VlcwpJTUdfTUFYID0gMTI4MCAgICAgICAgICAgICAgICAgICAgICAgICAgICAjIHBsdXMgZ3JhbmQgY8O0dMOpLCBlbiBwaXhlbHMKSU1HX0I2NF9NQVggPSA3NjAwMDAgICAgICAgICAgICAgICAgICAgICAgIyB1biBkb2N1bWVudCBGaXJlc3RvcmUgZmFpdCAxIE1pbwoKCmRlZiBpbWFnZV9yZWFkeShjaGVtaW4pOgogICAgIiIiUHLDqXBhcmUgdW5lIGltYWdlIHBvdXIgbGUgY2FuYWwgOiBKUEVHLCAxMjgwIHB4IGF1IHBsdXMsIGJhc2U2NCBib3Juw6kuCgogICAgRWxsZSBlc3QgUkVGT1JNQVTDiUUgKHBhcyBzZXVsZW1lbnQgY29wacOpZSkgcGFyY2UgcXUndW4gZG9jdW1lbnQgRmlyZXN0b3JlCiAgICBuZSBwZXV0IHBhcyBkw6lwYXNzZXIgMSBNaW8gOiB1bmUgcGhvdG8gZGUgdMOpbMOpcGhvbmUgbid5IHRpZW5kcmFpdCBwYXMuCiAgICAiIiIKICAgIHRyeToKICAgICAgICBmcm9tIFBJTCBpbXBvcnQgSW1hZ2UKICAgIGV4Y2VwdCBJbXBvcnRFcnJvcjogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICMgUGlsbG93IGFic2VudAogICAgICAgIHdpdGggb3BlbihjaGVtaW4sICJyYiIpIGFzIGY6CiAgICAgICAgICAgIGJydXQgPSBmLnJlYWQoKQogICAgICAgIGlmIGxlbihicnV0KSA+IElNR19CNjRfTUFYICogMyAvLyA0OgogICAgICAgICAgICByZXR1cm4gTm9uZQogICAgICAgIG1pbWUgPSAiaW1hZ2UvcG5nIiBpZiBicnV0Wzo0XSA9PSBiIlx4ODlQTkciIGVsc2UgImltYWdlL2pwZWciCiAgICAgICAgcmV0dXJuIHsibmFtZSI6IG9zLnBhdGguYmFzZW5hbWUoY2hlbWluKSwgIm1pbWUiOiBtaW1lLAogICAgICAgICAgICAgICAgImI2NCI6IGJhc2U2NC5iNjRlbmNvZGUoYnJ1dCkuZGVjb2RlKCJhc2NpaSIpfQogICAgdHJ5OgogICAgICAgIGltID0gSW1hZ2Uub3BlbihjaGVtaW4pLmNvbnZlcnQoIlJHQiIpCiAgICBleGNlcHQgRXhjZXB0aW9uIGFzIGVycjoKICAgICAgICBwcmludCgiICAgKGltYWdlIGlsbGlzaWJsZSA6ICVzKSIgJSBlcnIsIGZpbGU9c3lzLnN0ZGVycikKICAgICAgICByZXR1cm4gTm9uZQogICAgaW0udGh1bWJuYWlsKChJTUdfTUFYLCBJTUdfTUFYKSkKICAgIG5vbSA9IG9zLnBhdGguc3BsaXRleHQob3MucGF0aC5iYXNlbmFtZShjaGVtaW4pKVswXSArICIuanBnIgogICAgZm9yIHEgaW4gKDgyLCA3MiwgNjIsIDUyLCA0Mik6CiAgICAgICAgYnVmID0gaW8uQnl0ZXNJTygpCiAgICAgICAgaW0uc2F2ZShidWYsICJKUEVHIiwgcXVhbGl0eT1xLCBvcHRpbWl6ZT1UcnVlKQogICAgICAgIGI2NCA9IGJhc2U2NC5iNjRlbmNvZGUoYnVmLmdldHZhbHVlKCkpLmRlY29kZSgiYXNjaWkiKQogICAgICAgIGlmIGxlbihiNjQpIDw9IElNR19CNjRfTUFYOgogICAgICAgICAgICByZXR1cm4geyJuYW1lIjogbm9tLCAibWltZSI6ICJpbWFnZS9qcGVnIiwgImI2NCI6IGI2NH0KICAgIHByaW50KCIgICAoaW1hZ2UgdHJvcCBsb3VyZGUgbcOqbWUgY29tcHJlc3PDqWUgOiAlcykiICUgY2hlbWluLCBmaWxlPXN5cy5zdGRlcnIpCiAgICByZXR1cm4gTm9uZQoKCmRlZiBpbWFnZXNfcmVhZHkoY2hlbWlucyk6CiAgICBvdXQgPSBbXQogICAgZm9yIGMgaW4gKGNoZW1pbnMgb3IgW10pWzozXToKICAgICAgICBpZiBub3Qgb3MucGF0aC5leGlzdHMoYyk6CiAgICAgICAgICAgIGRpZSgiaW1hZ2UgaW50cm91dmFibGUgOiAlcyIgJSBjKQogICAgICAgIGltID0gaW1hZ2VfcmVhZHkoYykKICAgICAgICBpZiBpbToKICAgICAgICAgICAgb3V0LmFwcGVuZChpbSkKICAgIHJldHVybiBvdXQKCgpkZWYgaW1hZ2VfbHVlKGltLCBwcmVmaXhlKToKICAgICIiIsOJY3JpdCBzdXIgbGUgZGlzcXVlIHVuZSBpbWFnZSByZcOndWUgZXQgcmVuZCBzb24gY2hlbWluIChvdSAiIikuIiIiCiAgICBiNjQgPSAoaW0gb3Ige30pLmdldCgiYjY0Iikgb3IgIiIKICAgIGlmIG5vdCBiNjQ6CiAgICAgICAgcmV0dXJuICIiCiAgICBvcy5tYWtlZGlycyhJTUdfRElSLCBleGlzdF9vaz1UcnVlKQogICAgbWltZSA9IChpbS5nZXQoIm1pbWUiKSBvciAiaW1hZ2UvanBlZyIpLmxvd2VyKCkKICAgIGV4dCA9IHsiaW1hZ2UvcG5nIjogIi5wbmciLCAiaW1hZ2UvZ2lmIjogIi5naWYiLCAiaW1hZ2Uvd2VicCI6ICIud2VicCJ9LmdldChtaW1lLCAiLmpwZyIpCiAgICBub20gPSByZS5zdWIociJbXkEtWmEtejAtOS5fLV0rIiwgIi0iLCBzdHIoaW0uZ2V0KCJuYW1lIikgb3IgImltYWdlIikpCiAgICBpZiBub3Qgbm9tLmxvd2VyKCkuZW5kc3dpdGgoZXh0KToKICAgICAgICBub20gPSBvcy5wYXRoLnNwbGl0ZXh0KG5vbSlbMF0gKyBleHQKICAgIGNoZW1pbiA9IG9zLnBhdGguam9pbihJTUdfRElSLCAiJXMtJXMiICUgKHByZWZpeGUsIG5vbSkpCiAgICB0cnk6CiAgICAgICAgd2l0aCBvcGVuKGNoZW1pbiwgIndiIikgYXMgZjoKICAgICAgICAgICAgZi53cml0ZShiYXNlNjQuYjY0ZGVjb2RlKGI2NCkpCiAgICBleGNlcHQgRXhjZXB0aW9uIGFzIGVycjoKICAgICAgICBwcmludCgiICAgKGltYWdlIG5vbiBlbnJlZ2lzdHLDqWUgOiAlcykiICUgZXJyLCBmaWxlPXN5cy5zdGRlcnIpCiAgICAgICAgcmV0dXJuICIiCiAgICByZXR1cm4gY2hlbWluCgoKZGVmIGRpZShtc2csIGNvZGU9MSk6CiAgICBwcmludCgi4pyXICIgKyBtc2csIGZpbGU9c3lzLnN0ZGVycikKICAgIHN5cy5leGl0KGNvZGUpCgoKIyAtLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tIHJlcGxpIDogYWNjw6hzIGRpcmVjdApfRElSRUNUX1RPS0VOID0geyJpZCI6ICIiLCAiZXhwaXJlIjogMH0KX0RJUkVDVF9ESVQgPSBGYWxzZSAgICAgICAgIyBhLXQtb24gZMOpasOgIHByw6l2ZW51IHF1ZSBsJ29uIHBhc3NlIGVuIGRpcmVjdCA/CgoKZGVmIF9mYl90b2tlbigpOgogICAgIiIiVW5lIHNlc3Npb24gYW5vbnltZSBkdSBwcm9qZXQgKGF1Y3VuIG1vdCBkZSBwYXNzZSwgYXVjdW5lIGRvbm7DqWUgcHJpdsOpZSkuIiIiCiAgICBpZiBfRElSRUNUX1RPS0VOWyJpZCJdIGFuZCBfRElSRUNUX1RPS0VOWyJleHBpcmUiXSA+IHRpbWUudGltZSgpICsgNjA6CiAgICAgICAgcmV0dXJuIF9ESVJFQ1RfVE9LRU5bImlkIl0KICAgIHVybCA9ICgiaHR0cHM6Ly9pZGVudGl0eXRvb2xraXQuZ29vZ2xlYXBpcy5jb20vdjEvYWNjb3VudHM6c2lnblVwP2tleT0iICsgRkJfS0VZKQogICAgY29kZSwgcmVzID0gX2h0dHAoIlBPU1QiLCB1cmwsIHsicmV0dXJuU2VjdXJlVG9rZW4iOiBUcnVlfSkKICAgIHRva2VuID0gKHJlcyBvciB7fSkuZ2V0KCJpZFRva2VuIikgb3IgIiIKICAgIGlmIGNvZGUgPj0gMzAwIG9yIG5vdCB0b2tlbjoKICAgICAgICByZXR1cm4gIiIKICAgIF9ESVJFQ1RfVE9LRU5bImlkIl0gPSB0b2tlbgogICAgX0RJUkVDVF9UT0tFTlsiZXhwaXJlIl0gPSB0aW1lLnRpbWUoKSArIGZsb2F0KChyZXMgb3Ige30pLmdldCgiZXhwaXJlc0luIikgb3IgMzYwMCkKICAgIHJldHVybiB0b2tlbgoKCmRlZiBfZmJfdXJsKGNoZW1pbj0iIiwgcXVlcnk9IiIpOgogICAgcmV0dXJuICgiaHR0cHM6Ly9maXJlc3RvcmUuZ29vZ2xlYXBpcy5jb20vdjEvcHJvamVjdHMvIiArIEZCX1BST0pFVAogICAgICAgICAgICArICIvZGF0YWJhc2VzLyhkZWZhdWx0KS9kb2N1bWVudHMvIiArIGNoZW1pbi5sc3RyaXAoIi8iKSArIHF1ZXJ5KQoKCmRlZiBfZmJfZW5fZGlyZWN0KHF1b2kpOgogICAgZ2xvYmFsIF9ESVJFQ1RfRElUCiAgICBpZiBub3QgX0RJUkVDVF9ESVQ6CiAgICAgICAgX0RJUkVDVF9ESVQgPSBUcnVlCiAgICAgICAgcHJpbnQoIiAgIChwYXNzZXJlbGxlIGluZGlzcG9uaWJsZSDigJQgbGlhaXNvbiBkaXJlY3RlIMOgIGxhIGJhc2UgcG91ciAlcykiICUgcXVvaSkKICAgICAgICBzeXMuc3Rkb3V0LmZsdXNoKCkKCgpkZWYgX2ZiX3ZhbChjaGFtcCk6CiAgICBpZiBub3QgaXNpbnN0YW5jZShjaGFtcCwgZGljdCk6CiAgICAgICAgcmV0dXJuIE5vbmUKICAgIGZvciBjbGUgaW4gKCJzdHJpbmdWYWx1ZSIsICJ0aW1lc3RhbXBWYWx1ZSIpOgogICAgICAgIGlmIGNsZSBpbiBjaGFtcDoKICAgICAgICAgICAgcmV0dXJuIGNoYW1wW2NsZV0KICAgIGZvciBjbGUgaW4gKCJpbnRlZ2VyVmFsdWUiLCAiZG91YmxlVmFsdWUiKToKICAgICAgICBpZiBjbGUgaW4gY2hhbXA6CiAgICAgICAgICAgIHRyeToKICAgICAgICAgICAgICAgIHJldHVybiBmbG9hdChjaGFtcFtjbGVdKSBpZiBjbGUgPT0gImRvdWJsZVZhbHVlIiBlbHNlIGludChjaGFtcFtjbGVdKQogICAgICAgICAgICBleGNlcHQgKFR5cGVFcnJvciwgVmFsdWVFcnJvcik6CiAgICAgICAgICAgICAgICByZXR1cm4gTm9uZQogICAgaWYgImJvb2xlYW5WYWx1ZSIgaW4gY2hhbXA6CiAgICAgICAgcmV0dXJuIGJvb2woY2hhbXBbImJvb2xlYW5WYWx1ZSJdKQogICAgcmV0dXJuIE5vbmUKCgpkZWYgX2ZiX2ZpZWxkKHZhbGV1cik6CiAgICBpZiBpc2luc3RhbmNlKHZhbGV1ciwgYm9vbCk6CiAgICAgICAgcmV0dXJuIHsiYm9vbGVhblZhbHVlIjogdmFsZXVyfQogICAgaWYgaXNpbnN0YW5jZSh2YWxldXIsIGludCk6CiAgICAgICAgcmV0dXJuIHsiaW50ZWdlclZhbHVlIjogc3RyKHZhbGV1cil9CiAgICBpZiBpc2luc3RhbmNlKHZhbGV1ciwgZmxvYXQpOgogICAgICAgIHJldHVybiB7ImRvdWJsZVZhbHVlIjogdmFsZXVyfQogICAgcmV0dXJuIHsic3RyaW5nVmFsdWUiOiAiIiBpZiB2YWxldXIgaXMgTm9uZSBlbHNlIHN0cih2YWxldXIpfQoKCmRlZiBfZmJfYW55KGNoYW1wKToKICAgICIiIlZhbGV1ciBGaXJlc3RvcmUgY29tcGzDqHRlIDogZ8OocmUgYXVzc2kgbGVzIHRhYmxlYXV4IGV0IG9iamV0cyAoaW1hZ2VzKS4iIiIKICAgIGlmIG5vdCBpc2luc3RhbmNlKGNoYW1wLCBkaWN0KToKICAgICAgICByZXR1cm4gTm9uZQogICAgaWYgImFycmF5VmFsdWUiIGluIGNoYW1wOgogICAgICAgIHZhbHMgPSAoY2hhbXBbImFycmF5VmFsdWUiXSBvciB7fSkuZ2V0KCJ2YWx1ZXMiKSBvciBbXQogICAgICAgIHJldHVybiBbX2ZiX2FueSh2KSBmb3IgdiBpbiB2YWxzXQogICAgaWYgIm1hcFZhbHVlIiBpbiBjaGFtcDoKICAgICAgICBjaGFtcHMgPSAoY2hhbXBbIm1hcFZhbHVlIl0gb3Ige30pLmdldCgiZmllbGRzIikgb3Ige30KICAgICAgICByZXR1cm4gZGljdCgoaywgX2ZiX2FueShjaGFtcHNba10pKSBmb3IgayBpbiBjaGFtcHMpCiAgICByZXR1cm4gX2ZiX3ZhbChjaGFtcCkKCgpkZWYgX2ZiX2RvYyhkb2N1bWVudCk6CiAgICBjaGFtcHMgPSAoZG9jdW1lbnQgb3Ige30pLmdldCgiZmllbGRzIikgb3Ige30KICAgIG1lc3NhZ2UgPSBkaWN0KChjbGUsIF9mYl9hbnkoY2hhbXBzW2NsZV0pKSBmb3IgY2xlIGluIGNoYW1wcykKICAgIG1lc3NhZ2VbImlkIl0gPSBzdHIoKGRvY3VtZW50IG9yIHt9KS5nZXQoIm5hbWUiKSBvciAiIikuc3BsaXQoIi8iKVstMV0KICAgIHJldHVybiBtZXNzYWdlCgoKZGVmIF9mYl90cmllKG1lc3NhZ2VzKToKICAgIG1lc3NhZ2VzLnNvcnQoa2V5PWxhbWJkYSBtOiBtLmdldCgidHMiKSBvciAwKQogICAgcmV0dXJuIG1lc3NhZ2VzCgoKZGVmIGZiX2xpc3Qobj0zMCk6CiAgICAiIiJMJ2hpc3RvcmlxdWUgcsOpY2VudCAobCfDqXF1aXZhbGVudCBkaXJlY3QgZGUgQkFTRS9yZWFkKS4iIiIKICAgIHRva2VuID0gX2ZiX3Rva2VuKCkKICAgIGlmIG5vdCB0b2tlbjoKICAgICAgICByZXR1cm4gTm9uZQogICAgY29ycHMgPSB7InN0cnVjdHVyZWRRdWVyeSI6IHsKICAgICAgICAiZnJvbSI6IFt7ImNvbGxlY3Rpb25JZCI6IEZCX1NVQn1dLAogICAgICAgICJvcmRlckJ5IjogW3siZmllbGQiOiB7ImZpZWxkUGF0aCI6ICJ0cyJ9LCAiZGlyZWN0aW9uIjogIkRFU0NFTkRJTkcifV0sCiAgICAgICAgImxpbWl0IjogbWF4KDEsIG1pbigyMDAsIGludChuKSkpLAogICAgfX0KICAgIGNvZGUsIHJlcyA9IF9odHRwKCJQT1NUIiwgX2ZiX3VybChGQl9ET0MgKyAiOnJ1blF1ZXJ5IiksIGNvcnBzLAogICAgICAgICAgICAgICAgICAgICAgaGVhZGVycz17IkF1dGhvcml6YXRpb24iOiAiQmVhcmVyICIgKyB0b2tlbn0pCiAgICBpZiBjb2RlID49IDMwMCBvciBub3QgaXNpbnN0YW5jZShyZXMsIGxpc3QpOgogICAgICAgIHJldHVybiBOb25lCiAgICAjIERFU0NFTkRJTkcgbmUgcmVuZCBxdWUgbGEgdHJhbmNoZSByw6ljZW50ZSA6IG9uIGxhIHJlbWV0IGRhbnMgbCdvcmRyZS4KICAgIHJldHVybiBfZmJfdHJpZShbX2ZiX2RvYyh4WyJkb2N1bWVudCJdKSBmb3IgeCBpbiByZXMgaWYgaXNpbnN0YW5jZSh4LCBkaWN0KSBhbmQgeC5nZXQoImRvY3VtZW50IildKQoKCmRlZiBmYl9tYXJxdWVfbHVzKG1lc3NhZ2VzKToKICAgIHRva2VuID0gX2ZiX3Rva2VuKCkKICAgIGlmIG5vdCB0b2tlbjoKICAgICAgICByZXR1cm4KICAgIGZvciBtZXNzYWdlIGluIG1lc3NhZ2VzOgogICAgICAgIGlmIG5vdCBtZXNzYWdlLmdldCgiaWQiKToKICAgICAgICAgICAgY29udGludWUKICAgICAgICB1cmwgPSBfZmJfdXJsKEZCX0RPQyArICIvIiArIEZCX1NVQiArICIvIiArIG1lc3NhZ2VbImlkIl0sCiAgICAgICAgICAgICAgICAgICAgICAiP3VwZGF0ZU1hc2suZmllbGRQYXRocz1zdGF0ZSZ1cGRhdGVNYXNrLmZpZWxkUGF0aHM9cmVhZEF0IikKICAgICAgICBfaHR0cCgiUEFUQ0giLCB1cmwsIHsiZmllbGRzIjogeyJzdGF0ZSI6IF9mYl9maWVsZCgicmVhZCIpLCAicmVhZEF0IjogX2ZiX2ZpZWxkKGludCh0aW1lLnRpbWUoKSAqIDEwMDApKX19LAogICAgICAgICAgICAgIGhlYWRlcnM9eyJBdXRob3JpemF0aW9uIjogIkJlYXJlciAiICsgdG9rZW59KQoKCmRlZiBmYl91bnJlYWQoKToKICAgICIiIkxlcyBtZXNzYWdlcyBOT04gTFVTLCBwYXIgcmVxdcOqdGUgZmlsdHLDqWUgKGZhY3R1csOpZSBhdSBkb2N1bWVudCByZW5kdSA6CiAgICB1bmUgYXR0ZW50ZSBxdWkgbmUgdHJvdXZlIHJpZW4gbmUgY29uc29tbWUgZG9uYyByaWVuKS4iIiIKICAgIHRva2VuID0gX2ZiX3Rva2VuKCkKICAgIGlmIG5vdCB0b2tlbjoKICAgICAgICByZXR1cm4gTm9uZQogICAgY29ycHMgPSB7InN0cnVjdHVyZWRRdWVyeSI6IHsKICAgICAgICAiZnJvbSI6IFt7ImNvbGxlY3Rpb25JZCI6IEZCX1NVQn1dLAogICAgICAgICJ3aGVyZSI6IHsiZmllbGRGaWx0ZXIiOiB7ImZpZWxkIjogeyJmaWVsZFBhdGgiOiAic3RhdGUifSwgIm9wIjogIkVRVUFMIiwKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICJ2YWx1ZSI6IHsic3RyaW5nVmFsdWUiOiAibmV3In19fSwKICAgICAgICAibGltaXQiOiA1MCwKICAgIH19CiAgICBjb2RlLCByZXMgPSBfaHR0cCgiUE9TVCIsIF9mYl91cmwoRkJfRE9DICsgIjpydW5RdWVyeSIpLCBjb3JwcywKICAgICAgICAgICAgICAgICAgICAgIGhlYWRlcnM9eyJBdXRob3JpemF0aW9uIjogIkJlYXJlciAiICsgdG9rZW59KQogICAgaWYgY29kZSA+PSAzMDAgb3Igbm90IGlzaW5zdGFuY2UocmVzLCBsaXN0KToKICAgICAgICByZXR1cm4gTm9uZQogICAgbWVzc2FnZXMgPSBbX2ZiX2RvYyh4WyJkb2N1bWVudCJdKSBmb3IgeCBpbiByZXMgaWYgaXNpbnN0YW5jZSh4LCBkaWN0KSBhbmQgeC5nZXQoImRvY3VtZW50IildCiAgICBtZXNzYWdlcyA9IFttIGZvciBtIGluIG1lc3NhZ2VzIGlmIG0uZ2V0KCJyb2xlIikgPT0gInVzZXIiXQogICAgcmV0dXJuIF9mYl90cmllKG1lc3NhZ2VzKQoKCmRlZiBmYl9yZXN1bHRhdChyZXFfaWQpOgogICAgIiIiTGUgbWVzc2FnZSDCqyB0b29sIMK7IMOpY3JpdCBlbiByw6lwb25zZSDDoCBsYSBjb21tYW5kZSByZXFfaWQgKGxlIHBsdXMgcsOpY2VudCkuIiIiCiAgICBpZiBub3QgcmVxX2lkOgogICAgICAgIHJldHVybiBOb25lCiAgICB0b2tlbiA9IF9mYl90b2tlbigpCiAgICBpZiBub3QgdG9rZW46CiAgICAgICAgcmV0dXJuIE5vbmUKICAgIGNvcnBzID0geyJzdHJ1Y3R1cmVkUXVlcnkiOiB7CiAgICAgICAgImZyb20iOiBbeyJjb2xsZWN0aW9uSWQiOiBGQl9TVUJ9XSwKICAgICAgICAid2hlcmUiOiB7ImZpZWxkRmlsdGVyIjogeyJmaWVsZCI6IHsiZmllbGRQYXRoIjogInJlcSJ9LCAib3AiOiAiRVFVQUwiLAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgInZhbHVlIjogeyJzdHJpbmdWYWx1ZSI6IHN0cihyZXFfaWQpfX19LAogICAgICAgICJsaW1pdCI6IDEwLAogICAgfX0KICAgIGNvZGUsIHJlcyA9IF9odHRwKCJQT1NUIiwgX2ZiX3VybChGQl9ET0MgKyAiOnJ1blF1ZXJ5IiksIGNvcnBzLAogICAgICAgICAgICAgICAgICAgICAgaGVhZGVycz17IkF1dGhvcml6YXRpb24iOiAiQmVhcmVyICIgKyB0b2tlbn0pCiAgICBpZiBjb2RlID49IDMwMCBvciBub3QgaXNpbnN0YW5jZShyZXMsIGxpc3QpOgogICAgICAgIHJldHVybiBOb25lCiAgICBkb2NzID0gW19mYl9kb2MoeFsiZG9jdW1lbnQiXSkgZm9yIHggaW4gcmVzIGlmIGlzaW5zdGFuY2UoeCwgZGljdCkgYW5kIHguZ2V0KCJkb2N1bWVudCIpXQogICAgb3V0aWxzID0gW20gZm9yIG0gaW4gZG9jcyBpZiBtLmdldCgicm9sZSIpID09ICJ0b29sIl0KICAgIG91dGlscy5zb3J0KGtleT1sYW1iZGEgbTogbS5nZXQoInRzIikgb3IgMCkKICAgIHJldHVybiBvdXRpbHNbLTFdIGlmIG91dGlscyBlbHNlIE5vbmUKCgpkZWYgX2ZiX2ltYWdlcyhpbWdzKToKICAgICIiIkxlcyBpbWFnZXMgZCd1biBtZXNzYWdlLCBhdSBmb3JtYXQgRmlyZXN0b3JlICh0YWJsZWF1IGQnb2JqZXRzKS4iIiIKICAgIHZhbHMgPSBbXQogICAgZm9yIGltIGluIGltZ3Mgb3IgW106CiAgICAgICAgdmFscy5hcHBlbmQoeyJtYXBWYWx1ZSI6IHsiZmllbGRzIjogewogICAgICAgICAgICAibmFtZSI6IF9mYl9maWVsZChpbS5nZXQoIm5hbWUiKSBvciAiaW1hZ2UiKSwKICAgICAgICAgICAgIm1pbWUiOiBfZmJfZmllbGQoaW0uZ2V0KCJtaW1lIikgb3IgImltYWdlL2pwZWciKSwKICAgICAgICAgICAgImI2NCI6IF9mYl9maWVsZChpbS5nZXQoImI2NCIpIG9yICIiKSwKICAgICAgICB9fX0pCiAgICByZXR1cm4geyJhcnJheVZhbHVlIjogeyJ2YWx1ZXMiOiB2YWxzfX0KCgpkZWYgZmJfd3JpdGUodGV4dGUsIGtpbmQ9InRleHQiLCByZXE9IiIsIG5vdGU9IiIsIGltZ3M9Tm9uZSk6CiAgICAiIiLDiWNyaXJlIHVuIG1lc3NhZ2UgZGFucyBsZSBjYW5hbCAobCfDqXF1aXZhbGVudCBkaXJlY3QgZGUgQkFTRS93cml0ZSkuIiIiCiAgICB0b2tlbiA9IF9mYl90b2tlbigpCiAgICBpZiBub3QgdG9rZW46CiAgICAgICAgcmV0dXJuIE5vbmUKICAgIGNoYW1wcyA9IHsKICAgICAgICAicm9sZSI6IF9mYl9maWVsZChGQl9ST0xFKSwgInRleHQiOiBfZmJfZmllbGQodGV4dGUpLCAia2luZCI6IF9mYl9maWVsZChraW5kKSwKICAgICAgICAic3RhdGUiOiBfZmJfZmllbGQoIm5ldyIpLCAic3JjIjogX2ZiX2ZpZWxkKCJzY3JpcHQiKSwgIm5vdGUiOiBfZmJfZmllbGQobm90ZSksCiAgICAgICAgInJlcSI6IF9mYl9maWVsZChyZXEpLCAidHMiOiBfZmJfZmllbGQoaW50KHRpbWUudGltZSgpICogMTAwMCkpLAogICAgfQogICAgaWYgaW1nczoKICAgICAgICBjaGFtcHNbImltZ3MiXSA9IF9mYl9pbWFnZXMoaW1ncykKICAgIGNvcnBzID0geyJmaWVsZHMiOiBjaGFtcHN9CiAgICBjb2RlLCByZXMgPSBfaHR0cCgiUE9TVCIsIF9mYl91cmwoRkJfRE9DICsgIi8iICsgRkJfU1VCKSwgY29ycHMsCiAgICAgICAgICAgICAgICAgICAgICBoZWFkZXJzPXsiQXV0aG9yaXphdGlvbiI6ICJCZWFyZXIgIiArIHRva2VufSkKICAgIGlmIGNvZGUgPj0gMzAwIG9yIG5vdCBpc2luc3RhbmNlKHJlcywgZGljdCk6CiAgICAgICAgcmV0dXJuIE5vbmUKICAgIHJldHVybiBzdHIocmVzLmdldCgibmFtZSIpIG9yICIiKS5zcGxpdCgiLyIpWy0xXQoKCmRlZiBmYl9zdGF0ZSgpOgogICAgIiIiTCfDqXRhdCBkdSBjYW5hbCA6IGxhIGZpY2hlLCBsZSBjb21wdGUgZGVzIG1lc3NhZ2VzLCBsZXMgbm9uLWx1cy4iIiIKICAgIHRva2VuID0gX2ZiX3Rva2VuKCkKICAgIGlmIG5vdCB0b2tlbjoKICAgICAgICByZXR1cm4gTm9uZQogICAgZW50ZXRlcyA9IHsiQXV0aG9yaXphdGlvbiI6ICJCZWFyZXIgIiArIHRva2VufQogICAgY29kZV9maWNoZSwgZmljaGUgPSBfaHR0cCgiR0VUIiwgX2ZiX3VybChGQl9ET0MpLCBoZWFkZXJzPWVudGV0ZXMpCiAgICBub20gPSAiYXJlbmEiCiAgICBpZiBjb2RlX2ZpY2hlIDwgMzAwIGFuZCBpc2luc3RhbmNlKGZpY2hlLCBkaWN0KToKICAgICAgICBjaGFtcHMgPSAoZmljaGUuZ2V0KCJmaWVsZHMiKSBvciB7fSkKICAgICAgICBub20gPSBfZmJfdmFsKGNoYW1wcy5nZXQoIm5hbWUiKSkgb3IgImFyZW5hIgogICAgdG90YWwgPSBOb25lCiAgICBjb3JwcyA9IHsic3RydWN0dXJlZEFnZ3JlZ2F0aW9uUXVlcnkiOiB7CiAgICAgICAgInN0cnVjdHVyZWRRdWVyeSI6IHsiZnJvbSI6IFt7ImNvbGxlY3Rpb25JZCI6IEZCX1NVQn1dfSwKICAgICAgICAiYWdncmVnYXRpb25zIjogW3siY291bnQiOiB7fX1dLAogICAgfX0KICAgIGNvZGUsIHJlcyA9IF9odHRwKCJQT1NUIiwgX2ZiX3VybChGQl9ET0MgKyAiOnJ1bkFnZ3JlZ2F0aW9uUXVlcnkiKSwgY29ycHMsIGhlYWRlcnM9ZW50ZXRlcykKICAgIGlmIGNvZGUgPCAzMDAgYW5kIGlzaW5zdGFuY2UocmVzLCBsaXN0KToKICAgICAgICBmb3IgeCBpbiByZXM6CiAgICAgICAgICAgIGFncmVnYXRzID0gKCh4IG9yIHt9KS5nZXQoInJlc3VsdCIpIG9yIHt9KS5nZXQoImFnZ3JlZ2F0ZUZpZWxkcyIpIG9yIHt9CiAgICAgICAgICAgIGZvciBjbGUgaW4gYWdyZWdhdHM6CiAgICAgICAgICAgICAgICB0cnk6CiAgICAgICAgICAgICAgICAgICAgdG90YWwgPSBpbnQoYWdyZWdhdHNbY2xlXS5nZXQoImludGVnZXJWYWx1ZSIpIG9yIDApCiAgICAgICAgICAgICAgICBleGNlcHQgKFR5cGVFcnJvciwgVmFsdWVFcnJvcik6CiAgICAgICAgICAgICAgICAgICAgdG90YWwgPSBOb25lCiAgICBub25fbHVzID0gZmJfdW5yZWFkKCkKICAgIHJldHVybiB7Im5vbSI6IG5vbSwgIm1lc3NhZ2VzIjogdG90YWwsCiAgICAgICAgICAgICJub25fbHVzIjogbGVuKG5vbl9sdXMpIGlmIG5vbl9sdXMgaXMgbm90IE5vbmUgZWxzZSBOb25lfQoKCiMgLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLSBhZmZpY2hhZ2UKZGVmIGhobW0odHNfbXMpOgogICAgaWYgbm90IHRzX21zOgogICAgICAgIHJldHVybiAiPyIKICAgIHRyeToKICAgICAgICByZXR1cm4gZGF0ZXRpbWUuZnJvbXRpbWVzdGFtcChmbG9hdCh0c19tcykgLyAxMDAwLjApLmFzdGltZXpvbmUoKS5zdHJmdGltZSgiJUg6JU06JVMiKQogICAgZXhjZXB0IChPdmVyZmxvd0Vycm9yLCBPU0Vycm9yLCBWYWx1ZUVycm9yKToKICAgICAgICByZXR1cm4gIj8iCgoKUk9MRV9UQUcgPSB7InVzZXIiOiAi8J+RpCB0b2kiLCAiYXJlbmEiOiAi4peGIGFyZW5hIiwgImFnZW50IjogIuKXhiBhZ2VudCIsICJ0b29sIjogIuKamSB2bSJ9CgoKZGVmIHNob3cobXNnLCBmdWxsPUZhbHNlLCBpbmRlbnQ9IiAgIik6CiAgICB0YWcgPSBST0xFX1RBRy5nZXQobXNnLmdldCgicm9sZSIpLCBtc2cuZ2V0KCJyb2xlIikpCiAgICBraW5kID0gbXNnLmdldCgia2luZCIpIG9yICJ0ZXh0IgogICAgdGV4dCA9IG1zZy5nZXQoInRleHQiKSBvciAiIgogICAgYm9keSA9IHRleHQgaWYgKGZ1bGwgb3IgbGVuKHRleHQpIDw9IDUwMCkgZWxzZSB0ZXh0Wzo1MDBdICsgIiDigKYoJWQgY2FyYWN0w6hyZXMgYXUgdG90YWwpIiAlIGxlbih0ZXh0KQogICAgaGVhZCA9ICIlc1slc10gJXMlcyIgJSAoaW5kZW50LCBoaG1tKG1zZy5nZXQoInRzIikpLCB0YWcsCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAoIiDCtyAiICsga2luZCkgaWYga2luZCAhPSAidGV4dCIgZWxzZSAiIikKICAgIGxpbmVzID0gW2hlYWRdCiAgICBmb3IgbGluZSBpbiAoYm9keS5zcGxpdGxpbmVzKCkgb3IgWyIiXSk6CiAgICAgICAgbGluZXMuYXBwZW5kKGluZGVudCArICIgICIgKyBsaW5lKQogICAgIyBpbWFnZXMgcmXDp3VlcyA6IGVucmVnaXN0csOpZXMgc3VyIGxlIGRpc3F1ZSwgY2hlbWluIGFmZmljaMOpCiAgICBmb3IgaSwgaW0gaW4gZW51bWVyYXRlKG1zZy5nZXQoImltZ3MiKSBvciBbXSk6CiAgICAgICAgaWYgbm90IGlzaW5zdGFuY2UoaW0sIGRpY3QpOgogICAgICAgICAgICBjb250aW51ZQogICAgICAgIHByZWYgPSAiJXMtJXMiICUgKG1zZy5nZXQoImlkIikgb3IgIm1zZyIsIGkgKyAxKQogICAgICAgIGNoZW1pbiA9IGltYWdlX2x1ZShpbSwgcHJlZikKICAgICAgICBrbyA9IChsZW4oaW0uZ2V0KCJiNjQiKSBvciAiIikgKiAzIC8vIDQpIC8vIDEwMjQKICAgICAgICBpZiBjaGVtaW46CiAgICAgICAgICAgIGxpbmVzLmFwcGVuZCgiJXMgIPCflrwgJXMg4oCUICVkIEtvIOKGkiAlcyIgJSAoaW5kZW50LCBpbS5nZXQoIm5hbWUiKSBvciAiaW1hZ2UiLCBrbywgY2hlbWluKSkKICAgICAgICBlbHNlOgogICAgICAgICAgICBsaW5lcy5hcHBlbmQoIiVzICDwn5a8ICVzIOKAlCAlZCBLbyAobm9uIGVucmVnaXN0csOpZSkiICUgKGluZGVudCwgaW0uZ2V0KCJuYW1lIikgb3IgImltYWdlIiwga28pKQogICAgcHJpbnQoIlxuIi5qb2luKGxpbmVzKSkKCgojIC0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLSBjb21tYW5kZXMKZGVmIGNtZF9yZWFkKGFyZ3MpOgogICAgY29kZSwgcmVzID0gYXBpKCIvcmVhZD9uPSVkIiAlIG1heCgxLCBhcmdzLm4pKQogICAgbXNncyA9IE5vbmUKICAgIGlmIGNvZGUgPCAzMDAgYW5kIGlzaW5zdGFuY2UocmVzLCBkaWN0KSBhbmQgcmVzLmdldCgib2siKToKICAgICAgICBtc2dzID0gcmVzLmdldCgibWVzc2FnZXMiKSBvciBbXQogICAgICAgIGNhbmFsID0gcmVzLmdldCgiY2FuYWwiKSBvciB7fQogICAgZWxzZToKICAgICAgICBfZmJfZW5fZGlyZWN0KCJsYSBsZWN0dXJlIikKICAgICAgICBtc2dzID0gZmJfbGlzdChhcmdzLm4pCiAgICAgICAgY2FuYWwgPSB7Im5hbWUiOiAiYXJlbmEiLCAiaWQiOiAiYXJlbmEifQogICAgICAgIGlmIG1zZ3MgaXMgTm9uZToKICAgICAgICAgICAgZGllKCJsZWN0dXJlIGltcG9zc2libGUgKCVzKSA6ICVzIiAlIChjb2RlLCAocmVzIG9yIHt9KS5nZXQoImVycm9yIiwgIj8iKSkpCiAgICBpZiBub3QgbXNnczoKICAgICAgICBwcmludCgiKGNhbmFsIHZpZGUpIikKICAgICAgICByZXR1cm4gMAogICAgcHJpbnQoImNhbmFsIMKrICVzIMK7ICglcykg4oCUICVkIG1lc3NhZ2UocyksIGR1IHBsdXMgYW5jaWVuIGF1IHBsdXMgcsOpY2VudCA6XG4iCiAgICAgICAgICAlIChjYW5hbC5nZXQoIm5hbWUiLCAiPyIpLCBjYW5hbC5nZXQoImlkIiwgIj8iKSwgbGVuKG1zZ3MpKSkKICAgIGZvciBtc2cgaW4gbXNnczoKICAgICAgICBzaG93KG1zZywgZnVsbD1hcmdzLmZ1bGwpCiAgICAgICAgcHJpbnQoKQogICAgdXNlcnMgPSBbbSBmb3IgbSBpbiBtc2dzIGlmIG0uZ2V0KCJyb2xlIikgPT0gInVzZXIiXQogICAgaWYgdXNlcnM6CiAgICAgICAgbGFzdCA9IHVzZXJzWy0xXQogICAgICAgIHByaW50KCJkZXJuaWVyIG1lc3NhZ2UgZGUgbCd1dGlsaXNhdGV1ciA6ICVzICglcykgwrcgw6l0YXQgJXMiCiAgICAgICAgICAgICAgJSAobGFzdC5nZXQoImlkIiwgIj8iKSwgaGhtbShsYXN0LmdldCgidHMiKSksIGxhc3QuZ2V0KCJzdGF0ZSIpIG9yICI/IikpCiAgICByZXR1cm4gMAoKCmRlZiBjbWRfc3RhdGUoYXJncyk6CiAgICBjb2RlLCByZXMgPSBhcGkoIi9zdGF0ZSIpCiAgICAjIFVuZSByw6lwb25zZSDCqyBvayDCuyBxdWkgcG9ydGUgdW5lIGVycmV1ciBkZSBsZWN0dXJlIGludGVybmUgbidlc3QgcGFzIHVuZQogICAgIyByw6lwb25zZSA6IG9uIHBhc3NlIGVuIGRpcmVjdCAobGEgcGFzc2VyZWxsZSBzYWl0IGxlIGRpcmUgZGVwdWlzIFA3NSkuCiAgICBpZiAoY29kZSA8IDMwMCBhbmQgaXNpbnN0YW5jZShyZXMsIGRpY3QpIGFuZCByZXMuZ2V0KCJvayIpCiAgICAgICAgICAgIGFuZCBub3QgcmVzLmdldCgiZXJyZXVyIikgYW5kIG5vdCByZXMuZ2V0KCJmaXJlc3RvcmUiKSk6CiAgICAgICAgYyA9IHJlcy5nZXQoImNhbmFsIikgb3Ige30KICAgICAgICBwcmludCgiY2FuYWwgwqsgJXMgwrsiICUgYy5nZXQoIm5hbWUiLCAiPyIpKQogICAgICAgIHByaW50KCIgIGlkZW50aWZpYW50IDogJXMiICUgYy5nZXQoImlkIiwgIj8iKSkKICAgICAgICBwcmludCgiICBtb2RlICAgICAgICA6ICVzIiAlIGMuZ2V0KCJtb2RlIiwgIj8iKSkKICAgICAgICBwcmludCgiICBtZXNzYWdlcyAgICA6ICVzIiAlIGMuZ2V0KCJtZXNzYWdlcyIsICI/IikpCiAgICAgICAgcHJpbnQoIiAgbm9uIGx1cyAgICAgOiAlcyIgJSBjLmdldCgidW5yZWFkIiwgIj8iKSkKICAgICAgICByZXR1cm4gMAogICAgX2ZiX2VuX2RpcmVjdCgibCfDqXRhdCIpCiAgICBldGF0ID0gZmJfc3RhdGUoKQogICAgaWYgbm90IGV0YXQ6CiAgICAgICAgZGllKCLDqXRhdCBpbmRpc3BvbmlibGUgKCVzKSA6ICVzIiAlIChjb2RlLCAocmVzIG9yIHt9KS5nZXQoImVycm9yIiwgIj8iKSkpCiAgICBwcmludCgiY2FuYWwgwqsgJXMgwrsiICUgZXRhdFsibm9tIl0pCiAgICBwcmludCgiICBpZGVudGlmaWFudCA6IGFyZW5hIikKICAgIHByaW50KCIgIG1vZGUgICAgICAgIDogYWdlbnQiKQogICAgcHJpbnQoIiAgbWVzc2FnZXMgICAgOiAlcyIgJSBldGF0WyJtZXNzYWdlcyJdKQogICAgcHJpbnQoIiAgbm9uIGx1cyAgICAgOiAlcyIgJSBldGF0WyJub25fbHVzIl0pCiAgICByZXR1cm4gMAoKCmRlZiBfYXR0ZW50ZV9kaXJlY3RlKGhvbGQsIGdhcCwgdDAsIHRpbWVvdXQpOgogICAgIiIiTCdhdHRlbnRlIHNhbnMgcGFzc2VyZWxsZSA6IHJlcXXDqnRlIGZpbHRyw6llIMKrIG5vbiBsdSDCuywgcHVpcyBwYXVzZS4iIiIKICAgIHJhdGUgPSAwCiAgICB3aGlsZSBUcnVlOgogICAgICAgIG1lc3NhZ2VzID0gZmJfdW5yZWFkKCkKICAgICAgICBpZiBtZXNzYWdlcyBpcyBOb25lOgogICAgICAgICAgICByYXRlICs9IDEKICAgICAgICAgICAgIyBPbiBuZSByw6lww6h0ZSBwYXMgbGEgbcOqbWUgbGlnbmUgdG91dGVzIGxlcyAxMCBzIDogdW4gcmFwcGVsIHBhcgogICAgICAgICAgICAjIG1pbnV0ZSBzdWZmaXQgKGxhIGJhc2UgcGV1dCDDqnRyZSBlbiBxdW90YSBhdHRlaW50IHBlbmRhbnQgZGVzIGhldXJlcykuCiAgICAgICAgICAgIGlmIHJhdGUgPT0gMSBvciByYXRlICUgNiA9PSAwOgogICAgICAgICAgICAgICAgcHJpbnQoIiAgIChsaWFpc29uIGRpcmVjdGUgaW5kaXNwb25pYmxlIDogYmFzZSBlbiBxdW90YSDigJQgbm91dmVsIGVzc2FpLCAiCiAgICAgICAgICAgICAgICAgICAgICAiJS4wZiBtaW4gZCdhdHRlbnRlKSIgJSAoKHRpbWUudGltZSgpIC0gdDApIC8gNjAuMCkpCiAgICAgICAgICAgICAgICBzeXMuc3Rkb3V0LmZsdXNoKCkKICAgICAgICBlbGlmIG1lc3NhZ2VzOgogICAgICAgICAgICBmYl9tYXJxdWVfbHVzKG1lc3NhZ2VzKQogICAgICAgICAgICBwcmludCgiXG7wn5OoIG1lc3NhZ2UgcmXDp3UgYXByw6hzICUuMGYgcyA6XG4iICUgKHRpbWUudGltZSgpIC0gdDApKQogICAgICAgICAgICBmb3IgbSBpbiBtZXNzYWdlczoKICAgICAgICAgICAgICAgIHNob3cobSwgZnVsbD1UcnVlLCBpbmRlbnQ9IiIpCiAgICAgICAgICAgICAgICBwcmludCgpCiAgICAgICAgICAgIHJldHVybiBtZXNzYWdlcwogICAgICAgIGlmIHRpbWVvdXQgYW5kIHRpbWUudGltZSgpIC0gdDAgPj0gdGltZW91dDoKICAgICAgICAgICAgcHJpbnQoIlxu4o+xICByaWVuIGRlIG5vdXZlYXUgYXByw6hzICUuMGYgbWluLiIgJSAodGltZW91dCAvIDYwLjApKQogICAgICAgICAgICByZXR1cm4gW10KICAgICAgICB0aW1lLnNsZWVwKG1heChnYXAsIGhvbGQgLy8gMyBpZiBob2xkIGVsc2UgZ2FwKSkKCgpkZWYgY21kX3dhaXQoYXJncyk6CiAgICAiIiJBdHRlbmQgdW4gbWVzc2FnZS4gQydlc3QgbGUgU0NSSVBUIHF1aSBnw6hyZSBsYSBkdXLDqWUgOiBsZSBzZXJ2ZXVyIG5lCiAgICBibG9xdWUgcXVlIEhPTEQgc2Vjb25kZXMsIG9uIGVuY2hhw65uZSBkb25jIGxlcyBhdHRlbnRlcyBhdmVjIHVuZSBwYXVzZS4iIiIKICAgIGhvbGQgPSBtYXgoMCwgbWluKDI1LCBpbnQoYXJncy5ob2xkKSkpCiAgICBnYXAgPSBtYXgoMSwgaW50KGFyZ3MuZ2FwKSkKICAgIHQwID0gdGltZS50aW1lKCkKICAgIGZpbmkgPSBhcmdzLnRpbWVvdXQgYW5kIGFyZ3MudGltZW91dCA+IDAKICAgIHByaW50KCLij7MgY2FuYWwgwqsgYXJlbmEgwrsgOiBhdHRlbnRlIGQndW4gbWVzc2FnZSVz4oCmIgogICAgICAgICAgJSAoIiAoJS4wZiBtaW4gYXUgbWF4aW11bSkiICUgKGFyZ3MudGltZW91dCAvIDYwLjApIGlmIGZpbmkgZWxzZSAiIikpCiAgICBwcmludCgiICAgKGxlIHNlcnZldXIgdGllbnQgJWRzLCBwdWlzIG9uIHJlcHJlbmQgOyDCqyBzdG9wIMK7IGFycsOqdGUgbGVzIGF0dGVudGVzKSIgJSBob2xkKQogICAgc3lzLnN0ZG91dC5mbHVzaCgpCiAgICB3aGlsZSBUcnVlOgogICAgICAgIGNvZGUsIHJlcyA9IGFwaSgiL3dhaXQ/dGltZW91dD0lZCIgJSBob2xkLCB0aW1lb3V0PWhvbGQgKyAzMCkKICAgICAgICBpZiBjb2RlID49IDMwMCBvciBub3QgaXNpbnN0YW5jZShyZXMsIGRpY3QpIG9yIG5vdCByZXMuZ2V0KCJvayIpOgogICAgICAgICAgICBpZiBjb2RlID09IDAgb3IgY29kZSA+PSA0Mjk6CiAgICAgICAgICAgICAgICBfZmJfZW5fZGlyZWN0KCJsJ2F0dGVudGUiKQogICAgICAgICAgICAgICAgbXNncyA9IF9hdHRlbnRlX2RpcmVjdGUoaG9sZCwgZ2FwLCB0MCwgYXJncy50aW1lb3V0IGlmIGZpbmkgZWxzZSAwKQogICAgICAgICAgICAgICAgaWYgbXNnczoKICAgICAgICAgICAgICAgICAgICB0ZXh0ZSA9ICIgIi5qb2luKChtLmdldCgidGV4dCIpIG9yICIiKS5zdHJpcCgpLmxvd2VyKCkgZm9yIG0gaW4gbXNncykuc3RyaXAoKQogICAgICAgICAgICAgICAgICAgIGlmIHRleHRlIGluICgic3RvcCIsICJhcnLDqnRlIiwgImFycmV0ZSIpOgogICAgICAgICAgICAgICAgICAgICAgICBwcmludCgi4pagIMKrIHN0b3AgwrsgOiBsJ3V0aWxpc2F0ZXVyIGRlbWFuZGUgbCdhcnLDqnQgZGVzIGF0dGVudGVzLiIpCiAgICAgICAgICAgICAgICAgICAgICAgIHJldHVybiAzCiAgICAgICAgICAgICAgICAgICAgcHJpbnQoIijDoCB0cmFpdGVyIDsgwqsgd3JpdGUgwrsgcsOpcG9uZHJhIGF1dG9tYXRpcXVlbWVudCDDoCAlcykiCiAgICAgICAgICAgICAgICAgICAgICAgICAgJSAobXNnc1stMV0uZ2V0KCJpZCIpIG9yICI/IikpCiAgICAgICAgICAgICAgICAgICAgcmV0dXJuIDAKICAgICAgICAgICAgICAgIHJldHVybiAwCiAgICAgICAgICAgIGVyciA9IChyZXMgb3Ige30pLmdldCgiZXJyb3IiLCAiPyIpCiAgICAgICAgICAgIHByaW50KCIgICAoJXMpIOKAlCBub3V2ZWwgZXNzYWkgZGFucyAlZHMiICUgKGVyciwgZ2FwKSkKICAgICAgICAgICAgc3lzLnN0ZG91dC5mbHVzaCgpCiAgICAgICAgICAgIGlmIGZpbmkgYW5kIHRpbWUudGltZSgpIC0gdDAgPj0gYXJncy50aW1lb3V0OgogICAgICAgICAgICAgICAgcHJpbnQoIlxu4o+xICByaWVuIGRlIG5vdXZlYXUgYXByw6hzICUuMGYgbWluLiIgJSAoYXJncy50aW1lb3V0IC8gNjAuMCkpCiAgICAgICAgICAgICAgICByZXR1cm4gMAogICAgICAgICAgICB0aW1lLnNsZWVwKGdhcCkKICAgICAgICAgICAgY29udGludWUKICAgICAgICBtc2dzID0gcmVzLmdldCgibWVzc2FnZXMiKSBvciBbXQogICAgICAgIGlmIG1zZ3M6CiAgICAgICAgICAgIHByaW50KCJcbvCfk6ggbWVzc2FnZSByZcOndSBhcHLDqHMgJS4wZiBzIDpcbiIgJSAodGltZS50aW1lKCkgLSB0MCkpCiAgICAgICAgICAgIGZvciBtIGluIG1zZ3M6CiAgICAgICAgICAgICAgICBzaG93KG0sIGZ1bGw9VHJ1ZSwgaW5kZW50PSIiKQogICAgICAgICAgICAgICAgcHJpbnQoKQogICAgICAgICAgICB0ZXh0ZSA9ICIgIi5qb2luKChtLmdldCgidGV4dCIpIG9yICIiKS5zdHJpcCgpLmxvd2VyKCkgZm9yIG0gaW4gbXNncykuc3RyaXAoKQogICAgICAgICAgICBpZiB0ZXh0ZSBpbiAoInN0b3AiLCAiYXJyw6p0ZSIsICJhcnJldGUiKToKICAgICAgICAgICAgICAgIHByaW50KCLilqAgwqsgc3RvcCDCuyA6IGwndXRpbGlzYXRldXIgZGVtYW5kZSBsJ2FycsOqdCBkZXMgYXR0ZW50ZXMuIikKICAgICAgICAgICAgICAgIHJldHVybiAzCiAgICAgICAgICAgIHByaW50KCIow6AgdHJhaXRlciA7IMKrIHdyaXRlIMK7IHLDqXBvbmRyYSBhdXRvbWF0aXF1ZW1lbnQgw6AgJXMpIgogICAgICAgICAgICAgICAgICAlIChtc2dzWy0xXS5nZXQoImlkIikgb3IgIj8iKSkKICAgICAgICAgICAgcmV0dXJuIDAKICAgICAgICBpZiBmaW5pIGFuZCB0aW1lLnRpbWUoKSAtIHQwID49IGFyZ3MudGltZW91dDoKICAgICAgICAgICAgcHJpbnQoIlxu4o+xICByaWVuIGRlIG5vdXZlYXUgYXByw6hzICUuMGYgbWluIOKAlCByZWxhbmNlIHVuIHdhaXQgc2kgYmVzb2luLiIKICAgICAgICAgICAgICAgICAgJSAoYXJncy50aW1lb3V0IC8gNjAuMCkpCiAgICAgICAgICAgIHJldHVybiAwCiAgICAgICAgdGltZS5zbGVlcChnYXApCgoKZGVmIGNtZF93cml0ZShhcmdzKToKICAgICIiIsOJY3JpdCB1biBtZXNzYWdlLiBQb3VyIHVuZSBDT01NQU5ERSAoLS1qc29uKSwgcmVuZCBzb24gcsOpc3VsdGF0IDogbGEKICAgIHBhc3NlcmVsbGUgYXR0ZW5kIGVsbGUtbcOqbWUgbCdleMOpY3V0aW9uIGV0IHJlbnZvaWUgbGEgc29ydGllIDsgc2lub24sIG9uCiAgICBpbnRlcnJvZ2UgbGUgY2FuYWwganVzcXUnw6AgY2UgcXVlIGxlIHLDqXN1bHRhdCBhcnJpdmUuIiIiCiAgICB0ZXh0ZSA9IGFyZ3MudGV4dGUKICAgIGlmIHRleHRlID09ICItIjoKICAgICAgICB0ZXh0ZSA9IHN5cy5zdGRpbi5yZWFkKCkKICAgIGlmIHRleHRlIGlzIE5vbmU6CiAgICAgICAgaWYgZ2V0YXR0cihhcmdzLCAiaW1hZ2UiLCBOb25lKToKICAgICAgICAgICAgdGV4dGUgPSAiIiAgICAgICAgICAgICAgICAgICAgICAjIG1lc3NhZ2UgZmFpdCB1bmlxdWVtZW50IGQnaW1hZ2VzCiAgICAgICAgZWxzZToKICAgICAgICAgICAgZGllKCJyaWVuIMOgIMOpY3JpcmUgKGRvbm5lIHVuIHRleHRlLCBvdSDCqyAtIMK7IHBvdXIgbGlyZSBsJ2VudHLDqWUgc3RhbmRhcmQpIikKICAgIGtpbmQgPSAidGV4dCIKICAgIGlmIGFyZ3MuanNvbjoKICAgICAgICB0cnk6CiAgICAgICAgICAgIHBhcnNlZCA9IGpzb24ubG9hZHModGV4dGUpCiAgICAgICAgZXhjZXB0IFZhbHVlRXJyb3IgYXMgZXJyOgogICAgICAgICAgICBkaWUoIkpTT04gaW52YWxpZGUgOiAlcyIgJSBlcnIpCiAgICAgICAgaWYgbm90IGlzaW5zdGFuY2UocGFyc2VkLCBkaWN0KToKICAgICAgICAgICAgZGllKCJsZSBKU09OIGRvaXQgw6p0cmUgdW4gb2JqZXQgKHBhciBleGVtcGxlIHtcImFjdGlvblwiOlwiYmFzaFwiLFwiY29tbWFuZFwiOlwibHNcIn0pIikKICAgICAgICBpZiAiYWN0aW9uIiBub3QgaW4gcGFyc2VkOgogICAgICAgICAgICBkaWUoImlsIG1hbnF1ZSDCqyBhY3Rpb24gwrsgKGJhc2gsIGZldGNoIG91IGZpbmFsKSIpCiAgICAgICAgdGV4dGUgPSBqc29uLmR1bXBzKHBhcnNlZCwgZW5zdXJlX2FzY2lpPUZhbHNlKQogICAgICAgIGtpbmQgPSAiY2FsbCIKCiAgICAjIFBvdXIgdW5lIGNvbW1hbmRlLCBvbiBkZW1hbmRlIMOgIGxhIHBhc3NlcmVsbGUgZCdhdHRlbmRyZSBsZSByw6lzdWx0YXQKICAgICMgKDIwIHMgYXUgbWF4aW11bSkgOiBsYSBzb3J0aWUgcmV2aWVudCBkb25jIERBTlMgbGEgcsOpcG9uc2UgZHUgd3JpdGUuCiAgICBhdHRlbnRlX3NlcnZldXIgPSBpbnQobWF4KDAsIG1pbigyMCwgYXJncy53YWl0KSkpIGlmIGtpbmQgPT0gImNhbGwiIGVsc2UgMAogICAgaW1ncyA9IGltYWdlc19yZWFkeShnZXRhdHRyKGFyZ3MsICJpbWFnZSIsIE5vbmUpIG9yIFtdKQogICAgY29ycHMgPSB7InRleHQiOiB0ZXh0ZSwgImtpbmQiOiBraW5kLCAicmVxIjogYXJncy5yZXEgb3IgIiIsICJub3RlIjogIiJ9CiAgICBpZiBpbWdzOgogICAgICAgIGNvcnBzWyJpbWdzIl0gPSBpbWdzCiAgICBpZiBhdHRlbnRlX3NlcnZldXI6CiAgICAgICAgY29ycHNbIndhaXQiXSA9IGF0dGVudGVfc2VydmV1cgogICAgY29kZSwgcmVzID0gYXBpKCIvd3JpdGUiLCBjb3JwcywgdGltZW91dD02MCArIGF0dGVudGVfc2VydmV1cikKCiAgICBtc2dfaWQgPSBOb25lCiAgICBvdXRpbCA9IE5vbmUKICAgIGlmIGNvZGUgPCAzMDAgYW5kIGlzaW5zdGFuY2UocmVzLCBkaWN0KSBhbmQgcmVzLmdldCgib2siKToKICAgICAgICBtc2dfaWQgPSByZXMuZ2V0KCJpZCIpIG9yICI/IgogICAgICAgIG91dGlsID0gcmVzLmdldCgidG9vbCIpIG9yIE5vbmUKICAgIGVsc2U6CiAgICAgICAgX2ZiX2VuX2RpcmVjdCgibCfDqWNyaXR1cmUiKQogICAgICAgIG1zZ19pZCA9IGZiX3dyaXRlKHRleHRlLCBraW5kLCBhcmdzLnJlcSBvciAiIiwgIiIsIGltZ3MpCiAgICAgICAgaWYgbXNnX2lkIGlzIE5vbmU6CiAgICAgICAgICAgIGRpZSgiw6ljcml0dXJlIGltcG9zc2libGUgKCVzKSA6ICVzIiAlIChjb2RlLCAocmVzIG9yIHt9KS5nZXQoImVycm9yIiwgIj8iKSkpCiAgICBwcmludCgi4pyOIMOpY3JpdCBkYW5zIGxlIGNhbmFsIDogJXMgKHLDqXBvbnNlIMOgICVzKSVzIgogICAgICAgICAgJSAobXNnX2lkLCBhcmdzLnJlcSBvciAi4oCUIiwKICAgICAgICAgICAgICgiIMK3ICVkIGltYWdlKHMpIiAlIGxlbihpbWdzKSkgaWYgaW1ncyBlbHNlICIiKSkKCiAgICBpZiBub3Qgb3V0aWwgYW5kIGtpbmQgPT0gImNhbGwiIGFuZCBhcmdzLndhaXQgPiAwOgogICAgICAgICMgTGEgcGFzc2VyZWxsZSBuJ2EgcmllbiByZW5kdSA6IG9uIGF0dGVuZCBpY2ksIGVuIGludGVycm9nZWFudCBsZSBjYW5hbC4KICAgICAgICB0MCA9IHRpbWUudGltZSgpCiAgICAgICAgc3lzLnN0ZG91dC5mbHVzaCgpCiAgICAgICAgd2hpbGUgdGltZS50aW1lKCkgLSB0MCA8IGFyZ3Mud2FpdDoKICAgICAgICAgICAgb3V0aWwgPSBmYl9yZXN1bHRhdChtc2dfaWQpCiAgICAgICAgICAgIGlmIG5vdCBvdXRpbDoKICAgICAgICAgICAgICAgIGNvZGUyLCByMiA9IGFwaSgiL3JlYWQ/bj0yMCIpCiAgICAgICAgICAgICAgICBmb3IgbSBpbiAoKHIyIG9yIHt9KS5nZXQoIm1lc3NhZ2VzIikgb3IgW10pIGlmIGNvZGUyIDwgMzAwIGVsc2UgW106CiAgICAgICAgICAgICAgICAgICAgaWYgbS5nZXQoInJvbGUiKSA9PSAidG9vbCIgYW5kIChtLmdldCgicmVxIikgb3IgIiIpID09IG1zZ19pZDoKICAgICAgICAgICAgICAgICAgICAgICAgb3V0aWwgPSBtCiAgICAgICAgICAgICAgICAgICAgICAgIGJyZWFrCiAgICAgICAgICAgIGlmIG91dGlsOgogICAgICAgICAgICAgICAgYnJlYWsKICAgICAgICAgICAgdGltZS5zbGVlcCgyKQoKICAgIGlmIG91dGlsOgogICAgICAgIHByaW50KCJcbuKamSByw6lzdWx0YXQgZGUgbGEgY29tbWFuZGUgOiIpCiAgICAgICAgc2hvdyhvdXRpbCwgZnVsbD1UcnVlLCBpbmRlbnQ9IiIpCiAgICAgICAgcmV0dXJuIDAKCiAgICBwcmludCgiVm90cmUgbWVzc2FnZSBhIMOpdMOpIGVudm95w6kgw6AgbCd1dGlsaXNhdGV1ciBzdXIgbCdhcHAgISIpCiAgICBpZiBraW5kID09ICJjYWxsIiBhbmQgYXJncy53YWl0ID4gMDoKICAgICAgICBwcmludCgiKGF1Y3VuIHLDqXN1bHRhdCBwb3VyIGwnaW5zdGFudCA6IGwnYXBwbGljYXRpb24gZXN0LWVsbGUgb3V2ZXJ0ZSwgIgogICAgICAgICAgICAgICJldCBsYSBkaXNjdXNzaW9uIGVuIG1vZGUgYWdlbnQgPykiKQogICAgcmV0dXJuIDAKCgpkZWYgY21kX2RvYyhhcmdzKToKICAgIHByaW50KEJBU0UucnNwbGl0KCIvIiwgMSlbMF0pCiAgICByZXR1cm4gMAoKCmRlZiBjbWRfc2VsZnRlc3QoYXJncyk6CiAgICBwcmludCgiY2FuYWwgOiAlcyIgJSBCQVNFKQogICAgY29kZSwgcmVzID0gYXBpKCIvc3RhdGUiKQogICAgaWYgY29kZSA+PSAzMDAgb3Igbm90IChyZXMgb3Ige30pLmdldCgib2siKToKICAgICAgICBfZmJfZW5fZGlyZWN0KCJsJ8OpdGF0IikKICAgICAgICBldGF0ID0gZmJfc3RhdGUoKQogICAgICAgIGlmIG5vdCBldGF0OgogICAgICAgICAgICBkaWUoIsOpdGF0IGluZGlzcG9uaWJsZSAoJXMpIDogJXMiICUgKGNvZGUsIChyZXMgb3Ige30pLmdldCgiZXJyb3IiLCAiPyIpKSkKICAgICAgICBwcmludCgiaWRlbnRpZmlhbnQgOiBhcmVuYSDCtyBtb2RlIDogYWdlbnQgwrcgbWVzc2FnZXMgOiAlcyIgJSBldGF0WyJtZXNzYWdlcyJdKQogICAgZWxzZToKICAgICAgICBjID0gcmVzLmdldCgiY2FuYWwiKSBvciB7fQogICAgICAgIHByaW50KCJpZGVudGlmaWFudCA6ICVzIMK3IG1vZGUgOiAlcyDCtyBtZXNzYWdlcyA6ICVzIgogICAgICAgICAgICAgICUgKGMuZ2V0KCJpZCIpLCBjLmdldCgibW9kZSIpLCBjLmdldCgibWVzc2FnZXMiKSkpCiAgICB0ZXh0ZSA9ICJ0ZXN0IGNhbmFsIHotY2FuYWwucHkgIiArIHN0cihpbnQodGltZS50aW1lKCkpKQogICAgY29kZSwgdyA9IGFwaSgiL3dyaXRlIiwgeyJ0ZXh0IjogdGV4dGUsICJraW5kIjogInRleHQifSkKICAgIG9rX3cgPSBjb2RlIDwgMzAwIGFuZCAodyBvciB7fSkuZ2V0KCJvayIpCiAgICBpZiBub3Qgb2tfdzoKICAgICAgICBva193ID0gZmJfd3JpdGUodGV4dGUsICJ0ZXh0IikgaXMgbm90IE5vbmUKICAgIHByaW50KCLDqWNyaXR1cmUgOiAlcyVzIiAlICgiT0siIGlmIG9rX3cgZWxzZSAiw4lDSEVDIiwgIiIgaWYgb2tfdyBlbHNlICIg4oCUICVzIiAlICh3IG9yIHt9KS5nZXQoImVycm9yIikpKQogICAgY29kZSwgciA9IGFwaSgiL3JlYWQ/bj01IikKICAgIG1lc3NhZ2VzID0gKHIgb3Ige30pLmdldCgibWVzc2FnZXMiKSBpZiAoY29kZSA8IDMwMCBhbmQgaXNpbnN0YW5jZShyLCBkaWN0KSkgZWxzZSBOb25lCiAgICBpZiBtZXNzYWdlcyBpcyBOb25lOgogICAgICAgIG1lc3NhZ2VzID0gZmJfbGlzdCg1KSBvciBbXQogICAgb2tfciA9IGFueSgobS5nZXQoInRleHQiKSBvciAiIikgPT0gdGV4dGUgZm9yIG0gaW4gbWVzc2FnZXMpCiAgICBwcmludCgicmVsZWN0dXJlIDogJXMiICUgKCJPSyIgaWYgb2tfciBlbHNlICLDiUNIRUMiKSkKICAgIHJldHVybiAwIGlmIChva193IGFuZCBva19yKSBlbHNlIDEKCgpkZWYgbWFpbigpOgogICAgcCA9IGFyZ3BhcnNlLkFyZ3VtZW50UGFyc2VyKAogICAgICAgIHByb2c9InotY2FuYWwucHkiLAogICAgICAgIGRlc2NyaXB0aW9uPSJMZSBjYW5hbCBlbnRyZSBBcmVuYSBldCBsJ2FwcGxpY2F0aW9uIChzeXN0w6htZSBaLUNhbmFsKS4iLAogICAgICAgIGZvcm1hdHRlcl9jbGFzcz1hcmdwYXJzZS5SYXdEZXNjcmlwdGlvbkhlbHBGb3JtYXR0ZXIsCiAgICAgICAgZXBpbG9nPV9fZG9jX18uc3BsaXQoIlNvdXMtY29tbWFuZGVzIilbMV0gaWYgIlNvdXMtY29tbWFuZGVzIiBpbiBfX2RvY19fIGVsc2UgIiIpCiAgICBzdWIgPSBwLmFkZF9zdWJwYXJzZXJzKGRlc3Q9ImNtZCIpCgogICAgdyA9IHN1Yi5hZGRfcGFyc2VyKCJ3YWl0IiwgaGVscD0iYXR0ZW5kcmUgdW4gbWVzc2FnZSBkZSBsJ3V0aWxpc2F0ZXVyIikKICAgIHcuYWRkX2FyZ3VtZW50KCItLWhvbGQiLCB0eXBlPWludCwgZGVmYXVsdD1IT0xELCBoZWxwPSJkdXLDqWUgZCd1bmUgYXR0ZW50ZSBjw7R0w6kgc2VydmV1ciAoZMOpZmF1dCAyNSBzKSIpCiAgICB3LmFkZF9hcmd1bWVudCgiLS1nYXAiLCB0eXBlPWludCwgZGVmYXVsdD1HQVAsIGhlbHA9InBhdXNlIGVudHJlIGRldXggYXR0ZW50ZXMgKGTDqWZhdXQgNSBzKSIpCiAgICB3LmFkZF9hcmd1bWVudCgiLS10aW1lb3V0IiwgdHlwZT1mbG9hdCwgZGVmYXVsdD0wLCBoZWxwPSJkdXLDqWUgdG90YWxlIG1heCAoMCA9IHNhbnMgZmluKSIpCiAgICB3LnNldF9kZWZhdWx0cyhmdW5jPWNtZF93YWl0KQoKICAgIHIgPSBzdWIuYWRkX3BhcnNlcigicmVhZCIsIGhlbHA9ImFmZmljaGVyIGwnaGlzdG9yaXF1ZSBkdSBjYW5hbCIpCiAgICByLmFkZF9hcmd1bWVudCgiLS1uIiwgdHlwZT1pbnQsIGRlZmF1bHQ9MzAsIGhlbHA9Im5vbWJyZSBkZSBtZXNzYWdlcyAoZMOpZmF1dCAzMCkiKQogICAgci5hZGRfYXJndW1lbnQoIi0tYWxsIiwgYWN0aW9uPSJzdG9yZV90cnVlIiwgZGVzdD0iZnVsbCIsIGhlbHA9InRleHRlIGludMOpZ3JhbCAoc2FucyB0cm9uY2F0dXJlKSIpCiAgICByLnNldF9kZWZhdWx0cyhmdW5jPWNtZF9yZWFkKQoKICAgIGEgPSBzdWIuYWRkX3BhcnNlcigid3JpdGUiLCBoZWxwPSJlbnZveWVyIHVuIG1lc3NhZ2UgdmVycyBsJ2FwcGxpY2F0aW9uIikKICAgIGEuYWRkX2FyZ3VtZW50KCJ0ZXh0ZSIsIG5hcmdzPSI/IiwgZGVmYXVsdD1Ob25lLCBoZWxwPSJsZSB0ZXh0ZSwgb3UgwqsgLSDCuyBwb3VyIHN0ZGluIikKICAgIGEuYWRkX2FyZ3VtZW50KCItLWpzb24iLCBhY3Rpb249InN0b3JlX3RydWUiLCBoZWxwPSJsZSB0ZXh0ZSBlc3QgdW4gSlNPTiBkJ2FjdGlvbiAoYmFzaC9mZXRjaC9maW5hbCkiKQogICAgYS5hZGRfYXJndW1lbnQoIi0tcmVxIiwgZGVmYXVsdD0iIiwgaGVscD0iaWQgZHUgbWVzc2FnZSBhdXF1ZWwgb24gcsOpcG9uZCAoc2lub24gYXVjdW4pIikKICAgIGEuYWRkX2FyZ3VtZW50KCItLXdhaXQiLCB0eXBlPWZsb2F0LCBkZWZhdWx0PTIwLCBoZWxwPSJhdHRlbnRlIGQndW5lIHLDqXBvbnNlIGRlIGxhIFZNIChzKSIpCiAgICBhLmFkZF9hcmd1bWVudCgiLS1pbWFnZSIsIGFjdGlvbj0iYXBwZW5kIiwgZGVmYXVsdD1bXSwgbWV0YXZhcj0iRklDSElFUiIsCiAgICAgICAgICAgICAgICAgICBoZWxwPSJqb2ludCB1bmUgaW1hZ2UgYXUgbWVzc2FnZSAocsOpcMOpdGFibGUsIHRyb2lzIGF1IHBsdXMpIikKICAgIGEuc2V0X2RlZmF1bHRzKGZ1bmM9Y21kX3dyaXRlKQoKICAgIHMgPSBzdWIuYWRkX3BhcnNlcigic3RhdGUiLCBoZWxwPSLDqXRhdCBkdSBjYW5hbCIpCiAgICBzLnNldF9kZWZhdWx0cyhmdW5jPWNtZF9zdGF0ZSkKCiAgICBkID0gc3ViLmFkZF9wYXJzZXIoImRvYyIsIGhlbHA9ImwnYWRyZXNzZSBkZSBsYSBkb2N1bWVudGF0aW9uIikKICAgIGQuc2V0X2RlZmF1bHRzKGZ1bmM9Y21kX2RvYykKCiAgICB0ID0gc3ViLmFkZF9wYXJzZXIoInNlbGZ0ZXN0IiwgaGVscD0iYWxsZXItcmV0b3VyIGQnw6ljcml0dXJlL2xlY3R1cmUiKQogICAgdC5zZXRfZGVmYXVsdHMoZnVuYz1jbWRfc2VsZnRlc3QpCgogICAgYXJncyA9IHAucGFyc2VfYXJncygpCiAgICBpZiBub3QgZ2V0YXR0cihhcmdzLCAiZnVuYyIsIE5vbmUpOgogICAgICAgIHAucHJpbnRfaGVscCgpCiAgICAgICAgcmV0dXJuIDAKICAgIHJldHVybiBhcmdzLmZ1bmMoYXJncykKCgppZiBfX25hbWVfXyA9PSAiX19tYWluX18iOgogICAgdHJ5OgogICAgICAgIHN5cy5leGl0KG1haW4oKSkKICAgIGV4Y2VwdCBLZXlib2FyZEludGVycnVwdDoKICAgICAgICBwcmludCgiXG4oaW50ZXJyb21wdSkiKQogICAgICAgIHN5cy5leGl0KDEzMCkK";
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
      const isDiff = !!DIFF_MAP[model];
      /* Première image jointe (si l'utilisateur en a joint une) : le relais
         ModelScope doit voir exactement la même demande que Groq. */
      const msImage = (normImageList(body)[0] || {}).b64 || "";
      const gcfg = GOOGLE_MAP[model];
      if (!isGroq && !isPoll && !isIfm && !isDiff && !(gcfg && !gcfg.kind)) return json({ error: "streaming non supporté pour ce modèle" }, 400);

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
        } else if (isDiff) {
          const rq = {
            model: DIFF_MAP[model],
            messages: [{ role: "user", content: prompt }],
            stream: true,
            diffusing: true,
          };
          /* Mercury 3 rend jusqu'à 131 072 tokens : c'est le défaut. */
          const mt = Number(params.max_completion_tokens || params.max_tokens || 0) || 131072;
          rq.max_completion_tokens = Math.min(131072, Math.max(1, mt));
          if (params.temperature !== undefined && params.temperature !== null && params.temperature !== "") rq.temperature = Number(params.temperature);
          if (params.reasoning_effort) rq.reasoning_effort = String(params.reasoning_effort);
          up = await fetch(MERCURY_URL + "/chat/completions", {
            method: "POST",
            headers: { "Content-Type": "application/json", Authorization: "Bearer " + env.INCEPTION_API_KEY },
            body: JSON.stringify(rq),
          });
        } else if (isIfm) {
          const rq = { model: IFM_MAP[model], messages: [{ role: "user", content: prompt }], stream: true };
          /* K2 Horizon a 524 288 tokens de contexte ; le plafond de sortie
             accepté par la passerelle est 524 000. C'est le défaut. */
          rq.max_tokens = Math.min(524000, Math.max(16, Number(params.max_tokens || params.max_completion_tokens || 0) || 524000));
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
        if (isDiff) {
          let m3 = t.slice(0, 300);
          try { const j3 = JSON.parse(t); m3 = (j3.error && (j3.error.message || j3.error)) || m3; } catch {}
          if (Array.isArray(m3)) m3 = JSON.stringify(m3).slice(0, 300);
          return json({ error: "Mercury HTTP " + up.status + " — " + m3 }, up.status === 429 ? 429 : 502);
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
          let lastDiff = "";
          let diffCut = false;
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
                if (isDiff) {
                  const ch0 = j.choices && j.choices[0];
                  const d = ch0 && ch0.delta && ch0.delta.content;
                  if (d !== undefined && d !== null && d !== "") {
                    lastDiff = String(d);
                    const pr = (j.diffusion_meta && typeof j.diffusion_meta.diffusion_progress === "number") ? j.diffusion_meta.diffusion_progress : null;
                    send({ d: lastDiff, p: pr });
                  }
                  if (ch0 && ch0.finish_reason === "length") diffCut = true;
                } else if (isGroq || isPoll || isIfm) {
                  const ch0 = j.choices && j.choices[0];
                  /* Le raisonnement arrive souvent AVANT le texte (K2 Horizon
                     notamment) : on le relaie à part, l'app en fait une zone
                     « Thinking » repliable. */
                  const rz = ch0 && ch0.delta && ch0.delta.reasoning;
                  if (rz) send({ r: rz });
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
          /* Dernier état = la réponse finale : l'app le rend en markdown. */
          if (isDiff && lastDiff) {
            send({ t: lastDiff + (diffCut ? "\n\n*(réponse interrompue : limite de tokens atteinte — relancez pour continuer)*" : ""), fin: 1 });
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
        const imgs = canalImgs(b.imgs);
        if (!text && !imgs.length && kind !== "tool") return json({ error: "texte vide" }, 400);
        const now = Date.now();
        const w = await arenaWrite(env, {
          role: fsStr(role), text: fsStr(text), ts: fsInt(now), req: fsStr(b.req || ""),
          /* Un résultat de commande est terminal : rien à traiter côté app. */
          state: fsStr(kind === "tool" ? "done" : "new"), kind: fsStr(kind), src: fsStr("worker"),
          note: fsStr(b.note || ""), imgs: fsNewVal(imgs),
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
    if (url.pathname === "/api/ms/daily" || url.pathname === "/api/ms/models" || url.pathname === "/api/ms/state"
        || url.pathname === "/api/ms/probe") {
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
      /* Sonde de diagnostic (temporaire) : interroge ModelScope avec un
         identifiant EXACT et rend son verdict — aucun re-mappage implicite. */
      if (url.pathname === "/api/ms/probe" && request.method === "POST") {
        let b = {};
        try { b = await request.json(); } catch { /* corps illisible */ }
        const id = String(b.model || "").trim();
        if (!id) return json({ ok: false, error: "model manquant" }, 400);
        const kind = String(b.kind || "text");
        const key = msKey(env);
        const H = { "Content-Type": "application/json", Authorization: "Bearer " + key, "User-Agent": BROWSER_UA };
        if (kind === "image") {
          const r = await fetch(MS_HOSTS[0] + "/images/generations", {
            method: "POST",
            headers: Object.assign({ "X-ModelScope-Async-Mode": "true" }, H),
            body: JSON.stringify({ model: id, prompt: String(b.prompt || "un chat roux"), size: "1328x1328" }),
          });
          const t = await r.text();
          return json({ ok: r.ok, model: id, kind: kind, status: r.status, body: String(t).slice(0, 400) });
        }
        const r = await fetch(MS_HOSTS[0] + "/chat/completions", {
          method: "POST", headers: H,
          body: JSON.stringify({ model: id, messages: [{ role: "user", content: "Réponds : ok" }], max_tokens: 8 }),
        });
        const t = await r.text();
        let out = "";
        try { out = String(JSON.parse(t).choices[0].message.content || ""); } catch { out = ""; }
        return json({ ok: r.ok, model: id, kind: "text", status: r.status,
          output: out.slice(0, 80), body: r.ok ? "" : String(t).slice(0, 300) });
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

/* ===================== Modèles : disponibilité ==========================
   Un modèle ne « tombe » jamais seul : il passe par une plateforme, et
   parfois par une seconde quand la première refuse. Cette section dit, pour
   chaque modèle, la chaîne réellement prévue par le Worker, puis la teste
   pour de vrai (un appel minuscule), en gardant le message d'erreur de
   chaque plateforme essayée avant celle qui a répondu. */

function modelChain(env, id, kind) {
  const hasMS = !!msKey(env);
  if (kind === "image") {
    if (MS_IMG[id]) return ["ModelScope"];
    if (IMG_MODELS[id]) return ["Cloudflare Workers AI"];
    if (HF_IMG_MODELS[id]) return ["Hugging Face (Space)"];
    if (POLL_IMG_MAP[id]) return ["Pollinations"];
    return ["Roboflow (OpenRouter)"];
  }
  if (kind === "video") {
    if (POLL_VID_MAP[id]) return ["Pollinations (vidéo)"];
    return ["Hugging Face (Space)"];
  }
  if (kind === "live") return ["Google (Gemini API)"];
  const out = [];
  if (GROQ_MAP[id]) {
    out.push("Groq");
    if (hasMS && MS_TEXT[id]) out.push("ModelScope");
    return out;
  }
  if (POLL_TEXT_MAP[id]) {
    out.push("Pollinations");
    if (POLL_FB[id]) out.push("Roboflow");
    return out;
  }
  if (IFM_MAP[id]) return ["IFM"];
  if (DIFF_MAP[id]) return ["Inception (Mercury)"];
  if (hasMS && MS_FIRST[id] && MS_TEXT[id]) return ["ModelScope", "Roboflow"];
  out.push("Roboflow");
  if (hasMS && MS_TEXT[id]) out.push("ModelScope");
  const spec = MODELS[id];
  if (spec && spec.or) out.push("OpenRouter");
  return out;
}

/* Quelle plateforme a réellement répondu, d'après le champ « via ». */
function modelPlatformOf(chain, via) {
  const v = String(via || "");
  if (v === "modelscope") return "ModelScope";
  if (v === "openrouter") return "OpenRouter";
  if (v === "roboflow") return "Roboflow";
  if (v === "cloudflare") return "Cloudflare Workers AI";
  if (v === "hf") return "Hugging Face (Space)";
  if (v === "pollinations") return "Pollinations";
  if (v === "ifm") return "IFM";
  if (v === "inception") return "Inception (Mercury)";
  return chain.length ? chain[0] : "";
}

/* L'erreur du Worker contient déjà les messages des plateformes précédentes
   (« … · … ») : on la découpe pour l'afficher ligne par ligne. */
function modelErrors(error) {
  return String(error || "").split(" · ").map(function (x) { return x.trim(); }).filter(Boolean);
}

    /* ---- Crédits : ce qui reste dans chaque compte ------------------- */
    /* Les noms des secrets et l'état de chacun — jamais les valeurs. */
    /* ---- Roboflow : un jeton, un essai -----------------------------
       Envoie UNE requête au service de workflows avec le jeton donné et
       rend la réponse brute : statut, durée, en-têtes, début du corps.
       La clé n'est jamais renvoyée (quatre derniers caractères). */
    /* La clé du fournisseur, quand elle existe dans les secrets du Worker :
       le bloc l'utilise alors directement, sans passer par la clé prêtée par
       Roboflow (donc sans consommer de crédits d'espace de travail). */
    const providerKey = (env, c) => {
      if (!c || c.orFallback) return "";
      if (c.family === "claude") return String(env.ANTHROPIC_API_KEY || "").trim();
      if (c.family === "openai") return String(env.OPENAI_API_KEY || "").trim();
      return "";
    };
    /* Un pixel blanc de 1×1 : Roboflow exige un champ image, même pour un
       modèle de texte. Sert quand l'appelant n'en fournit pas. */
    const RF_PX = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==";
    const rfTryToken = async (tok, model, imageIn, promptTxt) => {
      const image = String(imageIn || "") || RF_PX;
      const spec = MODELS[model] || null;
      if (!spec) return { error: "modèle inconnu : " + model };
      const step = {
        type: spec.block, name: "m", prompt: String(promptTxt || "ping"),
        [spec.vf || "model_version"]: spec.vv || model,
        api_key: "rf_key:account", task_type: "unconstrained",
      };
      if (!spec.noimg && image) step.images = "$inputs.image";
      const payload = {
        api_key: tok,
        inputs: image ? { image: { type: "base64", value: image } } : {},
        specification: {
          version: "1.0",
          inputs: image ? [{ type: "WorkflowImage", name: "image" }] : [],
          steps: [step],
          outputs: [{ type: "JsonField", name: "output", selector: "$steps.m.output" }],
        },
      };
      const t0 = Date.now();
      let r;
      try {
        r = await fetch(ROBOFLOW_URL, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });
      } catch (e) {
        return { error: "Roboflow injoignable : " + String(e && e.message ? e.message : e), ms: Date.now() - t0 };
      }
      const texte = await r.text();
      const tous = {};
      r.headers.forEach(function (v, k) { tous[k] = v; });
      return {
        ok: r.status < 300, statut: r.status, ms: Date.now() - t0, texte: texte, entetes: tous,
        jeton: "…" + String(tok).slice(-4), bloc: spec.block,
      };
    };

    /* ---- Roboflow : l'enquête -------------------------------------
       Une seule requête vers le service de workflows, avec le même corps
       qu'une conversation, et la réponse BRUTE (statut + début du corps).
       La clé n'est jamais renvoyée : seulement ses quatre derniers
       caractères. Sert à comprendre un refus (pare-feu, quota, jeton). */
    if (url.pathname === "/api/rf/probe" && request.method === "POST") {
      if (env.ACCESS_CODE && request.headers.get("X-Access-Code") !== env.ACCESS_CODE) {
        return json({ error: "code d'accès invalide" }, 401);
      }
      let b = {};
      try { b = await request.json(); } catch { /* corps illisible */ }
      /* Les jetons ajoutés depuis l'application vivent dans l'objet durable :
         sans cette relecture, la sonde ne voyait que ceux du Worker et
         répondait pour le premier jeton en croyant tester un autre. */
      await rfExtraTokens(env, true);
      const toks = roboflowTokens(env);
      if (!toks.length) return json({ error: "aucun jeton Roboflow dans le Worker" }, 400);
      /* `fin` choisit un jeton précis ; sinon le premier. */
      const fin = String(b.fin || "");
      const tok = fin ? (toks.filter(t => String(t).slice(-4) === fin)[0] || toks[0]) : toks[0];
      const out = await rfTryToken(tok, String(b.model || "qwen-3-8-27b"), String(b.image || ""), String(b.prompt || "ping"));
      if (out.error) return json({ ok: false, error: out.error }, 502);
      return json({
        ok: out.ok, modele: String(b.model || "qwen-3-8-27b"), bloc: out.bloc, statut: out.statut,
        ms: out.ms, entetes: out.entetes, jeton: out.jeton,
        debut: String(out.texte).replace(/\s+/g, " ").slice(0, 900),
        html: /<html|Attention Required|Just a moment/i.test(out.texte),
        credits: roboflowCreditInfo(out.texte),
      });
    }

    /* ---- Roboflow : le compte --------------------------------------
       La fiche de l'espace de travail, filtrée : on ne garde que ce qui
       parle de plan, de cycle et de crédits. La clé reste au Worker. */
    if (url.pathname === "/api/rf/info" && request.method === "GET") {
      if (env.ACCESS_CODE && request.headers.get("X-Access-Code") !== env.ACCESS_CODE) {
        return json({ error: "code d'accès invalide" }, 401);
      }
      const toks = roboflowTokens(env);
      if (!toks.length) return json({ error: "aucun jeton Roboflow dans le Worker" }, 400);
      let r, txt = "";
      const chemin = String(new URL(request.url).searchParams.get("p") || "").replace(/^\/+/, "");
      if (chemin && (!/^[A-Za-z0-9._\/-]+$/.test(chemin) || /^workflows/i.test(chemin))) {
        return json({ error: "chemin refusé" }, 400);
      }
      try {
        r = await fetch("https://api.roboflow.com/" + chemin + (chemin ? "?" : "") + "api_key=" + encodeURIComponent(toks[0]),
          { headers: { "Accept": "application/json", "User-Agent": BROWSER_UA } });
        txt = await r.text();
      } catch (e) {
        return json({ error: "Roboflow injoignable : " + String(e && e.message ? e.message : e) }, 502);
      }
      let j = null;
      try { j = JSON.parse(txt); } catch { /* pas du JSON */ }
      if (!j) return json({ ok: false, statut: r.status, debut: String(txt).slice(0, 400) });
      const cle = /plan|cycle|billing|credit|period|reset|renew|expire|trial|subscript|seat|usage/i;
      const interest = {};
      for (const k of Object.keys(j)) if (cle.test(k)) interest[k] = j[k];
      return json({ ok: r.ok, statut: r.status, cles: Object.keys(j), interessants: interest });
    }

    /* ---- Roboflow : qui est de côté, et pourquoi ---------------------
       Le détail des mises de côté, sans aucune clé : les quatre derniers
       caractères du jeton et la raison. Sert à comprendre un refus. */
    if (url.pathname === "/api/rf/bans" && request.method === "GET") {
      if (env.ACCESS_CODE && request.headers.get("X-Access-Code") !== env.ACCESS_CODE) {
        return json({ error: "code d'accès invalide" }, 401);
      }
      const now = Date.now();
      let doBans = null;
      try {
        if (env.MISSION) {
          const stub = env.MISSION.get(env.MISSION.idFromName("secrets"));
          const r = await stub.fetch("https://do/rf-bans", { method: "GET" });
          const j = await r.json().catch(() => null);
          if (j && j.bans) doBans = j.bans;
        }
      } catch (e) { /* pas de stockage partagé */ }
      const vus = {};
      for (const src of [RF_BAN, doBans || {}]) {
        for (const k of Object.keys(src)) {
          const e = src[k] || {};
          if (!(Number(e.until || 0) > now)) continue;
          vus[k] = { jusqu: new Date(Number(e.until)).toISOString(), min: Math.round((Number(e.until) - now) / 60000), why: String(e.why || "") };
        }
      }
      const compte = {};
      for (const k of Object.keys(vus)) compte[vus[k].why || "?"] = (compte[vus[k].why || "?"] || 0) + 1;
      return json({ ok: true, total: roboflowTokens(env).length, deCote: Object.keys(vus).length, compte: compte, jetons: vus });
    }

    /* ---- Roboflow : « réessayer maintenant » ------------------------
       Efface les mises de côté (les deux caches) : au prochain message,
       chaque jeton est réessayé pour de vrai. Sert après une recharge de
       crédits, un changement de plan ou l'ajout d'une clé. */
    if (url.pathname === "/api/rf/forget" && request.method === "POST") {
      if (env.ACCESS_CODE && request.headers.get("X-Access-Code") !== env.ACCESS_CODE) {
        return json({ error: "code d'accès invalide" }, 401);
      }
      RF_BAN = {};
      RF_GOOD = "";
      RF_LIST_CACHE = { at: 0, n: 0, data: null };
      let ok = false;
      try {
        if (env.MISSION) {
          const stub = env.MISSION.get(env.MISSION.idFromName("secrets"));
          await stub.fetch("https://do/rf-bans", { method: "POST", body: JSON.stringify({ clear: true }) });
          ok = true;
        }
      } catch (e) { /* le cache local est déjà vide */ }
      return json({ ok: true, vide: ok, jetons: roboflowTokens(env).length });
    }

    /* ---- Roboflow : les jetons, un par un --------------------------
       Chaque jeton appartient à un espace de travail, et chaque espace a son
       propre quota de crédits. On affiche donc, par jeton : son espace, sa
       validité, et — quand Roboflow le dit — le compteur du cycle. */
    if (url.pathname === "/api/rf/tokens" && request.method === "GET") {
      if (env.ACCESS_CODE && request.headers.get("X-Access-Code") !== env.ACCESS_CODE) {
        return json({ error: "code d'accès invalide" }, 401);
      }
      await rfExtraTokens(env);
      const toks = roboflowTokens(env);
      const force = new URL(request.url).searchParams.get("fresh") === "1";
      if (!force && RF_LIST_CACHE.at && (Date.now() - RF_LIST_CACHE.at) < 60000 && RF_LIST_CACHE.n === toks.length) {
        return json(RF_LIST_CACHE.data);
      }
      const extras = RF_EXTRA.tokens.slice();
      const rows = await Promise.all(toks.map(async (t, i) => {
        const out = { n: i + 1, fin: String(t).slice(-4), source: extras.indexOf(t) >= 0 ? "ajouté" : "Worker" };
        try {
          const r = await fetch("https://api.roboflow.com/?api_key=" + encodeURIComponent(t),
            { headers: { "Accept": "application/json", "User-Agent": BROWSER_UA } });
          const txt = await r.text();
          let j = null;
          try { j = JSON.parse(txt); } catch { /* pas du JSON */ }
          out.statut = r.status;
          out.ok = !!(r.ok && j && j.workspace);
          out.workspace = (j && j.workspace) || "";
          out.message = out.ok ? "" : ((j && j.error && (j.error.message || j.error.type))
            || String(txt).slice(0, 140) || ("HTTP " + r.status));
        } catch (e) {
          out.ok = false; out.statut = 0;
          out.message = "injoignable : " + String(e && e.message ? e.message : e);
        }
        return out;
      }));
      const espaces = {};
      for (const r of rows) if (r.workspace) espaces[r.workspace] = (espaces[r.workspace] || 0) + 1;
      const data = { ok: true, jetons: rows, espaces: espaces, total: rows.length };
      RF_LIST_CACHE = { at: Date.now(), n: toks.length, data: data };
      return json(data);
    }
    /* Ajout / retrait d'un jeton, depuis l'application. Le jeton n'est jamais
       renvoyé : seul son espace et ses quatre derniers caractères sortent. */
    if (url.pathname === "/api/rf/tokens" && request.method === "POST") {
      if (env.ACCESS_CODE && request.headers.get("X-Access-Code") !== env.ACCESS_CODE) {
        return json({ error: "code d'accès invalide" }, 401);
      }
      let b = {};
      try { b = await request.json(); } catch { /* corps illisible */ }
      const remove = String(b.remove || "").trim();
      if (remove) {
        const j = await rfExtraWrite(env, { remove: remove });
        RF_LIST_CACHE.at = 0;
        return json({ ok: true, retire: remove.slice(-4), n: (j && j.n) || 0 });
      }
      const add = String(b.add || "").trim();
      if (!add) return json({ error: "jeton manquant" }, 400);
      if (add.length < 8 || /\s/.test(add)) return json({ error: "ce n'est pas un jeton Roboflow (trop court ou avec des espaces)" }, 400);
      let ws = "";
      try {
        const r = await fetch("https://api.roboflow.com/?api_key=" + encodeURIComponent(add),
          { headers: { "Accept": "application/json", "User-Agent": BROWSER_UA } });
        const txt = await r.text();
        let j = null;
        try { j = JSON.parse(txt); } catch { /* pas du JSON */ }
        if (!r.ok || !j || !j.workspace) {
          const why = (j && j.error && (j.error.message || j.error.type)) || String(txt).slice(0, 140) || ("HTTP " + r.status);
          return json({ ok: false, error: "jeton refusé par Roboflow : " + why }, 400);
        }
        ws = String(j.workspace);
      } catch (e) {
        return json({ ok: false, error: "Roboflow injoignable : " + String(e && e.message ? e.message : e) }, 502);
      }
      const j2 = await rfExtraWrite(env, { add: add });
      RF_LIST_CACHE.at = 0;
      return json({ ok: true, ajoute: add.slice(-4), workspace: ws, n: (j2 && j2.n) || 0 });
    }
    /* L'essai d'un jeton : un seul appel minuscule, avec ce jeton et lui seul.
       C'est la seule façon de savoir si un espace peut encore dépenser. */
    if (url.pathname === "/api/rf/tokens/test" && request.method === "POST") {
      if (env.ACCESS_CODE && request.headers.get("X-Access-Code") !== env.ACCESS_CODE) {
        return json({ error: "code d'accès invalide" }, 401);
      }
      let b = {};
      try { b = await request.json(); } catch { /* corps illisible */ }
      await rfExtraTokens(env);
      const toks = roboflowTokens(env);
      const fin = String(b.fin || "").slice(-4);
      const tok = fin ? toks.filter(t => String(t).slice(-4) === fin)[0] : toks[0];
      if (!tok) return json({ ok: false, error: "jeton introuvable" }, 404);
      const out = await rfTryToken(tok, String(b.model || "qwen-3-8-27b"), String(b.image || ""), "ping");
      if (out.error) return json({ ok: false, fin: fin, error: out.error }, 502);
      const credits = roboflowCreditInfo(out.texte);
      return json({
        ok: out.ok, fin: String(tok).slice(-4), statut: out.statut, ms: out.ms,
        credits: credits,
        epuise: !!credits,
        message: out.ok ? "" : (credits ? roboflowCreditError(out.texte)
          : (roboflowPlanError(out.texte) || String(out.texte).replace(/\s+/g, " ").slice(0, 300))),
        plan: !out.ok && !!roboflowPlanError(out.texte),
      });
    }

    /* ---- Modèles : la chaîne des plateformes, sans aucun appel ------
       Un Worker ne peut pas s'appeler lui-même (Cloudflare refuse une
       requête vers sa propre zone : « error code: 1042 »). L'essai réel se
       fait donc depuis l'application, en s'adressant à /api/chat comme une
       vraie conversation ; ici, on ne fait que dire, pour chaque modèle,
       l'ordre des plateformes que le Worker essaiera. */
    if (url.pathname === "/api/models/chains" && (request.method === "POST" || request.method === "GET")) {
      if (env.ACCESS_CODE && request.headers.get("X-Access-Code") !== env.ACCESS_CODE) {
        return json({ error: "code d'accès invalide" }, 401);
      }
      let b = {};
      if (request.method === "POST") { try { b = await request.json(); } catch { /* corps illisible */ } }
      let ids = Array.isArray(b.models) ? b.models : [];
      if (!ids.length) {
        /* Sans liste : tous les modèles connus du Worker, avec leur genre. */
        ids = Object.keys(MODELS).map(function (k) { return { id: k, kind: "text" }; });
        for (const k of Object.keys(MS_IMG)) ids.push({ id: k, kind: "image" });
        for (const k of Object.keys(IMG_MODELS)) ids.push({ id: k, kind: "image" });
        for (const k of Object.keys(HF_IMG_MODELS)) ids.push({ id: k, kind: "image" });
        for (const k of Object.keys(POLL_IMG_MAP)) ids.push({ id: k, kind: "image" });
        for (const k of Object.keys(POLL_VID_MAP)) ids.push({ id: k, kind: "video" });
      }
      const out = {};
      for (const it of ids.slice(0, 200)) {
        const id = typeof it === "string" ? it : String((it && it.id) || "");
        const kind = typeof it === "string" ? "text" : String((it && it.kind) || "text");
        if (!id) continue;
        out[id] = { kind: kind, chain: modelChain(env, id, kind) };
      }
      return json({ ok: true, modeles: out });
    }

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
      const VM_ORIGIN = "https://vmai.hugdu77777.workers.dev";
      const VM_BASE = VM_ORIGIN + "/api/linux";
      const sub = url.pathname.slice("/api/agent/".length);
      const H = { "Authorization": "Bearer " + (env.VMAI_PASS || "") };
      const corsH = { "Access-Control-Allow-Origin": "*" };
      /* Le même chemin, mais sur le tunnel publié directement (sans passer
         par la passerelle). Le premier tunnel qui répond gagne. */
      const vmDirect = async (sub, init) => {
        const urls = await vmTunnelList();
        for (const base of urls) {
          try {
            const r = await fetch(new Request(base + sub, init));
            if (r.status >= 500) continue;                       /* tunnel mort */
            /* 404 en texte brut = le bureau a répondu mais pas cette route :
               on tente le tunnel suivant plutôt que de rendre « 404 » sec. */
            const ct = String(r.headers.get("content-type") || "");
            if (r.status === 404 && ct.indexOf("json") < 0) continue;
            return r;
          } catch (e) { /* tunnel suivant */ }
        }
        return null;
      };
      /* Liaison de service (VMAI) : appel worker-à-worker direct, insensible
         à l'état du sous-domaine workers.dev. Repli : fetch HTTP classique.
         Et si la passerelle répond « daemon_pending_cycle » (son état des
         tunnels a du retard), la même requête repart par le tunnel direct. */
      if (sub === "debug") {
        const urls = await vmTunnelList();
        const essais = [];
        for (const base of urls) {
          const t0 = Date.now();
          try {
            const r = await fetch(base + "/api/linux/exec", {
              method: "POST", headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ command: "hostname", user: "kasm-user", timeout: 15 }),
            });
            const txt = await r.text();
            essais.push({ base: base, statut: r.status, ms: Date.now() - t0, corps: txt.slice(0, 160) });
          } catch (e) {
            essais.push({ base: base, erreur: String(e && e.message ? e.message : e), ms: Date.now() - t0 });
          }
        }
        let gateway = null;
        try {
          const rq = new Request(VM_BASE + "/exec", { method: "POST",
            headers: { ...H, "Content-Type": "application/json" },
            body: JSON.stringify({ command: "hostname", user: "kasm-user", timeout: 15 }) });
          const r = env.VMAI && typeof env.VMAI.fetch === "function" ? await env.VMAI.fetch(rq) : await fetch(rq);
          gateway = { statut: r.status, corps: (await r.text()).slice(0, 160) };
        } catch (e) { gateway = { erreur: String(e && e.message ? e.message : e) }; }
        return json({ ok: true, tunnels: urls, direct: essais, passerelle: gateway });
      }
      const vmFetch = async (u, init) => {
        /* Le tunnel expose les mêmes chemins que la passerelle (« /api/linux/exec ») :
           on garde donc le chemin complet, seul l'hôte change. */
        const sub = u.startsWith(VM_ORIGIN) ? u.slice(VM_ORIGIN.length) : u;
        let up = null, err = null;
        try {
          const rq = new Request(u, init);
          up = env.VMAI && typeof env.VMAI.fetch === "function" ? await env.VMAI.fetch(rq) : await fetch(rq);
        } catch (e) { err = e; }
        let pending = false;
        if (up) {
          try {
            if (up.status >= 500) pending = true;
            else {
              /* Toutes les routes /api/linux/* répondent du JSON. Une page
                 d'erreur Cloudflare (« error code: 1016 »), un texte brut ou
                 l'état « daemon_pending_cycle » veulent donc dire que la
                 passerelle n'a pas joint le bureau : on retente en direct. */
              const txt = await up.clone().text();
              if (txt.indexOf("daemon_pending_cycle") >= 0) pending = true;
              else if (txt && txt.charAt(0) !== "{" && txt.charAt(0) !== "[") pending = true;
              else if (/^\s*(<!doctype|<html|error code:)/i.test(txt)) pending = true;
            }
          } catch (e) { /* réponse illisible : on la garde telle quelle */ }
        }
        if (!up || pending) {
          const direct = await vmDirect(sub, init);
          if (direct) return direct;
          if (!up) throw err || new Error("VM injoignable");
        }
        return up;
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
        /* Diagnostic : l'état des tunnels vu par la passerelle, et la liste
           que nous connaissons. Sert à savoir sur QUELLE VM on est branché. */
        if (sub === "tunnels") {
          let gw = null;
          try {
            const r = await vmFetch(VM_ORIGIN + "/api/tunnels", { headers: H });
            gw = await r.json().catch(() => null);
          } catch (e) { gw = { error: String(e && e.message ? e.message : e) }; }
          const mine = await vmTunnelList();
          return json({ ok: true, passerelle: gw, utilises: mine });
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

    /* Les jetons ajoutés depuis l'application entrent dans la rotation. */
    try { await rfExtraTokens(env); } catch (e) { /* on garde ceux du Worker */ }

    const params = body.params && typeof body.params === "object" ? body.params : {};
    /* Diagnostic (aucune valeur de jeton révélée) : `params._tokens` renvoie le
       nombre de jetons de la rotation — sert aux vérifications de déploiement. */
    if (params._tokens === true) {
      return json({ ok: true, tokens: roboflowTokens(env).length });
    }
    const rawImage = String(body.image ?? "");

    /* --- Canal Pollinations (texte, compatible OpenAI, vision incluse) --- */
    if (DIFF_MAP[model]) {
      if (!env.INCEPTION_API_KEY) return json({ error: "Mercury : clé absente du worker (INCEPTION_API_KEY)" }, 500);
      const rq = { model: DIFF_MAP[model], messages: [{ role: "user", content: prompt }] };
      const mt = Number(params.max_completion_tokens || params.max_tokens || 0) || 131072;
      rq.max_completion_tokens = Math.min(131072, Math.max(1, mt));
      if (params.temperature !== undefined && params.temperature !== null && params.temperature !== "") rq.temperature = Number(params.temperature);
      if (params.reasoning_effort) rq.reasoning_effort = String(params.reasoning_effort);
      let mup;
      try {
        mup = await fetch(MERCURY_URL + "/chat/completions", {
          method: "POST",
          headers: { "Content-Type": "application/json", Authorization: "Bearer " + env.INCEPTION_API_KEY },
          body: JSON.stringify(rq),
        });
      } catch (err) { return json({ error: `Mercury injoignable : ${err.message}` }, 502); }
      const mtxt = await mup.text();
      let mj; try { mj = JSON.parse(mtxt); } catch { mj = null; }
      if (!mup.ok) {
        let msg = mtxt.slice(0, 300);
        if (mj && mj.error) msg = mj.error.message || JSON.stringify(mj.error).slice(0, 300);
        if (Array.isArray(msg)) msg = JSON.stringify(msg).slice(0, 300);
        return json({ error: `Mercury HTTP ${mup.status} — ${msg}` }, mup.status === 429 ? 429 : 502);
      }
      const out = String(mj?.choices?.[0]?.message?.content ?? "").trim();
      if (!out) return json({ error: "Mercury : réponse vide" }, 502);
      return json({ output: out, model, applied: ["mercury"] });
    }

    if (IFM_MAP[model]) {
      if (!env.IFM_API_KEY) return json({ error: "IFM : clé absente du worker (IFM_API_KEY)" }, 500);
      const rq = { model: IFM_MAP[model], messages: [{ role: "user", content: prompt }] };
      rq.max_tokens = Math.min(524000, Math.max(16, Number(params.max_tokens || params.max_completion_tokens || 0) || 524000));
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
      const rz2 = String(ij?.choices?.[0]?.message?.reasoning ?? "");
      return json(rz2 ? { output: out, reasoning: rz2, model, applied: ["ifm"] } : { output: out, model, applied: ["ifm"] });
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
      : (spec.chain || (spec.or && !spec.direct ? [spec, orCand] : [spec])));
    /* Rotation des jetons : chaque voie est rejouée avec le jeton suivant. */
    const calls = acrossTokens(env, cands);

    /* Une tentative : liste (jeton, bloc) essayés dans l'ordre, avec le filet
       anti-pare-feu et le repli OpenRouter sur erreur. Renvoie soit le
       résultat, soit le message d'erreur déjà rédigé. */
    const attempt = async (list) => {
      let safeTried = false;
      let tries = 0;
      if (!list.length) return { ok: false, error: rfWhyPhrase(env) || "Roboflow : aucun jeton disponible", tries };
      /* Ce que chaque jeton a répondu, pour pouvoir le dire en une phrase :
         « sur 18 jetons, aucun ne peut servir — 4 espaces à sec, 14 au plan
         gratuit ». */
      const tally = { credit: 0, plan: 0, scopes: 0, other: 0, last: "" };
      const rfSummary = () => rfWhyPhrase(env);
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
        if (!c.nokey) step.api_key = providerKey(env, c) || "rf_key:account";
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
          /* Ce jeton vient de répondre : il passera devant les autres. */
          RF_GOOD = String(c._token || "").slice(-4);
          delete RF_BAN[RF_GOOD];
          return { ok: true, used: c, data, tries };
        }

        let detail = t.slice(0, 400);
        try { const j = JSON.parse(t); detail = j.inner_error_message || j.message || detail; } catch { /* texte brut */ }
        /* Limite de plan (clés OpenRouter gérées) : la vraie raison aussi. */
        const plan = roboflowPlanError(t);
        if (plan) {
          rfBan(String(c._token || "").slice(-4), 6 * 3600 * 1000, "plan gratuit");
          try { if (ctx && ctx.waitUntil) ctx.waitUntil(rfPushBan(env, String(c._token || "").slice(-4))); } catch (e) {}
          tally.plan++; tally.last = plan;
          continue;
        }
        /* Crédits épuisés : un autre jeton a peut-être, lui, encore des
           crédits — on passe au suivant au lieu de conclure. */
        const credit = roboflowCreditError(t);
        if (credit) {
          rfBan(String(c._token || "").slice(-4), 30 * 60 * 1000, "crédits épuisés");
          try { if (ctx && ctx.waitUntil) ctx.waitUntil(rfPushBan(env, String(c._token || "").slice(-4))); } catch (e) {}
          tally.credit++; tally.last = credit;
          continue;
        }
        /* Jeton sans les droits requis (bloc Anthropic / Google direct) :
           même chose, le suivant a peut-être un plan qui l'ouvre. */
        if (r.status === 403 && /Unauthorized access to roboflow API|required scopes/i.test(t)) {
          rfBan(String(c._token || "").slice(-4), 6 * 3600 * 1000, "scopes");
          try { if (ctx && ctx.waitUntil) ctx.waitUntil(rfPushBan(env, String(c._token || "").slice(-4))); } catch (e) {}
          tally.scopes++; tally.last = detail;
          continue;
        }
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
        tally.other++; tally.last = detail;
        if (i >= list.length || list.slice(i).every((x) => skip.has(x.block))) {
          const resume = rfSummary();
          if (resume) return { ok: false, error: resume + " Dernier refus : " + detail, tries };
          const nt = roboflowTokens(env).length;
          const suffix = c.orFallback
            ? " (les deux voies Roboflow et OpenRouter ont échoué" + (nt > 1 ? ", sur les " + nt + " jetons" : "") + ")"
            : (nt > 1 ? " (essayé sur les " + nt + " jetons)" : "");
          return { ok: false, error: `Roboflow HTTP ${r.status} — ${detail}${suffix}`, tries };
        }
        await new Promise((res) => setTimeout(res, 120));
      }
      return { ok: false, error: rfSummary() || "Roboflow : aucune voie disponible", tries };
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
    /* Claude ne passe que par son bloc dédié : quand plus rien ne peut
       servir, on dit quoi faire au lieu de laisser un refus sec. */
    if (!res.ok && spec.family === "claude") {
      const aCle = !!String(env.ANTHROPIC_API_KEY || "").trim();
      res.error += aCle
        ? " La clé Anthropic du Worker a été refusée elle aussi : vérifie qu'elle est valide et qu'elle a encore du crédit."
        : " Aucun espace n'a de crédits, et Roboflow ne prête sa clé Anthropic qu'aux plans payants :"
          + " ajoute une clé Anthropic dans les secrets du Worker (ANTHROPIC_API_KEY) — le bloc l'utilise directement,"
          + " sans consommer de crédits d'espace — ou recharge les crédits d'un espace de travail.";
    }
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
    if (spec.or && !spec.direct && !forceOr && !(res.used && res.used.orFallback) && looksLikeImage(output)) {
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
