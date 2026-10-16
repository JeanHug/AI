#!/usr/bin/env node
/* z-canal — client du canal « __NAME__ » (id : __ID__) de l'application AI Agent.
 *
 * Ce canal relie CET agent a l'application : l'utilisateur ecrit dans l'app,
 * l'agent recoit ses messages, peut repondre — et si le canal est en mode
 * « agent », il peut aussi executer des commandes sur la VM puis renvoyer
 * leur resultat dans l'application.
 *
 *   node z-canal.js wait              attend un message (rend la main des qu'il arrive)
 *   node z-canal.js read [n]          l'historique recent (ne marque rien)
 *   node z-canal.js write "texte"     ecrit un message dans l'application
 *   node z-canal.js write --json '{"action":"bash","command":"ls -la ~"}'
 *   node z-canal.js state             l'etat du canal
 *
 * Un message ne revient JAMAIS deux fois : des qu'un « wait » le rend, le canal
 * le marque « lu ». C'est ce script qui gere l'attente (le serveur ne bloque pas
 * plus de 25 s) : wait -> rien ? pause de 5 s -> wait.
 */
const BASE = "__BASE__";
const ID = "__ID__";
const NAME = "__NAME__";
const MODE = "__MODE__";
const HOLD = 25, GAP = 5000;

async function call(path, data) {
  const url = BASE + "/" + ID + path;
  const opt = { headers: { "User-Agent": "z-canal/1.0" } };
  if (data !== undefined) {
    opt.method = "POST";
    opt.headers["Content-Type"] = "application/json";
    opt.body = JSON.stringify(data);
  }
  try {
    const r = await fetch(url, opt);
    const j = await r.json().catch(() => ({}));
    return j && typeof j === "object" ? j : { ok: false, error: "reponse illisible" };
  } catch (e) { return { ok: false, error: String((e && e.message) || e) }; }
}
const hhmm = (ts) => new Date(ts || 0).toLocaleTimeString("fr-FR");
function show(m) {
  const role = { user: "toi", agent: "agent", tool: "vm" }[m.role] || m.role;
  console.log("  [" + hhmm(m.ts) + "] " + role);
  String(m.text || "").split("\n").forEach((l) => console.log("    " + l));
}
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function cmdWait() {
  console.log("canal « " + NAME + " » (" + ID + ") : attente d'un message...");
  for (;;) {
    const rep = await call("/wait?timeout=" + HOLD);
    if (!rep.ok) { console.log("  (" + (rep.error || "echec") + ") — nouvel essai"); await sleep(GAP); continue; }
    const msgs = rep.messages || [];
    if (!msgs.length) { await sleep(GAP); continue; }
    console.log("");
    msgs.forEach(show);
    console.log("");
    return 0;
  }
}
async function cmdRead(n) {
  const rep = await call("/read?n=" + (n || 30));
  if (!rep.ok) { console.log("lecture impossible : " + (rep.error || "?")); return 1; }
  console.log("canal « " + NAME + " » : " + (rep.messages || []).length + " message(s)");
  (rep.messages || []).forEach(show);
  return 0;
}
async function cmdWrite(args) {
  if (!args.length) { console.log("rien a ecrire (texte, ou --json <objet>)"); return 1; }
  let data;
  if (args[0] === "--json") {
    let o; try { o = JSON.parse(args.slice(1).join(" ")); } catch (e) { console.log("JSON invalide : " + e.message); return 1; }
    data = { text: JSON.stringify(o), kind: "call" };
  } else data = { text: args.join(" "), kind: "text" };
  const rep = await call("/write", data);
  if (!rep.ok) { console.log("ecriture impossible : " + (rep.error || "?")); return 1; }
  console.log("ecrit dans le canal : " + (rep.id || "?"));
  return 0;
}
async function cmdState() {
  const rep = await call("/state");
  if (!rep.ok) { console.log("etat indisponible : " + (rep.error || "?")); return 1; }
  const c = rep.canal || {};
  console.log("canal « " + (c.name || NAME) + " »");
  console.log("  id       : " + (c.id || ID));
  console.log("  mode     : " + (c.mode || MODE));
  console.log("  messages : " + (c.messages === undefined ? "?" : c.messages));
  console.log("  non lus  : " + (c.unread === undefined ? "?" : c.unread));
  return 0;
}
(async () => {
  const [cmd, ...rest] = process.argv.slice(2);
  const code = cmd === "wait" ? await cmdWait()
    : cmd === "read" ? await cmdRead(Number(rest[0]) || 30)
    : cmd === "write" ? await cmdWrite(rest)
    : cmd === "state" ? await cmdState()
    : (console.log("usage : node z-canal.js wait|read|write|state"), 1);
  process.exit(code);
})();
