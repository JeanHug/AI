#!/usr/bin/env -S npx tsx
/* z-canal — client du canal « __NAME__ » (id : __ID__) de l'application AI Agent.
 *
 *   npx tsx z-canal.ts wait | read [n] | write "texte" | write --json '{...}' | state
 *
 * Un message ne revient jamais deux fois : le canal le marque « lu » des qu'un
 * « wait » le rend. C'est ce script qui boucle (le serveur tient 25 s, pas plus).
 */
const BASE: string = "__BASE__";
const ID: string = "__ID__";
const NAME: string = "__NAME__";
const MODE: string = "__MODE__";
const HOLD = 25, GAP = 5000;

type CanalMessage = { id?: string; role?: string; text?: string; ts?: number; state?: string };

async function call(path: string, data?: unknown): Promise<any> {
  const opt: RequestInit = { headers: { "User-Agent": "z-canal/1.0" } };
  if (data !== undefined) {
    opt.method = "POST";
    (opt.headers as Record<string, string>)["Content-Type"] = "application/json";
    opt.body = JSON.stringify(data);
  }
  try {
    const r = await fetch(BASE + "/" + ID + path, opt);
    return await r.json().catch(() => ({ ok: false, error: "reponse illisible" }));
  } catch (e) { return { ok: false, error: String(e) }; }
}
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));
function show(m: CanalMessage): void {
  const role = ({ user: "toi", agent: "agent", tool: "vm" } as Record<string, string>)[m.role || ""] || m.role;
  console.log("  [" + new Date(m.ts || 0).toLocaleTimeString("fr-FR") + "] " + role);
  String(m.text || "").split("\n").forEach((l) => console.log("    " + l));
}
async function main(): Promise<number> {
  const [cmd, ...rest] = process.argv.slice(2);
  if (cmd === "wait") {
    console.log("canal « " + NAME + " » (" + ID + ") : attente d'un message...");
    for (;;) {
      const rep = await call("/wait?timeout=" + HOLD);
      if (!rep.ok) { await sleep(GAP); continue; }
      const msgs: CanalMessage[] = rep.messages || [];
      if (!msgs.length) { await sleep(GAP); continue; }
      console.log("");
      msgs.forEach(show);
      return 0;
    }
  }
  if (cmd === "read") {
    const rep = await call("/read?n=" + (Number(rest[0]) || 30));
    (rep.messages || []).forEach(show);
    return rep.ok ? 0 : 1;
  }
  if (cmd === "write") {
    const data = rest[0] === "--json"
      ? { text: JSON.stringify(JSON.parse(rest.slice(1).join(" "))), kind: "call" }
      : { text: rest.join(" "), kind: "text" };
    const rep = await call("/write", data);
    console.log(rep.ok ? "ecrit dans le canal : " + rep.id : "echec : " + rep.error);
    return rep.ok ? 0 : 1;
  }
  if (cmd === "state") { console.log(JSON.stringify((await call("/state")).canal, null, 2)); return 0; }
  console.log("usage : npx tsx z-canal.ts wait|read|write|state");
  return 1;
}
main().then((c) => process.exit(c));
