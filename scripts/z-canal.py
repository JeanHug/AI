#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""z-canal.py — le canal entre Arena et l'application (Firebase Firestore).

Le canal vit dans Firestore, sous la forme d'une liste de messages :

    chan/main/box/{id} = { role, text, ts, req, state, kind, src }

      role   : « user »  = message écrit par l'application (toi)
               « arena » = message écrit par Arena (moi)
               « tool »  = résultat d'une commande, renvoyé par l'app
      text   : le texte (ou le JSON d'une commande, tel quel)
      ts     : date en millisecondes (l'horloge de celui qui écrit)
      req    : id du message auquel celui-ci répond (chaîne vide sinon)
      state  : « new » puis « read » (le lecteur a pris le message)
      kind   : « text » (message normal) ou « call » (JSON d'action)

Sous-commandes
--------------
  wait  [--timeout 1700] [--interval 4]   attend un message de l'utilisateur
  read  [--n 30] [--all] [--since ID]     affiche l'historique du canal
  write <texte|-> [--json] [--req ID] [--wait 15]
                                          envoie un message vers l'app
  state                                   état du canal (en un coup d'œil)
  selftest                                aller-retour d'écriture/lecture

Exemples
--------
  python3 z-canal.py read                 # l'historique complet, du plus ancien au plus récent
  python3 z-canal.py wait                 # bloque jusqu'à ton prochain message
  python3 z-canal.py write "Je regarde ça."
  python3 z-canal.py write --json '{"action":"bash","command":"ls -l /home/user"}'
  python3 z-canal.py write - < reponse.md # le texte arrive par l'entrée standard

Rien n'est secret ici : le script lit la configuration Firebase dans index.html
(la clé d'API web est publique par nature) et s'authentifie en anonyme, comme
le ferait un visiteur du site. Aucune clé privée n'est stockée.
"""

import argparse
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone

HOME = os.path.dirname(os.path.abspath(__file__))
APP = os.path.join(HOME, "index.html")          # source de la config Firebase
CACHE_DIR = "/tmp/.zcanal"                      # jetons et état local (hors workspace)
TOKEN_FILE = os.path.join(CACHE_DIR, "token.json")
STATE_FILE = os.path.join(CACHE_DIR, "state.json")
UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/120.0.0.0 Safari/537.36")

# --------------------------------------------------------------- configuration
FALLBACK = {
    "apiKey": "AIzaSyC58J1paO0ULhTW0RuDfUePVeuOUmG5fZY",
    "projectId": "ai-agent-1234a",
}


def firebase_config():
    """Lit apiKey / projectId dans index.html (aucune copie de jeton)."""
    cfg = dict(FALLBACK)
    try:
        with open(APP, encoding="utf-8") as fh:
            src = fh.read()
        for key in ("apiKey", "projectId"):
            m = re.search(key + r'\s*:\s*"([^"]+)"', src)
            if m:
                cfg[key] = m.group(1)
    except OSError:
        pass
    return cfg


CFG = firebase_config()
API_KEY = CFG["apiKey"]
PROJECT = CFG["projectId"]
FS = "https://firestore.googleapis.com/v1/projects/%s/databases/(default)/documents" % PROJECT
BOX = FS + "/chan/main/box"
MAIN = FS + "/chan/main"
AUTH = "https://identitytoolkit.googleapis.com/v1/accounts:signUp?key=" + API_KEY


def _mkdir():
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
    except OSError:
        pass


def _http(method, url, body=None, token=None, timeout=45):
    data = None
    headers = {"User-Agent": UA}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    if token:
        headers["Authorization"] = "Bearer " + token
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
            return resp.status, (json.loads(raw) if raw.strip() else {})
    except urllib.error.HTTPError as err:
        raw = err.read().decode("utf-8", "replace")
        try:
            parsed = json.loads(raw)
        except ValueError:
            parsed = {"error": {"message": raw[:300]}}
        return err.code, parsed
    except Exception as err:                                  # réseau
        return 0, {"error": {"message": str(err)}}


def token(force=False):
    """Jeton d'authentification anonyme, mis en cache ~50 minutes."""
    _mkdir()
    now = time.time()
    if not force and os.path.exists(TOKEN_FILE):
        try:
            with open(TOKEN_FILE, encoding="utf-8") as fh:
                saved = json.load(fh)
            if saved.get("expires", 0) > now + 60 and saved.get("token"):
                return saved["token"]
        except (OSError, ValueError):
            pass
    status, res = _http("POST", AUTH, {"returnSecureToken": True})
    tok = res.get("idToken")
    if not tok:
        die("connexion anonyme refusée par Firebase (%s) : %s"
            % (status, (res.get("error") or {}).get("message", "")))
    with open(TOKEN_FILE, "w", encoding="utf-8") as fh:
        json.dump({"token": tok, "expires": now + 3000}, fh)
    os.chmod(TOKEN_FILE, 0o600)
    return tok


def die(msg, code=1):
    print("✗ " + msg, file=sys.stderr)
    sys.exit(code)


# ------------------------------------------------------------------- Firestore
def _val(field):
    """Convertit une valeur Firestore en valeur Python."""
    if not isinstance(field, dict):
        return None
    for key in ("stringValue", "integerValue", "doubleValue", "booleanValue"):
        if key in field:
            v = field[key]
            if key == "integerValue":
                return int(v)
            if key == "doubleValue":
                return float(v)
            if key == "booleanValue":
                return bool(v)
            return v
    if "timestampValue" in field:
        return field["timestampValue"]
    return None


def _fields(fields):
    return {k: _val(v) for k, v in (fields or {}).items()}


def _doc(doc):
    out = _fields(doc.get("fields"))
    out["id"] = doc["name"].rsplit("/", 1)[-1]
    out["created"] = doc.get("createTime", "")
    out["updated"] = doc.get("updateTime", "")
    return out


def box_list(n=30, since=None):
    """Les n derniers messages, du PLUS ANCIEN au plus récent."""
    url = BOX + "?key=%s&orderBy=ts%%20desc&pageSize=%d" % (API_KEY, max(1, n))
    if since:
        url = BOX + "?key=%s&pageSize=200" % API_KEY
    status, res = _http("GET", url, token=token())
    if status == 401:
        status, res = _http("GET", url, token=token(force=True))
    if status >= 300:
        die("lecture du canal impossible (%s) : %s"
            % (status, (res.get("error") or {}).get("message", "")))
    docs = [_doc(d) for d in res.get("documents", [])]
    docs.sort(key=lambda d: (d.get("ts") or 0, d.get("created") or ""))
    if since:
        ids = [d["id"] for d in docs]
        if since in ids:
            docs = docs[ids.index(since) + 1:]
    return docs


def box_add(role, text, req="", kind="text", src="script", extra=None):
    fields = {
        "role": {"stringValue": role},
        "text": {"stringValue": text},
        "ts": {"integerValue": str(int(time.time() * 1000))},
        "req": {"stringValue": req or ""},
        "state": {"stringValue": "new"},
        "kind": {"stringValue": kind},
        "src": {"stringValue": src},
    }
    for k, v in (extra or {}).items():
        fields[k] = {"stringValue": str(v)}
    body = {"fields": fields}
    url = BOX + "?key=" + API_KEY
    status, res = _http("POST", url, body, token=token())
    if status == 401:
        status, res = _http("POST", url, body, token=token(force=True))
    if status >= 300:
        die("écriture dans le canal impossible (%s) : %s"
            % (status, (res.get("error") or {}).get("message", "")))
    return _doc(res)


def box_patch(doc_id, fields, mask=None):
    parts = mask or list(fields.keys())
    url = "%s/%s?key=%s&%s" % (BOX, doc_id, API_KEY,
                               "&".join("updateMask.fieldPaths=" + urllib.parse.quote(p) for p in parts))
    flat = {k: (v if isinstance(v, dict) else {"stringValue": str(v)}) for k, v in fields.items()}
    status, res = _http("PATCH", url, {"fields": flat}, token=token())
    if status == 401:
        status, res = _http("PATCH", url, {"fields": flat}, token=token(force=True))
    if status >= 300:
        die("mise à jour impossible (%s) : %s" % (status, (res.get("error") or {}).get("message", "")))
    return _doc(res)


def box_delete(doc_id):
    url = "%s/%s?key=%s" % (BOX, doc_id, API_KEY)
    return _http("DELETE", url, None, token=token())


def main_patch(fields):
    flat = {k: (v if isinstance(v, dict) else {"stringValue": v}) for k, v in fields.items()}
    url = MAIN + "?key=%s&%s" % (API_KEY, "&".join("updateMask.fieldPaths=" + p for p in flat))
    return _http("PATCH", url, {"fields": flat}, token=token())


# ------------------------------------------------------------------- affichage
def hhmm(ts_ms):
    if not ts_ms:
        return "?"
    d = datetime.fromtimestamp(ts_ms / 1000.0).astimezone()
    return d.strftime("%H:%M:%S")


def write_state(**kw):
    """Retient l'id du dernier message reçu (pour « write » qui répond)."""
    _mkdir()
    try:
        state = {}
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE, encoding="utf-8") as fh:
                state = json.load(fh)
        state.update(kw)
        with open(STATE_FILE, "w", encoding="utf-8") as fh:
            json.dump(state, fh)
    except (OSError, ValueError):
        pass


def read_state():
    try:
        with open(STATE_FILE, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


ROLE_TAG = {"user": "👤 toi", "arena": "◆ arena", "tool": "⚙ vm"}


def show(msg, full=False, indent="  "):
    tag = ROLE_TAG.get(msg.get("role"), msg.get("role"))
    state = msg.get("state") or "?"
    kind = msg.get("kind") or "text"
    body = msg.get("text") or ""
    if not full and len(body) > 500:
        body = body[:500] + " …(%d caractères au total)" % len(msg.get("text") or "")
    lines = ["%s[%s] %s · %s%s" % (indent, hhmm(msg.get("ts")), tag, state,
                                   (" · " + kind) if kind != "text" else "")]
    for line in body.splitlines() or [""]:
        lines.append(indent + "  " + line)
    print("\n".join(lines))


# ------------------------------------------------------------------ commandes
def marquer_vus(docs):
    """Retient les messages que l'agent vient de voir (état local du script)."""
    st = read_state()
    vus = list(st.get("seen_ids") or [])
    for m in docs:
        if m.get("role") == "user" and m["id"] not in vus:
            vus.append(m["id"])
    write_state(seen_ids=vus[-400:])


def cmd_read(args):
    docs = box_list(args.n)
    if not docs:
        print("(canal vide)")
        return
    print("canal arena — %d message(s), du plus ancien au plus récent :\n" % len(docs))
    for msg in docs:
        show(msg, full=args.full)
        print()
    users = [m for m in docs if m.get("role") == "user"]
    if users:
        m = users[-1]
        print("dernier message de l'utilisateur : %s (%s) · état %s"
              % (m["id"], hhmm(m.get("ts")), m.get("state")))
    # Les messages affichés sont désormais VUS : « wait » ne les redira pas.
    marquer_vus(users)


def cmd_state(args):
    docs = box_list(60)
    users = [m for m in docs if m.get("role") == "user"]
    neufs = [m for m in users if (m.get("state") or "new") == "new"]
    arenas = [m for m in docs if m.get("role") == "arena"]
    tools = [m for m in docs if m.get("role") == "tool"]
    print("canal arena — état")
    print("  messages            : %d (toi %d · arena %d · vm %d)"
          % (len(docs), len(users), len(arenas), len(tools)))
    print("  non lus par arena   : %d" % len(neufs))
    if docs:
        print("  dernier message     : [%s] %s" % (hhmm(docs[-1].get("ts")),
                                                   ROLE_TAG.get(docs[-1].get("role"), "?")))
    st = read_state()
    if st.get("last_user"):
        print("  dernier vu          : %s à %s" % (st.get("last_user"), st.get("last_user_at", "?")))
    if neufs:
        print("\n  à traiter :")
        for m in neufs:
            first = (m.get("text") or "").strip().splitlines()
            print("   • %s  %s" % (m["id"], (first[0] if first else "")[:90]))


def cmd_wait(args):
    """Attend un message de l'utilisateur.

    Le point de départ est l'ÉTAT DU SCRIPT (les messages déjà vus), pas le
    contenu du canal : un message arrivé pendant une absence est donc rendu
    tout de suite, dès la première vérification.
    """
    deja = set(read_state().get("seen_ids") or [])
    t0 = time.time()
    interval = max(2, args.interval)
    print("⏳ canal arena : attente d'un message (jusqu'à %.0f min, toutes les %d s)…"
          % (args.timeout / 60.0, interval))
    sys.stdout.flush()
    main_patch({"arenaWaiting": "true", "arenaWaitingSince": str(int(time.time() * 1000))})
    try:
        while True:
            docs = box_list(12)
            neufs = [m for m in docs if m.get("role") == "user" and m["id"] not in deja]
            if neufs:
                msg = neufs[-1]
                for m in neufs:
                    deja.add(m["id"])
                box_patch(msg["id"], {"state": "read", "readAt": str(int(time.time() * 1000))})
                write_state(last_user=msg["id"], last_user_at=hhmm(msg.get("ts")),
                            seen_ids=list(deja))
                print("\n📨 message reçu après %.0f s :\n" % (time.time() - t0))
                for m in neufs[:-1]:
                    print("  (il y en a un autre, plus récent : %s)" % m["id"])
                show(msg, full=True, indent="")
                if (msg.get("text") or "").strip().lower() in ("stop", "arrête", "arrete"):
                    print("\n■ « stop » : l'utilisateur demande l'arrêt des attentes.")
                    return 3
                print("\n(à traiter ; « write » répondra automatiquement à %s)" % msg["id"])
                return 0
            if time.time() - t0 >= args.timeout:
                print("\n⏱  rien de nouveau après %.0f min — relance un wait si besoin."
                      % (args.timeout / 60.0))
                return 0
            time.sleep(interval)
    finally:
        main_patch({"arenaWaiting": "false"})


def cmd_write(args):
    texte = args.texte
    if texte == "-":
        texte = sys.stdin.read()
    if texte is None:
        die("rien à écrire (donne un texte, ou « - » pour lire l'entrée standard)")
    kind = "text"
    if args.json:
        try:
            parsed = json.loads(texte)
        except ValueError as err:
            die("JSON invalide : %s" % err)
        if not isinstance(parsed, dict):
            die("le JSON doit être un objet (par exemple {\"action\":\"bash\",\"command\":\"ls\"})")
        if "action" not in parsed:
            die("il manque « action » (bash, fetch ou final)")
        texte = json.dumps(parsed, ensure_ascii=False)
        kind = "call"
    req = args.req
    if not req:
        docs = box_list(20)
        users = [m for m in docs if m.get("role") == "user"]
        if users:
            req = users[-1]["id"]
    msg = box_add("arena", texte, req=req, kind=kind)
    print("✎ écrit dans le canal : %s (réponse à %s)" % (msg["id"], req or "—"))
    if args.wait <= 0:
        print("Votre message a été envoyé à l'utilisateur sur l'app !")
        return 0
    # Une commande exécutée par l'app renvoie son résultat (role « tool ») :
    # on l'attend un peu, pour pouvoir l'afficher ici même.
    t0 = time.time()
    sys.stdout.flush()
    while time.time() - t0 < args.wait:
        for m in box_list(6):
            if m.get("role") == "tool" and (m.get("req") or "") == msg["id"]:
                print("\n⚙ réponse de la VM :")
                show(m, full=True, indent="")
                return 0
        time.sleep(3)
    print("Votre message a été envoyé à l'utilisateur sur l'app !"
          " (aucune réponse de la VM pour l'instant : l'app n'est peut-être pas ouverte)")
    return 0


def cmd_selftest(args):
    print("canal : %s" % BOX)
    print("projet : %s" % PROJECT)
    msg = box_add("arena", "test canal z-canal.py " + str(int(time.time())), kind="text")
    docs = box_list(3)
    ok = any(d["id"] == msg["id"] for d in docs)
    print("écriture puis relecture : %s" % ("OK" if ok else "ÉCHEC"))
    box_delete(msg["id"])
    print("nettoyage : message de test supprimé")
    return 0 if ok else 1


def main():
    p = argparse.ArgumentParser(
        prog="z-canal.py",
        description="Canal entre Arena et l'application (Firebase).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Sous-commandes")[1] if "Sous-commandes" in __doc__ else "")
    sub = p.add_subparsers(dest="cmd")

    w = sub.add_parser("wait", help="attendre un message de l'utilisateur")
    w.add_argument("--timeout", type=float, default=1700, help="durée max en secondes (défaut 1700)")
    w.add_argument("--interval", type=float, default=4, help="intervalle de vérification en secondes")
    w.set_defaults(func=cmd_wait)

    r = sub.add_parser("read", help="afficher l'historique du canal")
    r.add_argument("--n", type=int, default=30, help="nombre de messages (défaut 30)")
    r.add_argument("--all", action="store_true", dest="full", help="texte intégral (sans troncature)")
    r.add_argument("--since", default=None, help="messages après cet id")
    r.set_defaults(func=cmd_read)

    a = sub.add_parser("write", help="envoyer un message vers l'app")
    a.add_argument("texte", nargs="?", default=None, help="le texte, ou « - » pour stdin")
    a.add_argument("--json", action="store_true", help="le texte est un JSON d'action (bash/fetch/final)")
    a.add_argument("--req", default="", help="id du message auquel on répond (sinon le dernier)")
    a.add_argument("--wait", type=float, default=20, help="attente d'une réponse de la VM (s)")
    a.set_defaults(func=cmd_write)

    s = sub.add_parser("state", help="état du canal")
    s.set_defaults(func=cmd_state)

    t = sub.add_parser("selftest", help="aller-retour d'écriture/lecture")
    t.set_defaults(func=cmd_selftest)

    args = p.parse_args()
    if not getattr(args, "func", None):
        p.print_help()
        return 0
    return args.func(args)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n(interrompu)")
        sys.exit(130)
