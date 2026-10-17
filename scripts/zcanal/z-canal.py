#!/usr/bin/env python3
"""z-canal — client du canal « __NAME__ » (id : __ID__) de l'application AI Agent.

Ce canal relie CET agent a l'application : l'utilisateur ecrit dans l'app,
l'agent recoit ses messages, peut repondre — et si le canal est en mode
« agent », il peut aussi executer des commandes sur la VM puis renvoyer leur
resultat dans l'application.

  python3 z-canal.py wait              attend un message (rend la main des qu'il arrive)
  python3 z-canal.py read              l'historique recent (ne marque rien)
  python3 z-canal.py write "texte"     ecrit un message dans l'application
  python3 z-canal.py write --json '{"action":"bash","command":"ls -la ~"}'
  python3 z-canal.py state             l'etat du canal

Un message ne revient JAMAIS deux fois : des qu'un « wait » le rend, le canal
le marque « lu ». C'est ce script qui gere l'attente (le serveur ne bloque pas
plus de 25 secondes) : wait -> rien ? pause courte -> wait. N'attendez pas avec
une boucle serree : 5 secondes de pause quand il n'y a rien suffisent, et pour
un echanges toutes les 10 secondes le total reste tres loin des quotas.
"""
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

BASE = "__BASE__"
ID = "__ID__"
NAME = "__NAME__"
MODE = "__MODE__"
WAIT_HOLD = 25          # secondes tenues par le serveur sur un « wait »
WAIT_GAP = 5            # pause entre deux « wait » quand il n'y a rien


def call(path, data=None, timeout=60):
    url = BASE + "/" + ID + path
    headers = {"User-Agent": "z-canal/1.0"}
    body = None
    if data is not None:
        body = json.dumps(data).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=body, headers=headers,
                                 method="POST" if body else "GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as rep:
            raw = rep.read().decode("utf-8", "replace")
            return json.loads(raw or "{}")
    except urllib.error.HTTPError as err:
        try:
            return json.loads(err.read().decode("utf-8", "replace") or "{}")
        except Exception:
            return {"ok": False, "error": "HTTP %s" % err.code}
    except Exception as err:                       # reseau, DNS, delai...
        return {"ok": False, "error": str(err)}


def hhmm(ts):
    try:
        return time.strftime("%H:%M:%S", time.localtime((ts or 0) / 1000.0))
    except Exception:
        return "?"


def show(msg, indent="  "):
    role = {"user": "toi", "agent": "agent", "tool": "vm"}.get(msg.get("role"), msg.get("role"))
    print("%s[%s] %s" % (indent, hhmm(msg.get("ts")), role))
    for line in str(msg.get("text") or "").splitlines() or [""]:
        print("%s  %s" % (indent, line))


def cmd_wait(_args):
    """Attend un message. Rend la main des qu'il en arrive un."""
    print("canal « %s » (%s) : attente d'un message..." % (NAME, ID))
    while True:
        rep = call("/wait?timeout=%d" % WAIT_HOLD, timeout=WAIT_HOLD + 30)
        if not rep.get("ok"):
            print("  (%s) — nouvel essai dans %d s" % (rep.get("error", "echec"), WAIT_GAP))
            time.sleep(WAIT_GAP)
            continue
        msgs = rep.get("messages") or []
        if not msgs:
            time.sleep(WAIT_GAP)
            continue
        print("")
        for m in msgs:
            show(m)
        print("")
        return 0


def cmd_read(args):
    n = int(args[0]) if args else 30
    rep = call("/read?n=%d" % n)
    if not rep.get("ok"):
        print("lecture impossible : %s" % rep.get("error", "raison inconnue"))
        return 1
    msgs = rep.get("messages") or []
    print("canal « %s » : %d message(s)" % (NAME, len(msgs)))
    for m in msgs:
        show(m)
    return 0


def cmd_write(args):
    if not args:
        print("rien a ecrire (donnez un texte, ou --json <objet>)")
        return 1
    if args[0] == "--json":
        try:
            payload = json.loads(" ".join(args[1:]))
        except ValueError as err:
            print("JSON invalide : %s" % err)
            return 1
        if not isinstance(payload, dict):
            print("le JSON doit etre un objet")
            return 1
        data = {"text": json.dumps(payload, ensure_ascii=False), "kind": "call"}
        # une commande : la passerelle attend son execution et rend le resultat
        data["wait"] = 20
    else:
        data = {"text": " ".join(args), "kind": "text"}
    rep = call("/write", data, timeout=90)
    if not rep.get("ok"):
        print("ecriture impossible : %s" % rep.get("error", "raison inconnue"))
        return 1
    print("ecrit dans le canal : %s" % rep.get("id", "?"))
    if rep.get("tool"):
        outil = rep["tool"]
        print("")
        print("resultat de la commande :")
        print("  " + str(outil.get("text") or "").replace("\n", "\n  "))
        return 0
    if data.get("kind") == "call":
        print("(aucun resultat : l'application est-elle ouverte, et la discussion en mode agent ?)")
    return 0


def cmd_state(_args):
    rep = call("/state")
    if not rep.get("ok"):
        print("etat indisponible : %s" % rep.get("error", "raison inconnue"))
        return 1
    c = rep.get("canal") or {}
    print("canal « %s »" % (c.get("name") or NAME))
    print("  id       : %s" % (c.get("id") or ID))
    print("  mode     : %s" % (c.get("mode") or MODE))
    print("  messages : %s" % c.get("messages", "?"))
    print("  non lus  : %s" % c.get("unread", "?"))
    return 0


def main(argv):
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(__doc__)
        return 0
    cmd, rest = argv[0], argv[1:]
    if cmd == "wait":
        return cmd_wait(rest)
    if cmd == "read":
        return cmd_read(rest)
    if cmd == "write":
        return cmd_write(rest)
    if cmd == "state":
        return cmd_state(rest)
    print("commande inconnue : %s (wait, read, write, state)" % cmd)
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
