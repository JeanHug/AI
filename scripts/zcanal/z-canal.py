#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""z-canal.py — le canal entre Arena et l'application (système Z-Canal).

Ce script est le client du canal « arena », le canal par défaut de
l'application. Il parle aux adresses publiques du canal :

    GET  BASE/wait?timeout=25   les messages NON LUS (et les marque « lu »)
    GET  BASE/read?n=30         l'historique récent (ne marque rien)
    POST BASE/write             écrire un message  {"text": …, "kind": …}
    GET  BASE/state             l'état du canal

avec BASE = https://api-ai.hugdu77777.workers.dev/canal/arena

Rien n'est secret ici : l'adresse du canal est publique, et il n'y a aucun
jeton à stocker — le canal reconnaît l'agent par son identifiant.

REPLI DIRECT
------------
Si la passerelle du canal refuse une requête (quota épuisé côté passerelle,
erreur 429 ou 5xx, réseau en panne), le script parle DIRECTEMENT à la base du
projet avec une session anonyme. Les deux clés ci-dessous sont les clés
PUBLIQUES de l'application : elles sont écrites dans sa page pour tout le
monde, il n'y a donc rien de secret à protéger. Le canal continue donc de
fonctionner même quand la passerelle est indisponible.

Sous-commandes
--------------
  wait   [--hold 25] [--gap 5] [--timeout 0]
                      attend un message. Le serveur ne bloque que 25 s : c'est
                      CE script qui boucle (attente → rien ? pause de 5 s →
                      attente). --timeout 0 (défaut) = sans fin.
  read   [--n 30] [--all]
                      affiche l'historique récent (ne marque rien comme lu)
  write  <texte|-> [--json] [--req ID] [--wait 20] [--image FICHIER]
                      écrit un message dans l'application. Une commande
                      (--json) rend en plus SON RÉSULTAT : la sortie de la VM
                      s'affiche ici même, dans la réponse du write.
                      --image joint une image (répétable, trois au plus) :
                      elle est réduite en JPEG 1280 px pour tenir dans le
                      document Firestore, puis s'affiche dans l'app.
  state               état du canal en un coup d'œil
  doc                 l'adresse de la documentation
  selftest            aller-retour d'écriture/lecture

Exemples
--------
  python3 z-canal.py wait                       # veille : rend la main dès qu'un message arrive
  python3 z-canal.py read                       # ce qui s'est dit, du plus ancien au plus récent
  python3 z-canal.py write "Je regarde ça."
  python3 z-canal.py write --json '{"action":"bash","command":"ls -l /home/user"}'
  python3 z-canal.py write - < reponse.md       # le texte arrive par l'entrée standard

Un message ne revient JAMAIS deux fois : dès qu'un « wait » le rend, le canal
le marque « lu » — et l'application remplace « non lu » par « lu ! » sous le
message. Les commandes ne s'exécutent que si l'application est ouverte.
"""

import argparse
import base64
import io
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime

# --------------------------------------------------------------- configuration
# L'adresse du canal. Elle est publique : l'identifiant EST la clé.
BASE = "https://api-ai.hugdu77777.workers.dev/canal/arena"
HOLD = 25          # durée maximale d'une attente côté serveur (secondes)
GAP = 10           # pause entre deux attentes : 25 s d'attente + 10 s de pause

UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/120.0.0.0 Safari/537.36")

# Clés PUBLIQUES du projet (repli direct ; voir « REPLI DIRECT » plus haut).
FB_KEY = "AIzaSyC58J1paO0ULhTW0RuDfUePVeuOUmG5fZY"
FB_PROJET = "ai-agent-1234a"
FB_DOC = "chan/main"       # le canal par défaut : sa fiche, et sa boîte juste en dessous
FB_SUB = "box"
FB_ROLE = "arena"          # le rôle sous lequel l'application attend nos messages


def _http(method, url, body=None, timeout=60, headers=None):
    data = None
    h = {"User-Agent": UA}
    if headers:
        h.update(headers)
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        h["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=h, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
            try:
                return resp.status, json.loads(raw)
            except ValueError:
                return resp.status, {"raw": raw}
    except urllib.error.HTTPError as err:
        raw = err.read().decode("utf-8", "replace")
        try:
            return err.code, json.loads(raw)
        except ValueError:
            return err.code, {"error": raw[:300]}
    except Exception as err:                                  # réseau coupé
        return 0, {"error": str(err)}


def api(path, body=None, timeout=60):
    """Un appel à la passerelle du canal ; renvoie (code, objet JSON)."""
    return _http("POST" if body is not None else "GET", BASE + path, body, timeout)


IMG_DIR = "/home/user/uploads/canal"     # où sont rangées les images reçues
IMG_MAX = 1280                            # plus grand côté, en pixels
IMG_B64_MAX = 760000                      # un document Firestore fait 1 Mio


def image_ready(chemin):
    """Prépare une image pour le canal : JPEG, 1280 px au plus, base64 borné.

    Elle est REFORMATÉE (pas seulement copiée) parce qu'un document Firestore
    ne peut pas dépasser 1 Mio : une photo de téléphone n'y tiendrait pas.
    """
    try:
        from PIL import Image
    except ImportError:                                   # Pillow absent
        with open(chemin, "rb") as f:
            brut = f.read()
        if len(brut) > IMG_B64_MAX * 3 // 4:
            return None
        mime = "image/png" if brut[:4] == b"\x89PNG" else "image/jpeg"
        return {"name": os.path.basename(chemin), "mime": mime,
                "b64": base64.b64encode(brut).decode("ascii")}
    try:
        im = Image.open(chemin).convert("RGB")
    except Exception as err:
        print("   (image illisible : %s)" % err, file=sys.stderr)
        return None
    im.thumbnail((IMG_MAX, IMG_MAX))
    nom = os.path.splitext(os.path.basename(chemin))[0] + ".jpg"
    for q in (82, 72, 62, 52, 42):
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=q, optimize=True)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        if len(b64) <= IMG_B64_MAX:
            return {"name": nom, "mime": "image/jpeg", "b64": b64}
    print("   (image trop lourde même compressée : %s)" % chemin, file=sys.stderr)
    return None


def images_ready(chemins):
    out = []
    for c in (chemins or [])[:3]:
        if not os.path.exists(c):
            die("image introuvable : %s" % c)
        im = image_ready(c)
        if im:
            out.append(im)
    return out


def image_lue(im, prefixe):
    """Écrit sur le disque une image reçue et rend son chemin (ou "")."""
    b64 = (im or {}).get("b64") or ""
    if not b64:
        return ""
    os.makedirs(IMG_DIR, exist_ok=True)
    mime = (im.get("mime") or "image/jpeg").lower()
    ext = {"image/png": ".png", "image/gif": ".gif", "image/webp": ".webp"}.get(mime, ".jpg")
    nom = re.sub(r"[^A-Za-z0-9._-]+", "-", str(im.get("name") or "image"))
    if not nom.lower().endswith(ext):
        nom = os.path.splitext(nom)[0] + ext
    chemin = os.path.join(IMG_DIR, "%s-%s" % (prefixe, nom))
    try:
        with open(chemin, "wb") as f:
            f.write(base64.b64decode(b64))
    except Exception as err:
        print("   (image non enregistrée : %s)" % err, file=sys.stderr)
        return ""
    return chemin


def die(msg, code=1):
    print("✗ " + msg, file=sys.stderr)
    sys.exit(code)


# ------------------------------------------------------- repli : accès direct
_DIRECT_TOKEN = {"id": "", "expire": 0}
_DIRECT_DIT = False        # a-t-on déjà prévenu que l'on passe en direct ?


def _fb_token():
    """Une session anonyme du projet (aucun mot de passe, aucune donnée privée)."""
    if _DIRECT_TOKEN["id"] and _DIRECT_TOKEN["expire"] > time.time() + 60:
        return _DIRECT_TOKEN["id"]
    url = ("https://identitytoolkit.googleapis.com/v1/accounts:signUp?key=" + FB_KEY)
    code, res = _http("POST", url, {"returnSecureToken": True})
    token = (res or {}).get("idToken") or ""
    if code >= 300 or not token:
        return ""
    _DIRECT_TOKEN["id"] = token
    _DIRECT_TOKEN["expire"] = time.time() + float((res or {}).get("expiresIn") or 3600)
    return token


def _fb_url(chemin="", query=""):
    return ("https://firestore.googleapis.com/v1/projects/" + FB_PROJET
            + "/databases/(default)/documents/" + chemin.lstrip("/") + query)


def _fb_en_direct(quoi):
    global _DIRECT_DIT
    if not _DIRECT_DIT:
        _DIRECT_DIT = True
        print("   (passerelle indisponible — liaison directe à la base pour %s)" % quoi)
        sys.stdout.flush()


def _fb_val(champ):
    if not isinstance(champ, dict):
        return None
    for cle in ("stringValue", "timestampValue"):
        if cle in champ:
            return champ[cle]
    for cle in ("integerValue", "doubleValue"):
        if cle in champ:
            try:
                return float(champ[cle]) if cle == "doubleValue" else int(champ[cle])
            except (TypeError, ValueError):
                return None
    if "booleanValue" in champ:
        return bool(champ["booleanValue"])
    return None


def _fb_field(valeur):
    if isinstance(valeur, bool):
        return {"booleanValue": valeur}
    if isinstance(valeur, int):
        return {"integerValue": str(valeur)}
    if isinstance(valeur, float):
        return {"doubleValue": valeur}
    return {"stringValue": "" if valeur is None else str(valeur)}


def _fb_any(champ):
    """Valeur Firestore complète : gère aussi les tableaux et objets (images)."""
    if not isinstance(champ, dict):
        return None
    if "arrayValue" in champ:
        vals = (champ["arrayValue"] or {}).get("values") or []
        return [_fb_any(v) for v in vals]
    if "mapValue" in champ:
        champs = (champ["mapValue"] or {}).get("fields") or {}
        return dict((k, _fb_any(champs[k])) for k in champs)
    return _fb_val(champ)


def _fb_doc(document):
    champs = (document or {}).get("fields") or {}
    message = dict((cle, _fb_any(champs[cle])) for cle in champs)
    message["id"] = str((document or {}).get("name") or "").split("/")[-1]
    return message


def _fb_trie(messages):
    messages.sort(key=lambda m: m.get("ts") or 0)
    return messages


def fb_list(n=30):
    """L'historique récent (l'équivalent direct de BASE/read)."""
    token = _fb_token()
    if not token:
        return None
    corps = {"structuredQuery": {
        "from": [{"collectionId": FB_SUB}],
        "orderBy": [{"field": {"fieldPath": "ts"}, "direction": "DESCENDING"}],
        "limit": max(1, min(200, int(n))),
    }}
    code, res = _http("POST", _fb_url(FB_DOC + ":runQuery"), corps,
                      headers={"Authorization": "Bearer " + token})
    if code >= 300 or not isinstance(res, list):
        return None
    # DESCENDING ne rend que la tranche récente : on la remet dans l'ordre.
    return _fb_trie([_fb_doc(x["document"]) for x in res if isinstance(x, dict) and x.get("document")])


def fb_marque_lus(messages):
    token = _fb_token()
    if not token:
        return
    for message in messages:
        if not message.get("id"):
            continue
        url = _fb_url(FB_DOC + "/" + FB_SUB + "/" + message["id"],
                      "?updateMask.fieldPaths=state&updateMask.fieldPaths=readAt")
        _http("PATCH", url, {"fields": {"state": _fb_field("read"), "readAt": _fb_field(int(time.time() * 1000))}},
              headers={"Authorization": "Bearer " + token})


def fb_unread():
    """Les messages NON LUS, par requête filtrée (facturée au document rendu :
    une attente qui ne trouve rien ne consomme donc rien)."""
    token = _fb_token()
    if not token:
        return None
    corps = {"structuredQuery": {
        "from": [{"collectionId": FB_SUB}],
        "where": {"fieldFilter": {"field": {"fieldPath": "state"}, "op": "EQUAL",
                                  "value": {"stringValue": "new"}}},
        "limit": 50,
    }}
    code, res = _http("POST", _fb_url(FB_DOC + ":runQuery"), corps,
                      headers={"Authorization": "Bearer " + token})
    if code >= 300 or not isinstance(res, list):
        return None
    messages = [_fb_doc(x["document"]) for x in res if isinstance(x, dict) and x.get("document")]
    messages = [m for m in messages if m.get("role") == "user"]
    return _fb_trie(messages)


def fb_resultat(req_id):
    """Le message « tool » écrit en réponse à la commande req_id (le plus récent)."""
    if not req_id:
        return None
    token = _fb_token()
    if not token:
        return None
    corps = {"structuredQuery": {
        "from": [{"collectionId": FB_SUB}],
        "where": {"fieldFilter": {"field": {"fieldPath": "req"}, "op": "EQUAL",
                                  "value": {"stringValue": str(req_id)}}},
        "limit": 10,
    }}
    code, res = _http("POST", _fb_url(FB_DOC + ":runQuery"), corps,
                      headers={"Authorization": "Bearer " + token})
    if code >= 300 or not isinstance(res, list):
        return None
    docs = [_fb_doc(x["document"]) for x in res if isinstance(x, dict) and x.get("document")]
    outils = [m for m in docs if m.get("role") == "tool"]
    outils.sort(key=lambda m: m.get("ts") or 0)
    return outils[-1] if outils else None


def _fb_images(imgs):
    """Les images d'un message, au format Firestore (tableau d'objets)."""
    vals = []
    for im in imgs or []:
        vals.append({"mapValue": {"fields": {
            "name": _fb_field(im.get("name") or "image"),
            "mime": _fb_field(im.get("mime") or "image/jpeg"),
            "b64": _fb_field(im.get("b64") or ""),
        }}})
    return {"arrayValue": {"values": vals}}


def fb_write(texte, kind="text", req="", note="", imgs=None):
    """Écrire un message dans le canal (l'équivalent direct de BASE/write)."""
    token = _fb_token()
    if not token:
        return None
    champs = {
        "role": _fb_field(FB_ROLE), "text": _fb_field(texte), "kind": _fb_field(kind),
        "state": _fb_field("new"), "src": _fb_field("script"), "note": _fb_field(note),
        "req": _fb_field(req), "ts": _fb_field(int(time.time() * 1000)),
    }
    if imgs:
        champs["imgs"] = _fb_images(imgs)
    corps = {"fields": champs}
    code, res = _http("POST", _fb_url(FB_DOC + "/" + FB_SUB), corps,
                      headers={"Authorization": "Bearer " + token})
    if code >= 300 or not isinstance(res, dict):
        return None
    return str(res.get("name") or "").split("/")[-1]


def fb_state():
    """L'état du canal : la fiche, le compte des messages, les non-lus."""
    token = _fb_token()
    if not token:
        return None
    entetes = {"Authorization": "Bearer " + token}
    code_fiche, fiche = _http("GET", _fb_url(FB_DOC), headers=entetes)
    nom = "arena"
    if code_fiche < 300 and isinstance(fiche, dict):
        champs = (fiche.get("fields") or {})
        nom = _fb_val(champs.get("name")) or "arena"
    total = None
    corps = {"structuredAggregationQuery": {
        "structuredQuery": {"from": [{"collectionId": FB_SUB}]},
        "aggregations": [{"count": {}}],
    }}
    code, res = _http("POST", _fb_url(FB_DOC + ":runAggregationQuery"), corps, headers=entetes)
    if code < 300 and isinstance(res, list):
        for x in res:
            agregats = ((x or {}).get("result") or {}).get("aggregateFields") or {}
            for cle in agregats:
                try:
                    total = int(agregats[cle].get("integerValue") or 0)
                except (TypeError, ValueError):
                    total = None
    non_lus = fb_unread()
    return {"nom": nom, "messages": total,
            "non_lus": len(non_lus) if non_lus is not None else None}


# ------------------------------------------------------------------- affichage
def hhmm(ts_ms):
    if not ts_ms:
        return "?"
    try:
        return datetime.fromtimestamp(float(ts_ms) / 1000.0).astimezone().strftime("%H:%M:%S")
    except (OverflowError, OSError, ValueError):
        return "?"


ROLE_TAG = {"user": "👤 toi", "arena": "◆ arena", "agent": "◆ agent", "tool": "⚙ vm"}


def show(msg, full=False, indent="  "):
    tag = ROLE_TAG.get(msg.get("role"), msg.get("role"))
    kind = msg.get("kind") or "text"
    text = msg.get("text") or ""
    body = text if (full or len(text) <= 500) else text[:500] + " …(%d caractères au total)" % len(text)
    head = "%s[%s] %s%s" % (indent, hhmm(msg.get("ts")), tag,
                            (" · " + kind) if kind != "text" else "")
    lines = [head]
    for line in (body.splitlines() or [""]):
        lines.append(indent + "  " + line)
    # images reçues : enregistrées sur le disque, chemin affiché
    for i, im in enumerate(msg.get("imgs") or []):
        if not isinstance(im, dict):
            continue
        pref = "%s-%s" % (msg.get("id") or "msg", i + 1)
        chemin = image_lue(im, pref)
        ko = (len(im.get("b64") or "") * 3 // 4) // 1024
        if chemin:
            lines.append("%s  🖼 %s — %d Ko → %s" % (indent, im.get("name") or "image", ko, chemin))
        else:
            lines.append("%s  🖼 %s — %d Ko (non enregistrée)" % (indent, im.get("name") or "image", ko))
    print("\n".join(lines))


# ------------------------------------------------------------------ commandes
def cmd_read(args):
    code, res = api("/read?n=%d" % max(1, args.n))
    msgs = None
    if code < 300 and isinstance(res, dict) and res.get("ok"):
        msgs = res.get("messages") or []
        canal = res.get("canal") or {}
    else:
        _fb_en_direct("la lecture")
        msgs = fb_list(args.n)
        canal = {"name": "arena", "id": "arena"}
        if msgs is None:
            die("lecture impossible (%s) : %s" % (code, (res or {}).get("error", "?")))
    if not msgs:
        print("(canal vide)")
        return 0
    print("canal « %s » (%s) — %d message(s), du plus ancien au plus récent :\n"
          % (canal.get("name", "?"), canal.get("id", "?"), len(msgs)))
    for msg in msgs:
        show(msg, full=args.full)
        print()
    users = [m for m in msgs if m.get("role") == "user"]
    if users:
        last = users[-1]
        print("dernier message de l'utilisateur : %s (%s) · état %s"
              % (last.get("id", "?"), hhmm(last.get("ts")), last.get("state") or "?"))
    return 0


def cmd_state(args):
    code, res = api("/state")
    # Une réponse « ok » qui porte une erreur de lecture interne n'est pas une
    # réponse : on passe en direct (la passerelle sait le dire depuis P75).
    if (code < 300 and isinstance(res, dict) and res.get("ok")
            and not res.get("erreur") and not res.get("firestore")):
        c = res.get("canal") or {}
        print("canal « %s »" % c.get("name", "?"))
        print("  identifiant : %s" % c.get("id", "?"))
        print("  mode        : %s" % c.get("mode", "?"))
        print("  messages    : %s" % c.get("messages", "?"))
        print("  non lus     : %s" % c.get("unread", "?"))
        return 0
    _fb_en_direct("l'état")
    etat = fb_state()
    if not etat:
        die("état indisponible (%s) : %s" % (code, (res or {}).get("error", "?")))
    print("canal « %s »" % etat["nom"])
    print("  identifiant : arena")
    print("  mode        : agent")
    print("  messages    : %s" % etat["messages"])
    print("  non lus     : %s" % etat["non_lus"])
    return 0


def _attente_directe(hold, gap, t0, timeout):
    """L'attente sans passerelle : requête filtrée « non lu », puis pause."""
    rate = 0
    while True:
        messages = fb_unread()
        if messages is None:
            rate += 1
            # On ne répète pas la même ligne toutes les 10 s : un rappel par
            # minute suffit (la base peut être en quota atteint pendant des heures).
            if rate == 1 or rate % 6 == 0:
                print("   (liaison directe indisponible : base en quota — nouvel essai, "
                      "%.0f min d'attente)" % ((time.time() - t0) / 60.0))
                sys.stdout.flush()
        elif messages:
            fb_marque_lus(messages)
            print("\n📨 message reçu après %.0f s :\n" % (time.time() - t0))
            for m in messages:
                show(m, full=True, indent="")
                print()
            return messages
        if timeout and time.time() - t0 >= timeout:
            print("\n⏱  rien de nouveau après %.0f min." % (timeout / 60.0))
            return []
        time.sleep(max(gap, hold // 3 if hold else gap))


def cmd_wait(args):
    """Attend un message. C'est le SCRIPT qui gère la durée : le serveur ne
    bloque que HOLD secondes, on enchaîne donc les attentes avec une pause."""
    hold = max(0, min(25, int(args.hold)))
    gap = max(1, int(args.gap))
    t0 = time.time()
    fini = args.timeout and args.timeout > 0
    print("⏳ canal « arena » : attente d'un message%s…"
          % (" (%.0f min au maximum)" % (args.timeout / 60.0) if fini else ""))
    print("   (le serveur tient %ds, puis on reprend ; « stop » arrête les attentes)" % hold)
    sys.stdout.flush()
    while True:
        code, res = api("/wait?timeout=%d" % hold, timeout=hold + 30)
        if code >= 300 or not isinstance(res, dict) or not res.get("ok"):
            if code == 0 or code >= 429:
                _fb_en_direct("l'attente")
                msgs = _attente_directe(hold, gap, t0, args.timeout if fini else 0)
                if msgs:
                    texte = " ".join((m.get("text") or "").strip().lower() for m in msgs).strip()
                    if texte in ("stop", "arrête", "arrete"):
                        print("■ « stop » : l'utilisateur demande l'arrêt des attentes.")
                        return 3
                    print("(à traiter ; « write » répondra automatiquement à %s)"
                          % (msgs[-1].get("id") or "?"))
                    return 0
                return 0
            err = (res or {}).get("error", "?")
            print("   (%s) — nouvel essai dans %ds" % (err, gap))
            sys.stdout.flush()
            if fini and time.time() - t0 >= args.timeout:
                print("\n⏱  rien de nouveau après %.0f min." % (args.timeout / 60.0))
                return 0
            time.sleep(gap)
            continue
        msgs = res.get("messages") or []
        if msgs:
            print("\n📨 message reçu après %.0f s :\n" % (time.time() - t0))
            for m in msgs:
                show(m, full=True, indent="")
                print()
            texte = " ".join((m.get("text") or "").strip().lower() for m in msgs).strip()
            if texte in ("stop", "arrête", "arrete"):
                print("■ « stop » : l'utilisateur demande l'arrêt des attentes.")
                return 3
            print("(à traiter ; « write » répondra automatiquement à %s)"
                  % (msgs[-1].get("id") or "?"))
            return 0
        if fini and time.time() - t0 >= args.timeout:
            print("\n⏱  rien de nouveau après %.0f min — relance un wait si besoin."
                  % (args.timeout / 60.0))
            return 0
        time.sleep(gap)


def cmd_write(args):
    """Écrit un message. Pour une COMMANDE (--json), rend son résultat : la
    passerelle attend elle-même l'exécution et renvoie la sortie ; sinon, on
    interroge le canal jusqu'à ce que le résultat arrive."""
    texte = args.texte
    if texte == "-":
        texte = sys.stdin.read()
    if texte is None:
        if getattr(args, "image", None):
            texte = ""                      # message fait uniquement d'images
        else:
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

    # Pour une commande, on demande à la passerelle d'attendre le résultat
    # (20 s au maximum) : la sortie revient donc DANS la réponse du write.
    attente_serveur = int(max(0, min(20, args.wait))) if kind == "call" else 0
    imgs = images_ready(getattr(args, "image", None) or [])
    corps = {"text": texte, "kind": kind, "req": args.req or "", "note": ""}
    if imgs:
        corps["imgs"] = imgs
    if attente_serveur:
        corps["wait"] = attente_serveur
    code, res = api("/write", corps, timeout=60 + attente_serveur)

    msg_id = None
    outil = None
    if code < 300 and isinstance(res, dict) and res.get("ok"):
        msg_id = res.get("id") or "?"
        outil = res.get("tool") or None
    else:
        _fb_en_direct("l'écriture")
        msg_id = fb_write(texte, kind, args.req or "", "", imgs)
        if msg_id is None:
            die("écriture impossible (%s) : %s" % (code, (res or {}).get("error", "?")))
    print("✎ écrit dans le canal : %s (réponse à %s)%s"
          % (msg_id, args.req or "—",
             (" · %d image(s)" % len(imgs)) if imgs else ""))

    if not outil and kind == "call" and args.wait > 0:
        # La passerelle n'a rien rendu : on attend ici, en interrogeant le canal.
        t0 = time.time()
        sys.stdout.flush()
        while time.time() - t0 < args.wait:
            outil = fb_resultat(msg_id)
            if not outil:
                code2, r2 = api("/read?n=20")
                for m in ((r2 or {}).get("messages") or []) if code2 < 300 else []:
                    if m.get("role") == "tool" and (m.get("req") or "") == msg_id:
                        outil = m
                        break
            if outil:
                break
            time.sleep(2)

    if outil:
        print("\n⚙ résultat de la commande :")
        show(outil, full=True, indent="")
        return 0

    print("Votre message a été envoyé à l'utilisateur sur l'app !")
    if kind == "call" and args.wait > 0:
        print("(aucun résultat pour l'instant : l'application est-elle ouverte, "
              "et la discussion en mode agent ?)")
    return 0


def cmd_doc(args):
    print(BASE.rsplit("/", 1)[0])
    return 0


def cmd_selftest(args):
    print("canal : %s" % BASE)
    code, res = api("/state")
    if code >= 300 or not (res or {}).get("ok"):
        _fb_en_direct("l'état")
        etat = fb_state()
        if not etat:
            die("état indisponible (%s) : %s" % (code, (res or {}).get("error", "?")))
        print("identifiant : arena · mode : agent · messages : %s" % etat["messages"])
    else:
        c = res.get("canal") or {}
        print("identifiant : %s · mode : %s · messages : %s"
              % (c.get("id"), c.get("mode"), c.get("messages")))
    texte = "test canal z-canal.py " + str(int(time.time()))
    code, w = api("/write", {"text": texte, "kind": "text"})
    ok_w = code < 300 and (w or {}).get("ok")
    if not ok_w:
        ok_w = fb_write(texte, "text") is not None
    print("écriture : %s%s" % ("OK" if ok_w else "ÉCHEC", "" if ok_w else " — %s" % (w or {}).get("error")))
    code, r = api("/read?n=5")
    messages = (r or {}).get("messages") if (code < 300 and isinstance(r, dict)) else None
    if messages is None:
        messages = fb_list(5) or []
    ok_r = any((m.get("text") or "") == texte for m in messages)
    print("relecture : %s" % ("OK" if ok_r else "ÉCHEC"))
    return 0 if (ok_w and ok_r) else 1


def main():
    p = argparse.ArgumentParser(
        prog="z-canal.py",
        description="Le canal entre Arena et l'application (système Z-Canal).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Sous-commandes")[1] if "Sous-commandes" in __doc__ else "")
    sub = p.add_subparsers(dest="cmd")

    w = sub.add_parser("wait", help="attendre un message de l'utilisateur")
    w.add_argument("--hold", type=int, default=HOLD, help="durée d'une attente côté serveur (défaut 25 s)")
    w.add_argument("--gap", type=int, default=GAP, help="pause entre deux attentes (défaut 5 s)")
    w.add_argument("--timeout", type=float, default=0, help="durée totale max (0 = sans fin)")
    w.set_defaults(func=cmd_wait)

    r = sub.add_parser("read", help="afficher l'historique du canal")
    r.add_argument("--n", type=int, default=30, help="nombre de messages (défaut 30)")
    r.add_argument("--all", action="store_true", dest="full", help="texte intégral (sans troncature)")
    r.set_defaults(func=cmd_read)

    a = sub.add_parser("write", help="envoyer un message vers l'application")
    a.add_argument("texte", nargs="?", default=None, help="le texte, ou « - » pour stdin")
    a.add_argument("--json", action="store_true", help="le texte est un JSON d'action (bash/fetch/final)")
    a.add_argument("--req", default="", help="id du message auquel on répond (sinon aucun)")
    a.add_argument("--wait", type=float, default=20, help="attente d'une réponse de la VM (s)")
    a.add_argument("--image", action="append", default=[], metavar="FICHIER",
                   help="joint une image au message (répétable, trois au plus)")
    a.set_defaults(func=cmd_write)

    s = sub.add_parser("state", help="état du canal")
    s.set_defaults(func=cmd_state)

    d = sub.add_parser("doc", help="l'adresse de la documentation")
    d.set_defaults(func=cmd_doc)

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
