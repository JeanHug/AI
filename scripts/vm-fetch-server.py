#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vm-fetch-server.py — service de lecture web de la VM (outil « fetch » de l'agent).

Tourne DANS le conteneur (127.0.0.1:3011), lancé en arrière-plan par
setup-desktop.sh et surveillé par vm-fetch-watchdog.sh. Il garde UN Chrome
headless ouvert via Playwright : chaque lecture ne paie donc jamais le
démarrage du navigateur (~0,5 s) — c'est la clé de la vitesse.

API (JSON, 127.0.0.1 uniquement) :
  GET  /health   → {ok, browser, chrome, uptime_s}
  POST /fetch    → {url, mode:"html"|"text", wait_ms?, timeout_ms?, selector?}
                   → {ok, url, status, title, mode, chars, path, content, ms}

Optimisations : images / polices / médias bloqués ; CSS bloqué en mode texte ;
contexte persistant (cookies) ; page fermée après chaque lecture ; relance
automatique du navigateur s'il tombe. Le contenu renvoyé est COMPLET (aucune
troncature) et une copie est toujours écrite dans ~/.agtfetch/.
"""
import json
import os
import re
import threading
import time
import traceback
import urllib.parse
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer

PORT = int(os.environ.get("VMFETCH_PORT", "3011"))
SAVE_DIR = os.path.expanduser("~/.agtfetch")
UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/139.0.0.0 Safari/537.36")
CHROME_CANDIDATES = [
    "/usr/local/bin/google-chrome",
    "/usr/bin/google-chrome",
    "/usr/bin/google-chrome-stable",
    "/usr/bin/chromium",
    "/usr/bin/chromium-browser",
]
LAUNCH_ARGS = [
    "--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu",
    "--disable-background-networking", "--disable-sync", "--no-first-run",
    "--disable-extensions", "--disable-default-apps", "--mute-audio",
    "--disable-features=Translate,BackForwardCache,AcceptCHFrame,MediaRouter,OptimizationHints",
]

_state = {"pw": None, "browser": None, "ctx": None, "started": time.time(), "fetches": 0}
_lock = threading.Lock()          # l'API sync de Playwright se sérialise


def chrome_path():
    for p in CHROME_CANDIDATES:
        if os.path.exists(p):
            return p
    return None


def launch():
    from playwright.sync_api import sync_playwright
    st = _state
    if st["pw"] is None:
        st["pw"] = sync_playwright().start()
    exe = chrome_path()
    if exe:
        st["browser"] = st["pw"].chromium.launch(executable_path=exe, headless=True, args=LAUNCH_ARGS)
    else:
        st["browser"] = st["pw"].chromium.launch(headless=True, args=LAUNCH_ARGS)
    st["ctx"] = st["browser"].new_context(
        user_agent=UA, locale="fr-FR", viewport={"width": 1366, "height": 900},
        ignore_https_errors=True,
    )


def ensure():
    st = _state
    if st["browser"] is not None:
        try:
            if st["browser"].is_connected():
                return
        except Exception:
            pass
    launch()


def reset():
    st = _state
    for key in ("ctx", "browser"):
        try:
            obj = st.get(key)
            if obj is not None:
                obj.close()
        except Exception:
            pass
        st[key] = None


def slug(url):
    parts = urllib.parse.urlparse(url)
    raw = (parts.netloc + parts.path) or "page"
    return (re.sub(r"[^A-Za-z0-9._-]+", "_", raw)[:80].strip("_") or "page")


def do_fetch(body):
    url = str(body.get("url") or "").strip()
    if not re.match(r"^https?://", url, re.I):
        raise ValueError("url invalide (http:// ou https:// requis)")
    mode = "html" if str(body.get("mode") or "text").lower() == "html" else "text"
    try:
        wait_ms = min(max(int(body.get("wait_ms") or 700), 0), 15000)
    except Exception:
        wait_ms = 700
    try:
        timeout_ms = min(max(int(body.get("timeout_ms") or 45000), 3000), 180000)
    except Exception:
        timeout_ms = 45000
    selector = str(body.get("selector") or "").strip()[:200]

    t0 = time.time()
    with _lock:
        ensure()
        page = _state["ctx"].new_page()
        try:
            blocked = {"image", "media", "font"}
            if mode == "text":
                blocked.add("stylesheet")      # inutile pour du texte → plus rapide

            def on_route(route):
                try:
                    if route.request.resource_type in blocked:
                        route.abort()
                    else:
                        route.continue_()
                except Exception:
                    pass

            page.route("**/*", on_route)
            resp = page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
            if selector:
                try:
                    page.wait_for_selector(selector, timeout=min(timeout_ms, 10000))
                except Exception:
                    pass
            if wait_ms:
                page.wait_for_timeout(wait_ms)
            try:
                page.wait_for_load_state("networkidle", timeout=4000)
            except Exception:
                pass
            status = resp.status if resp else 0
            title = page.title() or ""
            final_url = page.url
            if mode == "html":
                content = page.content()
            else:
                content = page.evaluate("() => (document.body ? document.body.innerText : '')") or ""
        finally:
            try:
                page.close()
            except Exception:
                pass

    content = str(content)
    os.makedirs(SAVE_DIR, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = os.path.join(SAVE_DIR, stamp + "-" + slug(final_url) + ("." + ("html" if mode == "html" else "txt")))
    try:
        with open(path, "w", encoding="utf-8", errors="replace") as fh:
            fh.write(content)
    except Exception:
        path = ""
    _state["fetches"] += 1
    return {
        "ok": True,
        "requested": url,
        "url": final_url,
        "status": status,
        "title": title,
        "mode": mode,
        "chars": len(content),
        "path": path,
        "content": content,
        "ms": int((time.time() - t0) * 1000),
    }


class Handler(BaseHTTPRequestHandler):
    server_version = "VMFetch/1.0"
    protocol_version = "HTTP/1.0"      # connexion fermée après chaque réponse

    def log_message(self, *args):
        pass

    def _send(self, code, obj):
        raw = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        if self.path.startswith("/health"):
            return self._send(200, {
                "ok": True,
                "service": "vm-fetch",
                "browser": bool(_state["browser"]),
                "chrome": chrome_path(),
                "fetches": _state["fetches"],
                "uptime_s": int(time.time() - _state["started"]),
            })
        self._send(404, {"ok": False, "error": "route inconnue"})

    def do_POST(self):
        if not self.path.startswith("/fetch"):
            return self._send(404, {"ok": False, "error": "route inconnue"})
        try:
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length) or b"{}")
            if not isinstance(body, dict):
                body = {}
        except Exception:
            return self._send(400, {"ok": False, "error": "JSON invalide"})
        try:
            return self._send(200, do_fetch(body))
        except Exception as exc:
            reset()      # navigateur peut-être cassé : relance propre au prochain appel
            return self._send(500, {
                "ok": False,
                "error": str(exc),
                "trace": traceback.format_exc()[-600:],
            })


def main():
    try:
        ensure()          # préchauffage : le premier fetch ne paie pas le démarrage
        print("[vm-fetch] navigateur préchauffé (%s)" % (chrome_path() or "chromium playwright"), flush=True)
    except Exception as exc:
        print("[vm-fetch] préchauffage impossible : %s" % exc, flush=True)
    # Serveur MONO-THREAD : l'API sync de Playwright ne supporte pas de
    # changer de thread entre deux appels (le navigateur est préchauffé).
    srv = HTTPServer(("127.0.0.1", PORT), Handler)
    print("[vm-fetch] en écoute sur http://127.0.0.1:%d" % PORT, flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
