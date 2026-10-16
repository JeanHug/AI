#!/usr/bin/env python3
"""Réinjecte dans api-ai/worker.js les sources canoniques :
  - index.html                -> APP_HTML (chaîne JS, round-trip exact)
  - agent-docs/*.md           -> AGENT_DOC_B64 / INFOS_DOC_B64 / USER_DOC_B64
  - scripts/zcanal/*          -> ZCANAL_PY_B64 / JS / SH / TS / DOC (base64)
Usage : python3 scripts/embed-app-docs.py
"""
import base64, io, json, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # .ai/
# Worker canonique (workspace) + copie du dépôt : les deux sont écrites.
WF      = "/home/user/worker.js"
WF_REPO = os.path.join(ROOT, "api-ai", "worker.js")
APP     = "/home/user/index.html"          # source unique de vérité (workspace)
DOCS    = {"AGENT_DOC_B64": os.path.join(ROOT, "scripts", "agent-docs", "agent.md"),
           "INFOS_DOC_B64": os.path.join(ROOT, "scripts", "agent-docs", "infos.md"),
           "USER_DOC_B64":  os.path.join(ROOT, "scripts", "agent-docs", "user.md")}
ZC = os.path.join(ROOT, "scripts", "zcanal")
ZCANAL = {"ZCANAL_PY_B64":  os.path.join(ZC, "z-canal.py"),
          "ZCANAL_JS_B64":  os.path.join(ZC, "z-canal.js"),
          "ZCANAL_SH_B64":  os.path.join(ZC, "z-canal.sh"),
          "ZCANAL_TS_B64":  os.path.join(ZC, "z-canal.ts"),
          "ZCANAL_DOC_B64": os.path.join(ZC, "z-canal-doc.md")}

def js_str(text):
    out = json.dumps(text, ensure_ascii=False)
    return out.replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")

s = io.open(WF, encoding="utf-8").read()

# index.html : la source locale à la racine fait foi
assert os.path.exists(APP), "index.html introuvable : " + APP
html = io.open(APP, encoding="utf-8").read()
i = s.index("/*APP_HTML_START*/") + len("/*APP_HTML_START*/")
j = s.index("/*APP_HTML_END*/")
s = s[:i] + "\nconst APP_HTML = " + js_str(html) + ";\n" + s[j:]

# gabarits base64 (docs markdown + scripts z-canal)
B64 = {}
B64.update(DOCS)
B64.update(ZCANAL)
for const, path in B64.items():
    b64 = base64.b64encode(io.open(path, "rb").read()).decode()
    a = s.index("const %s = \"" % const)
    b = s.index("\";", a)
    s = s[:a] + "const %s = \"%s" % (const, b64) + s[b:]

io.open(WF, "w", encoding="utf-8").write(s)
io.open(WF_REPO, "w", encoding="utf-8").write(s)
io.open(os.path.join(ROOT, "index.html"), "w", encoding="utf-8").write(html)
io.open(os.path.join(ROOT, "api-ai", "index.html"), "w", encoding="utf-8").write(html)

# contrôles
s2 = io.open(WF, encoding="utf-8").read()
ok_app = js_str(html) in s2
print("index.html ->", APP, "(%d car.) | round-trip :" % len(html), ok_app)
for const, path in B64.items():
    a = s2.index("const %s = \"" % const) + len("const %s = \"" % const)
    b = s2.index("\";", a)
    dec = base64.b64decode(s2[a:b]).decode("utf-8")
    loc = io.open(path, encoding="utf-8").read()
    print("%-15s -> %6d car. | exact : %s" % (path.split("/")[-1], len(loc), dec == loc))
print("worker.js :", len(s2), "car.")
