#!/usr/bin/env python3
"""Réinjecte dans api-ai/worker.js les sources canoniques :
  - index.html            -> APP_HTML (chaîne JS, round-trip exact)
  - agent-docs/*.md       -> AGENT_DOC_B64 / INFOS_DOC_B64 / USER_DOC_B64
Usage : python3 scripts/embed-app-docs.py
"""
import base64, io, json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # .ai/
# Worker canonique (workspace) + copie du dépôt : les deux sont écrites.
WF   = "/home/user/worker.js"
WF_REPO = os.path.join(ROOT, "api-ai", "worker.js")
APP = '/home/user/index.html'          # source unique de vérité (workspace)
DOCS = {"AGENT_DOC_B64": "agent.md", "INFOS_DOC_B64": "infos.md", "USER_DOC_B64": "user.md"}

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

# docs markdown
for const, name in DOCS.items():
    p = os.path.join(ROOT, "scripts", "agent-docs", name)
    b64 = base64.b64encode(io.open(p, "rb").read()).decode()
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
print("index.html ->", APP, f"({len(html)} car.) | round-trip :", ok_app)
for const, name in DOCS.items():
    a = s2.index("const %s = \"" % const) + len("const %s = \"" % const)
    b = s2.index("\";", a)
    dec = base64.b64decode(s2[a:b]).decode("utf-8")
    loc = io.open(os.path.join(ROOT, "scripts", "agent-docs", name), encoding="utf-8").read()
    print(f"{name:9s} -> {len(loc)} car. | exact :", dec == loc)
print("worker.js :", len(s2), "car.")
