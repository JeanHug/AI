#!/usr/bin/env bash
# vm-fetch-restart.sh — (re)démarre le service de lecture web de l'agent.
#
# À lancer DEPUIS le conteneur (docker exec … bash -lc /home/kasm-user/bin/vm-fetch-restart.sh).
# Le motif [v] évite que pkill se tue lui-même, et ce script est le seul endroit
# où le nom du service apparaît de façon « littérale » : sûr partout.
set -u
BIN="/home/kasm-user/bin/vm-fetch-server.py"
[ -f "$BIN" ] || BIN="$HOME/bin/vm-fetch-server.py"

pkill -f "[v]m-fetch-server.py" 2>/dev/null || true
sleep 1
setsid nohup python3 "$BIN" > /tmp/vm-fetch.log 2>&1 < /dev/null &
sleep 2
if curl -s -m 15 http://127.0.0.1:3011/health | grep -q '"ok"'; then
  echo "vm-fetch : en ligne"
else
  echo "vm-fetch : démarrage en cours (voir /tmp/vm-fetch.log)"
fi
