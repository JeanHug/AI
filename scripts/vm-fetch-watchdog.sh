#!/usr/bin/env bash
# vm-fetch-watchdog.sh — surveille le service de lecture web de l'agent (port 3011)
# et le relance s'il ne répond plus (vérification toutes les 30 s).
#
# Tourne DANS le conteneur (comme le service lui-même) : le daemon de l'hôte
# recrée le service au démarrage, ce watchdog couvre les arrêts en cours de vie.
set -u
RESTART="/home/kasm-user/bin/vm-fetch-restart.sh"
[ -f "$RESTART" ] || RESTART="$HOME/bin/vm-fetch-restart.sh"

echo $$ > /tmp/vm-fetch-watchdog.pid
trap 'rm -f /tmp/vm-fetch-watchdog.pid' EXIT

while true; do
  if ! curl -s -m 5 -o /dev/null "http://127.0.0.1:3011/health" 2>/dev/null; then
    echo "[watchdog $(date -Is)] service muet -> relance"
    bash "$RESTART" 2>&1 | sed 's/^/  /'
  fi
  sleep 30
done
