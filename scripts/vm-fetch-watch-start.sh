#!/usr/bin/env bash
# vm-fetch-watch-start.sh — (re)démarre le watchdog, en un seul exemplaire.
#
# Le nom de ce script diffère volontairement de celui du watchdog : son propre
# « pkill »/« pgrep » ne peut donc pas se confondre avec le processus surveillé
# (leçon apprise : une commande lancée en ligne de commande qui contient le nom
# du service se détecte elle-même).
set -u
W="/home/kasm-user/bin/vm-fetch-watchdog.sh"
[ -f "$W" ] || W="$HOME/bin/vm-fetch-watchdog.sh"
P="/tmp/vm-fetch-watchdog.pid"

pkill -f "[v]m-fetch-watchdog.sh" 2>/dev/null || true
sleep 1
setsid nohup bash "$W" >> /tmp/vm-fetch-watchdog.log 2>&1 < /dev/null &
sleep 1
if [ -f "$P" ] && kill -0 "$(cat "$P")" 2>/dev/null; then
  echo "watchdog : en marche (pid $(cat "$P"))"
else
  echo "watchdog : échec du démarrage (voir /tmp/vm-fetch-watchdog.log)"
fi
