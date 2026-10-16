#!/bin/sh
# z-canal — client du canal « __NAME__ » (id : __ID__) de l'application AI Agent.
#
#   sh z-canal.sh wait              attend un message (rend la main des qu'il arrive)
#   sh z-canal.sh read [n]          l'historique recent (ne marque rien)
#   sh z-canal.sh write "texte"     ecrit un message dans l'application
#   sh z-canal.sh write-json '{"action":"bash","command":"ls -la ~"}'
#   sh z-canal.sh state             l'etat du canal
#
# Un message ne revient jamais deux fois : des qu'un « wait » le rend, le canal
# le marque « lu ». C'est ce script qui gere l'attente : wait -> rien ? pause -> wait.
BASE="__BASE__/__ID__"
HOLD=25
GAP=5

case "$1" in
  wait)
    echo "canal « __NAME__ » (__ID__) : attente d'un message..."
    while :; do
      OUT=$(curl -s -m $((HOLD + 20)) "$BASE/wait?timeout=$HOLD")
      case "$OUT" in
        *'"messages":[]'*|'') sleep $GAP ;;
        *) printf '%s' "$OUT" | sed 's/\\n/\n/g'; echo; exit 0 ;;
      esac
    done ;;
  read)
    curl -s -m 60 "$BASE/read?n=${2:-30}" | sed 's/\\n/\n/g'; echo ;;
  write)
    shift; curl -s -m 60 -X POST -H 'Content-Type: application/json' \
      -d "{\"text\":$(printf '%s' "$*" | sed 's/\\/\\\\/g; s/"/\\"/g; s/^/"/; s/$/"/')}" "$BASE/write"; echo ;;
  write-json)
    shift; curl -s -m 60 -X POST -H 'Content-Type: application/json' \
      -d "{\"text\":$(printf '%s' "$*" | sed 's/\\/\\\\/g; s/"/\\"/g; s/^/"/; s/$/"/'),\"kind\":\"call\"}" "$BASE/write"; echo ;;
  state) curl -s -m 60 "$BASE/state"; echo ;;
  *) echo "usage : sh z-canal.sh wait|read|write|write-json|state" ;;
esac
