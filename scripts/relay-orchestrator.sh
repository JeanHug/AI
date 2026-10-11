#!/usr/bin/env bash
set -e

RELAY_CYCLE="${RELAY_CYCLE:-1}"
SYNC_INTERVAL_MINS="${SYNC_INTERVAL_MINS:-10}"
RELAY_TRIGGER_MINS="${RELAY_TRIGGER_MINS:-315}"
RELAY_SHUTDOWN_MINS="${RELAY_SHUTDOWN_MINS:-345}"

echo "=================================================="
echo "    ORCHESTRATEUR DE RELAIS BI-VM (LINUX & ANDROID) "
echo "=================================================="

START_TIME=$(date +%s)
NEXT_BACKUP_TIME=$(( START_TIME + SYNC_INTERVAL_MINS * 60 ))
TRIGGER_TIME=$(( START_TIME + RELAY_TRIGGER_MINS * 60 ))
SHUTDOWN_TIME=$(( START_TIME + RELAY_SHUTDOWN_MINS * 60 ))
RELAY_TRIGGERED=false
CONSECUTIVE_TUNNEL_FAILS=0

cleanup_and_exit() {
  echo "Arrêt ordonné et sauvegarde finale 100%..."
  ./scripts/backup-sync.sh backup || true
  pkill -f cloudflared || true
  pkill -f "nokey@localhost.run" || true
  docker stop kasm_desktop redroid ws_scrcpy 2>/dev/null || true
  exit 0
}

trap cleanup_and_exit SIGTERM SIGINT

while true; do
  NOW=$(date +%s)
  ELAPSED_MINS=$(( (NOW - START_TIME) / 60 ))

  # 1. Sauvegarde périodique 100%
  if [ "$NOW" -ge "$NEXT_BACKUP_TIME" ]; then
    echo "[$(date +'%T')] Sauvegarde périodique complète..."
    ./scripts/backup-sync.sh backup || true
    NEXT_BACKUP_TIME=$(( NOW + SYNC_INTERVAL_MINS * 60 ))
  fi

  # 2. Vérification de santé du tunnel direct .lhr.life
  LHR_URL=$(grep -o 'https://[-a-zA-Z0-9_.]*\.lhr\.life' /tmp/localhost_run.log 2>/dev/null | head -n1 || true)
  if [ -n "$LHR_URL" ]; then
    HTTP_CODE=$(curl -s -k -o /dev/null -w "%{http_code}" --max-time 8 "$LHR_URL" || echo "000")
    if [ "$HTTP_CODE" = "200" ] || [ "$HTTP_CODE" = "302" ] || [ "$HTTP_CODE" = "401" ] || [ "$HTTP_CODE" = "403" ]; then
      CONSECUTIVE_TUNNEL_FAILS=0
    else
      CONSECUTIVE_TUNNEL_FAILS=$(( CONSECUTIVE_TUNNEL_FAILS + 1 ))
      echo "[$(date +'%T')] ⚠️ Alerte tunnel .lhr.life (Code: $HTTP_CODE, Échecs: $CONSECUTIVE_TUNNEL_FAILS/3)"
      if [ "$CONSECUTIVE_TUNNEL_FAILS" -ge 3 ]; then
        echo "[$(date +'%T')] 🔄 Tunnel .lhr.life non fonctionnel. Redéploiement automatique du tunnel..."
        ./scripts/start-tunnel.sh || true
        CONSECUTIVE_TUNNEL_FAILS=0
      fi
    fi
  fi

  # 3. Relais automatique vers le prochain runner GitHub
  if [ "$NOW" -ge "$TRIGGER_TIME" ] && [ "$RELAY_TRIGGERED" = false ]; then
    NEXT_CYCLE=$(( RELAY_CYCLE + 1 ))
    echo "Lancement du relais vers le cycle $NEXT_CYCLE"
    
    PAYLOAD=$(cat <<JSON
{
  "ref": "main",
  "inputs": {
    "relay_cycle": "$NEXT_CYCLE",
    "session_id": "${SESSION_ID:-session-main}"
  }
}
JSON
)
    curl -s -X POST \
      -H "Accept: application/vnd.github.v3+json" \
      -H "Authorization: Bearer ${GH_TOKEN}" \
      "https://api.github.com/repos/${GITHUB_REPOSITORY}/actions/workflows/vm-relay.yml/dispatches" \
      -d "$PAYLOAD" || true
    RELAY_TRIGGERED=true
  fi

  if [ "$NOW" -ge "$SHUTDOWN_TIME" ]; then
    cleanup_and_exit
  fi

  sleep 25
done
