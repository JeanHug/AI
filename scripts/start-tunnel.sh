#!/usr/bin/env bash
set -e

echo "=================================================="
echo "   DÉMARRAGE DU TUNNEL CLOUDFLARE POUR VM LINUX   "
echo "=================================================="

# 1. Installation de Cloudflared si manquant
if ! command -v cloudflared &>/dev/null; then
  echo "Installation de Cloudflared..."
  sudo curl -L --output /usr/local/bin/cloudflared https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
  sudo chmod +x /usr/local/bin/cloudflared
fi

# Nettoyage des anciens processus
pkill -f cloudflared || true
pkill -f "nokey@localhost.run" || true

# 2. Lancement du tunnel Cloudflare (priorité absolue)
echo "Lancement du Quick Tunnel Cloudflare..."
cloudflared tunnel --url http://127.0.0.1:3000 --no-autoupdate > /tmp/quick_tunnel_linux.log 2>&1 &

# 3. Lancement de localhost.run comme secours
echo "Lancement du tunnel localhost.run..."
ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ServerAliveInterval=15 -R 80:localhost:3000 nokey@localhost.run > /tmp/localhost_run_linux.log 2>&1 &

# 4. Récupération des URLs
RAW_CF_URL=""
for i in {1..40}; do
  RAW_CF_URL=$(grep -o 'https://[-a-zA-Z0-9_.]*\.trycloudflare\.com' /tmp/quick_tunnel_linux.log | head -n1 || true)
  if [ -n "$RAW_CF_URL" ]; then
    break
  fi
  sleep 1
done

RAW_LHR_URL=""
for i in {1..20}; do
  RAW_LHR_URL=$(grep -o 'https://[-a-zA-Z0-9_.]*\.lhr\.life' /tmp/localhost_run_linux.log | head -n1 || true)
  if [ -n "$RAW_LHR_URL" ]; then
    break
  fi
  sleep 1
done

PRIMARY_URL="$RAW_CF_URL"
if [ -z "$PRIMARY_URL" ]; then
  PRIMARY_URL="$RAW_LHR_URL"
fi

echo "=================================================="
echo "🎯 URL CLOUDFLARE LINUX      : $RAW_CF_URL"
echo "🎯 URL LOCALHOST.RUN LINUX   : $RAW_LHR_URL"
echo "🎯 URL PRINCIPALE ACTIVE     : $PRIMARY_URL"
echo "=================================================="

# 5. Publication de l'URL sur la branche tunnel-url
if [ -n "$GH_TOKEN" ] && [ -n "$GITHUB_REPOSITORY" ] && [ -n "$PRIMARY_URL" ]; then
  TMP_URL_REPO=$(mktemp -d)
  cd "$TMP_URL_REPO"
  git init
  git config user.name "VM-Tunnel-Auto"
  git config user.email "bot@vm.tunnel.local"
  git remote add origin "https://x-access-token:${GH_TOKEN}@github.com/${GITHUB_REPOSITORY}.git"
  git checkout -B tunnel-url
  
  git pull origin tunnel-url --rebase 2>/dev/null || true

  echo "$PRIMARY_URL" > current_url.txt
  echo "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" > updated_at.txt
  
  cat << JSON > tunnels.json
{
  "primary": "$PRIMARY_URL",
  "cloudflare": "$RAW_CF_URL",
  "lhrLife": "$RAW_LHR_URL",
  "status": "online",
  "updated_at": "$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
}
JSON

  cat << JSON > tunnels-linux.json
{
  "name": "Ubuntu Linux Desktop",
  "primary": "$PRIMARY_URL",
  "cloudflare": "$RAW_CF_URL",
  "lhrLife": "$RAW_LHR_URL",
  "status": "online",
  "updated_at": "$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
}
JSON

  git add current_url.txt updated_at.txt tunnels.json tunnels-linux.json
  git commit -m "chore(tunnel): active Cloudflare Linux URL [$PRIMARY_URL]" || true
  git push --force origin tunnel-url 2>&1 | sed 's/'"$GH_TOKEN"'/REDACTED/g' || true
  cd /
  rm -rf "$TMP_URL_REPO"
  echo " URL Cloudflare Linux publiée avec succès sur tunnel-url !"
fi
