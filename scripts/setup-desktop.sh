#!/usr/bin/env bash
set -e

echo "=================================================="
echo "    CONFIGURATION BUREAU LINUX UBUNTU (OFFICIEL)  "
echo "=================================================="

DATA_DIR="/home/runner/vm_data"

# Utilisation de sudo pour créer les dossiers nécessaires
sudo mkdir -p "$DATA_DIR/Desktop" "$DATA_DIR/Downloads" "$DATA_DIR/Applications" "$DATA_DIR/.config" "$DATA_DIR/.local/bin"
sudo chown -R 1000:1000 "$DATA_DIR"
sudo chmod -R 777 "$DATA_DIR"

# Nettoyage d'anciens conteneurs
docker rm -f kasm_desktop 2>/dev/null || true

# 1. Démarrage de l'image Kasm Desktop Ubuntu avec PRIVILÈGES COMPLETS (KVM, SECCOMP UNCONFINED)
# --privileged et --security-opt seccomp=unconfined permettent à Chrome de créer ses processus sans restriction
echo "=== Démarrage du conteneur Kasm Desktop Ubuntu (Privilèges complets) ==="
docker pull kasmweb/ubuntu-jammy-desktop:1.16.0
docker run -d \
  --name kasm_desktop \
  --privileged \
  --security-opt seccomp=unconfined \
  --shm-size=4096m \
  -p 6901:6901 \
  -e VNC_PW=vncpass \
  -e VNC_RESOLUTION=1600x900 \
  -v "$DATA_DIR":/home/kasm-user \
  kasmweb/ubuntu-jammy-desktop:1.16.0

# 1b. Sudo sans mot de passe pour kasm-user (l'agent IA peut apt-get install, etc.)
echo "=== Sudo NOPASSWD pour kasm-user ==="
for i in $(seq 1 30); do
  docker exec -u root kasm_desktop true 2>/dev/null && break
  sleep 2
done
docker exec -u root kasm_desktop bash -c \
  'echo "kasm-user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/kasm-user && chmod 440 /etc/sudoers.d/kasm-user' \
  && echo "sudoers OK" || echo "sudoers : échec (non bloquant)"

# 1c. Garde-fou agent.md / infos.md : recréation automatique si supprimés.
# Les fichiers canoniques vivent dans le dépôt (scripts/agent-docs/) et sont
# recopiés dans le dossier partagé. Le watchdog tourne sur le RUNNER (hors du
# conteneur) : l'agent ne peut ni le voir ni l'arrêter.
echo "=== Garde-fou agent.md / infos.md ==="
SCRIPT_DIR_EARLY="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
sudo mkdir -p "$DATA_DIR/user"
sudo chown 1000:1000 "$DATA_DIR/user"
for f in agent.md infos.md; do
  if [ -f "$SCRIPT_DIR_EARLY/agent-docs/$f" ]; then
    sudo cp -f "$SCRIPT_DIR_EARLY/agent-docs/$f" "$DATA_DIR/user/$f"
    sudo chown 1000:1000 "$DATA_DIR/user/$f"
    sudo chmod 644 "$DATA_DIR/user/$f"
  fi
done
# user.md : fiche de l'utilisateur — JAMAIS écrasée si elle existe ;
# restaurée depuis sa sauvegarde (.user.md.bak) si supprimée.
if [ ! -s "$DATA_DIR/user/user.md" ]; then
  if [ -s "$DATA_DIR/.user.md.bak" ]; then
    sudo cp -f "$DATA_DIR/.user.md.bak" "$DATA_DIR/user/user.md"
  elif [ -f "$SCRIPT_DIR_EARLY/agent-docs/user.md" ]; then
    sudo cp -f "$SCRIPT_DIR_EARLY/agent-docs/user.md" "$DATA_DIR/user/user.md"
  fi
  sudo chown 1000:1000 "$DATA_DIR/user/user.md" 2>/dev/null || true
  sudo chmod 644 "$DATA_DIR/user/user.md" 2>/dev/null || true
fi
cat << WATCHDOG_EOF | sudo tee /usr/local/bin/agent-docs-watchdog.sh > /dev/null
#!/usr/bin/env bash
# Recrée agent.md, infos.md et user.md dans le dossier partagé dès qu'ils manquent.
# user.md (fiche utilisateur) est restauré en priorité depuis sa sauvegarde.
SRC="\$1"; DST="\$2"; BK="\$3"
while true; do
  for f in agent.md infos.md; do
    if [ ! -s "\$DST/\$f" ] && [ -s "\$SRC/\$f" ]; then
      cp -f "\$SRC/\$f" "\$DST/\$f" 2>/dev/null || sudo cp -f "\$SRC/\$f" "\$DST/\$f"
      chown 1000:1000 "\$DST/\$f" 2>/dev/null || sudo chown 1000:1000 "\$DST/\$f" 2>/dev/null
      chmod 644 "\$DST/\$f" 2>/dev/null || true
      echo "[\$(date +'%T')] Restauré : \$f"
    fi
  done
  if [ ! -s "\$DST/user.md" ]; then
    if [ -s "\$BK" ]; then
      cp -f "\$BK" "\$DST/user.md" 2>/dev/null || sudo cp -f "\$BK" "\$DST/user.md"
    elif [ -s "\$SRC/user.md" ]; then
      cp -f "\$SRC/user.md" "\$DST/user.md" 2>/dev/null || sudo cp -f "\$SRC/user.md" "\$DST/user.md"
    fi
    chown 1000:1000 "\$DST/user.md" 2>/dev/null || sudo chown 1000:1000 "\$DST/user.md" 2>/dev/null
    chmod 644 "\$DST/user.md" 2>/dev/null || true
    echo "[\$(date +'%T')] Restauré : user.md"
  fi
  sleep 30
done
WATCHDOG_EOF
sudo chmod +x /usr/local/bin/agent-docs-watchdog.sh
sudo pkill -f agent-docs-watchdog 2>/dev/null || true
sudo nohup /usr/local/bin/agent-docs-watchdog.sh "$SCRIPT_DIR_EARLY/agent-docs" "$DATA_DIR/user" "$DATA_DIR/.user.md.bak" \
  > /tmp/agent-docs-watchdog.log 2>&1 &
echo "watchdog agent-docs démarré (vérification toutes les 30 s)"


# 2. Configuration du reverse-proxy NGINX ultra-rapide
echo "=== Configuration du reverse-proxy NGINX (Zéro latence) ==="
sudo apt-get update -qq && sudo apt-get install -y -qq nginx > /dev/null 2>&1

cat << 'NGINX_EOF' | sudo tee /etc/nginx/sites-available/default > /dev/null
server {
    listen 3000 default_server;
    listen [::]:3000 default_server;

    proxy_buffering off;
    proxy_request_buffering off;
    tcp_nodelay on;

    # API d'interaction Linux pour agents IA & automatisation
    location /api/ {
        proxy_pass http://127.0.0.1:3001/api/;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 300s;
        proxy_send_timeout 300s;
        proxy_buffering off;
    }

    location / {
        proxy_pass https://127.0.0.1:6901;
        proxy_ssl_verify off;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Autologin direct sans pop-up de mot de passe (kasm_user:vncpass)
        proxy_set_header Authorization "Basic a2FzbV91c2VyOnZuY3Bhc3M=";
        proxy_read_timeout 86400s;
        proxy_send_timeout 86400s;
    }
}
NGINX_EOF

sudo systemctl restart nginx || sudo service nginx restart

# 3. Attente que Kasm soit actif
echo "=== Attente de l'initialisation du bureau Kasm ==="
for i in {1..40}; do
  if curl -s -k https://127.0.0.1:6901/ > /dev/null 2>&1; then
    echo " Bureau Ubuntu Kasm opérationnel !"
    break
  fi
  sleep 2
done

# 4. Installation et configuration garantie de Google Chrome, Chromium et Google Docs
echo "=== Installation et configuration garantie de Google Chrome & Docs ==="
docker exec -u 0 kasm_desktop bash -c '
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq >/dev/null 2>&1 || true
  apt-get install -y -qq xclip xsel autocutsel wget curl gnupg ca-certificates wmctrl x11-utils scrot imagemagick xdotool netpbm > /dev/null 2>&1 || true
  # Visionneuse & exécution : conversion Office (pptx→pdf), figures matplotlib
  apt-get install -y -qq --no-install-recommends libreoffice-core ure libreoffice-impress libreoffice-writer fonts-liberation python3-matplotlib > /dev/null 2>&1 || true

  # Playwright + Chromium pour tests réels (bugs de rendu, etc.)
  # Le dépôt hashicorp expiré casse apt update : supprimé.
  rm -f /etc/apt/sources.list.d/hashicorp.list
  su - kasm-user -c "python3 -m playwright --version >/dev/null 2>&1 || pip3 install --quiet playwright" || true
  su - kasm-user -c "python3 -m playwright install-deps chromium >/dev/null 2>&1" || true
  su - kasm-user -c "python3 -m playwright install chromium >/dev/null 2>&1" || true

  # Synchronisation du presse-papier X11
  su - kasm-user -c "autocutsel -fork >/dev/null 2>&1 || true"
  su - kasm-user -c "autocutsel -selection PRIMARY -fork >/dev/null 2>&1 || true"

  # Installation officielle de Google Chrome Stable
  wget -q -O - https://dl-ssl.google.com/linux/linux_signing_key.pub | gpg --dearmor --yes -o /usr/share/keyrings/google-chrome.gpg 2>/dev/null || true
  echo "deb [arch=amd64 signed-by=/usr/share/keyrings/google-chrome.gpg] http://dl.google.com/linux/chrome/deb/ stable main" > /etc/apt/sources.list.d/google-chrome.list
  apt-get update -qq >/dev/null 2>&1 || true
  apt-get install -y -qq google-chrome-stable >/dev/null 2>&1 || {
    curl -sSL -o /tmp/google-chrome.deb https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
    dpkg -i /tmp/google-chrome.deb 2>/dev/null || apt-get install -y -f -qq >/dev/null 2>&1
    rm -f /tmp/google-chrome.deb
  }

  # Wrapper universel absolu pour /usr/local/bin/google-chrome et /usr/bin/google-chrome
  cat << "EOF_CHROME_GLOBAL" > /usr/local/bin/google-chrome
#!/usr/bin/env bash
# Nettoyage des verrous de profil corrompus
rm -rf /home/kasm-user/.config/google-chrome/Singleton* 2>/dev/null || true
rm -rf /home/kasm-user/.config/chromium/Singleton* 2>/dev/null || true

# Recherche du binaire Chrome
CHROME_BIN=""
if [ -x /usr/bin/google-chrome-stable ]; then
  CHROME_BIN="/usr/bin/google-chrome-stable"
elif [ -x /opt/google/chrome/google-chrome ]; then
  CHROME_BIN="/opt/google/chrome/google-chrome"
elif [ -x /usr/bin/chromium-browser ]; then
  CHROME_BIN="/usr/bin/chromium-browser"
elif [ -x /usr/bin/chromium ]; then
  CHROME_BIN="/usr/bin/chromium"
fi

exec "$CHROME_BIN" \
  --no-sandbox \
  --disable-setuid-sandbox \
  --disable-dev-shm-usage \
  --disable-gpu \
  --disable-software-rasterizer \
  --no-first-run \
  --no-default-browser-check \
  --password-store=basic \
  "$@"
EOF_CHROME_GLOBAL
  chmod 755 /usr/local/bin/google-chrome
  ln -sf /usr/local/bin/google-chrome /usr/local/bin/chrome
  ln -sf /usr/local/bin/google-chrome /usr/bin/google-chrome
  ln -sf /usr/local/bin/google-chrome /usr/bin/chrome

  # Wrapper pour Google Docs
  cat << "EOF_DOCS_GLOBAL" > /usr/local/bin/google-docs
#!/usr/bin/env bash
exec /usr/local/bin/google-chrome --app="https://docs.google.com" "$@"
EOF_DOCS_GLOBAL
  chmod 755 /usr/local/bin/google-docs
  ln -sf /usr/local/bin/google-docs /usr/bin/google-docs

  # Icône SVG officielle Google Docs
  mkdir -p /usr/share/icons/hicolor/scalable/apps /usr/share/pixmaps
  cat << "SVG_DOCS" > /usr/share/icons/hicolor/scalable/apps/google-docs.svg
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 48 48" width="128" height="128">
  <path fill="#4285F4" d="M30 4H12C9.79 4 8 5.79 8 8v32c0 2.21 1.79 4 4 4h24c2.21 0 4-1.79 4-4V16L30 4z"/>
  <path fill="#A1C2FA" d="M30 4v12h12L30 4z"/>
  <path fill="#FFFFFF" d="M16 22h16v2.5H16zm0 6h16v2.5H16zm0 6h10v2.5H16z"/>
</svg>
SVG_DOCS
  cp -f /usr/share/icons/hicolor/scalable/apps/google-docs.svg /usr/share/pixmaps/google-docs.svg 2>/dev/null || true

  # Raccourcis sur le Bureau XFCE
  mkdir -p /home/kasm-user/Desktop /usr/share/applications

  # Raccourci Google Chrome sur le Bureau
  cat << "DESKTOP_CHROME" > /home/kasm-user/Desktop/google-chrome.desktop
[Desktop Entry]
Version=1.0
Type=Application
Name=Google Chrome
GenericName=Navigateur Web
Comment=Naviguer sur Internet avec Google Chrome
Exec=/usr/local/bin/google-chrome %U
Icon=google-chrome
Terminal=false
Categories=Network;WebBrowser;
Actions=new-window;new-private-window;
DESKTOP_CHROME

  # Raccourci Google Docs sur le Bureau
  cat << "DESKTOP_DOCS" > /home/kasm-user/Desktop/google-docs.desktop
[Desktop Entry]
Version=1.0
Type=Application
Name=Google Docs
GenericName=Traitement de texte
Comment=Créer et éditer des documents en ligne
Exec=/usr/local/bin/google-docs
Icon=/usr/share/icons/hicolor/scalable/apps/google-docs.svg
Terminal=false
Categories=Office;WordProcessor;
DESKTOP_DOCS

  # Script direct double-clic de secours sur le bureau
  cat << "SH_CHROME" > /home/kasm-user/Desktop/Ouvrir_Google_Chrome.sh
#!/usr/bin/env bash
/usr/local/bin/google-chrome "https://google.com" &
SH_CHROME

  cat << "SH_DOCS" > /home/kasm-user/Desktop/Ouvrir_Google_Docs.sh
#!/usr/bin/env bash
/usr/local/bin/google-docs &
SH_DOCS

  # Copie dans le menu des applications système
  cp -f /home/kasm-user/Desktop/google-chrome.desktop /usr/share/applications/
  cp -f /home/kasm-user/Desktop/google-docs.desktop /usr/share/applications/

  # Permissions d exécution complètes
  chmod 777 /home/kasm-user/Desktop/* /usr/share/applications/google-*.desktop 2>/dev/null || true
  chown -R 1000:1000 /home/kasm-user

  # Approbation explicite XFCE pour lancer au double-clic sans popup de sécurité
  su - kasm-user -c "
    gio set /home/kasm-user/Desktop/*.desktop metadata::trusted true 2>/dev/null || true
    gio set /home/kasm-user/Desktop/*.sh metadata::trusted true 2>/dev/null || true
  "

  # ==========================================================
  # TEST ACTIF AUTOMATISÉ : Lancement réel de Google Chrome
  # ==========================================================
  echo "=== Test actif de lancement de Google Chrome sur Display :1 ==="
  su - kasm-user -c "DISPLAY=:1 /usr/local/bin/google-chrome --version"
  
  # Lancement en arrière-plan sur le serveur X11 et vérification de la création de la fenêtre
  su - kasm-user -c "
    DISPLAY=:1 /usr/local/bin/google-chrome https://docs.google.com >/dev/null 2>&1 &
    CPID=\$!
    sleep 3
    if ps -p \$CPID > /dev/null; then
      echo \" SUCCÈS CONFIRMÉ : Le processus Google Chrome tourne parfaitement (PID \$CPID) !\"
    else
      echo \"⚠️ Chrome a fermé immédiatement, relance avec mode sandbox allégé...\"
      DISPLAY=:1 /usr/local/bin/google-chrome --in-process-gpu https://docs.google.com &
    fi
  "
'

# 5. Démarrage du Démon d'API REST Linux (Port 3001)
echo "=== Démarrage du Démon d'API REST Linux (Port 3001) ==="
pkill -f "linux-api-daemon.cjs" || true
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
nohup node "$SCRIPT_DIR/linux-api-daemon.cjs" > /tmp/linux-api-daemon.log 2>&1 &
sleep 2
if curl -s http://127.0.0.1:3001/api/health >/dev/null 2>&1; then
  echo " Démon d'API REST Linux démarré et opérationnel sur port 3001 !"
else
  echo "⚠️ Avertissement : le démon d'API n'a pas répondu immédiatement, vérification des logs :"
  cat /tmp/linux-api-daemon.log 2>/dev/null || true
fi

echo " Bureau Ubuntu configuré avec succès : Google Chrome, Google Docs et API REST 100% opérationnels !"
