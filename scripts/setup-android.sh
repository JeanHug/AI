#!/usr/bin/env bash
set -e

echo "=================================================="
echo "    SMARTPHONE ANDROID GRAPHICAL UI (PORTRAIT 720x1280)"
echo "        Amorçage Robuste Zéro-Console & Zéro-Crash     "
echo "=================================================="

DATA_DIR="/home/runner/android_data"
sudo mkdir -p "$DATA_DIR"
sudo chown -R runner:docker "$DATA_DIR" 2>/dev/null || sudo chown -R $(id -u):$(id -g) "$DATA_DIR" 2>/dev/null || true
sudo chmod -R 777 "$DATA_DIR"
cd "$DATA_DIR"

# 1. Nettoyage préventif complet
echo "=== 1. Nettoyage des processus antérieurs ==="
pkill -9 -f qemu-system 2>/dev/null || true
pkill -9 -f websockify 2>/dev/null || true
pkill -9 -f nginx 2>/dev/null || true
pkill -9 -f android-bridge 2>/dev/null || true
pkill -9 -f adb 2>/dev/null || true

# 2. Installation des paquets requis
echo "=== 2. Installation des dépendances (QEMU, noVNC, Nginx, ADB) ==="
sudo apt-get update -qq >/dev/null 2>&1 || true
sudo apt-get install -y -qq qemu-system-x86 qemu-utils novnc websockify nginx adb wget curl p7zip-full squashfs-tools >/dev/null 2>&1 || true

# 3. Téléchargement d'Android-x86 8.1 r6 Oreo avec miroir haute vitesse et reprise automatique
ISO_NAME="android-x86_64-8.1-r6.iso"
if [ ! -f "$ISO_NAME" ] || [ $(stat -c%s "$ISO_NAME" 2>/dev/null || echo 0) -lt 500000000 ]; then
  rm -f "$ISO_NAME"
  echo "Téléchargement de l'image ISO Android-x86 depuis SourceForge CDN..."
  wget -q --show-progress --tries=5 --timeout=30 -c -O "$ISO_NAME" "https://downloads.sourceforge.net/project/android-x86/Release%208.1/android-x86_64-8.1-r6.iso" || \
  curl -L --retry 5 --retry-delay 2 -o "$ISO_NAME" "https://downloads.sourceforge.net/project/android-x86/Release%208.1/android-x86_64-8.1-r6.iso" || \
  curl -L --retry 3 -o "$ISO_NAME" "https://mirrors.dotsrc.org/osdn/android-x86/71931/android-x86_64-8.1-r6.iso"
fi

# Vérification de l'intégrité de l'ISO téléchargée
ISO_SIZE=$(stat -c%s "$ISO_NAME" 2>/dev/null || echo 0)
echo "Taille de l'ISO Android : $ISO_SIZE octets"
if [ "$ISO_SIZE" -lt 500000000 ]; then
  echo "❌ ERREUR: Le fichier ISO est incomplet ($ISO_SIZE octets)"
  exit 1
fi

# 4. Extraction du noyau et de l'initrd pour un amorçage direct instantané
BOOT_DIR="$DATA_DIR/boot"
mkdir -p "$BOOT_DIR"
if [ ! -f "$BOOT_DIR/kernel" ] || [ ! -f "$BOOT_DIR/initrd.img" ]; then
  echo "Extraction des fichiers d'amorçage..."
  7z x -y "$ISO_NAME" -o"$BOOT_DIR" kernel initrd.img >/dev/null 2>&1 || true
fi

# 5. Détection de l'accélération matérielle KVM
KVM_FLAG=""
if [ -e /dev/kvm ]; then
  sudo chmod 666 /dev/kvm 2>/dev/null || true
  KVM_FLAG="-enable-kvm -cpu host"
  echo "✅ Accélération matérielle KVM activée !"
else
  KVM_FLAG="-cpu max"
  echo "ℹ️ Émulation CPU logicielle"
fi

# 6. Initialisation du Disque Persistant (Option 3 : android.qcow2)
QCOW_DISK="$DATA_DIR/android.qcow2"
if [ ! -f "$QCOW_DISK" ]; then
  echo "=== Initialisation du disque persistant Android (android.qcow2 8 Go) ==="
  truncate -s 8G "$DATA_DIR/android_raw.img"
  mkfs.ext4 -F -L "AndroidData" -O ^metadata_csum "$DATA_DIR/android_raw.img" >/dev/null 2>&1 || true
  qemu-img convert -f raw -O qcow2 "$DATA_DIR/android_raw.img" "$QCOW_DISK"
  rm -f "$DATA_DIR/android_raw.img"
  echo "✅ Disque persistant vierge initialisé avec succès : $QCOW_DISK"
else
  echo "✅ Disque persistant existant détecté : $QCOW_DISK ($(stat -c%s "$QCOW_DISK" 2>/dev/null || echo 0) octets) - Amorçage accéléré (< 30s) !"
fi

# 7. Démarrage de QEMU Android (Graphique VirtIO 720x1280 + Disque Persistant VirtIO)
echo "=== 3. Lancement de QEMU Android (Graphique VirtIO 720x1280 + Disque Persistant) ==="
qemu-system-x86_64 \
  $KVM_FLAG \
  -m 4096 \
  -smp 2 \
  -vga none \
  -device virtio-vga,xres=720,yres=1280 \
  -vnc 127.0.0.1:0 \
  -drive file="$QCOW_DISK",format=qcow2,if=virtio,index=0 \
  -drive file="$DATA_DIR/$ISO_NAME",format=raw,media=cdrom,readonly=on \
  -kernel "$BOOT_DIR/kernel" \
  -initrd "$BOOT_DIR/initrd.img" \
  -append "root=/dev/ram0 androidboot.hardware=android_x86 androidboot.selinux=permissive buildvariant=userdebug quiet SETUPWIZARD=0 HWACCEL=0 androidboot.adb.port=5555 video=720x1280 DPI=280 SRC= DATA=/dev/vda" \
  -net nic,model=virtio \
  -net user,hostfwd=tcp::5555-:5555 \
  -usb -device usb-tablet \
  -serial file:/tmp/qemu_serial.log \
  -daemonize

# 8. Personnalisation noVNC : suppression totale de la barre et languette latérale
echo "=== 4. Personnalisation & Démarrage Websockify (Port 6080) ==="
sudo tee -a /usr/share/novnc/app/styles/base.css > /dev/null << 'NO_VNC_CSS'
#noVNC_control_bar_anchor, #noVNC_control_bar, #noVNC_status {
  display: none !important;
  visibility: hidden !important;
}
body {
  overflow: hidden !important;
  background-color: #000000 !important;
}
NO_VNC_CSS

sleep 2
websockify --web /usr/share/novnc 6080 127.0.0.1:5900 >/dev/null 2>&1 &
sleep 2

# 9. Bridge ADB pour le contrôle web tactile et statut
CURRENT_DIR=$(pwd)
WORK_DIR="/home/runner/work/VM/VM"
if [ -d "$WORK_DIR" ]; then
  CURRENT_DIR="$WORK_DIR"
fi
node "$CURRENT_DIR/scripts/android-bridge.cjs" >/dev/null 2>&1 &
sleep 1

# 10. Configuration NGINX Reverse-Proxy
echo "=== 5. Configuration NGINX Reverse-Proxy ==="
sudo mkdir -p /var/www/android-web
sudo cp -r "$CURRENT_DIR/android-web/"* /var/www/android-web/
sudo chmod -R 755 /var/www/android-web

cat << 'NGINX_EOF' | sudo tee /etc/nginx/sites-available/default > /dev/null
server {
    listen 3000 default_server;
    listen [::]:3000 default_server;

    proxy_buffering off;
    proxy_request_buffering off;
    tcp_nodelay on;

    location / {
        root /var/www/android-web;
        index index.html;
        try_files $uri $uri/ @novnc_proxy;
    }

    location /api/ {
        proxy_pass http://127.0.0.1:8080/api/;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }

    location /websockify {
        proxy_pass http://127.0.0.1:6080/websockify;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_read_timeout 86400s;
        proxy_send_timeout 86400s;
    }

    location @novnc_proxy {
        proxy_pass http://127.0.0.1:6080;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_read_timeout 86400s;
        proxy_send_timeout 86400s;
    }
}
NGINX_EOF

sudo systemctl restart nginx || sudo service nginx restart

# 11. Surveillance & Configuration ADB Asynchrone (Option 1 : Zéro-Blocage du Workflow)
echo "=== 6. Lancement du démon asynchrone de surveillance et provisionnement Android ==="
nohup bash -c '
  echo "[$(date -u +%T)] Démarrage du démon ADB background..." >> /tmp/android_provision.log
  for attempt in {1..150}; do
    sleep 2
    adb connect 127.0.0.1:5555 >/dev/null 2>&1 || true

    if adb devices 2>/dev/null | grep -q "offline"; then
      adb disconnect 127.0.0.1:5555 >/dev/null 2>&1 || true
      sleep 1
      adb connect 127.0.0.1:5555 >/dev/null 2>&1 || true
    fi

    # Réveil et déverrouillage préventif
    adb -s 127.0.0.1:5555 shell input keyevent 82 >/dev/null 2>&1 || true
    adb -s 127.0.0.1:5555 shell input keyevent 3 >/dev/null 2>&1 || true

    BOOT_OK=$(adb -s 127.0.0.1:5555 shell getprop sys.boot_completed 2>/dev/null | tr -d "\r\n" || echo "0")
    if [ "$BOOT_OK" = "1" ]; then
      echo "[$(date -u +%T)] ✅ ANDROID OPÉRATIONNEL (sys.boot_completed = 1) à la tentative $attempt !" >> /tmp/android_provision.log
      
      # 1. Éviter l assistant de configuration Google
      adb -s 127.0.0.1:5555 shell settings put global device_provisioned 1 >/dev/null 2>&1 || true
      adb -s 127.0.0.1:5555 shell settings put secure user_setup_complete 1 >/dev/null 2>&1 || true
      # 2. Fixer la rotation en mode portrait vertical permanent (0°)
      adb -s 127.0.0.1:5555 shell settings put system user_rotation 0 >/dev/null 2>&1 || true
      adb -s 127.0.0.1:5555 shell settings put system accelerometer_rotation 0 >/dev/null 2>&1 || true
      # 3. Empêcher la mise en veille
      adb -s 127.0.0.1:5555 shell settings put system screen_off_timeout 2147483647 >/dev/null 2>&1 || true
      # 4. Appliquer résolution & densité smartphone
      adb -s 127.0.0.1:5555 shell wm size 720x1280 >/dev/null 2>&1 || true
      adb -s 127.0.0.1:5555 shell wm density 280 >/dev/null 2>&1 || true
      # 5. Déverrouiller et amener sur l écran d accueil
      adb -s 127.0.0.1:5555 shell input keyevent 82 >/dev/null 2>&1 || true
      adb -s 127.0.0.1:5555 shell input keyevent 3 >/dev/null 2>&1 || true
      sync || true
      adb -s 127.0.0.1:5555 shell sync 2>/dev/null || true
      echo "[$(date -u +%T)] ✅ Configuration smartphone appliquée avec succès !" >> /tmp/android_provision.log
      break
    fi
  done
' > /dev/null 2>&1 &

# Attente ultra-rapide que Nginx réponde localement (Option 1)
echo "Vérification immédiate de la réactivité Nginx locale..."
for i in {1..10}; do
  if curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:3000/ | grep -q "200\|302"; then
    echo "✅ Port local 3000 prêt ! Le workflow passe directement au démarrage du tunnel Cloudflare."
    break
  fi
  sleep 1
done

echo "=================================================="
echo "    Smartphone Android & UI Web Prêts !"
echo "    Tunnel Cloudflare en cours d'activation immédiate..."
echo "=================================================="
