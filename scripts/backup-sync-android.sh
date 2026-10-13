#!/usr/bin/env bash
set -e

ACTION="${1:-backup}"
SESSION_ID="${2:-session-android}"
DATA_DIR="/home/runner/android_data"
BRANCH="vm-data-android"
QCOW_DISK="$DATA_DIR/android.qcow2"
ARCHIVE_NAME="android.qcow2.zst"

if [ -z "$GH_TOKEN" ] || [ -z "$GITHUB_REPOSITORY" ]; then
  echo "⚠️ Identifiants GitHub manquants, synchronisation ignorée."
  exit 0
fi

sudo mkdir -p "$DATA_DIR"
sudo chmod -R 777 "$DATA_DIR" 2>/dev/null || true

TMP_REPO=$(mktemp -d)

case "$ACTION" in
  restore)
    echo "=================================================="
    echo "   RESTAURATION DISQUE PERSISTANT ANDROID 14      "
    echo "=================================================="
    cd "$TMP_REPO"
    git init -q
    git config user.name "VM-Android-Sync"
    git config user.email "bot@vm.android.local"
    git remote add origin "https://x-access-token:${GH_TOKEN}@github.com/${GITHUB_REPOSITORY}.git"

    if git fetch origin "$BRANCH" --depth=1 2>/dev/null; then
      git checkout -B "$BRANCH" origin/"$BRANCH"
      sudo apt-get update -qq >/dev/null 2>&1 && sudo apt-get install -y -qq zstd tar >/dev/null 2>&1 || true

      if ls ${ARCHIVE_NAME}.part_* 1>/dev/null 2>&1; then
        echo " Reconstitution du disque persistant Android (${ARCHIVE_NAME}) depuis les fragments..."
        cat ${ARCHIVE_NAME}.part_* > "/tmp/$ARCHIVE_NAME"
        echo " Décompression zstd du disque virtuel vers $QCOW_DISK..."
        zstd -d -f "/tmp/$ARCHIVE_NAME" -o "$QCOW_DISK" 2>/dev/null || true
        rm -f "/tmp/$ARCHIVE_NAME"
        echo "✅ Disque persistant $QCOW_DISK restauré avec succès !"
      elif [ -f "$ARCHIVE_NAME" ]; then
        echo " Décompression zstd du disque virtuel vers $QCOW_DISK..."
        zstd -d -f "$ARCHIVE_NAME" -o "$QCOW_DISK" 2>/dev/null || true
        echo "✅ Disque persistant $QCOW_DISK restauré avec succès !"
      elif [ -f "android-data.tar.zst" ]; then
        echo " Archive héritée trouvée. Décompression..."
        sudo tar -I zstd -xf "android-data.tar.zst" -C "$DATA_DIR" 2>/dev/null || true
      else
        echo "ℹ️ Aucun disque persistant archivé sur la branche $BRANCH (première session)."
      fi
    else
      echo "ℹ️ Branche $BRANCH inexistante. Initialisation d'une nouvelle session Android 14."
    fi
    cd /
    rm -rf "$TMP_REPO"
    ;;

  backup)
    echo "=================================================="
    echo "   SAUVEGARDE DU DISQUE PERSISTANT ANDROID 14    "
    echo "=================================================="
    if [ ! -f "$QCOW_DISK" ]; then
      echo "ℹ️ Aucun fichier disque virtuel $QCOW_DISK à sauvegarder."
      exit 0
    fi

    # Flush des écritures sur le disque virtuel
    sync || true
    adb -s 127.0.0.1:5555 shell sync 2>/dev/null || true

    sudo apt-get update -qq >/dev/null 2>&1 && sudo apt-get install -y -qq zstd tar >/dev/null 2>&1 || true

    ARCHIVE_PATH="/tmp/$ARCHIVE_NAME"
    echo "Compression zstd rapide du disque persistant $QCOW_DISK..."
    zstd -3 -T0 -f "$QCOW_DISK" -o "$ARCHIVE_PATH"

    cd "$TMP_REPO"
    git init -q
    git config user.name "VM-Android-Sync"
    git config user.email "bot@vm.android.local"
    git remote add origin "https://x-access-token:${GH_TOKEN}@github.com/${GITHUB_REPOSITORY}.git"
    git checkout -B "$BRANCH"

    git pull origin "$BRANCH" --rebase 2>/dev/null || true

    # Nettoyage des anciens fragments
    rm -f ${ARCHIVE_NAME}* android-data.tar.zst 2>/dev/null || true

    ARCHIVE_SIZE=$(stat -c%s "$ARCHIVE_PATH" 2>/dev/null || echo 0)
    echo "Taille du disque compressé : $ARCHIVE_SIZE octets"

    # Découpage si > 50 Mo pour respecter la limite absolue GitHub de 100 Mo par fichier
    if [ "$ARCHIVE_SIZE" -gt 50000000 ]; then
      echo "Découpage en fragments de 45 Mo pour compatibilité GitHub..."
      split -b 45M "$ARCHIVE_PATH" "${ARCHIVE_NAME}.part_"
      git add ${ARCHIVE_NAME}.part_*
    else
      mv -f "$ARCHIVE_PATH" "$ARCHIVE_NAME"
      git add "$ARCHIVE_NAME"
    fi
    rm -f "$ARCHIVE_PATH"

    echo "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" > last_sync_android.txt
    git add last_sync_android.txt

    if ! git diff --staged --quiet; then
      git commit -m "chore(android-disk): persist android.qcow2 [$(date -u +'%Y-%m-%d %H:%M:%S UTC')]"
      git push --force origin "$BRANCH" 2>&1 | sed 's/'"$GH_TOKEN"'/REDACTED/g'
      echo "✅ Disque persistant Android synchronisé et sécurisé sur la branche $BRANCH !"
    else
      echo "ℹ️ Aucun changement dans le disque persistant Android."
    fi
    cd /
    rm -rf "$TMP_REPO"
    ;;

  *)
    echo "Usage: $0 {restore|backup} [session_id]"
    exit 1
    ;;
esac
