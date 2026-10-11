# 🖥️ Cloud Web VM (Linux XFCE) - Auto-Relay 24/7

Machine virtuelle Linux complète avec interface graphique bureau dans le navigateur web, persistance intégrale des données et **relais perpétuel sans coupure** (Blue-Green Deployment).

---

## ✨ Fonctionnalités

- **🖥️ Bureau Complet dans le Navigateur** : Interface fluide Ubuntu XFCE accessible en HTTPS depuis n'importe quel appareil (PC, smartphone, tablette) sans rien installer.
- **🔄 Relais Perpétuel (Blue-Green Relay)** : Contourne la limite des 6 heures de GitHub Actions grâce à un orchestrateur qui lance un runner jumeau à 5h15, synchronise l'état et passe le relais proprement à 5h45.
- **💾 Persistance Automatique** : Sauvegarde continue et incrémentale de tous vos fichiers, téléchargements et configurations vers la branche isolée `data-persist`.
- **🌐 Accès Sécurisé via Cloudflare Tunnel** : Aucune IP publique ni port à ouvrir, trafic sécurisé de bout en bout par Cloudflare Zero Trust.
- **🚀 Outils Préinstallés** : Navigateur Chromium, Terminal Bash, Git, Python 3, Curl, Nano, Htop, etc.

---

## 🛠️ Secrets Requis dans le Dépôt

Ces 3 secrets doivent être présents dans **Settings > Secrets and variables > Actions** :

| Secret | Description |
| :--- | :--- |
| **`GH_TOKEN`** | Personal Access Token GitHub avec permissions `repo` et `workflow` (pour auto-déclencher le runner suivant et sauvegarder l'état). |
| **`CLOUDFLARE_TOKEN`** | Jeton de tunnel Cloudflare Zero Trust (pour router le domaine web vers le port 3000 de la VM). |
| **`CLOUDFLARE_ID`** | Identifiant de votre compte / tunnel Cloudflare. |

---

## 🚀 Comment Démarrer la VM

1. Rendez-vous dans l'onglet **Actions** de ce dépôt.
2. Dans le menu de gauche, sélectionnez le workflow **`Web VM Persistent Relay`**.
3. Cliquez sur **Run workflow** (bouton en haut à droite) et validez.
4. Votre bureau web sera accessible en quelques instants via votre URL Cloudflare (ex: `https://vm.votredomaine.com`) !

---

## ⏱️ Chronologie du Cycle de Relais

```text
0h00 : Démarrage du Runner A -> Restauration des données -> Lancement de Webtop -> Activation Cloudflare
          |
          |  (Utilisation active de la VM & sauvegardes automatiques toutes les 10 min)
          v
5h15 : Le Runner A appelle l'API GitHub pour lancer le Runner B (Runner B s'installe en coulisses)
          |
          |  (30 minutes de chevauchement : vous continuez à utiliser la VM sur le Runner A)
          v
5h45 : Runner B est 100% prêt -> Runner A pousse la dernière sauvegarde -> Arrêt propre de A
          |
          v
5h45+ : Runner B prend le relais instantanément sur le tunnel Cloudflare
```
