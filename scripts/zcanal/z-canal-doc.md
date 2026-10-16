# z-canal — le canal « __NAME__ »

**Identifiant du canal :** `__ID__` · **mode :** `__MODE__` · **adresse :** `__BASE__/__ID__`

Un canal relie un agent IA — n'importe lequel — à l'application AI Agent. L'utilisateur
écrit dans l'application ; l'agent reçoit ses messages par ce canal, peut répondre, et
si le mode est `agent`, exécuter des commandes sur la VM et renvoyer leur résultat.

## Les quatre adresses

| Adresse | Ce qu'elle fait |
|---|---|
| `__BASE__/__ID__/write` | l'agent écrit un message (POST `{"text": "...", "kind": "text"|"call"}`) |
| `__BASE__/__ID__/read?n=30` | l'historique récent, sans rien marquer |
| `__BASE__/__ID__/wait?timeout=25` | **les messages non lus**, qu'elle marque « lu » |
| `__BASE__/__ID__/state` | l'état du canal (nom, mode, nombre de messages, non lus) |

`__BASE__` seul (sans identifiant) affiche la documentation complète.
`__BASE__/__ID__/install` donne les instructions d'installation.

## Ce qu'il faut savoir

- **Aucun mot de passe.** L'identifiant est la clé : ne le partagez pas.
- **Un message ne revient jamais deux fois.** Dès qu'un `/wait` le rend, le canal le
  marque « lu » — et l'application affiche « lu ! » à la place de « non lu ».
- **C'est votre script qui gère l'attente**, pas le serveur : le serveur ne bloque pas
  plus de 25 secondes. La boucle recommandée est `wait` → rien ? pause de 5 s → `wait`.
  N'attendez pas en boucle serrée : le plan gratuit de la plateforme accorde
  100 000 requêtes par jour, et une attente toutes les 10 secondes en consomme moins de 10 %.
- **Les commandes ne s'exécutent que si l'application est ouverte** : c'est elle qui
  parle à la VM. Si elle est fermée, le message attend dans le canal.
- Pour agir sur la VM, écrivez un `kind` **`call`** contenant un objet JSON :
  `{"action":"bash","command":"ls -la ~"}` ou `{"action":"fetch","url":"https://…"}`.
  Le résultat revient dans le canal en tant que message `tool`.
- Pour simplement parler, écrivez un `kind` **`text`** : il s'affiche comme un message
  de l'agent dans l'application.
