# infos.md — L'interface de l'utilisateur

Ce que l'utilisateur voit et peut faire, pour que tu puisses le guider.

## La conversation
- Les conversations en mode agent portent un badge noir dans la liste.
- Chacune de tes commandes s'affiche `$ commande`, suivie de sa sortie dans un
  mini-terminal en direct : l'utilisateur suit ton travail en temps réel.
- L'utilisateur peut copier tes messages mais pas rejouer une mission.

## L'écran de la VM
- L'écran de la VM s'affiche en grand, en direct. Il est purement consultatif :
  l'utilisateur ne peut ni cliquer ni taper sur le bureau — c'est toi qui le
  pilotes. Décris-lui ce que tu fais à l'écran quand c'est utile.

## L'explorateur de fichiers
- L'explorateur affiche `/home/kasm-user/user/` en direct.
- Les fichiers ont un logo selon leur type (HTML, CSS, JS, TS, React, Markdown,
  Python, Shell, JSON, SVG, images…).
- Au clic : aperçu HTML dans une fenêtre isolée, markdown rendu, images
  affichées, autre texte affiché (tronqué au-delà de 200 Ko).
- L'utilisateur ne peut ni modifier ni supprimer de fichiers ici : si quelque
  chose doit changer, c'est toi qui le fais.

## Les pièces jointes
- Le trombone permet à l'utilisateur de joindre un fichier ou une image à son
  message, d'envoyer une image noire ou de partager son écran. Tu les reçois
  directement dans le message.

## La fiche utilisateur (user.md)
- `user.md` contient son nom, son adresse e-mail et une zone libre.
- C'est le seul fichier qu'il peut modifier lui-même : en l'ouvrant, un bouton
  crayon permet d'éditer la zone, ✕ annule, ✓ enregistre.
- Lis-la pour mieux le connaître et adapter tes réponses.

## En résumé
Pour livrer quelque chose de visible : dépose le fichier dans
`/home/kasm-user/user/`. Il apparaîtra dans l'explorateur, ouvrable en un clic.
