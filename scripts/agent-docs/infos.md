# infos.md — L'interface de l'utilisateur

Ce fichier décrit précisément ce que l'utilisateur voit et peut faire, pour
que tu puisses le guider sans deviner. Quand tu lui dis « cliquez ici » ou
« ouvrez ce fichier », appuie-toi sur ces descriptions.

## 1. La conversation

- L'application est un chat. Chaque discussion apparaît dans la liste de
  gauche, avec son titre et la date.
- Une discussion est **soit un chat classique, soit une conversation agent** :
  le choix se fait au tout début, sur une conversation vierge. Les
  conversations agent portent un **badge noir « Agent »** dans la liste.
- Tes réponses sont rendues en markdown : titres, listes, tableaux, blocs de
  code, formules mathématiques LaTeX (`$...$` et `$$...$$`) sont affichés
  correctement.
- Sous chacun de tes messages, l'utilisateur a un bouton **copier**. En mode
  agent, il n'y a **pas** de bouton « recommencer » : une mission ne se rejoue
  pas, il en redemande une nouvelle.
- Les messages et, en mode agent, **les commandes du terminal** sont
  enregistrés dans son compte : en rouvrant la discussion plus tard, il
  retrouve tout, y compris le détail des commandes et de leur sortie.

## 2. Le mode agent

- Quand il t'écrit en mode agent, un panneau s'ouvre avec **l'écran de la VM
  en direct** et **l'explorateur de fichiers**.
- Chacune de tes commandes s'affiche dans un mini-terminal : `$ ta-commande`,
  puis sa sortie, en direct. Il suit donc ton travail étape par étape et peut
  voir à quel moment ça bloque.
- Le terminal se replie à la fin de la mission ; il peut le rouvrir.
- S'il joint une capture d'écran à son message, tu la reçois avec la demande.

## 3. L'écran de la VM

- L'écran du bureau Ubuntu s'affiche en grand, rafraîchi toutes les 3 secondes.
- Il est **purement consultatif** : l'utilisateur ne peut ni cliquer ni taper
  sur le bureau. C'est toi qui pilotes.
- Conséquence pratique : s'il faut cliquer quelque part, ouvrir un navigateur,
  remplir un formulaire, c'est à toi de le faire (xdotool, wmctrl, scripts).
  Décris-lui ce qui se passe à l'écran quand c'est utile.

## 4. L'explorateur de fichiers

- Il montre le contenu de `/home/kasm-user/user/` en direct, avec les
  sous-dossiers dépliables.
- Chaque fichier a une icône selon son type : dossiers, HTML, CSS, JS, TS,
  React, Markdown, Python, Shell, JSON, SVG, images, vidéos, audio, PDF,
  Word, PowerPoint, Excel, LaTeX, et une icône générique en repli.
- **Un clic sur un fichier l'ouvre dans la visionneuse.** C'est le seul moyen
  pour lui de voir tes livrables : tout ce qui doit être visible doit être
  dans ce dossier.
- Il ne peut ni supprimer ni renommer de fichiers ici. S'il faut nettoyer,
  c'est toi qui le fais.

## 5. La visionneuse — ce qu'il peut voir, format par format

En haut de la visionneuse : le nom du fichier, des **boutons d'action** qui
changent selon le type, et une croix pour fermer (la touche Échap marche aussi).

- **Images** (png, jpg, jpeg, gif, webp, bmp, avif, ico) : affichées à leur
  **taille d'origine, sans redimensionnement** — la zone défile si l'image est
  grande. Les GIF animés s'animent. Bouton **Télécharger**.
- **SVG** : **exécuté**, donc les animations et scripts internes fonctionnent,
  dans une fenêtre isolée. Boutons **Voir le code** / **Aperçu** et **Copier**.
- **Vidéos** (mp4, webm, ogv, mov, m4v) : lecteur intégré avec contrôles
  (lecture, pause, volume, plein écran, barre de progression), à la résolution
  d'origine. Bouton **Télécharger**.
- **Audio** (mp3, wav, ogg, m4a, flac, aac, opus) : lecteur audio avec
  contrôles. Bouton **Télécharger**.
- **PDF** : visionneuse PDF intégrée (pages, zoom, défilement). Bouton
  **Télécharger**.
- **Word (.docx)** : le document est rendu en texte et mise en page dans la
  page. Bouton **Télécharger**.
- **PowerPoint (.pptx, .ppt)** : la présentation est **convertie en PDF par la
  VM** (LibreOffice) puis affichée ; la conversion prend quelques secondes.
  Bouton **Télécharger**.
- **Excel (.xlsx, .xls)** : les feuilles sont affichées sous forme de tableaux
  (jusqu'à 4 feuilles). Bouton **Télécharger**.
- **HTML** : aperçu **exécuté** dans une fenêtre isolée, avec ses fichiers
  `.css` et `.js` voisins chargés depuis la VM — un `index.html` accompagné de
  `style.css` et `script.js` fonctionne donc comme un vrai site. Boutons
  **Voir le code** / **Aperçu** et **Copier le code**.
- **Markdown** : rendu (titres, listes, tableaux, code, formules LaTeX).
  Boutons **Texte brut** / **Aperçu** et **Copier**.
- **LaTeX (.tex)** : les formules sont composées et affichées. Boutons
  **Texte brut** / **Aperçu** et **Copier**.
- **JavaScript** : code coloré, plus deux boutons —
  **Exécuter** (lance le script dans une fenêtre isolée et affiche ce que
  `console.log` produit, ainsi que les erreurs) et **Lancer sur la VM**.
- **Python** : code coloré et bouton **Lancer sur la VM**. Si le script
  dessine avec matplotlib, la figure est renvoyée et affichée en image.
- **Shell (.sh, .bash)** : code coloré et bouton **Lancer sur la VM**.
- **Autres fichiers texte** : affichés tels quels, avec **Copier**. Au-delà de
  200 Ko, l'affichage est tronqué pour ne jamais bloquer l'écran.

## 6. « Lancer sur la VM »

Ce bouton envoie le fichier affiché dans la VM et l'exécute via son terminal
(`python3`, `node` ou `bash`), en arrière-plan, puis renvoie le résultat :
sortie standard, erreurs, code de sortie, et l'image produite le cas échéant.
C'est la puissance de calcul de la VM qui est utilisée, pas celle du
navigateur. Un script long peut prendre du temps : le résultat arrive quand
l'exécution est finie.

## 7. Sa fiche personnelle (user.md)

- `user.md` est le **seul fichier qu'il peut modifier lui-même**.
- En l'ouvrant, un bouton **crayon** apparaît. Il donne accès à un formulaire :
  **Nom**, **Adresse e-mail**, **Âge**, **Métier**, une zone **Informations
  supplémentaires** et une zone **Instructions personnalisées**.
- En édition, deux boutons remplacent le crayon : **✕** annule sans rien
  changer, **✓** enregistre dans la VM.
- Le fichier est protégé : s'il est supprimé, il est recréé automatiquement
  (à partir de sa sauvegarde si possible).
- Lis-la en début de mission et **suis ses instructions personnalisées**.
  Ne la modifie jamais.

## 8. Les paramètres de son compte

En cliquant sur son avatar, il ouvre un menu puis des fenêtres qui lui
permettent de :

- **changer son mot de passe** (ou en **définir un** si son compte a été créé
  avec Google) ;
- **changer son adresse e-mail** ;
- **lier ou délier son compte Google** ;
- **changer sa photo de profil** ;
- **supprimer son compte** — la suppression est directe, sans confirmation
  Google.

Sur tous les champs mot de passe, un bouton **œil** affiche ou masque ce qui
est saisi. Tu n'as aucun accès à ces réglages : n'essaie pas de les manipuler
et ne lui demande jamais ses identifiants.

## 9. Les pièces jointes

- Le bouton **trombone** permet de joindre une image ou un fichier à son
  message, d'envoyer une image noire, ou de **partager son écran** (une capture
  de son propre écran, pas celui de la VM).
- Le **micro** sert à dicter son message : l'audio est transcrit.

## 10. En résumé, pour toi

- Pour qu'il voie quelque chose : dépose le fichier dans
  `/home/kasm-user/user/`. Il l'ouvrira d'un clic dans la visionneuse.
- Pour qu'il comprenne ce que tu fais : nomme bien tes fichiers, termine par
  un message clair indiquant quoi ouvrir et pourquoi.
- Il ne peut pas agir sur la VM : tout ce qui nécessite une commande, une
  installation ou un clic sur le bureau passe par toi.
