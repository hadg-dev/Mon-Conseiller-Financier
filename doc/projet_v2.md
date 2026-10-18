# Projet de fin de formation Lead IA — Mon conseiller financier

## 1. Objectif du projet

Développer un conseiller financier basé sur un modèle de langage local, adapté à l'aide d'un fine-tuning LoRA, afin de répondre à des questions relatives aux finances personnelles.

Le projet vise à mettre en œuvre une chaîne MLOps complète, depuis la préparation des données jusqu'au déploiement cloud, en passant par l'entraînement, l'évaluation, le suivi des expériences et le versionnement du modèle.

Compte tenu du temps disponible, le périmètre sera volontairement limité. L'objectif principal est de développer et d'évaluer un conseiller conversationnel, sans connexion à des comptes bancaires réels ni développement d'un système complet de simulation financière.

## 2. Cas d'usage envisagés

### [1] Conseil en investissement

« J'ai X € à placer. Quelle stratégie d'investissement me recommandes-tu pour atteindre Y € dans 10 ans ? »

L'objectif est d'explorer les stratégies envisageables, les risques associés et les hypothèses nécessaires pour atteindre un objectif financier donné.

### [2] Conseil en gestion budgétaire

« À partir de mes revenus, de mes dépenses et de mes objectifs, comment puis-je mieux gérer mon argent ? »

L'objectif est de fournir des conseils personnalisés en matière de gestion budgétaire, d'épargne et de maîtrise des dépenses.

### [3] Analyse d'un budget mensuel

« Voici mon budget mensuel [...]. Je voudrais que tu catégorises mes dépenses, calcules mon reste à vivre et détermines combien je peux épargner. »

L'objectif est de catégoriser les dépenses, de calculer les indicateurs budgétaires pertinents et d'expliquer les résultats obtenus.

### [4] Explication de concepts financiers

« Explique-moi en quoi consistent le taux d'endettement, les intérêts composés et l'inflation. »

L'objectif est de vulgariser les principaux concepts financiers et de fournir des explications claires, structurées et adaptées au niveau de l'utilisateur.

### [5] Simulation de scénarios d'épargne

« Que se passe-t-il si j'épargne 200 € de plus par mois ? Projette mon épargne sur plusieurs années. »

L'objectif est de comparer différents scénarios d'épargne et d'expliquer leur impact sur le capital accumulé à terme.

## 3. Fonctionnalités envisagées

| Fonctionnalité | Cas d'usage associés | Statut |
|---|---|---|
| Analyse du budget | [1], [2] et [3] | À préciser selon le périmètre retenu |
| Conseiller financier conversationnel | [4], avec extension possible aux cas [1] et [2] | Fonctionnalité principale |
| Simulateur financier | [5] | Hors périmètre initial |

Le conseil en investissement et la gestion budgétaire pourront être abordés progressivement. Les réponses nécessitant des calculs devront obligatoirement s'appuyer sur des outils déterministes.

### Restrictions fonctionnelles

- Aucune connexion à de véritables comptes bancaires.
- Pas de simulateur financier complet dans la première version, afin de limiter le développement des outils de calcul.
- Pas de backend métier complet pour le projet de formation, compte tenu du temps disponible.
- Exposition du conseiller via une API FastAPI légère, si nécessaire pour le déploiement.

## 4. Datasets envisagés

### 4.1 Finance-Instruct-500k

Lien : https://huggingface.co/datasets/Josephgflowers/Finance-Instruct-500k

- Environ 518 185 lignes.
- Dataset conçu pour l'entraînement de modèles de langage sur des tâches financières, le raisonnement et les conversations à plusieurs tours.
- Volume important, qui nécessite une phase de sélection, de validation et de nettoyage.
- Présence potentielle de bruit et d'exemples peu pertinents pour les finances personnelles.

**Utilisation envisagée :** dataset complémentaire éventuel, après analyse de sa pertinence et de sa licence.

### 4.2 Agent Finance Reasoning

Lien : https://huggingface.co/datasets/snorkelai/agent-finance-reasoning

- Environ 357 exemples.
- Dataset axé sur le raisonnement financier et les interactions agentiques.
- Contient des traces d'interactions et des métadonnées associées.
- Présente un intérêt pour l'étude des appels d'outils (*tool calls*).

**Utilisation envisagée :** source complémentaire pour explorer la structuration des interactions avec des outils et les tâches de raisonnement financier.

### 4.3 Finance Reasoning SFT Dataset

Lien : https://huggingface.co/datasets/likhitjuttada/finance-reasoning-sft-dataset

- 651 exemples d'entraînement annoncés.
- Dataset orienté questions-réponses et raisonnement financier.
- Taille réduite, permettant une première expérimentation de fine-tuning.

**Utilisation envisagée :** dataset de départ pour entraîner un premier adapter LoRA, sous réserve de vérifier sa licence, la qualité des exemples et leur adéquation avec les cas d'usage retenus.

## 5. Stratégie de modélisation

### 5.1 Modèles de base

Deux modèles sont envisagés :

- Qwen3.5-9B.
- Gemma 4 12B.

Le choix définitif sera effectué après une phase de comparaison portant sur les performances initiales, la compatibilité avec les bibliothèques de fine-tuning, les besoins en mémoire GPU et la qualité des réponses sur un jeu de test représentatif.

### 5.2 Fine-tuning LoRA

L'approche privilégiée consiste à entraîner un adapter LoRA sur un modèle instruct adapté aux interactions conversationnelles.

L'objectif du fine-tuning est principalement d'améliorer la manière dont le modèle répond aux questions financières : structure des réponses, pédagogie, gestion des informations manquantes et prudence dans les recommandations.

Le fine-tuning ne doit pas être considéré comme un moyen de garantir l'exactitude des informations financières ou des calculs.

### 5.3 RAG

Un système RAG pourra être ajouté si le conseiller a besoin de retrouver des informations complémentaires issues de sources officielles.

Les documents pourront notamment porter sur les notions de finances personnelles, les dispositifs d'épargne et les règles applicables en France.

Le RAG permettra de fournir des réponses documentées et de tenir compte de sources actualisées sans devoir réentraîner le modèle à chaque évolution documentaire.

### 5.4 Calculs déterministes et appels d'outils

Tous les calculs financiers devront être exécutés par des outils déterministes appelés par le modèle.

Exemples :
- Calcul du reste à vivre.
- Calcul du taux d'épargne ou du taux d'endettement.
- Calcul des intérêts composés et des projections financières.

Le LLM pourra identifier le calcul nécessaire, extraire les paramètres pertinents, demander les informations manquantes et expliquer le résultat. Il ne devra pas inventer lui-même les résultats numériques.

Cette approche devra être démontrée par des tests fonctionnels vérifiant les appels d'outils et la justesse des résultats.

## 6. Organisation MLOps

### Étape 1 — Data pipeline

Objectif : construire un pipeline de préparation des données reproductible.

Travaux :
- Récupération d'un dataset depuis Hugging Face.
- valider sa licence et sa pertinence.
- Analyse exploratoire et validation des données.
- Nettoyage et filtrage des exemples non pertinents.
- Déduplication.
- Normalisation du format des conversations.
- Séparation des données d'entraînement, de validation et de test.
- Versionnement du dataset et documentation de sa provenance.

Livrables :
- Pipeline de préparation des données.
- Dataset nettoyé et versionné.
- Documentation des choix de filtrage et des licences.

### Étape 2 — Prototype et baseline

Objectif : définir une référence avant le fine-tuning.

Travaux :
- Tester les modèles de base sur des cas d'usage représentatifs.
- Comparer Qwen3.5-9B et Gemma 4 12B si le temps le permet.
- Constituer un jeu de test indépendant.
- Définir les critères d'évaluation.
- Vérifier la compatibilité des modèles avec l'environnement de training retenu.

Livrables :
- Baseline documentée.
- Jeu de test.
- Configuration initiale des expériences.

### Étape 3 — Training et expérimentation

Objectif : entraîner et évaluer un adapter LoRA.

Travaux :
- Définition des hyperparamètres.
- Exécution des entraînements sur une plateforme GPU à la demande, telle que Modal ou RunPod. Alternative EC2 typé GPU via AWS, et utiliser Train.
- Suivi des expériences avec MLflow.
- Enregistrement des paramètres, métriques, courbes de loss et artefacts.
- Réalisation de tests fonctionnels.
- Comparaison des résultats avec le modèle de base.

Métriques et critères à étudier :
- Loss d'entraînement et de validation.
- Qualité et pertinence des réponses.
- Respect du format attendu.
- Exactitude des résultats numériques obtenus via les outils.
- Capacité à reconnaître les informations manquantes.
- Latence et consommation des ressources, lorsque mesurables.

Livrables :
- Adapter LoRA entraîné.
- Expériences MLflow documentées.
- Rapport d'évaluation comparatif.

### Étape 4 — Packaging et versionnement du modèle

Objectif : rendre le modèle reproductible et déployable.

Travaux :
- Versionnement de l'adapter produit.
- Identification de la version du modèle de base.
- Conservation des hyperparamètres et de la configuration d'entraînement.
- Documentation du dataset et des limites du modèle.
- Préparation de l'environnement d'inférence.

Livrables :
- Adapter versionné.
- Configuration d'inférence.
- Model card et instructions de reproduction.

### Étape 5 — Industrialisation et déploiement AWS

Objectif : mettre en œuvre les principales briques MLOps et les technologies d'infrastructure vues en formation.

Travaux :
- Conteneurisation du service d'inférence.
- Exposition du conseiller via FastAPI.
- Publication de l'image dans Amazon ECR.
- Déploiement sur AWS avec Kubernetes, si le choix d'EKS est retenu.
- Packaging et déploiement des composants avec Helm.
- Provisioning de l'infrastructure avec Terraform.
- Gestion de la configuration et des secrets.
- Mise en place de la journalisation et de l'observabilité.
- Étude des aspects sécurité et coûts.

Livrables :
- Application déployable.
- Infrastructure décrite sous forme de code.
- Procédure de déploiement et de nettoyage.
- Documentation de l'observabilité et des mesures de sécurité.

## 7. Périmètre et priorités

Compte tenu du temps disponible, les priorités sont les suivantes :

**Priorité 1 — Socle du projet**
- Sélection et préparation du dataset.
- Baseline et évaluation.
- Entraînement d'un adapter LoRA.
- Suivi des expériences avec MLflow.

**Priorité 2 — Démonstration applicative**
- Service d'inférence.
- Conseiller conversationnel.
- Appels d'outils pour les calculs retenus.
- RAG si nécessaire.

**Priorité 3 — Industrialisation**
- Conteneurisation.
- Déploiement AWS.
- Helm et Terraform.
- Observabilité et sécurité.

Les fonctionnalités avancées, notamment le simulateur financier complet, la comparaison exhaustive de plusieurs modèles et le training distribué avec Ray, restent conditionnées au temps disponible.

## 8. Résultat attendu

Le projet devra démontrer la capacité à construire une chaîne IA reproductible, depuis la préparation des données jusqu'au déploiement d'un conseiller conversationnel.

Les principaux résultats attendus sont :
- Un adapter LoRA entraîné et versionné.
- Une évaluation mesurant son apport par rapport au modèle de base.
- Un suivi des expériences et des artefacts avec MLflow.
- Un conseiller capable d'expliquer des concepts financiers et de solliciter des outils déterministes lorsque des calculs sont nécessaires.
- Une application conteneurisée et un déploiement AWS documenté.
- Une présentation des limites du modèle, des risques de hallucination et des mesures mises en place pour les réduire.
