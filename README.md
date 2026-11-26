# API de classification d'intentions

Projet personnel de bout en bout : un modèle de classification de textes, de l'entraînement jusqu'à la mise en production (API, conteneur, tests automatisés, déploiement continu et suivi des prédictions).

**Démo en ligne :** https://intention-classification-api.onrender.com/docs

> L'hébergement est gratuit : après 15 minutes sans visite, le service s'endort et la première requête prend environ une minute.

## Objectif

Le modèle reçoit une phrase en français adressée à un assistant vocal et prédit l'intention de l'utilisateur parmi 60 possibles (régler une alarme, demander la météo, lancer de la musique, etc.).

L'objectif du projet n'est pas le score du modèle, mais la chaîne complète qui permet de le rendre utilisable et fiable en production.

## Architecture

```
git push
   │
   ├──> GitHub Actions : lance les tests (pytest)
   │
   └──> Render : construit l'image Docker et redéploie l'API
                      │
                      └──> API FastAPI ──> modèle (TF-IDF + régression logistique)
```

## Données

- Jeu de données : [MASSIVE](https://huggingface.co/datasets/mteb/amazon_massive_intent) (Amazon), configuration française
- 11 514 phrases d'entraînement, 2 033 de validation, 2 974 de test
- 60 intentions

## Modèle

- Vectorisation TF-IDF (mots et paires de mots) puis régression logistique, réunies dans un pipeline scikit-learn
- Résultats sur le jeu de validation : **accuracy 0,769** et **F1 macro 0,693**
- Modèle sauvegardé dans `model.pkl`

Ce modèle simple sert de référence. Il peut être remplacé par un modèle plus performant sans modifier le reste de la chaîne.

## Utilisation de l'API

| Route | Méthode | Rôle |
|---|---|---|
| `/health` | GET | Vérifie que le service fonctionne |
| `/predict` | POST | Prédit l'intention d'une phrase |
| `/docs` | GET | Documentation interactive |

Exemple de requête :

```bash
curl -X POST https://intention-classification-api.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "mets une alarme à sept heures"}'
```

Exemple de réponse :

```json
{
  "intention": "alarm_set",
  "confiance": 0.698,
  "fiable": true
}
```

Le champ `fiable` vaut `false` lorsque la confiance est inférieure au seuil fixé dans `main.py`. L'application appelante peut alors demander à l'utilisateur de reformuler au lieu d'agir sur une prédiction incertaine.

## Lancer le projet en local

Avec Docker :

```bash
docker build -t intention-api .
docker run -p 8000:8000 intention-api
```

Sans Docker :

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

L'API est ensuite disponible sur http://127.0.0.1:8000/docs.

## Tests et intégration continue

```bash
pip install pytest httpx2
pytest
```

Trois tests vérifient que le service répond, que la réponse a le bon format et que le modèle prédit correctement un cas simple. GitHub Actions les relance automatiquement à chaque `git push` (fichier `.github/workflows/tests.yml`).

## Déploiement

L'API est conteneurisée avec Docker et hébergée sur Render. Chaque `git push` sur la branche `main` déclenche la reconstruction de l'image et le redéploiement.

## Suivi en production

Chaque prédiction est enregistrée dans les journaux au format JSON : texte reçu, intention prédite, confiance, indicateur de fiabilité, date et heure (UTC). Ces journaux permettent de repérer les phrases sur lesquelles le modèle hésite et de suivre la part de prédictions peu fiables.

## Structure du dépôt

```
├── main.py                        # API FastAPI
├── model.pkl                      # modèle entraîné
├── test_main.py                   # tests
├── requirements.txt               # dépendances
├── Dockerfile                     # image Docker
└── .github/workflows/tests.yml    # intégration continue
```

## Technologies

Python, scikit-learn, FastAPI, Pydantic, Docker, pytest, GitHub Actions, Render

## Limites et pistes d'amélioration

- Le modèle est volontairement simple : un modèle de type CamemBERT donnerait de meilleurs scores.
- Les journaux ne sont pas conservés durablement : les stocker en base permettrait de détecter une dérive des données.
- Le seuil de confiance peut être ajusté à partir du jeu de validation.
- Un déploiement sur un cloud (AWS) est prévu.

## Auteur

Serigne Bassirou NDIAYE, Data Scientist — [GitHub](https://github.com/DST-NDIAYE)
