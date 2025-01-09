import requests
import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

JETON_API = "REDACTED"
URL_BASE = "https://api.football-data.org/v4/"
ENTETES = {"X-Auth-Token": JETON_API}

def recuperer_donnees(endpoint):
    url = URL_BASE + endpoint
    reponse = requests.get(url, headers=ENTETES)
    if reponse.status_code == 200:
        return reponse.json()
    else:
        print(f"Erreur: {reponse.status_code}, {reponse.text}")
        return None

def obtenir_liste_ligues():
    endpoint = "competitions"
    donnees = recuperer_donnees(endpoint)
    if donnees and "competitions" in donnees:
        return [
            {"code": ligue["code"], "nom": ligue["name"]}
            for ligue in donnees["competitions"]
            if ligue["code"] and ligue["name"]
        ]
    return []

def obtenir_donnees_ligue(code_ligue, saison):
    endpoint = f"competitions/{code_ligue}/matches?season={saison}"
    return recuperer_donnees(endpoint)

def obtenir_donnees_toutes_ligues_5_saisons():
    ligues = obtenir_liste_ligues()
    donnees_globales = []
    annee_actuelle = 2024
    saisons = [annee_actuelle - i for i in range(3)] # max 3 car + = payant

    for saison in saisons:
        for ligue in ligues:
            print(f"Récupération des données pour la ligue : {ligue['nom']} ({ligue['code']}), saison : {saison}")
            donnees_ligue = obtenir_donnees_ligue(ligue["code"], saison)
            if donnees_ligue:
                donnees_preparees = preparer_donnees(donnees_ligue)
                donnees_globales.append(donnees_preparees)
                time.sleep(10) # 10 car max 10req/min

    return pd.concat(donnees_globales, ignore_index=True) if donnees_globales else None

def preparer_donnees(donnees_matches):
    donnees = []
    for match in donnees_matches.get("matches", []):
        if match["status"] == "FINISHED":
            donnees.append({
                "equipe_domicile": match["homeTeam"]["name"],
                "equipe_exterieur": match["awayTeam"]["name"],
                "score_domicile": match["score"]["fullTime"]["home"],
                "score_exterieur": match["score"]["fullTime"]["away"],
                "date": match["utcDate"]
            })
    return pd.DataFrame(donnees)


def tracer_performance_equipes(donnees):
    performance = donnees.groupby("equipe_domicile")["score_domicile"].sum().sort_values(ascending=False).head(10)
    performance.plot(kind="bar", figsize=(10, 6))
    plt.title("Top 10 des équipes à domicile (buts marqués)")
    plt.xlabel("Équipe")
    plt.ylabel("Buts")
    plt.tight_layout()
    plt.show()

    moyennes_domicile = donnees.groupby("equipe_domicile")["score_domicile"].mean()
    plt.hist(moyennes_domicile, bins=10, edgecolor='black')
    plt.title("Répartition des scores moyens à domicile")
    plt.xlabel("Score moyen")
    plt.ylabel("Nombre d'équipes")
    plt.tight_layout()
    plt.show()

    plt.boxplot(donnees["score_domicile"].dropna(), vert=False, patch_artist=True)
    plt.title("Répartition des scores à domicile")
    plt.xlabel("Score")
    plt.tight_layout()
    plt.show()

def construire_modele_prediction(donnees):
    donnees["moyenne_buts_domicile"] = donnees.groupby("equipe_domicile")["score_domicile"].transform("mean")
    donnees["moyenne_buts_contre_domicile"] = donnees.groupby("equipe_domicile")["score_exterieur"].transform("mean")
    donnees["moyenne_buts_exterieur"] = donnees.groupby("equipe_exterieur")["score_exterieur"].transform("mean")
    donnees["moyenne_buts_contre_exterieur"] = donnees.groupby("equipe_exterieur")["score_domicile"].transform("mean")

    features = ["moyenne_buts_domicile", "moyenne_buts_contre_domicile", "moyenne_buts_exterieur", "moyenne_buts_contre_exterieur"]
    X = donnees[features]
    y_domicile = donnees["score_domicile"]
    y_exterieur = donnees["score_exterieur"]

    X_train_dom, X_test_dom, y_train_dom, y_test_dom = train_test_split(X, y_domicile, test_size=0.2, random_state=42)
    X_train_ext, X_test_ext, y_train_ext, y_test_ext = train_test_split(X, y_exterieur, test_size=0.2, random_state=42)

    modele_domicile = RandomForestRegressor(n_estimators=100, random_state=42)
    modele_domicile.fit(X_train_dom, y_train_dom)

    modele_exterieur = RandomForestRegressor(n_estimators=100, random_state=42)
    modele_exterieur.fit(X_train_ext, y_train_ext)

    print(f"Erreur quadratique moyenne (domicile): {mean_squared_error(y_test_dom, modele_domicile.predict(X_test_dom))}")
    print(f"Erreur quadratique moyenne (exterieur): {mean_squared_error(y_test_ext, modele_exterieur.predict(X_test_ext))}")

    return modele_domicile, modele_exterieur

def predire_score_match_interligue(modele_domicile, modele_exterieur, equipe_dom, equipe_ext, donnees):
    def calculer_moyenne(df, colonne):
        return df[colonne].mean() if not df[colonne].isnull().all() else 0

    moyenne_buts_domicile = calculer_moyenne(donnees[donnees['equipe_domicile'] == equipe_dom], 'moyenne_buts_domicile')
    moyenne_buts_contre_domicile = calculer_moyenne(donnees[donnees['equipe_domicile'] == equipe_dom], 'moyenne_buts_contre_domicile')
    moyenne_buts_exterieur = calculer_moyenne(donnees[donnees['equipe_exterieur'] == equipe_ext], 'moyenne_buts_exterieur')
    moyenne_buts_contre_exterieur = calculer_moyenne(donnees[donnees['equipe_exterieur'] == equipe_ext], 'moyenne_buts_contre_exterieur')

    features = pd.DataFrame([[moyenne_buts_domicile, moyenne_buts_contre_domicile, moyenne_buts_exterieur, moyenne_buts_contre_exterieur]],
                             columns=["moyenne_buts_domicile", "moyenne_buts_contre_domicile", "moyenne_buts_exterieur", "moyenne_buts_contre_exterieur"])
    
    score_dom = modele_domicile.predict(features)[0]
    score_ext = modele_exterieur.predict(features)[0]

    print(f"Score prédit : {equipe_dom} {int(score_dom)} - {int(score_ext)} {equipe_ext}")
    return int(score_dom), int(score_ext)

if __name__ == "__main__":
    print("\nTéléchargement des données pour toutes les ligues et les 5 dernières saisons...")
    donnees_matches = obtenir_donnees_toutes_ligues_5_saisons()

    if donnees_matches is not None:
        print("Préparation des données...")
        tracer_performance_equipes(donnees_matches)
        print("Création des modèles...\n")
        modele_domicile, modele_exterieur = construire_modele_prediction(donnees_matches)
    else:
        print("Erreur : les données n'ont pas pu être récupérées.")
        exit()

    equipes = sorted(set(donnees_matches["equipe_domicile"].unique()) | set(donnees_matches["equipe_exterieur"].unique()))
    print("\nÉquipes disponibles :\n", equipes)
    print("\n--------------------\n")
    equipe_dom = input("Choisissez une équipe à domicile > ")
    equipe_ext = input("Choisissez une équipe à l'extérieur > ")

    while equipe_dom not in equipes or equipe_ext not in equipes:
        print("Veuillez choisir des équipes valides.")
        equipe_dom = input("Choisissez une équipe à domicile > ")
        equipe_ext = input("Choisissez une équipe à l'extérieur > ")

    predire_score_match_interligue(modele_domicile, modele_exterieur, equipe_dom, equipe_ext, donnees_matches)