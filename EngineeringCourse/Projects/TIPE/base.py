import sqlite3 as sql
import math
import ast
import sqlite3
import itertools

db = str

def list_ingred(d:db)->list[str]:
    cnect = sql.connect(d)
    cur = cnect.cursor()
    cur.execute("select NER from test order by field1")
    data = cur.fetchall()
    cnect.close()
    return list(dict.fromkeys(ing for tup in data for ing in ast.literal_eval(tup[0])))


def creer_et_remplir_tables(d:db):
    # Connexion à la base de données
    conn = sql.connect(d)
    cursor = conn.cursor()

    # Supprimer les anciennes tables pour un nettoyage (optionnel)
    cursor.execute("DROP TABLE IF EXISTS ingredtest;")
    cursor.execute("DROP TABLE IF EXISTS jointabtest;")

    # Créer les tables 'ingred' et 'jointabtest' si elles n'existent pas
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS ingredtest (
        id INTEGER PRIMARY KEY,
        name TEXT UNIQUE
    );
    ''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS jointabtest (
        recipe_id INTEGER,
        ingredient_id INTEGER,
        FOREIGN KEY (recipe_id) REFERENCES test(field1),
        FOREIGN KEY (ingredient_id) REFERENCES ingredtest(id),
        PRIMARY KEY (recipe_id, ingredient_id)
    );
    ''')

    # Créer l'index après avoir créé les tables
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_ingredient_name ON ingredtest(name);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_recipe_id ON jointabtest(recipe_id);")

    # Récupérer toutes les recettes et leurs ingrédients NER
    cursor.execute("SELECT field1, NER FROM test")
    recettes = cursor.fetchall()

    # Créer des listes pour les insertions en masse
    ingredients_to_insert = []
    jointab_to_insert = set()  # Utiliser un set pour éviter les duplications

    # Traitement des recettes sans calculs de progression
    for i, recette in enumerate(recettes):
        field1 = recette[0]
        # Changed json.loads to ast.literal_eval
        ner_ingredients = ast.literal_eval(recette[1])

        for ingredient in ner_ingredients:
            # Ajouter l'ingrédient à la liste pour insertion en masse
            ingredients_to_insert.append((ingredient,))

            # Ajouter l'association (recipe_id, ingredient) dans un set pour éviter les doublons
            jointab_to_insert.add((field1, ingredient))

    # Insertion des ingrédients en masse
    cursor.executemany("INSERT OR IGNORE INTO ingredtest (name) VALUES (?)", ingredients_to_insert)

    # Récupérer les IDs des ingrédients nouvellement insérés
    cursor.execute("SELECT id, name FROM ingredtest")
    ingredient_ids = {name: id for id, name in cursor.fetchall()}

    # Construire la liste des insertions pour jointab, avec les IDs des ingrédients
    jointab_to_insert = [(field1, ingredient_ids[ingredient]) for field1, ingredient in jointab_to_insert]

    # Insertion des associations (jointab) en masse
    cursor.executemany("INSERT OR IGNORE INTO jointabtest (recipe_id, ingredient_id) VALUES (?, ?)", jointab_to_insert)

    # Valider les changements et fermer la connexion
    conn.commit()
    conn.close()

    print("Traitement terminé.")


def buildPMI(d: str):
    conn = sql.connect(d)
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS PMI(
            id INTEGER,
            ext_id INTEGER,
            pmi REAL
        );
    ''')

    # Nombre total de recettes (DISTINCT est préférable)
    cursor.execute("SELECT COUNT(DISTINCT recipe_id) FROM jointabtest")
    n_recipes = cursor.fetchone()[0]

    # Probabilités individuelles P(i)
    cursor.execute(f"""
        SELECT ingredient_id, COUNT(ingredient_id) * 1.0 / {n_recipes}
        FROM jointab
        GROUP BY ingredient_id
    """)
    # Transforme en dict pour accès rapide
    self_probas = dict(cursor.fetchall())

    # Probabilités conjointes P(i, j)
    cursor.execute(f"""
        SELECT
            j1.ingredient_id AS i,
            j2.ingredient_id AS j,
            COUNT(DISTINCT j1.recipe_id) * 1.0 / {n_recipes} AS pij
        FROM jointab j1
        JOIN jointab j2
            ON j1.recipe_id = j2.recipe_id AND j1.ingredient_id < j2.ingredient_id
        GROUP BY i, j
    """)
    ext_probas = cursor.fetchall()

    for i, j, pij in ext_probas:
        pi = self_probas.get(i)
        pj = self_probas.get(j)

        if pi and pj and pij > 0:
            pmi = math.log(pij / (pi * pj))
            cursor.execute(
                "INSERT INTO PMI (id, ext_id, pmi) VALUES (?, ?, ?)",
                (i, j, pmi)
            )

    conn.commit()
    conn.close()

d = "C:/Users/xavie/Documents/Etudes/TIPE/Salad/main.db"

def get_ingred(id):
    conn = sql.connect(d)
    cur = conn.cursor()

    cur.execute("SELECT name FROM ingred WHERE id={}".format(id))
    ingred = cur.fetchone()[0]

    conn.close()
    return ingred

def count_cooccurrences(db_path: str):
    """
    Counts the co-occurrences of each ingredient pair across all recipes
    and stores them in a new table named 'cooc'.

    Args:
        db_path (str): The path to the SQLite database file (e.g., 'main.db').
    """
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Crée la table cooc si elle n'éxiste pas
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cooc (
                id INTEGER NOT NULL,
                ext_id INTEGER NOT NULL,
                cooc_count INTEGER NOT NULL,
                PRIMARY KEY (id, ext_id),
                FOREIGN KEY (id) REFERENCES ingred(id),
                FOREIGN KEY (ext_id) REFERENCES ingred(id)
            );
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_cooc_id ON cooc (id);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_cooc_ext_id ON cooc (ext_id);")
        conn.commit()
        print("Table 'cooc' created or already exists.")

        # Recupère les liens ingrédients recètes
        cursor.execute("SELECT recipe_id, ingredient_id FROM jointab ORDER BY recipe_id;")
        recipe_ingredient_data = cursor.fetchall()

        # Création du dictionaire recète -> ingrédient
        recipes_ingredients = {}
        for recipe_id, ingred_id in recipe_ingredient_data:
            if recipe_id not in recipes_ingredients:
                recipes_ingredients[recipe_id] = []
            recipes_ingredients[recipe_id].append(ingred_id)

        # Calcul des co-occurrences
        cooccurrence_counts = {}
        for recipe_id, ingredient_list in recipes_ingredients.items():
            if len(ingredient_list) >= 2:
                for ing1, ing2 in itertools.combinations(sorted(ingredient_list), 2):
                    pair = (ing1, ing2)
                    if pair not in cooccurrence_counts:
                        cooccurrence_counts[pair] = 0
                    cooccurrence_counts[pair] += 1

        # Préparation des données à l'insertion multiple
        data_to_insert = []
        for (id1, id2), count in cooccurrence_counts.items():
            data_to_insert.append((id1, id2, count))

        # Nétoyage de la table
        cursor.execute("DELETE FROM cooc;")
        conn.commit()
        print("Existing data in 'cooc' table cleared.")

        # Insertion multiple
        cursor.executemany("INSERT INTO cooc (id, ext_id, cooc_count) VALUES (?, ?, ?)", data_to_insert)
        conn.commit()
        print(f"Successfully counted and inserted {len(data_to_insert)} co-occurrence pairs into 'cooc' table.")

    except sqlite3.Error as e:
        print(f"An SQLite error occurred: {e}")
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":

    creer_et_remplir_tables(d)
    #list_ingred(d)
    #buildPMI(d)

    #count_cooccurrences(d)

