import pandas as pd
import numpy as np
from scipy.optimize import linear_sum_assignment
from dotenv import load_dotenv
import os
import glob

load_dotenv()
API_KEY = os.getenv("API_KEY")

def read_data(file_path):
    archivos = glob.glob(file_path)
    if not archivos:
        raise FileNotFoundError("No se encontró ningún archivo en la ruta especificada.")    
    return pd.read_excel(archivos[0])

def preprocess_data(df):
    unified_data = []
    role_col = "Indícanos si eres mechón / mechona o de generación anterior 🤔 (Si te cambiaste a TEL este año, también cuentas como mechón/a 😊)"
    
    for idx, row in df.iterrows():
        role = row[role_col]
        
        if pd.isna(role): 
            continue
        
        if 'Mechón' in role:
            role_type = 'Mechón'
            name = row['Dinos tu nombre y apellido\xa0']
            email = row['Indícanos tu correo USM\xa0']
            deportes = str(row['¿Qué deportes te gusta practicar?'])
            juegos = str(row['¿Juegas alguno de estos juegos?'])
            series = str(row['Eres fan de las series de...'])
            pref = str(row['Me gusta mas...'])
            comida = str(row['¿Cuál de estas comidas te parece mas deliciosa? (Elige tu top 3)'])
            musica = str(row['Tipo de música favorita (Elige tu top 3)'])
            dieta = str(row['Tipo de dieta'])
            trago = str(row['Tienes algún trago favorito? (Elige tu top 3)'])
            hobby = str(row['¿Tienes algún Hobby?'])
            idiomas = str(row['¿Cuál de estos idiomas sabes o te gustaría aprender?\xa0'])
        else:
            role_type = 'Padrino'
            name = row['Dinos tu nombre y apellido']
            email = row['Indícanos tu correo USM']
            deportes = str(row['¿Qué deportes te gusta practicar?2'])
            juegos = str(row['¿Juegas alguno de estos juegos?2'])
            series = str(row['Eres fan de las series/películas de...'])
            pref = str(row['Me gusta mas...2'])
            comida = str(row['¿Cuál de estas comidas te parece mas deliciosa? (Elige tu top 3)2'])
            musica = str(row['Tipo de música favorita (Elige hasta tu top 3)'])
            dieta = str(row['Tipo de dieta2'])
            trago = str(row['Tienes algún trago favorito? (Elige tu top 3)2'])
            hobby = str(row['¿Tienes algún Hobby?2'])
            idiomas = str(row['¿Cuál de estos idiomas sabes o te gustaría aprender?\xa02'])
        
        unified_data.append({
            'Role': role_type, 'Name': name, 'Email': email,
            'Deportes': deportes, 'Juegos': juegos, 'Series': series,
            'Pref': pref, 'Comida': comida, 'Musica': musica,
            'Dieta': dieta, 'Trago': trago, 'Hobby': hobby, 'Idiomas': idiomas
        })

    df_unified = pd.DataFrame(unified_data).replace('nan', '')
    
    mechones = df_unified[df_unified['Role'] == 'Mechón'].reset_index(drop=True)
    padrinos = df_unified[df_unified['Role'] == 'Padrino'].reset_index(drop=True)

    return mechones, padrinos

df_crudo = read_data("data/*.xlsx") 
lista_mechones, lista_padrinos = preprocess_data(df_crudo)

"""
Para verificar que el proceso de unificación y limpieza de datos se realizó correctamente, se imprimen la cantidad
de mechones y padrinos encontrados en el DataFrame unificado.
"""

print(f"Mechones encontrados: {len(lista_mechones)}")
print(f"Padrinos encontrados: {len(lista_padrinos)}")

def parse_preferences(value): 
    if pd.isna(value) or value == '': 
            return set()
    return set([x.strip().lower() for x in str(value).split(';') if x.strip()])

def jaccard_similarity(set1, set2):
    if not set1 and not set2: 
        return 0
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0

def match(mechones, padrinos):
    cost_matrix = np.zeros((len(mechones), len(padrinos)))
    
    categorias = ['Deportes', 'Juegos', 'Series', 'Pref', 'Comida', 'Musica', 'Dieta', 'Trago', 'Hobby', 'Idiomas']
    
    for i, mechon in mechones.iterrows():
        for j, padrino in padrinos.iterrows():
            score = 0
            
            for cat in categorias:
                score += jaccard_similarity(
                    parse_preferences(mechon[cat]), 
                    parse_preferences(padrino[cat])
                )
            
            cost_matrix[i, j] = -score

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    matches = [
        {
            "Mechon": mechones.iloc[i]['Name'],
            "Padrino": padrinos.iloc[j]['Name']
        } 
        for i, j in zip(row_ind, col_ind)
    ]
    
    return matches

print("Generando emparejamientos")
matches = match(lista_mechones, lista_padrinos)
df_matches = pd.DataFrame(matches, columns=["Mechon", "Padrino"])
df_matches.to_csv("matches.csv", index=False, encoding='utf-8-sig')
print("Emparejamientos generados, visite matches.csv")