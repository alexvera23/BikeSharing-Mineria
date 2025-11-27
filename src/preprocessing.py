import pandas as pd
import numpy as np
import os
from sklearn.model_selection import StratifiedShuffleSplit

# ==========================================
# CONFIGURACIÓN DE RUTAS
# ==========================================
# Definimos rutas relativas para que funcione en cualquier computadora del equipo
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "day.csv")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

def cargar_datos(ruta):
    """Carga el dataset original"""
    if not os.path.exists(ruta):
        raise FileNotFoundError(f"No se encontró el archivo en: {ruta}. ¡Asegúrate de haberlo copiado!")
    print(f"--> Cargando datos desde: {ruta}")
    return pd.read_csv(ruta)

def limpieza_y_transformacion(df):
    """
    TEMA 2: ALMACÉN DE DATOS (ETL)
    Realiza limpieza básica y transformación de columnas.
    """
    print("--> Iniciando limpieza y transformación...")
    
    # 1. Renombrar columnas para que sean más intuitivas para el equipo
    nombres_nuevos = {
        'dteday': 'fecha',
        'yr': 'anio',
        'mnth': 'mes',
        'weathersit': 'clima_cat', # 1: Bueno, 2: Nublado, 3: Lluvia/Nieve
        'hum': 'humedad',
        'cnt': 'total_rentas'
    }
    df = df.rename(columns=nombres_nuevos)
    
    # 2. Verificar nulos
    nulos = df.isnull().sum().sum()
    if nulos > 0:
        print(f"   ATENCIÓN: Se encontraron {nulos} valores nulos. Rellenando con forward fill...")
        df = df.fillna(method='ffill')
    else:
        print("   Limpieza: No se encontraron valores nulos (Dataset de alta calidad).")

    # 3. INGENIERÍA DE CARACTERÍSTICAS (Ayuda para Integrante 4 - Naive Bayes)
    # Convertimos la variable numérica 'total_rentas' en Categoría: Baja, Media, Alta
    # Usamos cuartiles para definir los límites
    df['demanda_nivel'] = pd.cut(df['total_rentas'], 
                                 bins=[0, 3000, 6000, 99999], 
                                 labels=['Baja', 'Media', 'Alta'])
    
    return df

def generar_muestra_representativa(df):
    """
    TEMA 6: MUESTRA REPRESENTATIVA
    Divide los datos en Entrenamiento (70%) y Prueba (30%) usando Muestreo Estratificado.
    
    ¿Por qué estratificado? 
    Para asegurar que no tengamos un set de prueba que tenga SOLO días de invierno.
    Queremos que todas las estaciones (season) estén representadas igualitariamente.
    """
    print("--> Generando muestra representativa (Split Estratificado)...")
    
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    
    # Usamos 'season' como la categoría para estratificar
    for train_index, test_index in split.split(df, df["season"]):
        strat_train_set = df.loc[train_index]
        strat_test_set = df.loc[test_index]
    
    # Verificación de proporciones
    print(f"   Tamaño Entrenamiento: {len(strat_train_set)} días")
    print(f"   Tamaño Prueba: {len(strat_test_set)} días")
    
    return strat_train_set, strat_test_set

def guardar_datos(train, test):
    """Guarda los archivos procesados listos para los demás compañeros"""
    if not os.path.exists(PROCESSED_DIR):
        os.makedirs(PROCESSED_DIR)
        
    train_path = os.path.join(PROCESSED_DIR, "train_set.csv")
    test_path = os.path.join(PROCESSED_DIR, "test_set.csv")
    
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    
    print(f"--> ¡Éxito! Archivos generados en: {PROCESSED_DIR}")
    print("    1. train_set.csv (Usar para Árboles, Clustering y Reglas)")
    print("    2. test_set.csv (Usar SOLO para validar métricas finales)")

if __name__ == "__main__":
    # Flujo principal de ejecución
    try:
        # 1. Carga
        datos_crudos = cargar_datos(RAW_DATA_PATH)
        
        # 2. Limpieza
        datos_limpios = limpieza_y_transformacion(datos_crudos)
        
        # 3. Muestreo Representativo
        set_entrenamiento, set_prueba = generar_muestra_representativa(datos_limpios)
        
        # 4. Guardado
        guardar_datos(set_entrenamiento, set_prueba)
        
    except Exception as e:
        print(f"\n[ERROR CRÍTICO]: {e}")