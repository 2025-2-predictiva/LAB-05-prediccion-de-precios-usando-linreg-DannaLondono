#
# En este dataset se desea pronosticar el precio de vhiculos usados. El dataset
# original contiene las siguientes columnas:
#
# - Car_Name: Nombre del vehiculo.
# - Year: Año de fabricación.
# - Selling_Price: Precio de venta.
# - Present_Price: Precio actual.
# - Driven_Kms: Kilometraje recorrido.
# - Fuel_type: Tipo de combustible.
# - Selling_Type: Tipo de vendedor.
# - Transmission: Tipo de transmisión.
# - Owner: Número de propietarios.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# pronostico están descritos a continuación.
#
#
# Paso 1.
# Preprocese los datos.
# - Cree la columna 'Age' a partir de la columna 'Year'.
#   Asuma que el año actual es 2021.
# - Elimine las columnas 'Year' y 'Car_Name'.
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Escala las variables numéricas al intervalo [0, 1].
# - Selecciona las K mejores entradas.
# - Ajusta un modelo de regresion lineal.
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use el error medio absoluto
# para medir el desempeño modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas r2, error cuadratico medio, y error absoluto medio
# para los conjuntos de entrenamiento y prueba. Guardelas en el archivo
# files/output/metrics.json. Cada fila del archivo es un diccionario con
# las metricas de un modelo. Este diccionario tiene un campo para indicar
# si es el conjunto de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'metrics', 'dataset': 'train', 'r2': 0.8, 'mse': 0.7, 'mad': 0.9}
# {'type': 'metrics', 'dataset': 'test', 'r2': 0.7, 'mse': 0.6, 'mad': 0.8}
#

"""paso 1"""

# limpiar_datasets.py
import pandas as pd
from pathlib import Path

def limpiar_dataset_csv(csv_path):
    df = pd.read_csv(csv_path)

    # 1) Renombrar objetivo
    if 'Year' in df.columns:
        
        df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
        df["Age"] = 2021 - df["Year"]


        df = df.drop(columns=['Year'])
        df = df.drop(columns=['Car_Name'])
        df = df.drop(columns=['Present_Price'])


    df = df.dropna().reset_index(drop=True)
    return df


def cargar_y_procesar(train_csv="files/input/train_data.csv",
                      test_csv="files/input/test_data.csv",
                      ):
    """Carga y limpia train/test; opcionalmente guarda los CSV limpios."""
    df_train = limpiar_dataset_csv(train_csv)
    df_test = limpiar_dataset_csv(test_csv)


    return df_train, df_test

# Paso 2: dividir datasets en X/y (train y test)
from typing import Tuple, Optional

def dividir_en_xy(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    target_col: str = "Selling_Price"
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, Optional[pd.Series]]:
    """
    Separa características (X) y objetivo (y) para train y test.
    - Si df_test no contiene la columna objetivo, y_test será None.
    - Alinea X_train y X_test para tener exactamente las mismas columnas y en el mismo orden.
    """
    if target_col not in df_train.columns:
        raise ValueError(f"No se encontró la columna objetivo '{target_col}' en df_train. "
                         f"Columnas disponibles: {list(df_train.columns)}")

    # Separar X e y en train
    X_train = df_train.drop(columns=[target_col]).copy()
    y_train = pd.to_numeric(df_train[target_col], errors="coerce")
    # Separar X e y en test (si existe la columna)
    if target_col in df_test.columns:
        X_test = df_test.drop(columns=[target_col]).copy()
        y_test = pd.to_numeric(df_test[target_col], errors="coerce")
    else:
        X_test = df_test.copy()
        y_test = None

    # Alinear columnas de X_train y X_test (mismo set y mismo orden)
    common_cols = X_train.columns.intersection(X_test.columns)
    if len(common_cols) == 0:
        raise ValueError("X_train y X_test no comparten columnas. Revisa el preprocesamiento.")

    # Avisar si hay columnas faltantes/extras y alinear
    faltantes_en_test = [c for c in X_train.columns if c not in X_test.columns]
    extras_en_test = [c for c in X_test.columns if c not in X_train.columns]
    if faltantes_en_test or extras_en_test:
        print("[Aviso] Alineando columnas:")
        if faltantes_en_test:
            print(f" - Columnas presentes en train y faltantes en test: {faltantes_en_test}")
        if extras_en_test:
            print(f" - Columnas presentes en test y no en train: {extras_en_test}")

    X_train = X_train[common_cols].copy()
    X_test = X_test[common_cols].copy()
    
    return X_train, y_train, X_test, y_test 



from typing import List
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression


def crear_pipeline_rf(categorical_cols: List[str], numeric_cols: List[str], k: int) -> Pipeline:
    # Pipeline para categóricas: imputa y aplica OHE en formato denso
    cat_preprocess = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    # Pipeline para numéricas: imputa y escala a [0, 1]
    num_preprocess = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", MinMaxScaler())
    ])
    
    # Ensamble por columnas
    preprocessor = ColumnTransformer(transformers=[
        ("cat", cat_preprocess, categorical_cols),
        ("num", num_preprocess, numeric_cols),
    ],
    remainder="drop")  # descarta columnas no especificadas

    # Pipeline completo
    model = Pipeline(steps=[
        ("prep", preprocessor),
        ("select", SelectKBest(score_func=f_regression, k=k)),
        ("reg", LinearRegression())
    ])

    return model


import os
import gzip
import pickle
import numpy as np
def find_model_path():
    for p in MODEL_PATHS:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        "No encontré el modelo. Probé: " + ", ".join(MODEL_PATHS)
    )

def load_model(path):
    # Intenta joblib si está disponible; si falla, usa pickle+gzip
    try:
        import joblib
        return joblib.load(path)
    except Exception:
        with gzip.open(path, "rb") as f:
            return pickle.load(f)

POS_LABEL = None
def infer_setting(y):
    classes = np.unique(y)
    n_classes = len(classes)
    if n_classes <= 2:
        # Binario
        # Si POS_LABEL está definido lo usamos; si no, inferimos:
        if POS_LABEL is not None:
            pos_label = POS_LABEL
        else:
            # Heurística: si clases son numéricas y contienen 0 y 1 -> pos = 1
            try:
                numeric = np.array(classes, dtype=float)
                if set(numeric.tolist()) == {0.0, 1.0}:
                    pos_label = 1
                else:
                    # Tomamos la "mayor" etiqueta como positiva
                    pos_label = classes[-1]
            except Exception:
                # Para etiquetas no numéricas, tomamos la última en orden
                pos_label = classes[-1]
        return {"average": "binary", "pos_label": pos_label}
    
    else:
        # Multiclase: promediado ponderado
        return {"average": "weighted", "pos_label": None}


import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def _to_numpy_1d(y):
    """
    Convierte Series/DataFrames (incluyendo dtypes pandas nullable) a numpy 1D.
    """
    if isinstance(y, (pd.Series, pd.DataFrame)):
        y = y.to_numpy()
    y = np.asarray(y)
    if y.ndim > 1 and y.shape[1] == 1:
        y = y.ravel()
    return y


def compute_metrics(model, X, y, dataset_name):
    """
    Calcula las métricas de REGRESIÓN: r2, mse, mad (MAE) para el dataset indicado.
    Retorna un diccionario con el mismo formato base que tu función de clasificación.
    """
    # Convertir y y filtrar posibles NaN en el target
    y_np = _to_numpy_1d(y)
    if np.isnan(y_np).any():
        mask = ~np.isnan(y_np)
        X = X[mask]
        y_np = y_np[mask]

    # Predicción
    y_pred = model.predict(X)
    y_pred = _to_numpy_1d(y_pred)

    
    # Métricas
    r2 = r2_score(y_np, y_pred)
    mse = mean_squared_error(y_np, y_pred)
    mad = mean_absolute_error(y_np, y_pred)  # interpretado como MAE

    return {
        "type": "metrics",
        "dataset": dataset_name,
        "r2": float(r2),
        "mse": float(mse),
        "mad": float(mad),
    }




def ensure_output_dir(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

import json
def main():
    # 1) Modelo
    model_path = find_model_path()
    model = load_model(model_path)


    # 3) Métricas
    train_metrics = compute_metrics(model, X_train, y_train, "train")
    test_metrics  = compute_metrics(model, X_test,  y_test,  "test")

    # 4) Guardado en JSON Lines (una fila por diccionario)
    ensure_output_dir(OUTPUT_JSON)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        f.write(json.dumps(train_metrics, ensure_ascii=False) + "\n")
        f.write(json.dumps(test_metrics,  ensure_ascii=False) + "\n")

    print(f"✅ Métricas guardadas en: {OUTPUT_JSON}")
    print("Ejemplo de filas escritas:")
    print(train_metrics)
    print(test_metrics)


STRICT_EXAMPLE_TYPO = False   # True para replicar "predicte_1"
TYPO_LABEL = "1" 


from sklearn.model_selection import ParameterSampler, StratifiedKFold, GridSearchCV

if __name__ == "__main__":
    # Rutas por defecto (ajusta si es necesario)
    ruta_train = "files/input/train_data.csv/train_data.csv"
    ruta_test = "files/input/test_data.csv/test_data.csv"

    df_train, df_test = cargar_y_procesar(ruta_train, ruta_test)
    X_train, y_train, X_test, y_test =dividir_en_xy(df_train, df_test)
    print("Listo lindos")

    # 2) Definir columnas categóricas y numéricas
    cat_cols_explicit = ['Fuel_Type','Selling_type','Transmission']
    categorical_cols = [c for c in cat_cols_explicit if c in X_train.columns]
    # Numéricas = el resto
    numeric_cols = [c for c in X_train.columns if c not in categorical_cols]
    pipe = crear_pipeline_rf(categorical_cols, numeric_cols,20)
    
    
    #print([k for k in pipe.get_params().keys() if k.startswith('clf__')])

   
    from sklearn.model_selection import ParameterSampler, KFold
    from sklearn.metrics import balanced_accuracy_score
    from tqdm import tqdm
    import numpy as np

  
    param_dist = {
        
    "clf__C": np.logspace(-3, 3, 7),         # Regularización
        "clf__gamma": ["scale", "auto"],         # Parámetro del kernel RBF
        "clf__kernel": ["rbf"],                  # Tipo de kernel (puedes probar otros como 'linear', 'poly')
    }
    
    param_grid = {
        "reg__fit_intercept": [True, False],
        "reg__copy_X": [True],
        "reg__n_jobs": [None],        # generalmente None; en algunas versiones no tiene efecto
        "reg__positive": [False, True]  # True fuerza coeficientes positivos
    }

    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    grid = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,               
        scoring="neg_root_mean_squared_error",
        cv=cv,
        n_jobs=-1,
    )

    grid.fit(X_train, y_train)

    print("\n✅ Mejor balanced accuracy (CV): {:.4f}".format(grid.best_score_))
    print("🧪 Mejores hiperparámetros:", grid.best_params_)

    
    # 6) A partir de aquí, 'model' ES GridSearchCV (tu assert pasa)
    model = grid

    # 7) Guardar el objeto GridSearchCV completo (incluye best_estimator_)
    import pickle, gzip
    from pathlib import Path

    model_dir = Path("files/models")
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "model.pkl.gz"

    with gzip.open(model_path, "wb") as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"✅ Modelo (GridSearchCV) guardado en: {model_path}")

    # 8) (Opcional) Verificación explícita de tu requerimiento
    print(str(type(model)))  # debería contener 'GridSearchCV'
    assert "GridSearchCV" in str(type(model))


    
    

    ##### PASO 6 #####
    
    MODEL_PATHS = [
    "files/models/model.pkl.gz"
    ]   

    OUTPUT_JSON = "files/output/metrics.json"
    main()
   