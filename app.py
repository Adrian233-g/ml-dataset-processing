
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="ML Dataset Processing", layout="wide")

st.title("🔬 Procesamiento de Datasets en Machine Learning")
st.markdown("---")

    # Menú de navegación
menu = st.sidebar.selectbox(
    "Selecciona un ejercicio:",
    ["Inicio", "Ejercicio 1: Titanic", "Ejercicio 2: Student Performance", "Ejercicio 3: Iris"]
)

if menu == "Inicio":
    st.header("Bienvenido al Sistema de Procesamiento de Datos")
    st.markdown("""
    ### Objetivo del Proyecto
    Aplicar las etapas del procesamiento de datos:
    1. **Carga del dataset**
    2. **Exploración inicial** (info, describe, nulls, tipos de datos)
    3. **Limpieza de datos** (valores nulos, duplicados, outliers)
    4. **Codificación de variables categóricas**
    5. **Normalización o estandarización**
    6. **División en conjuntos de entrenamiento y prueba**
    
    ### Ejercicios Disponibles
    - 📊 **Ejercicio 1: Análisis del Dataset Titanic** - Predicción de supervivencia
    """)

elif menu == "Ejercicio 1: Titanic":
    st.header("📊 Ejercicio 1: Análisis del Dataset Titanic")
    st.markdown("**Objetivo:** Preparar los datos para predecir la supervivencia de pasajeros")
    st.markdown("---")
    
    # Cargar dataset
    st.subheader("1️⃣ Carga del Dataset")
    
    try:
        df = sns.load_dataset('titanic')
        st.success("✅ Dataset Titanic cargado exitosamente")
        st.write(f"Dimensiones originales: {df.shape}")
    except:
        st.error("Error al cargar el dataset")
        st.stop()
    
    # Exploración inicial
    st.subheader("2️⃣ Exploración Inicial")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Primeros 5 registros:**")
        st.dataframe(df.head())
    
    with col2:
        st.write("**Información del Dataset:**")
        st.text(f"Total de registros: {len(df)}")
        st.text(f"Total de columnas: {len(df.columns)}")
        st.write("**Columnas disponibles:**")
        st.write(list(df.columns))
    
    if st.checkbox("Mostrar información detallada"):
        col3, col4 = st.columns(2)
        with col3:
            st.write("**Estadísticas Descriptivas:**")
            st.dataframe(df.describe())
        with col4:
            st.write("**Valores Nulos:**")
            null_data = pd.DataFrame(df.isnull().sum(), columns=['Nulos'])
            null_data['Porcentaje'] = (null_data['Nulos'] / len(df) * 100).round(2)
            st.dataframe(null_data)
    
    # Limpieza de datos
    st.subheader("3️⃣ Limpieza de Datos")
    
    df_limpio = df.copy()
    
    # Eliminar columnas irrelevantes
    st.write("**Eliminando columnas irrelevantes:** Name, Ticket, Cabin")
    columnas_a_eliminar = ['passenger_id', 'who', 'adult_male', 'deck']
    df_limpio = df_limpio.drop(columnas_a_eliminar, axis=1, errors='ignore')
    st.success(f"Columnas restantes: {list(df_limpio.columns)}")
    
    # Manejo de valores nulos
    st.write("**Tratamiento de valores nulos:**")
    
    # Age: reemplazar con media
    edad_media = df_limpio['age'].mean()
    df_limpio['age'].fillna(edad_media, inplace=True)
    st.write(f"- Age: rellenados con media ({edad_media:.2f})")
    
    # Embarked: reemplazar con moda
    moda_embarked = df_limpio['embark_town'].mode()[0]
    df_limpio['embark_town'].fillna(moda_embarked, inplace=True)
    st.write(f"- Embarked: rellenados con moda ({moda_embarked})")
    
    # Eliminar filas con NaN en otras columnas
    df_limpio = df_limpio.dropna()
    st.success(f"Dataset después de limpieza: {df_limpio.shape}")
    
    # Eliminar duplicados
    duplicados = df_limpio.duplicated().sum()
    df_limpio = df_limpio.drop_duplicates()
    st.write(f"- Duplicados eliminados: {duplicados}")
    
    # Codificación de variables categóricas
    st.subheader("4️⃣ Codificación de Variables Categóricas")
    
    # Codificar Sex
    le_sex = LabelEncoder()
    df_limpio['sex_encoded'] = le_sex.fit_transform(df_limpio['sex'])
    st.write(f"- Sex codificado: {dict(zip(le_sex.classes_, le_sex.transform(le_sex.classes_)))}")
    
    # Codificar Embarked
    le_embarked = LabelEncoder()
    df_limpio['embark_town_encoded'] = le_embarked.fit_transform(df_limpio['embark_town'])
    st.write(f"- Embarked codificado: {dict(zip(le_embarked.classes_, le_embarked.transform(le_embarked.classes_)))}")
    
    # Codificar Pclass
    le_pclass = LabelEncoder()
    df_limpio['pclass_encoded'] = le_pclass.fit_transform(df_limpio['pclass'])
    
    # Codificar Survived
    le_survived = LabelEncoder()
    df_limpio['survived_encoded'] = le_survived.fit_transform(df_limpio['survived'])
    
    # Crear dataset procesado
    df_procesado = df_limpio[['pclass_encoded', 'sex_encoded', 'age', 'sibsp', 'parch', 'fare', 'embark_town_encoded', 'survived_encoded']].copy()
    df_procesado.columns = ['PClass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Survived']
    
    st.success("✅ Codificación completada")
    
    # Normalización/Estandarización
    st.subheader("5️⃣ Normalización/Estandarización")
    
    scaler = StandardScaler()
    columnas_numericas = ['Age', 'Fare']
    
    df_estandarizado = df_procesado.copy()
    df_estandarizado[columnas_numericas] = scaler.fit_transform(df_procesado[columnas_numericas])
    
    st.write("**Variables estandarizadas:** Age, Fare")
    st.write("Método: StandardScaler (media=0, desv. est.=1)")
    
    col_comp1, col_comp2 = st.columns(2)
    with col_comp1:
        st.write("**Antes de estandarización:**")
        st.dataframe(df_procesado[columnas_numericas].describe())
    with col_comp2:
        st.write("**Después de estandarización:**")
        st.dataframe(df_estandarizado[columnas_numericas].describe())
    
    # División de datos
    st.subheader("6️⃣ División en Conjuntos de Entrenamiento y Prueba")
    
    X = df_estandarizado.drop('Survived', axis=1)
    y = df_estandarizado['Survived']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    st.write("**División: 70% entrenamiento - 30% prueba**")
    
    col_div1, col_div2, col_div3 = st.columns(3)
    with col_div1:
        st.metric("Entrenamiento", f"{X_train.shape[0]} registros")
    with col_div2:
        st.metric("Prueba", f"{X_test.shape[0]} registros")
    with col_div3:
        st.metric("Total", f"{len(X)} registros")
    
    col_div_shape1, col_div_shape2 = st.columns(2)
    with col_div_shape1:
        st.write("**Shape Entrenamiento:**")
        st.code(f"X_train: {X_train.shape}\ny_train: {y_train.shape}")
    with col_div_shape2:
        st.write("**Shape Prueba:**")
        st.code(f"X_test: {X_test.shape}\ny_test: {y_test.shape}")
    
    # Salida esperada
    st.subheader("📋 Salida Esperada")
    
    st.write("**Primeros 5 registros procesados:**")
    st.dataframe(df_estandarizado.head())
    
    st.write("**Resumen del procesamiento:**")
    resumen = pd.DataFrame({
        'Etapa': ['Dataset Original', 'Después de Limpieza', 'Después de Codificación', 'Después de División (Train)'],
        'Registros': [len(df), len(df_limpio), len(df_procesado), len(X_train)],
        'Características': [len(df.columns), len(df_limpio.columns), len(df_procesado.columns), len(X_train.columns)]
    })
    st.dataframe(resumen, use_container_width=True)
    
    # Visualizaciones
    st.subheader("📊 Visualizaciones")
    
    col_viz1, col_viz2 = st.columns(2)
    
    with col_viz1:
        st.write("**Distribución de Supervivencia (Original)**")
        fig, ax = plt.subplots()
        df['survived'].value_counts().plot(kind='bar', ax=ax, color=['#ff6b6b', '#51cf66'])
        ax.set_title('Supervivencia en el Titanic')
        ax.set_xlabel('Sobrevivió')
        ax.set_ylabel('Cantidad')
        st.pyplot(fig)
    
    with col_viz2:
        st.write("**Distribución por Género**")
        fig, ax = plt.subplots()
        df.groupby('sex')['survived'].value_counts().unstack().plot(kind='bar', ax=ax, color=['#ff6b6b', '#51cf66'])
        ax.set_title('Supervivencia por Género')
        ax.set_xlabel('Género')
        ax.set_ylabel('Cantidad')
        st.pyplot(fig)

elif menu == "Ejercicio 2: Student Performance":
    st.header("📚 Ejercicio 2: Procesamiento del Dataset Student Performance")
    st.markdown("**Objetivo:** Procesar datos para predecir la nota final (G3) de estudiantes")
    st.markdown("---")
    
    # Cargar dataset
    st.subheader("1️⃣ Carga del Dataset")
    
    try:
        df = pd.read_csv('datos/student-mat.csv', sep=',')
        st.success("✅ Dataset Student Performance cargado exitosamente")
        st.write(f"Dimensiones originales: {df.shape}")
    except FileNotFoundError:
        st.error("❌ No se encontró el archivo 'datos/student-mat.csv'")
        st.write("Asegúrate de que el archivo esté en la carpeta 'datos' dentro de tu proyecto")
        st.stop()
    except Exception as e:
        st.error(f"Error al cargar el archivo: {e}")
        st.stop()
    
    # Exploración inicial
    st.subheader("2️⃣ Exploración Inicial")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Primeros 5 registros:**")
        st.dataframe(df.head())
    
    with col2:
        st.write("**Información del Dataset:**")
        st.text(f"Total de registros: {len(df)}")
        st.text(f"Total de columnas: {len(df.columns)}")
    
    if st.checkbox("Mostrar información detallada (Student)"):
        col3, col4 = st.columns(2)
        with col3:
            st.write("**Tipos de Datos:**")
            tipos = pd.DataFrame(df.dtypes, columns=['Tipo'])
            st.dataframe(tipos)
        with col4:
            st.write("**Valores Nulos:**")
            null_data = pd.DataFrame(df.isnull().sum(), columns=['Nulos'])
            null_data['Porcentaje'] = (null_data['Nulos'] / len(df) * 100).round(2)
            st.dataframe(null_data)
    
    # Análisis de variables categóricas
    st.subheader("3️⃣ Análisis de Variables Categóricas")
    
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    st.write(f"**Variables categóricas encontradas:** {len(cat_cols)}")
    
    col_cat1, col_cat2 = st.columns(2)
    
    with col_cat1:
        st.write(cat_cols[:len(cat_cols)//2 + 1])
        for col in cat_cols[:len(cat_cols)//2 + 1]:
            st.write(f"- **{col}:** {df[col].nunique()} valores únicos")
    
    with col_cat2:
        st.write(cat_cols[len(cat_cols)//2 + 1:])
        for col in cat_cols[len(cat_cols)//2 + 1:]:
            st.write(f"- **{col}:** {df[col].nunique()} valores únicos")
    
    # Limpieza de datos
    st.subheader("4️⃣ Limpieza de Datos")
    
    df_limpio = df.copy()
    
    # Eliminar duplicados
    duplicados_antes = len(df_limpio)
    df_limpio = df_limpio.drop_duplicates()
    duplicados_eliminados = duplicados_antes - len(df_limpio)
    st.write(f"**Duplicados eliminados:** {duplicados_eliminados}")
    
    # Detectar valores inconsistentes
    st.write("**Verificación de valores inconsistentes:**")
    
    # Age debe estar entre 15 y 22
    age_inconsistentes = df_limpio[(df_limpio['age'] < 15) | (df_limpio['age'] > 22)].shape[0]
    st.write(f"- Registros con edad fuera de rango [15-22]: {age_inconsistentes}")
    df_limpio = df_limpio[(df_limpio['age'] >= 15) & (df_limpio['age'] <= 22)]
    
    # Absences no debe ser negativo
    absences_negativas = df_limpio[df_limpio['absences'] < 0].shape[0]
    st.write(f"- Registros con ausencias negativas: {absences_negativas}")
    df_limpio = df_limpio[df_limpio['absences'] >= 0]
    
    # G1, G2, G3 deben estar entre 0 y 20
    for col in ['G1', 'G2', 'G3']:
        inconsistentes = df_limpio[(df_limpio[col] < 0) | (df_limpio[col] > 20)].shape[0]
        st.write(f"- Registros con {col} fuera de rango [0-20]: {inconsistentes}")
        df_limpio = df_limpio[(df_limpio[col] >= 0) & (df_limpio[col] <= 20)]
    
    st.success(f"Dataset después de limpieza: {df_limpio.shape}")
    
    # One Hot Encoding
    st.subheader("5️⃣ Codificación de Variables Categóricas (One Hot Encoding)")
    
    df_encoded = df_limpio.copy()
    
    # Seleccionar variables categóricas principales
    cat_to_encode = ['school', 'sex', 'address', 'famsize', 'Pstatus', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic']
    
    st.write(f"**Variables a codificar:** {', '.join(cat_to_encode)}")
    
    # Aplicar One Hot Encoding
    df_encoded = pd.get_dummies(df_encoded, columns=cat_to_encode, drop_first=True, dtype=int)
    
    st.success(f"✅ Codificación completada")
    st.write(f"Nuevas dimensiones: {df_encoded.shape}")
    
    # Normalización de variables numéricas
    st.subheader("6️⃣ Normalización/Estandarización de Variables Numéricas")
    
    columnas_numericas = ['age', 'absences', 'G1', 'G2']
    
    scaler = StandardScaler()
    df_normalizado = df_encoded.copy()
    df_normalizado[columnas_numericas] = scaler.fit_transform(df_encoded[columnas_numericas])
    
    st.write(f"**Variables normalizadas:** {', '.join(columnas_numericas)}")
    st.write("Método: StandardScaler (media=0, desv. est.=1)")
    
    col_norm1, col_norm2 = st.columns(2)
    with col_norm1:
        st.write("**Antes de normalización:**")
        st.dataframe(df_encoded[columnas_numericas].describe())
    with col_norm2:
        st.write("**Después de normalización:**")
        st.dataframe(df_normalizado[columnas_numericas].describe())
    
    # Separar X y y
    st.subheader("7️⃣ Separación de Características (X) y Variable Objetivo (y)")
    
    X = df_normalizado.drop('G3', axis=1)
    y = df_normalizado['G3']
    
    st.write(f"**Variable Objetivo (y):** G3 (Nota Final)")
    st.write(f"**Características (X):** {X.shape[1]} variables")
    
    col_sep1, col_sep2 = st.columns(2)
    with col_sep1:
        st.metric("Cantidad de características", X.shape[1])
    with col_sep2:
        st.metric("Registros totales", len(y))
    
    # División en entrenamiento y prueba
    st.subheader("8️⃣ División en Conjuntos de Entrenamiento y Prueba")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    st.write("**División: 80% entrenamiento - 20% prueba**")
    
    col_div1, col_div2, col_div3 = st.columns(3)
    with col_div1:
        st.metric("Entrenamiento", f"{X_train.shape[0]} registros")
    with col_div2:
        st.metric("Prueba", f"{X_test.shape[0]} registros")
    with col_div3:
        st.metric("Total", f"{len(X)} registros")
    
    col_div_shape1, col_div_shape2 = st.columns(2)
    with col_div_shape1:
        st.write("**Shape Entrenamiento:**")
        st.code(f"X_train: {X_train.shape}\ny_train: {y_train.shape}")
    with col_div_shape2:
        st.write("**Shape Prueba:**")
        st.code(f"X_test: {X_test.shape}\ny_test: {y_test.shape}")
    
    # Salida esperada
    st.subheader("📋 Salida Esperada")
    
    st.write("**Primeros 5 registros procesados:**")
    st.dataframe(df_normalizado.head())
    
    st.write("**Resumen del procesamiento:**")
    resumen = pd.DataFrame({
        'Etapa': ['Dataset Original', 'Después de Limpieza', 'Después de Codificación', 'Después de Normalización', 'Después de División (Train)'],
        'Registros': [len(df), len(df_limpio), len(df_encoded), len(df_normalizado), len(X_train)],
        'Características': [len(df.columns), len(df_limpio.columns), len(df_encoded.columns), len(df_normalizado.columns), len(X_train.columns)]
    })
    st.dataframe(resumen, use_container_width=True)
    
    # Reto adicional: Correlación entre G1, G2 y G3
    st.subheader("🎯 Reto Adicional: Correlación entre Notas (G1, G2, G3)")
    
    # Usar dataset sin normalizar para correlación
    correlaciones = df_limpio[['G1', 'G2', 'G3']].corr()
    
    st.write("**Matriz de Correlación:**")
    st.dataframe(correlaciones)
    
    col_corr1, col_corr2 = st.columns(2)
    
    with col_corr1:
        st.write("**Heatmap de Correlación:**")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(correlaciones, annot=True, cmap='coolwarm', center=0, ax=ax, cbar_kws={'label': 'Correlación'})
        ax.set_title('Correlación entre Notas G1, G2 y G3')
        st.pyplot(fig)
    
    with col_corr2:
        st.write("**Análisis de Correlación:**")
        st.write(f"- G1 vs G2: {correlaciones.loc['G1', 'G2']:.4f} (correlación fuerte)")
        st.write(f"- G1 vs G3: {correlaciones.loc['G1', 'G3']:.4f}")
        st.write(f"- G2 vs G3: {correlaciones.loc['G2', 'G3']:.4f} (mejor predictor)")
        st.info("💡 G2 tiene mayor correlación con G3, indicando que es un buen predictor de la nota final.")
    
    # Visualizaciones adicionales
    st.subheader("📊 Visualizaciones")
    
    col_viz1, col_viz2 = st.columns(2)
    
    with col_viz1:
        st.write("**Distribución de Notas Finales (G3)**")
        fig, ax = plt.subplots()
        ax.hist(df_limpio['G3'], bins=15, color='#4ecdc4', edgecolor='black', alpha=0.7)
        ax.set_title('Distribución de Notas Finales')
        ax.set_xlabel('Nota Final (G3)')
        ax.set_ylabel('Cantidad de Estudiantes')
        st.pyplot(fig)
    
    with col_viz2:
        st.write("**Relación G2 vs G3**")
        fig, ax = plt.subplots()
        ax.scatter(df_limpio['G2'], df_limpio['G3'], alpha=0.6, color='#ff6b6b')
        ax.set_title('Relación entre G2 y G3')
        ax.set_xlabel('Segunda Nota (G2)')
        ax.set_ylabel('Nota Final (G3)')
        st.pyplot(fig)

elif menu == "Ejercicio 3: Iris":
    st.header("🌸 Ejercicio 3: Preprocesamiento del Dataset Iris")
    st.markdown("**Objetivo:** Implementar un flujo completo de preprocesamiento y visualizar resultados")
    st.markdown("---")
    
    # Carga del dataset
    st.subheader("1️⃣ Carga del Dataset desde sklearn")
    
    iris = load_iris()
    df_iris = pd.DataFrame(iris.data, columns=iris.feature_names)
    df_iris['target'] = iris.target
    df_iris['target_name'] = df_iris['target'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
    
    st.success("✅ Dataset Iris cargado exitosamente")
    st.write(f"Dimensiones: {df_iris.shape}")
    
    # Exploración inicial
    st.subheader("2️⃣ Conversión a DataFrame y Nombres de Columnas")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Primeros 5 registros:**")
        st.dataframe(df_iris.head())
    
    with col2:
        st.write("**Información del Dataset:**")
        st.text(f"Total de registros: {len(df_iris)}")
        st.text(f"Total de características: {len(iris.feature_names)}")
        st.text(f"Clases: {iris.target_names.tolist()}")
        st.text(f"Distribución de clases:")
        for i, name in enumerate(iris.target_names):
            count = (df_iris['target'] == i).sum()
            st.text(f"  - {name}: {count} muestras")
    
    # Información detallada
    if st.checkbox("Mostrar información detallada (Iris)"):
        col3, col4 = st.columns(2)
        with col3:
            st.write("**Estadísticas sin Estandarizar:**")
            st.dataframe(df_iris[iris.feature_names].describe())
        with col4:
            st.write("**Nombres de Columnas:**")
            col_names = pd.DataFrame(iris.feature_names, columns=['Características'])
            st.dataframe(col_names)
    
    # Estandarización
    st.subheader("3️⃣ Estandarización con StandardScaler")
    
    scaler = StandardScaler()
    df_estandarizado = pd.DataFrame(
        scaler.fit_transform(df_iris[iris.feature_names]),
        columns=iris.feature_names
    )
    df_estandarizado['target'] = iris.target
    df_estandarizado['target_name'] = df_iris['target_name'].values
    
    st.write("**Método:** StandardScaler (media=0, desviación estándar=1)")
    
    col_stand1, col_stand2 = st.columns(2)
    
    with col_stand1:
        st.write("**Antes de Estandarización:**")
        st.dataframe(df_iris[iris.feature_names].describe())
    
    with col_stand2:
        st.write("**Después de Estandarización:**")
        st.dataframe(df_estandarizado[iris.feature_names].describe())
    
    st.success("✅ Estandarización completada")
    
    # División de datos
    st.subheader("4️⃣ División del Dataset (70% Entrenamiento, 30% Prueba)")
    
    X = df_estandarizado[iris.feature_names]
    y = df_estandarizado['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    col_div1, col_div2, col_div3 = st.columns(3)
    with col_div1:
        st.metric("Entrenamiento", f"{X_train.shape[0]} muestras")
    with col_div2:
        st.metric("Prueba", f"{X_test.shape[0]} muestras")
    with col_div3:
        st.metric("Total", f"{len(X)} muestras")
    
    col_div_shape1, col_div_shape2 = st.columns(2)
    with col_div_shape1:
        st.write("**Shape Entrenamiento:**")
        st.code(f"X_train: {X_train.shape}\ny_train: {y_train.shape}")
    with col_div_shape2:
        st.write("**Shape Prueba:**")
        st.code(f"X_test: {X_test.shape}\ny_test: {y_test.shape}")
    
    # Salida esperada
    st.subheader("📋 Salida Esperada")
    
    st.write("**Primeros 5 registros estandarizados:**")
    st.dataframe(df_estandarizado.head())
    
    st.write("**Resumen del procesamiento:**")
    resumen = pd.DataFrame({
        'Etapa': ['Dataset Original', 'Después de Estandarización', 'División (Train)', 'División (Test)'],
        'Registros': [len(df_iris), len(df_estandarizado), len(X_train), len(X_test)],
        'Características': [len(iris.feature_names), len(iris.feature_names), len(X_train.columns), len(X_test.columns)]
    })
    st.dataframe(resumen, use_container_width=True)
    
    # Visualizaciones
    st.subheader("📊 Visualizaciones")
    
    st.write("**Gráfico de Dispersión: Sepal Length vs Petal Length (diferenciado por clase)**")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Gráfico con datos originales
    colors = ['#ff6b6b', '#4ecdc4', '#45b7d1']
    for i, (target, name) in enumerate(zip([0, 1, 2], iris.target_names)):
        mask = df_iris['target'] == target
        ax1.scatter(
            df_iris[mask]['sepal length (cm)'],
            df_iris[mask]['petal length (cm)'],
            c=colors[i],
            label=name,
            s=100,
            alpha=0.7,
            edgecolors='black',
            linewidth=0.5
        )
    
    ax1.set_xlabel('Sepal Length (cm)', fontsize=11)
    ax1.set_ylabel('Petal Length (cm)', fontsize=11)
    ax1.set_title('Iris Dataset - Datos Originales', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Gráfico con datos estandarizados
    for i, (target, name) in enumerate(zip([0, 1, 2], iris.target_names)):
        mask = df_estandarizado['target'] == target
        ax2.scatter(
            df_estandarizado[mask]['sepal length (cm)'],
            df_estandarizado[mask]['petal length (cm)'],
            c=colors[i],
            label=name,
            s=100,
            alpha=0.7,
            edgecolors='black',
            linewidth=0.5
        )
    
    ax2.set_xlabel('Sepal Length (Estandarizado)', fontsize=11)
    ax2.set_ylabel('Petal Length (Estandarizado)', fontsize=11)
    ax2.set_title('Iris Dataset - Datos Estandarizados', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Análisis adicional
    st.subheader("📈 Análisis Adicional")
    
    col_ana1, col_ana2 = st.columns(2)
    
    with col_ana1:
        st.write("**Correlación entre Características:**")
        corr_matrix = df_iris[iris.feature_names].corr()
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax, 
                    cbar_kws={'label': 'Correlación'}, square=True)
        ax.set_title('Matriz de Correlación - Iris Dataset')
        st.pyplot(fig)
    
    with col_ana2:
        st.write("**Distribución de Características por Clase:**")
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        fig.suptitle('Distribución por Clase', fontsize=14, fontweight='bold')
        
        for idx, feature in enumerate(iris.feature_names):
            ax = axes[idx // 2, idx % 2]
            for i, (target, name) in enumerate(zip([0, 1, 2], iris.target_names)):
                mask = df_iris['target'] == target
                ax.hist(df_iris[mask][feature], alpha=0.6, label=name, color=colors[i], bins=15)
            ax.set_xlabel(feature)
            ax.set_ylabel('Frecuencia')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)