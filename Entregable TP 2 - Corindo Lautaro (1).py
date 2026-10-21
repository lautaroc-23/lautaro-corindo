#!/usr/bin/env python
# coding: utf-8

# In[47]:


# Importo la librería pandas, la herramienta esencial para trabajar con DataFrames.
import pandas as pd


# In[49]:


# Cargo el archivo .xlsx en un DataFrame llamado 'df_vino'.
df_vino = pd.read_excel("winemag-data-depurado.xlsx")


# In[50]:


# Reviso las primeras 5 filas para verificar el formato y la carga.
print("Revisión de las primeras filas:")
print(df_vino.head())


# In[7]:


print("\nDimensiones del conjunto de datos (filas, columnas):")
print(df_vino.shape)


# In[51]:


# Veo un resumen conciso: nombres de columnas, tipos de datos y conteo de valores no nulos.
print("\nEstructura y Tipos de Datos (info):")
df_vino.info()


# In[11]:


print("Nombres de todas las variables (columnas):")
print(df_vino.columns.tolist())	


# In[12]:


print("\nTipos de datos y Conteo de No Nulos:")
df_vino.info()	


# In[14]:


print("Resumen Estadístico de Variables Numéricas:")
print(df_vino[['points', 'price']].describe().T)


# In[15]:


print("\nTop 5 Países (country):")
print(df_vino['country'].value_counts().head(5))	


# In[16]:


print("\nTop 5 Variedades de Uva (variety):")
print(df_vino['variety'].value_counts().head(5))	


# In[17]:


print("\nTotal de Bodegas (winery) únicas:")
print(df_vino['winery'].nunique())	


# In[18]:


import matplotlib.pyplot as plt
import seaborn as sns	


# In[20]:


plt.figure(figsize=(10, 6))
sns.countplot(x='points', data=df_vino, palette='viridis')
plt.title('Distribución de Puntuaciones de Vino')
plt.xlabel('Puntuación (80-100)')	
plt.ylabel('Número de Reseñas')	
plt.show()


# In[21]:


plt.figure(figsize=(10, 6))
sns.histplot(df_vino['price'].dropna(), bins=50, kde=True)
plt.title('Distribución de Precios (Ignorando NAs)')	
plt.xlabel('Precio (USD)')	
plt.xlim(0, 300)
plt.show()


# In[22]:


plt.figure(figsize=(8, 4))
sns.boxplot(x=df_vino['price'].dropna())	
plt.title('Detección de Outliers en Precios')	
plt.show()


# In[66]:


# Gráfico: Top 10 Países por Número de Reseñas
top_countries = df_vino['country'].value_counts().nlargest(10)
plt.figure(figsize=(12, 6))
# Uso de 'hue' y 'legend=False' para satisfacer la buena práctica de seaborn
sns.barplot(x=top_countries.index, y=top_countries.values, palette='RdPu', hue=top_countries.index, legend=False)
plt.title('Top 10 Países por Número de Reseñas')
plt.ylabel('Número de Reseñas')
plt.xlabel('País')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


# In[26]:


print("Conteo de valores faltantes (NaN) por columna:")
print(df_vino.isnull().sum())


# In[29]:


import missingno as msno
msno.matrix(df_vino)
plt.title('Matriz de Datos Faltantes', fontsize=20)
plt.show()


# In[52]:


# Eliminación de filas con NAs en variables geográficas críticas (country, province, region_1). 
# El número de filas afectadas es ínfimo (menos del 0.05% del total). 
# Eliminar estas filas es el método más seguro para evitar sesgos por imputación en variables geográficas críticas.
df_vino.dropna(subset=['country', 'province', 'region_1'], inplace=True)
print(f"Filas restantes después de la eliminación: {df_vino.shape[0]}")


# In[62]:


# Dado el alto volumen de faltantes (30-60%), la eliminación resultaría en una pérdida masiva de datos y un sesgo potencial.
# La imputación categórica preserva los datos de otras columnas y crea una categoría que puede ser analíticamente significativa
# Imputación Categórica: Reemplazar NAs por una categoría 'Missing'
df_vino['designation'].fillna('Missing Designation', inplace=True)
df_vino['region_2'].fillna('No Region 2', inplace=True)
df_vino['taster_twitter_handle'].fillna('No Twitter', inplace=True)


# In[63]:


# El precio tiene un nivel moderado de faltantes (aprox. 6.9%). 
# Usar la mediana es la técnica de imputación avanzada más adecuada para distribuciones como el precio, ya que previene que los outliers sesguen los valores imputados.
# Imputación Numérica: Reemplazar NAs en 'price' con la mediana
mediana_precio = df_vino['price'].median()
df_vino['price'] = df_vino['price'].fillna(mediana_precio)
print(f"Precio Mediano usado para imputación: {mediana_precio}")


# In[35]:


# Verificar que no quedan NAs en las columnas clave
print("\nVerificación de NAs después del tratamiento:")
print(df_vino[['country', 'price', 'designation', 'region_2']].isnull().sum())


# In[40]:


# Re-importar librerías para mayor seguridad
import matplotlib.pyplot as plt	
import seaborn as sns


# In[41]:


plt.figure(figsize=(12, 5))
sns.boxplot(x=df_vino['points'])	
plt.title('Boxplot de Puntuaciones (Points)')	
plt.show()


# In[43]:


plt.figure(figsize=(12, 5))
# Usamos log(price) para mejor visualización, pero el análisis es sobre los datos originales.	
sns.boxplot(x=df_vino['price'])
plt.title('Boxplot de Precios (Price)')
plt.xlim(0, 500)
plt.show()


# In[67]:


# TRATAMIENTO DE OUTLIERS EN PRICE (CAPPING BASADO EN IQR)
Q1 = df_vino['price'].quantile(0.25)
Q3 = df_vino['price'].quantile(0.75)
IQR = Q3 - Q1
limite_superior = Q3 + 1.5 * IQR	
limite_inferior = Q1 - 1.5 * IQR	
print(f"Límite Inferior (Floor): {limite_inferior:.2f}")	
print(f"Límite Superior (Cap): {limite_superior:.2f}")


# In[68]:


# Aplicar Capping: Reemplazar valores por encima/debajo de los límites
df_vino['price'] = df_vino['price'].clip(lower=limite_inferior, upper=limite_superior)
print("Outliers en 'price' tratados mediante Capping.")


# In[69]:


# Visualización POST-TRATAMIENTO para confirmar el efecto	
plt.figure(figsize=(12, 5))	
sns.boxplot(x=df_vino['price'])	
plt.title('Boxplot de Precios POST-CAPPING')	
plt.show()


# In[ ]:




