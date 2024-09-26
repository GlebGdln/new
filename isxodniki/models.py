import numpy as np # линейная алгебра
import pandas as pd # обработка данных
 
data = pd.read_csv(r"C:\Users\glebg\OneDrive\proekt\yield_df.csv")

# Отображение основной информации и описательной статистики для каждого столбца
info = data.info()
description = data.describe(include='all')

info, description

# График для каждого столбца с соответствующей визуализацией

import matplotlib.pyplot as plt
import seaborn as sns

# Задаём стиль участков
sns.set_style("whitegrid")

# Увеливаем размер рисунка по умолчанию для лучшей читабельности 
plt.rcParams['figure.figsize'] = [10, 6]

## Площадь (столбчатая диаграмма 10 лучших областей)
plt.figure()
area_counts = data['Area'].value_counts().head(10)
sns.barplot(x=area_counts.values, y=area_counts.index, palette='coolwarm')
plt.title('Top 10 Areas')
plt.xlabel('Count')
plt.ylabel('Area')
plt.show() 

## Растение (столбчатая диаграмма растений)
plt.figure()
item_counts = data['Item'].value_counts()
sns.barplot(x=item_counts.values, y=item_counts.index, palette='viridis')
plt.title('Items Count')
plt.xlabel('Count')
plt.ylabel('Item')
plt.show() 

## Год (линейный график записей по годам)
plt.figure()
year_counts = data['Year'].value_counts().sort_index()
sns.lineplot(x=year_counts.index, y=year_counts.values, marker='o', color='b')
plt.title('Entries Over Years')
plt.xlabel('Year')
plt.ylabel('Number of Entries')
plt.show() 

## hg/ha_yield (гистограмма)
plt.figure()
sns.histplot(data['hg/ha_yield'], bins=30, kde=True, color='g')
plt.title('Distribution of hg/ha_yield')
plt.xlabel('hg/ha_yield')
plt.ylabel('Frequency')
plt.show() 

## среднее_количество_осадков_мм_в_год (гистограмма)
plt.figure()
sns.histplot(data['average_rain_fall_mm_per_year'], bins=30, kde=True, color='r')
plt.title('Distribution of Average Rainfall (mm/year)')
plt.xlabel('Average Rainfall (mm/year)')
plt.ylabel('Frequency')
plt.show() 

## пестициды тонн (гистограмма)
plt.figure()
sns.histplot(data['pesticides_tonnes'], bins=30, kde=True, color='m')
plt.title('Distribution of Pesticides (tonnes)')
plt.xlabel('Pesticides (tonnes)')
plt.ylabel('Frequency')
plt.show() 

## средняяя температура (гистограмма)
plt.figure()
sns.histplot(data['avg_temp'], bins=30, kde=True, color='c')
plt.title('Distribution of Average Temperature')
plt.xlabel('Average Temperature (°C)')
plt.ylabel('Frequency')
plt.show() #

# Создание корреляционной тепловой карты для числовых столбцов в наборе данных

# Рассчитать матрицу корреляции
corr_matrix = data.corr(numeric_only=True)

# Создаём тепловую карту для визуализации корреляционной матрицы
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Heatmap of Dataset')
plt.show() 

# Обучаем XGBoost и Random Forest прогнозированию урожайности на основе страны, товара, колва пестицидов, средней температуры и количества осадков

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor


# Переименуем столбцы для удобства

data_renamed = data.rename(columns={
    "hg/ha_yield": "Yield",
    "average_rain_fall_mm_per_year": "Rainfall",
    "pesticides_tonnes": "Pesticides",
    "avg_temp": "Avg_Temp"
})

# Удалиv столбец индекса, тк он не нужен
data_cleaned = data_renamed.drop(columns=["Unnamed: 0"])

# Кодировать категориальные переменные
le_country = LabelEncoder()
le_item = LabelEncoder()
data_cleaned['Country_Encoded'] = le_country.fit_transform(data_cleaned['Area'])
data_cleaned['Item_Encoded'] = le_item.fit_transform(data_cleaned['Item'])

# Определим признаки и целевую переменную
X = data_cleaned[['Country_Encoded', 'Item_Encoded', 'Pesticides', 'Avg_Temp', 'Rainfall']]
y = data_cleaned['Yield']

# Разделение данных на обучающие и тестовые наборы
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Инициализируем и обучаем модель XGBoost
xgb_model = XGBRegressor(objective='reg:squarederror', n_estimators=100)
xgb_model.fit(X_train, y_train)

rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)

# Прогноз на тестовом наборе
y_pred_xgb = xgb_model.predict(X_test)
y_pred_rf = rf_model.predict(X_test)


# Оценка моделей
rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
accuracy_xgb = xgb_model.score(X_test, y_test)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
accuracy_rf = rf_model.score(X_test, y_test)

print(f"RMSE XGBoot: {rmse_xgb}")
print(f"Acc XGBoost: {accuracy_xgb}")

print(f"RMSE RF: {rmse_rf}")
print(f"Acc RF: {accuracy_rf}")

plt.scatter(y_test, y_pred_xgb,s=10,color='#9B673C')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values XGB')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='green', linewidth = 4)
plt.show()

plt.scatter(y_test, y_pred_rf,s=10,color='#9B673C')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values RF')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='green', linewidth = 4)
plt.show()

import joblib

# Сохранение модели Random Forest
joblib.dump(rf_model, r"C:\Users\glebg\OneDrive\proekt\random_forest_model.pkl")

joblib.dump(xgb_model, r"C:\Users\glebg\OneDrive\proekt\xgb_model.pkl")

