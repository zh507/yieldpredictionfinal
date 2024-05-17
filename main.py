import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Loading the datasets
pesticides_df = pd.read_csv('data/pesticides.csv')
rainfall_df = pd.read_csv('data/rainfall.csv')
temp_df = pd.read_csv('data/temp.csv')
yield_df = pd.read_csv('data/yield.csv')

pd.options.display.width = 0

# Dropping columns that are not required
yield_df = yield_df.drop(['Year Code', 'Element Code', 'Element', 'Year Code', 'Area Code', 'Domain Code', 'Domain', 'Unit', 'Item Code'], axis=1)

# Renaming the Value column
yield_df.rename(columns={'Value': 'hg/ha_yield'}, inplace=True)

# Removing space in front of 'Area' column head
rainfall_df.rename(columns={' Area': 'Area'}, inplace=True)

# Using to_numeric and coercing errors to NaN
rainfall_df['average_rain_fall_mm_per_year'] = pd.to_numeric(rainfall_df['average_rain_fall_mm_per_year'], errors='coerce')

# Dropping empty rows
rainfall_df = rainfall_df.dropna()

# Merging yield and rainfall datasets by 'Year' and 'Area'
final_df = pd.merge(yield_df, rainfall_df, on=['Year', 'Area'])

# Dropping unwanted columns
pesticides_df = pesticides_df.drop(['Domain', 'Element', 'Item', 'Unit'], axis=1)

# Changing 'Value' column to 'pesticides_amount'
pesticides_df.rename(columns={'Value': 'pesticides_amount'}, inplace=True)

# Merging pesticides dataset with final dataset
final_df = pd.merge(final_df, pesticides_df, on=['Year', 'Area'])

# Changing 'year' and 'country' to more suitable headings
temp_df.rename(columns={'year': 'Year', 'country': 'Area'}, inplace=True)

# Dropping null values
temp_df = temp_df.dropna()

# Merging temperature dataset with final dataset
final_df = pd.merge(final_df, temp_df, on=['Year', 'Area'])

# Changing Rice, paddy to just Rice
final_df['Item'] = final_df['Item'].str.replace('Rice, paddy', 'Rice')

# Dropping Plantains and others from the dataset
final_df = final_df[final_df['Item'] != 'Plantains and others']

# Encoding categorical data and scaling numerical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['Year', 'average_rain_fall_mm_per_year', 'pesticides_amount', 'avg_temp']),
        ('cat', OneHotEncoder(), ['Area', 'Item'])
    ])

# Splitting the data into features and target variable
X = final_df.drop(['hg/ha_yield'], axis=1)
y = final_df['hg/ha_yield']

print(final_df.describe())

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initializing models
models = {
    "Linear Regression": LinearRegression(),
    "K-Neighbors Regressor": KNeighborsRegressor(),
    "Decision Tree Regressor": DecisionTreeRegressor(),
    "Gradient Boosting Regressor": GradientBoostingRegressor(random_state=42),
    "Random Forest Regressor": RandomForestRegressor()
}

# Training and evaluating each model
for model_name, model in models.items():
    # Creating a pipeline with the preprocessor and the regressor
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('regressor', model)])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"{model_name}:")
    print(f"  Mean Squared Error: {mse}")
    print(f"  R^2 Score: {r2}\n")

# Predictive System using Decision Tree Regressor
dtr = DecisionTreeRegressor(random_state=42)
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('regressor', dtr)])
pipeline.fit(X_train, y_train)

# Predictive System function
def prediction(Year, average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp, Area, Item):
    features = pd.DataFrame([[Year, average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp, Area, Item]],
                            columns=['Year', 'average_rain_fall_mm_per_year', 'pesticides_amount', 'avg_temp', 'Area', 'Item'])
    transformed_features = pipeline['preprocessor'].transform(features)
    predicted_yield = pipeline['regressor'].predict(transformed_features)
    return predicted_yield[0]

# Example prediction
result = prediction(1990, 1485.0, 121.0, 16.37, 'Albania', 'Maize')
print(f"Predicted Yield: {result}")
