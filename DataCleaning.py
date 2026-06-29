import pandas as pd

#read
df = pd.read_csv(r"C:\Users\joako\Desktop\bd\news.csv") #-------------------your db
# Initial EDA
print("--- Data Info ---")
print(df.info())
print("\n--- Descriptive Statistics ---")
print(df.describe())

# Create copy for cleaning
dfClean = df.copy()

# Missing values
print("\n--- Missing Values Count ---")
print(dfClean.isnull().sum())

print("\n--- Missing Values Percentage ---")
print((dfClean.isnull().sum() / len(dfClean)) * 100)

# Remove duplicates
print("\n--- Duplicate Count ---")
print(dfClean.duplicated().sum())

dfClean = dfClean.drop_duplicates()

print(f"\nClean dataset shape: {dfClean.shape}")
