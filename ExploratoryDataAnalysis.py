import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
# Load clean data from previous step
dfClean = pd.read_csv(r"")#------------------your db
# Eda of dfClean
print(f'\nDelitos frecuentes\n{dfClean['tipo_delito'].value_counts()}')
print(f'\nComunas frecuentes\n{dfClean['comuna'].value_counts()}')
print(f'\n{dfClean['descripcion'].value_counts()}')
