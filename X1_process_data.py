import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split

df = pd.read_csv('mushrooms.csv')
data = pd.get_dummies(df.drop(columns=['class']), df.columns[1:]).join(df['class'] == 'e')
data.to_csv('data_processed.csv', encoding='utf-8')

