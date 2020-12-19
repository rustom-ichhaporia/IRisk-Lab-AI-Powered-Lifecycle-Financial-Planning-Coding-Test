'''Imports'''

import pandas as pd
import numpy as np
import pickle

df = pd.read_csv('./test_data.csv')
model = pickle.load('model/path')

'''Output Function'''

AGE_INTERVAL = 10
MAX_AGE = 100

def mortality_distributions(people: pd.DataFrame):
    # Handle series instead of DataFrame
    if isinstance(people, pd.Series):
        people = people.to_frame().T

    # Remove weight column if present
    if 'wt' in people.columns: 
        people = people.drop(columns=['wt'])
    
    # Create list of ages to generate mortality predictions
    ages = [num for num in range(0, MAX_AGE, AGE_INTERVAL)]
    output = pd.DataFrame().reindex(columns=ages)

    # Reset age of individual and calculate mortality prediction
    for age in ages: 
        people['age'] = age
        output[age] = model.predict_proba(people)[:, 1]

    # Copy indices
    output = output.set_index(people.index)
    
    return output

print(mortality_distributions(df))