import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from apyori import apriori

dataset = pd.read_csv('Market_Basket_Optimisation.csv', header=None)

transactions = []

# transactions = dataset.values.tolist()  # This does not work because not every value is string

for i in range(0, len(dataset)):
    transactions.append([str(dataset.values[i, j]) for j in range(0, 20)])

rules = apriori(
    transactions=transactions,
    min_support=((3 * 7) / len(transactions)),
    min_confidence=0.2,
    min_lift=3,
    min_length=2,
    max_length=2,
)

results = list(rules)


def inspect(results):
    lhs = [tuple(result[2][0][0])[0] for result in results]
    rhs = [tuple(result[2][0][1])[0] for result in results]
    supports = [result[1] for result in results]
    confidences = [result[2][0][2] for result in results]
    lifts = [result[2][0][3] for result in results]
    return list(zip(lhs, rhs, supports, confidences, lifts))


resultsInDataFrame = pd.DataFrame(
    inspect(results),
    columns=[
        'Left Hand Side',
        'Right Hand Side',
        'Support',
        'Confidence',
        'Lift'
    ]
)

print(resultsInDataFrame)

resultsInDataFrame = resultsInDataFrame.nlargest(n=len(resultsInDataFrame), columns='Lift')

print(resultsInDataFrame)
