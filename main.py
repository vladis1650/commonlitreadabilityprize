import numpy as np 
import pandas as pd
from sklearn.linear_model import LinearRegression

#Read csv
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

def get_words (excerpt):
    return excerpt.lower().split()

def get_vector_x (excerpt: str, dict: dict) -> np.array:
    transformed_excerpt = [dict.get(i, 0) for i in get_words(excerpt)]
    x_vector = np.zeros(len(dict), dtype=int)
    for i in transformed_excerpt:
        if i == 0:
            continue
        x_vector[i - 1] = 1
    return x_vector


dict = set()
for excerpt in train_data['excerpt']:
    dict.update(get_words(excerpt))

dict = { word:index for index,word in enumerate(dict,1) }

X = train_data['excerpt'].transform(lambda x: get_vector_x(x, dict)).apply(pd.Series)
y = train_data['target']

model = LinearRegression()
model.fit(X, y)

X_test = test_data['excerpt'].transform(lambda x: get_vector_x(x, dict)).apply(pd.Series)
pred = model.predict(X_test)

#Save csv
df = pd.DataFrame({
    'id': test_data['id'],
    'target': pred
})

df.to_csv('res.csv')

