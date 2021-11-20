from joblib import dump, load
model = load('model')
predications = model.predict_proba([[ 0., 25.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.]])
print(predications)
print('Percentage that you have attacked by Covid-19', predications[0][1],"%")