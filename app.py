from joblib import dump, load
model = load('model')

# input symptoms in this manner:
# gender(Male = 1/Female=0), age_year, fever(Yes = 1/No = 0), cough(Yes = 1/No = 0), runny_nose(Yes = 1/No = 0), muscle_soreness(Yes = 1/No = 0), pneumonia(Yes = 1/No = 0), diarrhea(Yes = 1/No = 0), lung_infection(Yes = 1/No = 0), travel_history(Yes = 1/No = 0), isolation_treatment(Yes = 1/No = 0)
predications = model.predict_proba([[ 1., 47.,  0.,  1.,  0.,  0.,  1.,  0.,  1.,  1.,  0.]])
print(predications)
print('Percentage that you have attacked by Covid-19', predications[0][1],"%")