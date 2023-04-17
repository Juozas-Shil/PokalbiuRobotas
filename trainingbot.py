import random
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD


lemmatizer = WordNetLemmatizer()

# loading json file
intents = json.loads(open('intents.json', encoding='utf-8').read())

words = []
classes = []
documents = []
ignore_letters = ['?', '!', '.', ',']

# extracting data:
for intent in intents['intents']:
    for pattern in intent['patterns']:
        wordList = nltk.word_tokenize(pattern) # kiekviena sablona pavercia zodziu sarasu, o ne kaip eilute
        words.extend(wordList)
        documents.append((wordList, intent['tag'])) # kiekviena sablona itraukiam i documents lista kuris susyjes su tag
        if intent['tag'] not in classes: # jei tokio tag nera clases liste itraukiam i clase
            classes.append(intent['tag'])

#kiekvienas zodis paverciamas mazosiomis raidemis
#su kiekvienu zodziu atliekamas lematizavimas tai yra zodis paverciamas pagrindine jo forma.
# zodziai isfiltruojami ir jei sarase yra zodziai su ?, !, ., , tie simboliai panaikinami.
words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_letters]
words = sorted(list(set(words)))

classes = sorted(list(set(classes)))

#issaugojame kintamuosius words ir classes panaudojam wb kad galetume siuos failus perrasyti.
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

training = []
output_empty = [0] * len(classes)

for doc in documents: # iteruojam per documents lista ir kiekvienam priskiriam doc reiksme
    bag = []  # sukuriamas tuscias sarasas i kuri bus pridedami vektoriaus elementai
    word_patterns = doc[0] # priskiriam word_patterns kintamajam doc pirmaji elementa.
    # sukuriamas naujas sarasas mazosiomis raidemis is doc zodziu.
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        # sukuriamas vektorius bag kuriame pridedama reiksme 1 arba 0.
        bag.append(1) if word in word_patterns else bag.append(0)
    output_row = list(output_empty) # sukuriamas naujas sarasas kuris lygus output_empty sarasui
    # pagal doc reiksme nustatomas atitinkamas indeksas output_row sarase, kuriam priskiriam reiksme 1.
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row]) # pridedame i traininga vektorius bag ir output_row

random.shuffle(training) # sumaiso training duomenys atsitiktine tvarka
training = np.array(training) # pakeicia training sarasa i numpy masyva

train_x = list(training[:, 0]) # yra mokymosi duomenu matrica, kurioje kiekviena eilute yra matrica
train_y = list(training[:, 1]) # yra mokymosi kategoriju vektorius

# Neuroninio tinklo mokymosi modelis, algoritmas.
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))
sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model.h5')
