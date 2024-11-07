import pandas as pd
import tensorflow as tf
import keras

from sklearn.model_selection import train_test_split
from tabulate import tabulate

url ='https://github.com/allanspadini/curso-tensorflow-proxima-palavra/raw/main/dados/train.zip'

df = pd.read_csv(url, header=None, names=['ClassIndex', 'Titulo', 'Descricao'], compression='zip')
df['Texto'] = df['Titulo'] + ' ' + df['Descricao']
df['ClassIndex'] = df['ClassIndex'] -1

print(tabulate(df.head(), headers='keys', tablefmt='fancy_grid'))

seed = 4256
tamanho_vocabulario = 1000

x_train, x_test, y_train, y_test = train_test_split(df['Texto'].values, df['ClassIndex'].values,
                                                    test_size=0.2, random_state=seed)

encoder = keras.layers.TextVectorization(max_tokens=tamanho_vocabulario)
encoder.adapt(x_train)

vocab = encoder.get_vocabulary()

modelo = keras.Sequential()
modelo.add(encoder)
modelo.add(keras.layers.Embedding(input_dim=tamanho_vocabulario, output_dim=16, mask_zero=True))
modelo.add(keras.layers.GlobalAveragePooling1D())
modelo.add(keras.layers.Dense(16, activation='relu'))
modelo.add(keras.layers.Dense(4, activation='softmax'))

modelo.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4),
               loss=keras.losses.SparseCategoricalCrossentropy(),
               metrics=['accuracy'])

modelo.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

print(modelo.predict(x_test[:1]).argmax(axis=1))


