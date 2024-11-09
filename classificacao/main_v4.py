import keras
import keras_tuner as kt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, KFold
from tabulate import tabulate


def build_model(hp):
  model = keras.Sequential()
  model.add(encoder)
  model.add(keras.layers.Embedding(input_dim=len(encoder.get_vocabulary()),
                                   output_dim=hp.Int('embedding_dim', min_value=32, max_value=128, step=32),
                                   mask_zero=True))
  model.add(keras.layers.Bidirectional(keras.layers.LSTM(hp.Int('lstm_units', min_value=32, max_value=128, step=32), return_sequences=True)))
  model.add(keras.layers.Bidirectional(keras.layers.LSTM(hp.Int('lstm_units', min_value=32, max_value=128, step=32))))
  model.add(keras.layers.Dense(hp.Int('dense_units', min_value=32, max_value=128, step=32), activation='relu'))
  model.add(keras.layers.Dropout(rate=hp.Float('dropout', min_value=0.2, max_value=0.5, step=0.1)))
  model.add(keras.layers.Dense(4, activation='softmax'))

  model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4),
                 loss=keras.losses.SparseCategoricalCrossentropy(),
                 metrics=['accuracy'])

  return model

def run_tunner(x, y, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True)

    for train_index, val_index in kf.split(x):
        x_train_fold, x_val_fold = x[train_index], x[val_index]
        y_train_fold, y_val_fold = y[train_index], y[val_index]

        tuner.search(x_train_fold, y_train_fold, epochs=10, validation_data=(x_val_fold, y_val_fold), batch_size=2048)

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    print(f"""
              A pesquisa de hiperparâmetros foi concluída. O número ideal de dimensões de incorporação é {best_hps.get('embedding_dim')},
              o número ideal de unidades LSTM é {best_hps.get('lstm_units')}, e
              o número ideal de unidades densas é {best_hps.get('dense_units')},
              e a taxa de abandono ideal é {best_hps.get('dropout')}.
        """)

    return best_hps


url ='https://github.com/allanspadini/curso-tensorflow-proxima-palavra/raw/main/dados/train.zip'

df = pd.read_csv(url, header=None, names=['ClassIndex', 'Titulo', 'Descricao'], compression='zip')
df['Texto'] = df['Titulo'] + ' ' + df['Descricao']
df['ClassIndex'] = df['ClassIndex'] -1

print(tabulate(df.head(), headers='keys', tablefmt='fancy_grid'))

seed = 4256
tamanho_vocabulario = 1000

keras.utils.set_random_seed(seed)

x_train, x_test, y_train, y_test = train_test_split(df['Texto'].values, df['ClassIndex'].values,
                                                    test_size=0.2, random_state=seed)

encoder = keras.layers.TextVectorization(max_tokens=tamanho_vocabulario)
encoder.adapt(x_train, batch_size=2048)

tuner = kt.Hyperband(build_model,
                     objective='val_accuracy',
                     max_epochs=10,
                     factor=3,
                     directory='logs',
                     project_name='classification_optimization')

params = run_tunner(x_train, y_train)

modelo_final = keras.Sequential()
modelo_final.add(encoder)
modelo_final.add(keras.layers.Embedding(input_dim=tamanho_vocabulario, output_dim=params.get('embedding_dim'), mask_zero=True))
modelo_final.add(keras.layers.Bidirectional(keras.layers.LSTM(params.get('lstm_units'), return_sequences=True)))
modelo_final.add(keras.layers.Bidirectional(keras.layers.LSTM(params.get('lstm_units') // 2)))
modelo_final.add(keras.layers.Dense(params.get('dense_units'), activation='relu'))
modelo_final.add(keras.layers.Dropout(rate=params.get('dropout')))
modelo_final.add(keras.layers.Dense(4, activation='softmax'))

modelo_final.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4),
              loss=keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

modelo_final.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test), batch_size=4096)

y_pred = modelo_final.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)

matrix = confusion_matrix(y_test, y_pred_classes)

classes = np.unique(np.concatenate([y_test, y_pred_classes]))
class_labels = [f"Classe {cls}" for cls in classes]

df_cm = pd.DataFrame(matrix, index=class_labels, columns=class_labels)

plt.figure(figsize=(8, 6))
sns.heatmap(df_cm, annot=True, fmt="d", cmap="Blues", cbar=True)
plt.title("Matriz de Confusão")
plt.ylabel("Classe Real")
plt.xlabel("Classe Prevista")
plt.show()