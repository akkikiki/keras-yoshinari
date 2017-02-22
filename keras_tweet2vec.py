import numpy as np
np.random.seed(13)

from keras.models import Model
from keras.layers import Dense, Embedding, GRU, Input, merge
from keras.preprocessing.text import Tokenizer, base_filter
from keras.preprocessing import sequence
from keras.regularizers import l2
from keras.utils import np_utils
from keras.utils.visualize_util import model_to_dot, plot
from IPython.display import SVG
from matplotlib import pyplot as plt
plt.style.use("ggplot")

tweets = []
labels = []
#with open("./data/tweets.tsv") as f:
with open("./trainer_example_keras.txt") as f:
    for l in f:
        tweet, label = l.strip().split("\t")
        print(tweet, label)
        tweets.append(" ".join(list(tweet)))
        labels.append(int(label))
maxlen = 140

tokenizer = Tokenizer(filters="")
tokenizer.fit_on_texts(tweets)
X_train = tokenizer.texts_to_sequences(tweets)
print("Before sequence padding")
print(X_train)
print("After sequence padding")
print(sequence.pad_sequences(X_train, maxlen=140))
#print(X_train)
print(labels)
Y_train = np_utils.to_categorical(labels, len(set(labels)))
print(Y_train)
V = len(tokenizer.word_index) + 1
print(V)

l2_coef = 0.001
tweet = Input(shape=(maxlen,), dtype='int32')
x = Embedding(V, 80, input_length=maxlen, W_regularizer=l2(l=l2_coef))(tweet)
# input dim, output dimension
f = GRU(128, return_sequences=False, W_regularizer=l2(l=l2_coef), b_regularizer=l2(l=l2_coef), U_regularizer=l2(l=l2_coef))(x)
b = GRU(128, go_backwards=True, return_sequences=False, W_regularizer=l2(l=l2_coef), b_regularizer=l2(l=l2_coef), U_regularizer=l2(l=l2_coef))(x)
x = merge([f, b], mode="sum")
x = Dense(len(set(labels)), W_regularizer=l2(l=l2_coef), activation="softmax")(x)

tweet2vec = Model(input=tweet, output=x)

tweet2vec.compile(loss='categorical_crossentropy',
                  optimizer='RMSprop',
                  metrics=['accuracy'])

SVG(model_to_dot(tweet2vec, show_shapes=True).create(prog='dot', format='svg'))
plot(tweet2vec, show_shapes=True, to_file='model.png')

tweet2vec.fit(X_train, Y_train, nb_epoch=10, batch_size=32, validation_split=0.1)
