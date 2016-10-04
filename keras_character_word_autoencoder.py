import numpy as np
from keras.layers import Dense, Embedding, GRU, Input, merge
from keras.models import Model
np.random.seed(13)

#from keras.models import Model
#from keras.layers import Dense, Embedding, GRU, Input, merge
#from keras.preprocessing.text import Tokenizer, base_filter
#from keras.preprocessing import sequence
#from keras.regularizers import l2
#from keras.utils import np_utils
#from keras.utils.visualize_util import model_to_dot, plot
#from IPython.display import SVG
#from matplotlib import pyplot as plt
#plt.style.use("ggplot")


tweets = []
labels = []
max_words = 0
MAX = 1
word_vector_len = 0
input_layer = Input(shape=(100,), dtype='float32')
encoding_dim = 32  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

encoded = Dense(encoding_dim, activation='relu')(input_layer)
# this model maps an input to its encoded representation
encoder = Model(input=input_layer, output=encoded)

# "decoded" is the lossy reconstruction of the input
decoded = Dense(100, activation='sigmoid')(encoded)
# this model maps an input to its reconstruction
autoencoder = Model(input=input_layer, output=decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

encoded_input = Input(shape=(encoding_dim,))
decoder_layer = autoencoder.layers[-1]
decoder = Model(input=encoded_input, output=decoder_layer(encoded_input))



with open("/Users/Fujinuma/work/crisisNLP_LRL/data/lorelei/cca.100.en") as f:
    for l in f:
        columns = l.strip().split(" ")
        word_vector = columns[1:]
        word = columns[0]
        print(word_vector)
        print(np.array(word_vector))
        if word_vector_len == 0:
            word_vector_len = len(word_vector)
        input_layer = Input(shape=(len(word_vector),), dtype='int32')
        print(input_layer)
        max_words += 1
        numpy_array = np.array([word_vector])
        autoencoder.fit(numpy_array, numpy_array,
                nb_epoch=10,
                batch_size=256,
                shuffle=True,
                validation_data=(numpy_array, numpy_array))

        encoded_imgs = encoder.predict(numpy_array)
        decoded_imgs = decoder.predict(encoded_imgs)
        print(decoded_imgs)
        print(numpy_array)


        if max_words == MAX:
            break
 





#maxlen = 140
#
#tokenizer = Tokenizer(filters="")
#tokenizer.fit_on_texts(tweets)
#X_train = tokenizer.texts_to_sequences(tweets)
#print("Before sequence padding")
#print(X_train)
##X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
#print("After sequence padding")
#print(X_train)
#print(labels)
#Y_train = np_utils.to_categorical(labels, len(set(labels)))
#print(Y_train)
#V = len(tokenizer.word_index) + 1
#print(V)
#
#l2_coef = 0.001
#tweet = Input(shape=(maxlen,), dtype='int32')
#x = Embedding(V, 80, input_length=maxlen, W_regularizer=l2(l=l2_coef))(tweet)
## input dim, output dimension
#f = GRU(128, return_sequences=False, W_regularizer=l2(l=l2_coef), b_regularizer=l2(l=l2_coef), U_regularizer=l2(l=l2_coef))(x)
#b = GRU(128, go_backwards=True, return_sequences=False, W_regularizer=l2(l=l2_coef), b_regularizer=l2(l=l2_coef), U_regularizer=l2(l=l2_coef))(x)
#x = merge([f, b], mode="sum")
#x = Dense(len(set(labels)), W_regularizer=l2(l=l2_coef), activation="softmax")(x)
#
#tweet2vec = Model(input=tweet, output=x)
#
#tweet2vec.compile(loss='categorical_crossentropy',
#                  optimizer='RMSprop',
#                  metrics=['accuracy'])
#
#SVG(model_to_dot(tweet2vec, show_shapes=True).create(prog='dot', format='svg'))
#plot(tweet2vec, show_shapes=True, to_file='model.png')
#
#tweet2vec.fit(X_train, Y_train, nb_epoch=10, batch_size=32, validation_split=0.1)
