import random
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


alphabeta = list('abcdefghijklmnopqrstuvwxyz')
values = ['.-', '-...', '-.-.', '-..', '.', '..-.', '--.', '....', '..', '.---', '-.-', '.-..', '--', '-.', '---', '.--.', '--.-', '.-.', '...', '-','..-', '...-', '.--', '-..-', '-.--', '--..']
morse_dict = dict(zip(alphabeta, values))
def morse_encode(word):
    return '*'.join([morse_dict[w] for w in word])

word_len = 9
max_len_x = 4*word_len+ (word_len-1)
max_len_y = word_len

print( 'max_len_x=%d, max_len_y=%d' % (max_len_x, max_len_y))

def data_gen(n):
    with open('words_alpha.txt','r') as f:
        all_words = f.read().lower().split('\n')
        words = [word for word in all_words if len(word) == n]
        random.shuffle(words)
        
        g_out = lambda x: ' '*(max_len_y-len(x)) + x
        output_list = [g_out(word) for word in words]

        g_in = lambda x: morse_encode(x) + ' ' * (max_len_x - len( morse_encode(x)))
        input_list = [g_in(word) for word in words]

        return output_list, input_list

output_list, input_list = data_gen(9)
# print( output_list, input_list)

class CharTable(object):
    def __init__(self, chars):
        self.chars = chars
        self.char_indices = dict((c,i) for i,c in enumerate(chars))
        self.indices_char = dict((i,c) for i,c in enumerate(chars))

    def encode(self, token, num_rows):
        x = np.zeros((num_rows,len(self.chars)))
        for i, c in enumerate(token):
            x[i,self.char_indices[c]] = 1
        return x

    def decode(self, x, calc_argmax=True):
        if calc_argmax:
            x = x.argmax(axis=-1)
        return ' '.join( self.indices_char[i] for i in x)
    

chars_in = '*-. '
chars_out = 'abcdefghijklmnopqrstuvwxyz '
ctable_in = CharTable( chars_in )
ctable_out = CharTable( chars_out )

x = np.zeros((len(input_list), max_len_x, len(chars_in)))
y = np.zeros(( len(output_list), max_len_y, len(chars_out)))

for i,token in enumerate(input_list):
    x[i] = ctable_in.encode(token, max_len_x)

for i,token in enumerate(output_list):
    y[i] = ctable_out.encode(token, max_len_y)

m = len(x)//4

(x_train, x_val ) = x[:m], x[m:]
(y_train, y_val ) = y[:m], y[m:]

from keras.models import Sequential
from keras import layers

model = Sequential()
latent_dim = 256
model.add( layers.LSTM(latent_dim,input_shape=(max_len_x, len(chars_in))))
model.add( layers.RepeatVector(max_len_y))
model.add( layers.LSTM(latent_dim, return_sequences=True))
model.add(layers.TimeDistributed(layers.Dense(len(chars_out))))
model.add( layers.Activation('softmax'))
model.compile( loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
model.summary()

Epochs = 1
Batch_size = 1024

hist = model.fit( x_train, y_train, batch_size = Batch_size, epochs=Epochs, validation_data=(x_val, y_val))

model.save("morse.h5")

plt.figure(figsize=(20,5)) 
plt.subplot(121) 
plt.plot(hist.history['accuracy']) 
plt.plot(hist.history['val_accuracy']) 
plt.title('model accuracy') 
plt.ylabel('accuracy') 
plt.xlabel('epoch') 
plt.legend(['train','validation'], loc='upper left') 
plt.subplot(122) 
plt.plot(hist.history['loss']) 
plt.plot(hist.history['val_loss']) 
plt.title('model loss') 
plt.ylabel('loss') 
plt.xlabel('epoch') 
plt.legend(['train','validation'], loc='upper right') 
plt.show()