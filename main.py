from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.merging import concatenate

visible = Input(shape=(64, 64, 1))

# pierwsza odnoga ekstrakcji cech
conv_1 = Conv2D(filters=32, kernel_size=(4, 4), activation='relu')(visible)
pool_1 = MaxPooling2D(pool_size=(2, 2))(conv_1)
flat_1 = Flatten()(pool_1)

# druga odnoga ekstrakcji cech
conv_2 = Conv2D(filters=16, kernel_size=(8, 8), activation='relu')(visible)
pool_2 = MaxPooling2D(pool_size=(2, 2))(conv_2)
flat_2 = Flatten()(pool_2)

# połączenie warstw
merge = concatenate([flat_1, flat_2])

# połączenie siecią gęstą
hidden_1 = Dense(units=10, activation='relu')(merge)
output = Dense(units=1, activation='sigmoid')(hidden_1)


model = Model(inputs=visible, outputs=output)
plot_model(model)

from IPython.display import Image
plot_model(model, to_file='SIL.png')
Image('SIL.png')
model.summary()