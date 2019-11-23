from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose,UpSampling2D,MaxPooling2D
from tensorflow.keras.models import Model

def testConv():
    input = Input(shape=(3,3,3))
    y = Conv2DTranspose(30, 3, (2,2), padding='same' )(input)
    m = Model(inputs=input, outputs=y)
    return m

if __name__ == '__main__':
    testConv()