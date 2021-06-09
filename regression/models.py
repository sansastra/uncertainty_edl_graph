import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, RepeatVector, TimeDistributed, Dropout
from tensorflow.keras.activations import elu, relu

import evidential_deep_learning as edl


def get_model(INPUT_LEN, dim, TARGET_LEN=1, evidential=True):
    hidden_size = 128
    if evidential:
        model = tf.keras.Sequential([
            # Seq2Seq
            LSTM(hidden_size, input_shape=(INPUT_LEN, dim)),
            Dropout(0.1),
            RepeatVector(TARGET_LEN),
            LSTM(hidden_size, return_sequences=True),
            Dropout(0.1),
            Dense(hidden_size, activation=relu),
            # model.add(Dropout(0.2))
            TimeDistributed(Dense(dim, activation=relu)),
            #edl.layers.DenseNormal(2)
            edl.layers.DenseNormalGamma(dim),
        ])
        # Custom loss function to handle the custom regularizer coefficient
        def EvidentialRegressionLoss(true, pred):
            return edl.losses.EvidentialRegression(true, pred, coeff=1e-2)
        def RMSEloss(true, pred):
            return edl.losses.RMSE(true, pred)

        # Compile and fit the model!
        model.compile( optimizer=tf.keras.optimizers.Adam(1e-3), loss=EvidentialRegressionLoss) # RMSEloss
        return model
    else:
        model = tf.keras.Sequential([
            # Seq2Seq
            LSTM(hidden_size, input_shape=(INPUT_LEN, dim)),
            Dropout(0.1),
            RepeatVector(TARGET_LEN),
            LSTM(hidden_size, return_sequences=True),
            Dropout(0.1),
            Dense(hidden_size, activation=relu),
            # model.add(Dropout(0.2))
            TimeDistributed(Dense(dim, activation=relu)),
        ])

        # Compile and fit the model!
        model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
        return model