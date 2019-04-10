from keras.models import Sequential
from keras.layers import Dense

class feature_extractor:

    def __init__(self):
        # create model
        self.model = Sequential()
        self.model.add(Dense(12, input_dim=8, activation='relu'))
        self.model.add(Dense(100, activation='relu'))
        self.model.add(Dense(20, activation='softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    def get_features(self):
        pass

    def train(self, X, Y, X_validate, Y_validate):
        # Fit the model
        self.model.fit(X, Y, validation_data=(X_validate, Y_validate), epochs=10, batch_size=200, verbose=2)

    def test(self, X_test, Y_test):
        # Final evaluation of the model
        scores = self.model.evaluate(X_test, Y_test, verbose=0)
        print(scores)

    def save_model(self):
        pass