import PIL.Image
import PIL
from enum import Enum
import numpy as np
import os

class DoodleClass(Enum):
    Apple = "Apple"
    Pencil = "Pencil"
    Teapot = "Teapot"
    Bicycle = "Bicycle"
    Umbrella = "Umbrella"
    
    def all() -> list:
        return list(DoodleClass.__members__.values())
# class Guesser():
#     def __init__(self, layers: list[int]) -> None:
#         self.layers = layers
#         self.biases = [np.random.randn(y, 1) for y in layers[1:]]
#         self.weights = [np.random.randn(y, x)
#                         for x, y in zip(layers[:-1], layers[1:])]
#         self.lr = 0.01
#         self.epochs = 100
#         train = load_train()
#         random.shuffle(train)
#         self.train(train[0])
        
#     def guess(self, image: PIL.Image.Image): # -> list[tuple[DoodleClass, float]]:
#         input = [float(abs(int(i[0]) // 255 - 1)) for i in list(image.getdata())]
#         return self.feed(input)
        
#     def feed(self, input: list[float]):
#         layers = []
#         for b, w in zip(self.biases, self.weights):
#             input = sig(np.dot(w, input)+b)
#             layers.append(input)
        
#         return (layers, [sum(i) / len(i) for i in input])
    
#     def train(self, set: tuple[DoodleClass, list[float]]):
        
#         num_samples = np.shape(set[1])
#         for epoch in range(self.epochs):
#             index = DoodleClass.all().index(set[0])
#             y_target = [0.0 for i in range(len(DoodleClass))]
#             y_target[index] = 1.0
            
#             feed = self.feed(set[1])
#             y_pred = feed[1]
        
#             y_pred = np.array(y_pred)
#             y_target = np.array(y_target)
            
#             mse = ((y_pred - y_target) ** 2).mean()
#             print(f"Epoch {epoch+1}/{self.epochs}, MSE: {mse:.4f}")
            
#             layer1 = np.array(feed[0][0])
#             layer2 = np.array(feed[0][1])
#             X_train = np.array(set[1])
#             diff = y_pred - y_target
#             delta2 = diff * sigprime(layer2)
#             delta1 = np.dot(delta2, self.weights[1].T) * sigprime(layer1)
            
#             self.weights[1] += self.lr * np.dot(layer1.T, delta2) / num_samples
#             self.biases[1] += self.lr * np.mean(delta2, axis=0)
#             self.weights[0] += self.lr * np.dot(X_train.T, delta1) / num_samples
#             self.biases[0] += self.lr * np.mean(delta1, axis=0)
            
# def sig(x: float) -> float:
#     return 1 / (1 + math.e ** x)
# def sigprime(x: float) -> float:
#     return sig(x) * (1 - sig(x))

import helper

def load_train() -> list[tuple[list[float], list[float]]]:
    a = []
    for (r,_,f) in os.walk('Training'):
        for i in f:
            im = PIL.Image.open(f"{r}/{i}")
            imdat = helper.to_data(im)
            obj = i[:i.find('_')]
            index = list(DoodleClass.__members__.keys()).index(obj)
            l = [0.0 for i in range(5)]
            l[index] = 1.0
            a.append((l, imdat))
    return a

# guesser = Guesser([140625, 10, 10, 5])
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

def train(set: list[tuple[list[float], list[float]]]) -> LinearRegression:
    X = [i[1] for i in set]
    y = [i[0] for i in set]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X, y)
    return model
