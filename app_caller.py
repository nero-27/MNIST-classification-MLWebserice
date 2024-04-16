import requests
import matplotlib.pyplot as plt
from util import get_data
import numpy as np
import json

X,Y=get_data()
N=len(Y)

while True:
    i=np.random.choice(N)   # pick random data point
    r=requests.post("http://localhost:8888/predict", data={'input': X[i]}) # send it to server
    print(r.content)
    j=r.json()
    print(j)
    print("Target: ", Y[i])

    plt.imshow(X[i].reshape(28,28), cmap='gray')
    plt.show()

    response=input("Continue? (y/n) : ", )
    if response in ('n', 'N'):
        break