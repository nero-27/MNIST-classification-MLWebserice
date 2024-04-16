"""
Trainer file to train the model
"""

import numpy as np
from util import get_data
from sklearn.ensemble import RandomForestClassifier
import pickle

if __name__=='__main__':
    X,Y=get_data()

    # train on 25% of data
    Ntrain=len(Y)//4
    Xtrain, ytrain=X[:Ntrain], Y[:Ntrain]
    Xtest, ytest=X[Ntrain:], Y[Ntrain:]
    
    model=RandomForestClassifier()
    model.fit(Xtrain, ytrain)

    print("RFClassifier accuracy for MNIST: ", model.score(Xtest, ytest))

    # save the model
    with open('mymodel.pkl', 'wb') as f:
        pickle.dump(model, f)

   