import pickle
import numpy as np
import os
import json

import tornado.ioloop
import tornado.web

# check if model exists
if not os.path.exists("mymodel.pkl"):
    exit("Cannot run without a model")

with open("mymodel.pkl", 'rb') as f:
    model=pickle.load(f)

class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.write("Hello Tornado!")

class PredictionHandler(tornado.web.RequestHandler):
    def post(self):
        params=self.request.arguments   # arguments returns dictionary (strings)
        x=np.array(list(map(float, params['input'])))    # covert string to float again
        y=model.predict([x])[0]     # predict accepts matrix of size NxD and returns a matrix, y must be a vector so take 0th element
        self.write(json.dumps({'prediction': y.item()}))
        self.finish()


if __name__=='__main__':
    application=tornado.web.Application([
        (r"/", MainHandler),    
        (r"/predict", PredictionHandler) # handler to respons when on /predict url
    ])
    application.listen(8888)
    tornado.ioloop.IOLoop.current().start()
