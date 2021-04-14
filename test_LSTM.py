import time
from pricePredictionLSTM import *

lastval, nextval = trainAndPredict()
print("Last Val: {}\nNext Val: {}".format(lastval,nextval))