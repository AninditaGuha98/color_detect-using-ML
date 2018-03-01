
import numpy as np
import matplotlib.pyplot as pl
import math
import pickle
f=open("sample_data","rb")
X=pickle.load(f)
f.close()
f=open("sample_output","rb")
y=pickle.load(f)
f.close()
print(y)
X=X/1000.0
print(X)

# f=open("theta","rb")
# theta=pickle.load(f)
# f.close()
#pl.plot(theta,'ro')
#pl.show()


features=3

X=np.insert(X,0,[1],axis=1)   #input  trained

y=np.array(y)   #output trained
m=247             #No of examples
alpha=2     #learning rate
theta=np.array([0]*int(features+1)).reshape(int(features+1),1)    #initializing theta





def sigmoid(z):

    den = 1.0 + np.e ** (-1.0 * z)

    d = 1.0 / den
    #print(d)
    return d

def cost(theta):

    sigmoid_value=X.dot(theta)
                                                    #print(sigmoid_value)

                                                    # value1=sigmoid(sigmoid_value)
                                                    # value2=np.log(sigmoid(sigmoid_value))
                                                    # value3= 1-sigmoid(sigmoid_value)
                                                    # value4=np.log( 1-sigmoid(sigmoid_value))
                                                    #
                                                    # print("value1:",value1)
                                                    # print("value2",value2)
                                                    # print("value3",value3)
                                                    # print("value4",value4)

    J=(-1./m)*np.sum((y*(np.log(sigmoid(sigmoid_value))) + (1-y)*np.log( 1-sigmoid(sigmoid_value))   ))

    return J





