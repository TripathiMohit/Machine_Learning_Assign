import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from numpy.linalg import det, inv
from math import sqrt, pi
import scipy.io
import matplotlib.pyplot as plt
import pickle
import sys






def ldaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmat - A single d x d learnt covariance matrix 
    
    # IMPLEMENT THIS METHOD 
    unique_y = len(np.unique(y))
    classes = unique_y + 1
    mean_vectors = []
    combined = np.column_stack([X, y])
    for clas in range(1,classes):
        data = combined[np.where(combined[:,2] == clas)]
        mean_vectors.append(np.mean(data[:,0:2], axis=0))
    means = np.array(mean_vectors).T
    covmat = np.cov(X.T)
    return means,covmat

def qdaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmats - A list of k d x d learnt covariance matrices for each of the k classes
    
    # IMPLEMENT THIS METHOD
    unique_y = len(np.unique(y))
    classes = unique_y + 1
    mean_vectors = []
    covmats = []
    combined = np.column_stack([X, y])
    for clas in range(1,classes):
        data = combined[np.where(combined[:,2] == clas)]
        mean_vectors.append(np.mean(data[:,0:2], axis=0))
        data_new = data[:,[0,1]]
        covmat = np.cov(data_new.T)
        covmats.append(covmat)
    means = np.array(mean_vectors).T
    return means,covmats

def ldaTest(means,covmat,Xtest,ytest):
    # Inputs
    # means, covmat - parameters of the LDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels

    # IMPLEMENT THIS METHOD
    unique_y = means.shape[1]
    size = Xtest.shape[0]
    X_new = Xtest[:,np.newaxis]
    means_new = means[:,np.newaxis]
    m = means_new.T
    #print("line 73 means", m)
    
    invcovmat = np.linalg.inv(covmat)
    #print("line 76 invcovmat", invcovmat)
    ypred1 = []
    count = 0
    for i in range(size):
        Distance = []
        for j in range(unique_y):
            abc = X_new[i] - m[j]
            Dist = np.dot(np.dot(abc,invcovmat),abc.T)
            Distance.append(Dist)
        Min_dist = min(float(s) for s in Distance)
        Class = Distance.index(Min_dist) + 1
        ypred1.append(Class)
    
    ypred2 = np.array(ypred1)
    ypred = ypred2[:, np.newaxis]
    for k in range(size):
        if ypred[k] == ytest[k]:
            count = count + 1
    acc = count/size
    return acc,ypred

def qdaTest(means,covmats,Xtest,ytest):
    # Inputs
    # means, covmats - parameters of the QDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels

    # IMPLEMENT THIS METHOD
    unique_y = means.shape[1]
    size = Xtest.shape[0]
    
    X_new = Xtest[:,np.newaxis]
    means_new = means[:,np.newaxis]
    m = means_new.T
    ypred1 = []
    count = 0
    for i in range(size):
        Distance = []
        for j in range(unique_y):
            abc = X_new[i] - m[j]
            class_cov = covmats[j]
            invcovmat = np.linalg.inv(class_cov)
            Dist = np.dot(np.dot(abc,invcovmat),abc.T)
            Distance.append(Dist)
        Min_dist = min(float(s) for s in Distance)
        Class = Distance.index(Min_dist) + 1
        ypred1.append(Class)
    
    ypred2 = np.array(ypred1)
    ypred = ypred2[:, np.newaxis]
    for k in range(size):
        if ypred[k] == ytest[k]:
            count = count + 1
    acc = count/size
    return acc,ypred

def learnOLERegression(X,y):
    # Inputs:                                                         
    # X = N x d 
    # y = N x 1                                                               
    # Output: 
    # w = d x 1 
	
    # IMPLEMENT THIS METHOD                                                   
    w = np.dot(np.linalg.inv(np.dot(X.T,X)),np.dot(X.T,y))
    return w

def learnRidgeRegression(X,y,lambd):
    # Inputs:
    # X = N x d                                                               
    # y = N x 1 
    # lambd = ridge parameter (scalar)
    # Output:                                                                  
    # w = d x 1                                                                

    # IMPLEMENT THIS METHOD 
    d = X.shape[1]
    I = np.identity(d, dtype = None)
    w = np.dot(np.linalg.inv((np.dot(X.T,X)) + lambd * I),np.dot(X.T,y)) 
    return w


def testOLERegression(w,Xtest,ytest):
    # Inputs:
    # w = d x 1
    # Xtest = N x d
    # ytest = X x 1
    # Output:
    # mse
    
    # IMPLEMENT THIS METHOD
    N = Xtest.shape[0]
    m = 0
    error = 0
    X_new = Xtest[:,np.newaxis]
    for i in range(N):
        x_pred = np.dot(X_new[i],w)
        error = (ytest[i] - x_pred)**2
        m = m + error
    mse = (1/N)*(m)
    
    return mse

def regressionObjVal(w, X, y, lambd):

    # compute squared error (scalar) and gradient of squared error with respect
    # to w (vector) for the given data X and y and the regularization parameter
    # lambda                                                                  

    # IMPLEMENT THIS METHOD 
    #print("line 186", w.shape)
    w_new = np.asmatrix(w)
    #print("line 188", w.shape)
    a = y - np.dot(X,w_new.T)
    #print(a)
    error = 0.5*(np.dot((a).T,(a)) + lambd*np.dot(w_new,w_new.T))
    error_grad = np.squeeze(np.array(-(np.dot(X.T,a)) + lambd*w_new.T))
    #print("line 195", +error)
    #print("line 193",error_grad.shape)
    return error, error_grad

def mapNonLinear(x,p):
    # Inputs:                                                                  
    # x - a single column vector (N x 1)                                       
    # p - integer (>= 0)                                                       
    # Outputs:                                                                 
    # Xp - (N x (p+1)) 
	
    # IMPLEMENT THIS METHOD
    #print(x.shape)
    #x = np.array([1,2])
    #p = 2
    #print(x.shape)
    N = x.shape[0]
    lis = [np.power(x,i) for i in range(0,p+1)]
    lis = np.array(lis).reshape(p+1,N)
    Xp = np.transpose(lis)
    #print(Xp.shape)
    #Xp = np.array(lis)
    #print(Xp)
    #print(Xp.shape)
    return Xp






# Main script

# Problem 1
# load the sample data                                                                 
if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'),encoding = 'latin1')

# LDA
means,covmat = ldaLearn(X,y)
ldaacc,ldares = ldaTest(means,covmat,Xtest,ytest)
print('LDA Accuracy = '+str(ldaacc))
# QDA
means,covmats = qdaLearn(X,y)
qdaacc,qdares = qdaTest(means,covmats,Xtest,ytest)
print('QDA Accuracy = '+str(qdaacc))

# plotting boundaries
x1 = np.linspace(-5,20,100)
x2 = np.linspace(-5,20,100)
xx1,xx2 = np.meshgrid(x1,x2)
xx = np.zeros((x1.shape[0]*x2.shape[0],2))
xx[:,0] = xx1.ravel()
xx[:,1] = xx2.ravel()

fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)

zacc,zldares = ldaTest(means,covmat,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zldares.reshape((x1.shape[0],x2.shape[0])),alpha=0.3)
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest[:,0])
plt.title('LDA')

plt.subplot(1, 2, 2)

zacc,zqdares = qdaTest(means,covmats,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zqdares.reshape((x1.shape[0],x2.shape[0])),alpha=0.3)
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest[:,0])
plt.title('QDA')

plt.show()






# Problem 2
if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'),encoding = 'latin1')

# add intercept
X_i = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)
Xtest_i = np.concatenate((np.ones((Xtest.shape[0],1)), Xtest), axis=1)

w = learnOLERegression(X,y)
mle = testOLERegression(w,Xtest,ytest)
mle_train = testOLERegression(w,X,y)

w_i = learnOLERegression(X_i,y)
mle_i = testOLERegression(w_i,Xtest_i,ytest)
mle_train_i = testOLERegression(w_i,X_i,y)

print('Test MSE without intercept '+str(mle))
print('Test MSE with intercept '+str(mle_i))
print('Train MSE without intercept '+str(mle_train))
print('Train MSE with intercept '+str(mle_train_i))







# Problem 3
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
mses3_train = np.zeros((k,1))
mses3 = np.zeros((k,1))
for lambd in lambdas:
    w_l = learnRidgeRegression(X_i,y,lambd)
    mses3_train[i] = testOLERegression(w_l,X_i,y)
    mses3[i] = testOLERegression(w_l,Xtest_i,ytest)
    i = i + 1
#print("Mses3_train  in problem 3 is ", +mses3_train)
#print("Mses3_test  in problem 3 is ", +mses3)
fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(lambdas,mses3_train)
plt.title('MSE for Train Data')
plt.subplot(1, 2, 2)
plt.plot(lambdas,mses3)
plt.title('MSE for Test Data')

plt.show()



# Problem 4
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
mses4_train = np.zeros((k,1))
mses4 = np.zeros((k,1))
opts = {'maxiter' : 20}    # Preferred value.                                                
w_init = np.ones((X_i.shape[1],1))
for lambd in lambdas:
    args = (X_i, y, lambd)
    w_l = minimize(regressionObjVal, w_init, jac=True, args=args,method='CG', options=opts)
    w_l = np.transpose(np.array(w_l.x))
    w_l = np.reshape(w_l,[len(w_l),1])
    mses4_train[i] = testOLERegression(w_l,X_i,y)
    mses4[i] = testOLERegression(w_l,Xtest_i,ytest)
    i = i + 1

#print(mses4)    
fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(lambdas,mses4_train)
plt.plot(lambdas,mses3_train)
plt.title('MSE for Train Data')
plt.legend(['Using scipy.minimize','Direct minimization'])

plt.subplot(1, 2, 2)
plt.plot(lambdas,mses4)
plt.plot(lambdas,mses3)
plt.title('MSE for Test Data')
plt.legend(['Using scipy.minimize','Direct minimization'])
plt.show()



# Problem 5
pmax = 7
lambda_opt = 0.06 # REPLACE THIS WITH lambda_opt estimated from Problem 3
mses5_train = np.zeros((pmax,2))
mses5 = np.zeros((pmax,2))
for p in range(pmax):
    Xd = mapNonLinear(X[:,2],p)
    #print()
    Xdtest = mapNonLinear(Xtest[:,2],p)
    w_d1 = learnRidgeRegression(Xd,y,0)
    mses5_train[p,0] = testOLERegression(w_d1,Xd,y)
    mses5[p,0] = testOLERegression(w_d1,Xdtest,ytest)
    w_d2 = learnRidgeRegression(Xd,y,lambda_opt)
    mses5_train[p,1] = testOLERegression(w_d2,Xd,y)
    mses5[p,1] = testOLERegression(w_d2,Xdtest,ytest)
#print(mses5_train)
#print(mses5)
fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(range(pmax),mses5_train)
plt.title('MSE for Train Data')
plt.legend(('No Regularization','Regularization'))
plt.subplot(1, 2, 2)
plt.plot(range(pmax),mses5)
plt.title('MSE for Test Data')
plt.legend(('No Regularization','Regularization'))
plt.show()
