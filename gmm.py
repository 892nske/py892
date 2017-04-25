# from numpy.random import *
# from numpy import *
import matplotlib.pylab as plt
import numpy as np
import scipy.stats as sp

def mnd(_x, _mu, _sig):
    x = np.matrix(_x)
    mu = np.matrix(_mu)
    sig = np.matrix(_sig)
    a = np.sqrt(np.linalg.det(sig)*(2*np.pi)**sig.ndim)
    b = np.linalg.det(-0.5*(x-mu)*sig.I*(x-mu).T)
    return np.exp(b)/a


def LogLikelihood( X, MU, SIGMA, PI ):
	# X = np.matrix( _X)
	# MU = np.matrix(_MU)
	# SIGMA = np.matrix(_SIGMA)
	# PI = np.matrix(_PI)

	K = MU.shape[1]
	N = X.shape[0]
	pin = np.zeros((N,K))

	for k in range(K):
		pin[:,k] = mnd(X,MU[:,k].T, SIGMA[:,:,k]*PI[k])

	return np.log(pin.sum(axis=1)).sum(axis=0)

def Estep(X,MU, SIGMA, PI):
	K = MU.shape[1]
	N = X.shape[0]
	gamma = np.zeros((N,K))

	for k in range(K):
		gamma[:, k] = mnd(X, MU[:, k].T, SIGMA[:, :, k] * PI[k])

	return gamma/np.tile(gamma.sum(axis=1),(K,1)).T

def Mstep(gamma,X,MU, SIGMA):
	D = X.shape[1]
	K = MU.shape[1]
	N = X.shape[0]
	Nk = gamma.sum(axis=0)

	for d in range(D):
		MU[d,:] = (gamma * np.tile(X[:,d],(K,1)).T).sum(axis=0) / Nk

	for k in range(K):
		SIGMA[:,:,k] = np.dot( np.tile( gamma[:,k], (D,1) ) * ( X.T - np.tile( MU[:,k], (N,1) ).T ), ( X.T - np.tile( MU[:,k], (N,1) ).T ).T) / Nk[k]

	PI = Nk.T / N

	return (MU, SIGMA, PI)





x = np.array( [] )
y = np.array( [] )

for line in open("OFdata.txt","r"):
	data = line.split()
	x = np.append(x, float(data[1]))
	y = np.append(y, float(data[2]))

X = np.hstack((x.reshape(len(x),1),y.reshape(len(y),1)))
X = sp.stats.zscore(X,axis=0)

K = 2          # クラスタの数
D = 2          # データの要素ベクトルの次元
R = 20
N = len(X)

MU = np.array([[-1,1],[0.5,-2]])
SIGMA = np.dstack((np.eye((D)),np.eye((D))))
PI = np.ones((K,))
PI = PI/K

gamma = np.zeros((N,K))
for k in range(K):
	gamma[:, k] = mnd(X, np.tile(MU[:, k],(N,1)), SIGMA[:, :, k] * PI[k])
print(gamma)
print(X)

MUR = np.zeros((D,K,R+1))
SIGMAR = np.zeros((D,D,K,R+1))
PIR = np.zeros((K,R+1))
gammmaR = np.zeros((N,K,R+1))
LLR = np.zeros((R+1,1))

MUR[:,:,0] = MU
SIGMAR[:,:,:,0] = SIGMA
PIR[:,0] = PI
LLR[0] = LogLikelihood( X, MU, SIGMA, PI )

for r in range(1):
	gamma = Estep(X,MU, SIGMA, PI)
	# print(gamma)
	gammmaR[:,:,r] = gamma
	MU, SIGMA, PI = Mstep(gamma,X,MU, SIGMA)

	MUR[:,:,r+1] = MU
	SIGMAR[:,:,:,r+1] = SIGMA
	PIR[:,r+1]

	LLR[r+1] = LogLikelihood( X, MU, SIGMA, PI )




# plt.figure(1)
# for r in range(R):
# 	gamma=gammmaR[:,:,r]
# 	plt.subplot(4,5,r+1)
# 	plt.scatter(X[:, 0], X[:, 1], c = gamma[:,0],cmap='hsv')
# 	# plt.plot(X[:, 0], X[:, 1], color = [0,1,0])
# 	plt.plot(MUR[0,:,r],MUR[1,:,r],'gx')
#
# plt.show()

# print(gammmaR[:,:,0])