import numpy as np
import theano.tensor as tt

x = np.ones((2,3,4))/4.


y = np.array([[[1,2,3,4],
			   [1,1,1,1],
			   [1,1,1,1]],
			  [[4,3,2,1],
			   [1,1,1,1],
			   [2,2,2,2]]])


a = tt.as_tensor(x)
b = tt.as_tensor(y)
c = tt.pow(a,b)

d = tt.log(c)
e = tt.sum(d,axis=2) 
f = tt.exp(e)
g = f * e
h = tt.sum(g,axis=1)
print h.eval()

#m1, m2 = T.dmatrices('m1', 'm2')