import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D # <--- This is important for 3d plotting 
A = 1
x0 = 0
y0 = 0

sigma_X = 2
sigma_Y = 2

xg = np.linspace(-6,6,num=100)
yg = np.linspace(-6,6,num=100)

theta= np.pi

X, Y = np.meshgrid(xg,yg)

a = np.cos(theta)**2/(2*sigma_X**2) + np.sin(theta)**2/(2*sigma_Y**2);
b = -np.sin(2*theta)/(4*sigma_X**2) + np.sin(2*theta)/(4*sigma_Y**2);
c = np.sin(theta)**2/(2*sigma_X**2) + np.cos(theta)**2/(2*sigma_Y**2);

aXXdet = np.array([a*(Xi-x0)**2 for Xi in X],float)
bbXYdet = np.array([2*b*(Xi-x0)*(Y[ii]-y0) for ii,Xi in enumerate(X)],float)
cYYdet = np.array([c*(Yi-y0)**2 for Yi in Y],float)
Z = np.array([A*np.exp( - (ai + bbXYdet[i] + cYYdet[i])) for i,ai in enumerate(aXXdet)],float);

# plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap=cm.viridis)
ax.set_xlabel('21.6 mm')
ax.set_ylabel('21.6 mm')
ax.set_zlabel('weight')
ax.set_xticklabels([])
ax.set_yticklabels([])
plt.show()