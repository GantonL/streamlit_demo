import matplotlib.pylab as plt
import numpy as np
from numpy import pi
import streamlit as st

st.title('Aerodynamics basics')
st.write('Change the constants to see their effect over this sink & source model')

x_vec = np.linspace(-5, 5, 20)
y_vec = np.linspace(-5, 5, 20)

(x, y) = np.meshgrid(x_vec, y_vec)

m = st.slider('m', 0.5*pi, 3*pi, 3*pi)
a = st.slider('a',0.0, 2.0, 1.0, 0.1)
U = st.slider('U',1, 100, 20, 1)
r2 = np.sqrt((x+a)**2+y**2)
r1 = np.sqrt((x-a)**2+y**2)

psi = U*y + (m/(2*pi*np.arctan2(y, x+a))) - (m/(2*pi*np.arctan2(y, x-a)))
phi = U*x + m/(2*pi*np.log(np.sqrt(r2))) - m/(2*pi*np.log(np.sqrt(r1)))

u_vel = U + (m/2*pi)*(1/(np.log(np.sqrt(r2)))**2)*((x+y*a)/r2**2) - (m/2*pi)*(1/(np.log(np.sqrt(r1)))**2)*((x-y*a)/r1**2)
v_vel = (m/2*pi)*(1/(np.log(np.sqrt(r2)))**2)*(y/r2**2) - (m/2*pi)*(1/(np.log(np.sqrt(r1)))**2)*(y/r1**2)

u_vel_norm = u_vel/np.sqrt(u_vel**2+v_vel**2)
v_vel_norm = v_vel/np.sqrt(u_vel**2+v_vel**2)

plt.subplot(3, 1, 1)
plt.contour(x, y, psi, 20)
plt.xlabel('x')
plt.ylabel('y')

plt.subplot(3, 1, 2)
plt.contour(x, y, phi, 20)
plt.xlabel('x')
plt.ylabel('y')

plt.subplot(3, 1, 3)
plt.quiver(x, y, u_vel_norm, v_vel_norm)
plt.xlabel('x')
plt.ylabel('y')

st.pyplot(plt);
