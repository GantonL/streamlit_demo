import streamlit as st
import numpy as np
import matplotlib.pylab as plt

st.title('Aerodynamics Basics')
st.write('Change the constants to see their effect over the model')

x_vec = np.linspace(-10, 10, 20)
y_vec = np.linspace(-10, 10, 20)

(x, y) = np.meshgrid(x_vec, y_vec)

k = 100
gama = k/2*np.pi
U = 20
r = np.sqrt(x**2+y**2)
theta = np.arctan2(y, x)
R = np.sqrt(gama/U)

k = st.slider('k', 1, 500, 100, 1)
U = st.slider('U', 1, 100, 20, 1)

psi = U*np.sin(theta)*(r-(R**2/r))
phi = U*r*np.cos(theta) + gama*(np.cos(theta)/r)

u_vel = -(U+(gama/r**2))*np.sin(theta)
v_vel = (U-(gama/r**2))*np.cos(theta)

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
