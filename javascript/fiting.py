import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import linregress
import pandas as pd

# SiGe = pd.read_csv('VegardsLaw.dat')
# c = SiGe.Composition
# a = SiGe["Lattice Parameter"]
c=np.array([0.26,0.35,0.42])
a=np.array([473,493,513])
R = 8.314  # Universal gas constant in J/mol K

# Define the Arrhenius equation
def arrhenius(temperature, A, Q):
    return A * np.exp(-Q / (R * temperature))
m,b = np.polyfit(c,a,1)
fit_line =  m*c + b
plt.scatter(c,a)
plt.plot(c, fit_line,'r' )
plt.show()
residuals = a - fit_line
plt.scatter(c, residuals)
plt.show()
Fit = linregress(c,a)
print(Fit)
