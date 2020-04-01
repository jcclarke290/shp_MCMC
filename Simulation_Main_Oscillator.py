"""
Senior Honours Project - Simulation Final

Created: 08/03/2020 - James C. Clarke

Produces the Results that are used by the analysis program to calculate observables
Optionally Provides a Visualisation of the process occuring
"""

import math as m
import numpy as np
import matplotlib.pyplot as plt


#Structure:
#1. Decide on type of potential
#2. take in lattice length and mass and frequency values, base Nsep on lattice length
#3. enter iterations to run test simulation for and plot observables to get equilibriation
#4. cut the results array so that only measureable states are left in the array
#5. dump these results to file (preferably csv) with parameters in filename

"""
Classical Actions -------------------------------------------------------------------------------------------------------------
"""

def action_NN(change, left, right):
    #calculates the action in the region around the altered lattice point to save computation
    delta_s = 0.0
    left_change = (0.5 * mass_dim * (change - left)**2 + 0.5 * mass_dim * (omega_dim**2) * (left**2) )
    change_right = (0.5 * mass_dim * (right - change)**2 + 0.5 * mass_dim * (omega_dim**2) * (change**2) )
    delta_s = left_change + change_right
    return delta_s

def action_anharmonic_NN(change, left, right):
    delta_s = 0.0
    left_change = (0.5 * mass_dim * (change - left)**2 - 0.5 * mass_dim * (omega_dim**2) * (left**2) + 0.25 * (left**4))
    change_right = (0.5 * mass_dim * (right - change)**2 - 0.5 * mass_dim * (omega_dim**2) * (change**2) + 0.25 * (change**4))
    delta_s = left_change + change_right
    return delta_s
"""
-------------------------------------------------------------------------------------------------------------------------------
"""

"""
Wrap Functions ----------------------------------------------------------------------------------------------------------------
"""
def wrap_size(j, size):
    #produces the periodic boundary conditions of the lattice with a value for given size of lattice
    if j >= (size):
        while j >= (size):
            j = j - (size)
        return j

    elif j < 0:
        while j < 0:
            j = j + (size)
        return j

    else:
        #print("keep same")
        return j

def wrap(j):
    #produces the periodic boundary conditions of the lattice
    if j >= latt_size:
        while j >= latt_size:
            j = j - latt_size
        return j

    elif j < 0:
        while j < 0:
            j = j + latt_size
        return j

    else:
        #print("keep same")
        return j
"""
-------------------------------------------------------------------------------------------------------------------------------
"""

"""
Expected X^2 value ------------------------------------------------------------------------------------------------------------
"""

def calc_x_2_expect():
    R = 1 + ((omega_dim **2) *0.5) - (omega_dim*( m.pow( (1+ 0.25* omega_dim**2), 0.5)))
    preFactor = 1 / ((2*mass_dim*(omega_dim**2)) * m.pow( (1+ 0.25* omega_dim**2), 0.5))
    R_frac = (1 + m.pow(R, latt_size)) / (1- m.pow(R, latt_size))

    return preFactor*R_frac

"""
-------------------------------------------------------------------------------------------------------------------------------
"""

"""
Observables as a list ---------------------------------------------------------------------------------------------------------
"""

def avg_x_list(raw_lists):
    sums = np.sum(raw_lists, axis=1)
    averages = sums / len(raw_lists)
    return averages

def avg_x_2_list(raw_lists):
    squared_raw = raw_lists ** 2
    sums = np.sum(squared_raw, axis=1)
    averages = sums / len(raw_lists)
    return averages

"""
-------------------------------------------------------------------------------------------------------------------------------
"""

"""
Simulation Code ---------------------------------------------------------------------------------------------------------------
"""

func_handle = str(input("Which Potential do you want to Investigate?" + "\n" + "Harmonic (h)" + "\n" + "Anharmonic (a)" + "\n" + ":  "))
func_handle.lower()

if func_handle == 'a':
    gamma = float(input("Enter the value of gamma for the quartic term:  "))
    action_func = action_anharmonic_NN
    file_func = "anharmonic_" + str(gamma)
else:
    action_func = action_NN
    file_func = "harmonic"

latt_size = int(input("Enter the length of the lattice:  "))
spacing = 1/float(latt_size)
Nsep = int(latt_size / 10)
sweeps = 10000
h = 0.75
animate = str(input("Animate the evolution of the process?  (y/n):  "))
animate.lower()
    
if animate == 'y':
    fig = plt.figure()
    axes = plt.gca()
    plt.xlabel("Position")
    plt.ylabel("Lattice Time Slice")
    plt.title("Position values of all lattice points")

mass = float(input("Enter the mass of the particle in the potential:  "))
mass_dim = mass * spacing
omega = float(input("Enter the natural frequency, omega, of the potential:  "))
omega_dim = omega * spacing

lattice = np.zeros(latt_size, dtype=float)
input_cold = str(input("For a hot start, type 'h', anything else provides a cold start:  "))
input_cold.lower()
if input_cold == 'h':
    cold_tag = '-h'
    for i in range(latt_size):
        lattice[i] = np.random.normal()/spacing
else:
    cold_tag = '-c'

results = np.asarray([np.copy(lattice)])

accepts = 0
rejects = 0
trials = sweeps

for k in range(sweeps): #number of metropolis sweeps + equilibriation
    #MonteCarloMetropolis Code Begin
    for n in range(latt_size):
        #sweeps forwards accessing each lattice point once per sweep

        u = np.random.uniform(-1*h, h)
        trial_x = lattice[n] + u
        trial_lattice = np.copy(lattice)
        trial_lattice[n] = trial_x
        delta_s_NN = action_func(trial_x, lattice[wrap(n-1)], lattice[wrap(n+1)]) - action_func(lattice[n], lattice[wrap(n-1)], lattice[wrap(n+1)])
        if delta_s_NN < 0:
            lattice[n] = trial_x
            accepts += 1
        else:
            if np.random.uniform() <= m.exp(-1 * delta_s_NN):
                lattice[n] = trial_x
                accepts += 1
            else:
                rejects += 1

        
    if k%Nsep == 0:
        if animate == 'y':
            plt.cla()
            line = plt.plot(lattice, np.arange(0,latt_size))
            axes.axvline(linewidth=2, color='r')
            plt.xlabel("Position")
            plt.ylabel("Lattice Time Slice")
            plt.title("Position values of all lattice points")
            plt.draw()
            plt.pause(0.001)

        results = np.append(results, [lattice], axis=0)

if animate == 'y':
    plt.cla()               
    line = plt.plot(lattice, np.arange(0,latt_size))
    axes.axvline(linewidth=2, color='r')
    plt.xlabel("Position")
    plt.ylabel("Lattice Time Slice")
    plt.title("Position values of all lattice points - Final Position")
    plt.show()

avg_xs   = avg_x_list(np.copy(results))
avg_x_2s = avg_x_2_list(np.copy(results))

fig_x = plt.figure()
axes_x = plt.gca()
plt.plot(np.arange(0, len(avg_xs), 1), avg_xs)
plt.title("Evolution of Average x over simulation")
plt.xlabel("Iteration")
plt.ylabel("Average x")
axes_x.axhline(linewidth=2, color='r')

fig_x_2 = plt.figure()
axes_x_2 = plt.gca()
plt.plot(np.arange(0, len(avg_x_2_list(results)), 1), avg_x_2_list(results))
plt.title("Evolution of Average x^2 over simulation")
plt.xlabel("Iteration")
plt.ylabel("Average x^2")
expected_x_2 = calc_x_2_expect()
axes_x_2.axhline(expected_x_2, linewidth=2, color='r')
plt.show()

cut_off = int(input("Enter the point at which the system is in equilibrium"))
cut_results = np.split(results, [cut_off])
new_results = cut_results[1]


filename = file_func + "-results-m_" + str(mass) + "-w_" + str(omega) + "-Ntau_" + str(latt_size) + "-Nsep_" + str(Nsep)
if cold_tag == '-c':
    filename = filename + cold_tag

np.savetxt( (filename + ".csv"), new_results)
np.savetxt( (filename + ".txt"), new_results)

print("Finished, Results Sent to file with identifier: " + "\n" + filename)

"""
END
"""

