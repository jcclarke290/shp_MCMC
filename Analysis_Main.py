"""
Senior Honours Project - Analysis Final

Created: 08/03/2020 - James C. Clarke

Takes in results from a given file and then produces values of observables
Optionally provides autocorrelation data
"""

import math as m
import numpy as np
import matplotlib.pyplot as plt

"""
Jackknife Analysis Functions --------------------------------------------------------------------------------------------------
"""

def jackknife(average, values, function):
    squared_differences = np.zeros(len(values))
    for i in range(len(values)):
        ith_value = np.copy(values)
        ith_value = np.delete(ith_value, i, 0)
        avg_O_i = function(ith_value)
        squared_differences[i] = m.pow((avg_O_i - average), 2)
    return m.pow((np.sum(squared_differences)), 0.5)
        

def jackknife_param(value, value_list, function, param_extra):
    #extra parameter required for some observables hence reshuflle
    squared_differences = np.zeros(len(value_list))
    for i in range(len(value_list)):
        ith_value = np.copy(value_list)
        ith_value = np.delete(ith_value, i, 0)
        avg_O_i = function(ith_value, param_extra)
        squared_differences[i] = m.pow((avg_O_i - value), 2)
    return m.pow((np.sum(squared_differences)), 0.5)  

"""
-------------------------------------------------------------------------------------------------------------------------------
"""


"""
Integrated Autocorrelation ----------------------------------------------------------------------------------------------------
"""
def int_corr_sum(observables, t):
    A_t_i = 0.0
    for i in range(len(observables)-t):
        if len(observables) - t != 0:
            A_t_i = A_t_i + ((observables[i])*(observables[i+t]))

            A_t_i = A_t_i / (len(observables) - t)
        else:
            A_t_i = A_t_i + ((observables[i])*(observables[i+t]))
            A_t_i = A_t_i
    return A_t_i

def int_corr_check(observables):
    E_0 = np.sum(observables) / len(observables)
    observables = observables - E_0
    A_0_is = np.zeros(len(observables))
    for i in range(len(observables)):
        A_0_is[i] = observables[i]**2
    A_0 = np.sum(A_0_is) / (len(observables)-1)

    A_os = np.zeros(len(observables)-1)
    Error_A_os = np.copy(A_os)
    for t in range(1, len(observables)):
        A_os[t-1] = int_corr_sum(observables, t)
        Error_A_os[t-1] = jackknife_param(A_os[t-1], np.copy(observables), int_corr_sum, t)

    plt.errorbar(np.arange(1,len(observables),1), A_os, yerr=Error_A_os, fmt='bx-', capsize=2)
    plt.title("Integrated Correlation against seperation of values in Markov Chain")
    plt.xlabel("Markov Chain Seperation")
    plt.ylabel("Integrated Correlation")
    plt.show()

    new_nu = int(input("Enter the value of nu at which to cut off the calculation:  "))
    return new_nu

def int_corr(observables, nu):
    E_0 = np.sum(observables) / len(observables)
    observables = observables - E_0
    A_0 = 0.0
    for i in range(len(observables)):
        A_0_i = observables[i]**2
        A_0 = A_0 + A_0_i
    A_0 = A_0 / (len(observables)-1)

    t_int = 0.0
    for t in range(1, nu):
        A_t_i = 0.0
        for i in range(len(observables)-t):
            A_t_i = A_t_i + ( (observables[i]) * (observables[i+t]) )

        A_t_i = A_t_i / (len(observables) - t)
        t_int = t_int + A_t_i

    t_int = 0.5 + (t_int / A_0)
    return t_int

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
        return j

def wrap(j):
    #produces the periodic boundary conditions of the lattice
    if j >= time_slice:
        while j >= time_slice:
            j = j - time_slice
        return j

    elif j < 0:
        while j < 0:
            j = j + time_slice
        return j

    else:
        return j

"""
-------------------------------------------------------------------------------------------------------------------------------
"""


"""
Classical Action Functions ----------------------------------------------------------------------------------------------------
"""

def action(x_s, mass, freq, gamma):
    #used to calculate the full action of a given configuration 
    result = 0.0
    for i in range(len(x_s)):
        i_one = wrap_size((i+1), len(x_s))
        s_i = (0.5 * mass * (x_s[i_one] - x_s[i])**2 + 0.5 * mass * (freq**2) * (x_s[i]**2) )

        result = result + s_i
    return result

def action_anharmonic(x_s, mass, freq, gamma):
    #used to calculate the full action of a given configuration
    result = 0.0
    for i in range(len(x_s)):
        i_one = wrap_size((i+1), len(x_s))
        s_i = (0.5 * mass * (x_s[i_one] - x_s[i])**2 - 0.5 * mass * (freq**2) * (x_s[i]**2) + 0.25 * gamma *(x_s[i]**4) )

        result = result + s_i
    return result

"""
-------------------------------------------------------------------------------------------------------------------------------
"""


"""
Correlation Function ----------------------------------------------------------------------------------------------------------
"""

def g_del_tau_new(config, delta_tau):
    average_x = np.sum(config) / len(config)
    #average x at fixed distance is equivalent due to boundary cond.
    average_product = 0.0
    for i in range(len(config)):
        for j in range(len(config)):
            if (j-i)%len(config) == delta_tau:
                product_i = config[i] * config[j]
                average_product = average_product + product_i
    average_product = average_product / len(config)
    return (average_product)- m.pow(average_x, 2)

def correlation_func_HO_plot_single(config):
    cut_off =  int(len(config) / 4) + 1
    G_s = np.zeros(cut_off)
    errors = np.copy(G_s)
    for i in range(cut_off):
        G_s[i] = g_del_tau_new(config, i)
        errors[i] = jackknife_param(G_s[i], config, g_del_tau_new, i)

    
    plt.errorbar(np.arange(0, cut_off, 1), G_s, yerr=errors, capsize=2, fmt='x-')
    plt.title("Correlation Function value versus delta Tau")
    plt.xlabel("Seperation Delta Tau")
    plt.ylabel("Correlation Function Value")
    plt.show()
    neg_start = int(input("where should the values be cut?"))
    G_s = G_s[:neg_start]
    errors = errors[:neg_start]
    return [G_s, errors]

"""
-------------------------------------------------------------------------------------------------------------------------------
"""


"""
Observables -------------------------------------------------------------------------------------------------------------------
"""


def avg_x_list(raw_lists):
    sums = np.sum(raw_lists, axis=1)
    averages = sums / len(raw_lists)
    return averages

def avg_x_calc(raw_lists):
    sums = np.sum(raw_lists, axis=1)
    averages = sums / len(raw_lists)
    result_number = len(averages)
    return (np.sum(averages) / result_number)

def avg_x_2_list(raw_lists):
    squared_raw = raw_lists ** 2
    sums = np.sum(squared_raw, axis=1)
    averages = sums / len(raw_lists)
    return averages

def avg_x_2_calc(raw_lists):
    squared_raw = raw_lists ** 2
    sums = np.sum(squared_raw, axis=1)
    averages = sums / len(raw_lists)
    result_number = len(averages)
    return (np.sum(averages) / result_number)

def plot_ground_state(raw_lists, mass, freq, gamma):
    bins = np.arange(np.amin(raw_lists), np.amax(raw_lists) + 0.1, 0.1)
    y, bin_edges, _ = plt.hist(raw_lists.flatten(), bins=bins)
    plt.cla()
    total = np.sum(y)
    errors = np.sqrt(y) / (total*0.1)
    bin_centers = 0.5*(bin_edges[1:] + bin_edges[:-1])
    y_norm, _, _ = plt.hist(raw_lists.flatten(), bins=bins, density=True)
    plt.errorbar(bin_centers, y_norm, yerr = errors, fmt='o', capsize=2, marker = '.', drawstyle = 'steps-mid')

    if gamma == 0:
        y_expect = np.zeros(len(bin_centers))
        for i in range(len(y_expect)):
            y_expect[i] = m.pow( (mass*freq) / m.pi, 0.25) * m.exp((-1 * mass * freq * (bin_centers[i]**2))/2)
        plt.plot(bin_centers, y_expect, '--')
    
    plt.title("Distribution of position values over whole simluation")
    plt.xlabel("Position")
    plt.ylabel("Frequency")
    plt.show()


"""
-------------------------------------------------------------------------------------------------------------------------------
"""

"""
Expected Values ---------------------------------------------------------------------------------------------------------------
"""

def calc_x_2_expect(mass, freq, latt_size):
    R = 1 + ((freq **2) *0.5) - (freq*( m.pow( (1+ 0.25* freq**2), 0.5)))
    preFactor = 1 / ((2*mass*(freq**2)) * m.pow( (1+ 0.25* freq**2), 0.5))
    R_frac = (1 + m.pow(R, latt_size)) / (1- m.pow(R, latt_size))

    return preFactor*R_frac

"""
-------------------------------------------------------------------------------------------------------------------------------
"""




filename = str(input("Enter the filename of the data to read in:  "))

results_list = np.loadtxt(filename)

slices    = int(input("Enter the size of the lattice used:  "))
spacing   = 1 / slices
mass_val  = float(input("Enter the Mass used in this experiment:  ")) * spacing
freq_val  = float(input("Enter the Natural Frequency used in this experiment:  ")) * spacing
gamma_val = float(input("Enter the value of Gamma used (enter 0 if harmonic oscillator):  "))


#ask for autocorrelation
#ask for classical action
#ask for observables
#---> ask each observable in turn

#output all to file or as a graph

filename =  filename[:-4] + "-analysis.txt"

f_out = open(filename, 'w')
f_out.write("Ntau = " + str(slices) + "    Mass = " +str(mass_val)+ "   Frequency = " + str(freq_val) + "   Gamma = " + str(gamma_val) + "\n")

auto_inp = str(input("Perform Integrated Autocorrelation analysis for an observable? (Will take a long time):  "))
auto_inp.lower()
if auto_inp == 'y':
    avg_x_results = avg_x_list(np.copy(results_list))
    #only looks at the first quarter of the values in an attempt to speed up the process
    #expect the integrated autocorrelation to quickly deteriorate into noise
    avg_x_results = avg_x_results[:100]
    nu_val = int_corr_check(avg_x_results)
    int_corr_result = int_corr(avg_x_results, nu_val)
    int_corr_error = jackknife_param(int_corr_result, avg_x_results, int_corr, nu_val)
    string = "Integrated Autocorrelation (Avg_x): " + str(int_corr_result) + "      Error: " + str(int_corr_error) + "\n"
    f_out.write(string)
    

    avg_x_2_results = avg_x_2_list(np.copy(results_list))
    #only looks at the first quarter of the values in an attempt to speed up the process
    #expect the integrated autocorrelation to quickly deteriorate into noise
    avg_x_2_results = avg_x_2_results[:100]
    nu_val = int_corr_check(avg_x_2_results)
    int_corr_result = int_corr(avg_x_2_results, nu_val)
    int_corr_error = jackknife_param(int_corr_result, avg_x_2_results, int_corr, nu_val)
    string = "Integrated Autocorrelation (Avg_x^2): " + str(int_corr_result) + "      Error: " + str(int_corr_error) + "\n"
    f_out.write(string)


    corr_func_results = correlation_func_HO_plot_single(results_list[(int(len(results_list)/2))])
    corr_func_results = corr_func_results[0]
    nu_val = int_corr_check(corr_func_results)
    int_corr_result = int_corr(corr_func_results, nu_val)
    int_corr_error = jackknife_param(int_corr_result, corr_func_results, int_corr, nu_val)
    string = "Integrated Autocorrelation (Corr. Func.): " + str(int_corr_result) + "      Error: " + str(int_corr_error) + "\n"
    f_out.write(string)


class_action_inp = str(input("Show a graph of the classical action over the course of the simulation?"))
class_action_inp.lower()
if class_action_inp == 'y':
    if gamma_val != 0:
        action_func = action_anharmonic
    else:
        action_func = action
        class_actions = np.zeros(len(results_list))
        for i in range(len(results_list)):
            class_actions[i] = action_func(results_list[i], mass_val, freq_val, gamma_val)
        
        plt.plot(np.arange(0,len(results_list), 1), class_actions, 'bx-')
        plt.title("Evolution of the value of the Clasical Action over the measured paths in the Markov Chain")
        plt.xlabel("Markov Chain Index")
        plt.ylabel("Classical Action")
        plt.show()

values_inp = str(input("Enter 'y' to calculate the values of average position and average squared position:  "))
values_inp.lower()
if values_inp == 'y':
    avg_x   = avg_x_calc(np.copy(results_list))
    x_error = jackknife(avg_x, np.copy(results_list), avg_x_calc)
    avg_x_string = "Average Position: " + str(avg_x) + "    Error: " + str(x_error) + "\n"
    f_out.write(avg_x_string)
    if gamma_val == 0:
        f_out.write("Expected Value = 0" + "\n")

    avg_x_2   = avg_x_2_calc(np.copy(results_list))
    x_2_error = jackknife(avg_x_2, np.copy(results_list), avg_x_2_calc)
    avg_x_2_string = "Average Position: " + str(avg_x_2) + "    Error: " + str(x_2_error) + "\n"
    f_out.write(avg_x_2_string)
    if gamma_val == 0:
        f_out.write("Expected Value = " + str(calc_x_2_expect(mass_val, freq_val, slices)) + "\n")

ground_inp = str(input("Enter 'y' to show a plot of the ground state:  "))
ground_inp.lower()
if ground_inp == 'y':
    plot_ground_state(np.copy(results_list), mass_val, freq_val, gamma_val)

energy_diff_inp = str(input("Enter 'y' to output energy difference values for further analysis:  "))
energy_diff_inp.lower()
if energy_diff_inp == 'y':
    #select the middle value, should be fine to use any configuration as long as it's equilibriated
    results_energy = correlation_func_HO_plot_single(results_list[(int(len(results_list)/2))])
    energy_out = filename[:-4] + "-Energy Difference.csv"
    errors_out = filename[:-4] + "-Energy Errors.csv"
    energy_vals = np.log(np.absolute(results_energy[0]))
    error_vals = np.divide(results_energy[1], results_energy[0])

    np.savetxt(energy_out, energy_vals)
    np.savetxt(errors_out, error_vals)
    f_out.write("Expected value of first energy level transition = " + str(freq_val) + "\n")
    print("Correlation function data with corresponding errors written to seperate files.")

f_out.close()
print("Analysis Complete, File closed")
    
