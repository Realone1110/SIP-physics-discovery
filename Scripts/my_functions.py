import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from scipy.stats import gaussian_kde
import autograd.numpy as anp 
import sampyl as smp
from scipy.integrate import odeint
import pandas as pd
#import sunode.wrappers.as_pytensor as sun

def mean_squared_error(y_true, y_pred):
    """
    Calculates the Mean Squared Error (MSE) between two arrays.

    Args:
        y_true (numpy array): The true values.
        y_pred (numpy array): The predicted values.

    Returns:
        float: The Mean Squared Error between the true and predicted values.
    """
    y_true = np.array(y_true)  # Ensure the input is a numpy array
    y_pred = np.array(y_pred)  # Ensure the input is a numpy array

    # Calculate the MSE
    mse = np.mean((y_true - y_pred) ** 2)
    return mse


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

def plot_kde_matrix(data, titles, color='orange', name='SIP method'):
    """
    # Example usage
    data = np.array([[0, 1, 2],
                     [0, 1, 2],
                     [0, 1, 2],
                     [0, 1, 2],
                     [0, 1, 2]])
    
    titles = ['All Zeroes', 'Column 2', 'Column 3']
    plot_kde_matrix(data, titles)"""
    data = np.array(data)

    n, m = data.shape
    if len(titles) != m:
        raise ValueError("Length of titles vector must be equal to the number of columns in data.")
    
    fig, axes = plt.subplots(m, 1, figsize=(2.5, 1.5 * m), gridspec_kw={'hspace': 0})
    for i in range(m):
        if np.all(data[:, i] == data[0, i]):
            # All elements are the same
            k = data[0, i]
            axes[i].axvline(k, color=color, linestyle='-', linewidth=2)
            axes[i].set_ylim(0, 1)
            axes[i].set_xlim(k - 0.5, k + 0.5)
        else:
            kde = gaussian_kde(data[:, i])
            x = np.linspace(min(data[:, i]), max(data[:, i]), 1000)
            axes[i].plot(x, kde(x), color=color)
            axes[i].fill_between(x, kde(x), color=color, alpha=0.5)
            axes[i].set_xlim(min(data[:, i]) - 0.05, max(data[:, i]) + 0.05)
        
        mean_val = np.mean(data[:, i])
        axes[i].axvline(mean_val, color='red', linestyle='--', linewidth=1, label=f'mean= {mean_val.round(2)}')
        axes[i].set_ylabel(titles[i], rotation=0, labelpad=40)
        axes[i].legend(loc='upper right')
        axes[i].yaxis.set_label_position("left")
        axes[i].spines['top'].set_visible(False)
        axes[i].spines['right'].set_visible(False)
        axes[i].spines['left'].set_visible(False)
        axes[i].spines['bottom'].set_visible(True)
        axes[i].get_xaxis().set_visible(True)
        axes[i].tick_params(axis='x', which='both', bottom=True)
        axes[i].tick_params(axis='y', which='both', left=False, right=False)  # Remove vertical ticks
        axes[i].set_yticklabels([])  # Remove vertical tick labels

    plt.subplots_adjust(left=0.3)  # Adjust this as necessary to fit titles
    plt.xlabel('Value')  # Set common xlabel for the horizontal axis
    plt.suptitle(name)
    plt.show()



# Required Modification to run NUTS, FInd_MAP and Hamiltonian monte carlo
from autograd import numpy as anp
from autograd.extend import primitive, defvjp

@primitive
def safe_sqrt(x):
    """
    A safe version of sqrt that can handle ArrayBox types.

    Args:
        x (float or numpy array): The input value or array.

    Returns:
        float or numpy array: The square root of the input.
    """
    return x**0.5  # Using exponentiation as a substitute for sqrt.

def grad_safe_sqrt(ans, x):
    """
    Gradient of sqrt at x, using the chain rule.

    Args:
        ans (float or numpy array): The result of safe_sqrt(x).
        x (float or numpy array): The input value or array.

    Returns:
        function: The gradient function.
    """
    return lambda g: g * 0.5 * x**(-0.5)

# Linking the gradient function to safe_sqrt
defvjp(safe_sqrt, grad_safe_sqrt)

def safe_std(x, axis=None, keepdims=False):
    """
    Computes the standard deviation using a safe sqrt function.

    Args:
        x (numpy array): The input array.
        axis (int or tuple of int, optional): Axis or axes along which the standard deviation is computed.
        keepdims (bool, optional): If True, the axes which are reduced are left in the result as dimensions with size one.

    Returns:
        numpy array: The standard deviation of the input array.
    """
    mean = anp.mean(x, axis=axis, keepdims=True)
    var = anp.mean(anp.square(x - mean), axis=axis, keepdims=keepdims)
    std_dev = safe_sqrt(var)  # Using the custom safe_sqrt function.
    return std_dev

def generate_samples(mean_or_lower, std_or_upper, n_samples, distribution='normal'):
    """
    Generate samples from a specified distribution (normal or uniform) based on provided parameters.

    Args:
        mean_or_lower (float or numpy array): The mean (for normal) or lower bound (for uniform) of the distribution.
        std_or_upper (float or numpy array): The standard deviation (for normal) or upper bound (for uniform) of the distribution.
        n_samples (int): The number of samples to generate.
        distribution (str, optional): The type of distribution to sample from ('normal' or 'uniform'). Defaults to 'normal'.

    Returns:
        numpy array: The generated samples.
    """
    if distribution == 'normal':
        # Generate samples from a normal distribution
        if np.isscalar(mean_or_lower) and np.isscalar(std_or_upper):
            samples = np.random.normal(loc=mean_or_lower, scale=std_or_upper, size=n_samples)
        else:
            # Vectorized sampling for multiple sets of means and std deviations
            samples = np.random.normal(loc=mean_or_lower, scale=std_or_upper, size=(n_samples, len(mean_or_lower)))            
    elif distribution == 'uniform':
        # Generate samples from a uniform distribution
        if np.isscalar(mean_or_lower) and np.isscalar(std_or_upper):
            samples = np.random.uniform(low=mean_or_lower, high=std_or_upper, size=n_samples)
        else:
            # Vectorized sampling for multiple sets of bounds
            samples = np.random.uniform(low=mean_or_lower, high=std_or_upper, size=(n_samples, len(mean_or_lower)))            
    else:
        raise ValueError("Unsupported distribution type. Choose 'normal' or 'uniform'.")    
    return samples

def expand_2D_array(arr):
    """
    Expands a 2D array by pairing every element of each column with every element of all other columns.

    Args:
        arr (numpy array): The input 2D array.

    Returns:
        numpy array: The expanded 2D array.
    """
    # Number of columns
    n = arr.shape[1]    
    # List to hold the 1D arrays (columns)
    columns = [arr[:, i] for i in range(n)]    
    # Generate all combinations of pairs from these columns
    # itertools.product creates a Cartesian product, equivalent to a nested for-loop
    combinations = list(product(*columns))    
    # Convert list of tuples (combinations) back to a NumPy array
    expanded_array = np.array(combinations)    
    return expanded_array

def plot_paths(x_values, y_paths, show=False, color='blue', hl='r*', label='mean_path'):
    """
    Plots multiple y paths (each row in y_paths) against x_values on the same plot with a specified transparency.

    Args:
        x_values (numpy array): The x-axis values.
        y_paths (numpy array): The y-axis values for multiple paths.
        show (bool, optional): If True, displays the plot. Defaults to False.
        color (str, optional): Color for the paths. Defaults to 'blue'.
        hl (str, optional): Highlight style for the mean path. Defaults to 'r*'.
        label (str, optional): Label for the mean path. Defaults to 'mean_path'.

    Returns:
        None
    """
    # Calculate the mean path across all rows (mean for each column)
    mean_path = np.mean(y_paths, axis=0)
    
    # Set up the plot
    plt.figure(figsize=(10, 6))
    
    # Plot each path
    for path in y_paths:
        plt.plot(x_values, path, color=color, alpha=0.3)  # semi-transparent blue lines
    
    # Plot the mean path
    plt.plot(x_values, mean_path, hl, label=label)  # mean path in red
    
    # Adding title and labels
    plt.title('Multiple Y Paths with Mean Path')
    plt.xlabel('X Values')
    plt.ylabel('Y Values')
    plt.legend()
    if show:
        plt.show()

def Hudson_bay_data(plot=True):
    """
    Provides and optionally plots the Hudson Bay lynx and hare population data.

    Args:
        plot (bool, optional): If True, plots the population data. Defaults to True.

    Returns:
        tuple: Contains arrays for hare data, lynx data, the corresponding years and their respective scales.
    """
    # Define the data
    years_ = [1900, 1901, 1902, 1903, 1904, 1905, 1906, 1907, 1908, 1909,
             1910, 1911, 1912, 1913, 1914, 1915, 1916, 1917, 1918, 1919, 1920]
    
    lynx = [4, 6.1, 9.8, 35.2, 59.4, 41.7, 19, 13, 8.3, 9.1, 7.4, 8, 12.3, 
            19.5, 45.7, 51.1, 29.7, 15.8, 9.7, 10.1, 8.6]
    
    hare = [30, 47.2, 70.2, 77.4, 36.3, 20.6, 18.1, 21.4, 22, 25.4, 27.1,
            40.3, 57, 76.6, 52.3, 19.5, 11.2, 7.6, 14.6, 16.2, 24.7]
    
    # Prepare and scale data
    u_data =  np.array(hare)
    v_data = np.array(lynx)
    u_scale = np.std(u_data)
    v_scale = np.std(v_data)
    
    u_norm = u_data/u_scale
    v_norm = v_data/v_scale
    
    # Create a DataFrame
    data = {'Year': years_, 'Lynx': lynx, 'Hare': hare}
    df = pd.DataFrame(data)
    if plot:
        # Plot scaled data
        plt.figure()
        plt.title('Lotka-Volterra Model Dynamics')
        plt.plot(years_, u_norm,'r-x', label='Normed prey')
        plt.plot(years_, v_norm, 'b-x',label='Normed predator')
        
        plt.xlabel('Time')
        plt.ylabel('Population')
        plt.legend()
        plt.show()
        
        #plot actual data
        plt.plot(years_, hare, 'r-x',label = 'Hare')
        plt.plot(years_, lynx, 'b-x',label = 'Lynx')
        plt.legend()
        plt.xlabel('years')
        plt.ylabel('populations')
        plt.show()

    return u_data, v_data, u_scale, v_scale, years_


from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
def gaussian_process_interpolation(X_train, y_train, num_points, noise_level=0.1):
    """
    Trains a Gaussian Process model and uses it to interpolate new points.

    Args:
    X_train (numpy array): The input values for training.
    y_train (numpy array): The output values for training.
    num_points (int): The number of points to predict.
    noise_level (float): The noise level of the training data for regularization.

    Returns:
    numpy array, numpy array: The new input values and the predicted outputs.
    """
    # Ensure inputs are numpy arrays
    X_train = np.array(X_train).reshape(-1, 1)
    y_train = np.array(y_train).ravel()

    # Define the kernel: RBF kernel multiplied by a constant factor
    kernel = C(1.0, (1e-4, 1e1)) * RBF(1.0, (1e-4, 1e1))

    # Create the Gaussian Process model
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=noise_level**2)

    # Fit the model on the training data
    gp.fit(X_train, y_train)

    # Generate new points for interpolation
    X_new = np.linspace(X_train.min(), X_train.max(), num_points).reshape(-1, 1)

    # Make predictions
    y_new, sigma = gp.predict(X_new, return_std=True)

    # Optional: Plot the results
    plt.figure()
    plt.plot(X_train, y_train, 'r.', markersize=10, label='Training Data')
    plt.plot(X_new, y_new, 'b-', label='GPR Prediction')
    plt.fill_between(X_new.ravel(), y_new - 1.96 * sigma, y_new + 1.96 * sigma, color='blue', alpha=0.2, label='95% Confidence Interval')
    plt.xlabel('Input')
    plt.ylabel('Output')
    plt.title('Gaussian Process Regression Interpolation')
    plt.legend()
    plt.show()

    return X_new, y_new

import numpy as np

def save_array_to_csv(array, filename):
    """
    Save a numpy array to a CSV file.

    Parameters:
    array (np.ndarray): The array to save.
    filename (str): The name of the CSV file to save the array to.
    
    # Example usage
    arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    save_array_to_csv(arr, 'saved_array.csv')
    """
    np.savetxt(filename, array, delimiter=',')

def load_csv_to_array(filepath):
    """
    Load a CSV file into a numpy array.

    Parameters:
    filepath (str): The path of the CSV file to load.

    Returns:
    np.ndarray: The loaded array.

    # Example usage
    loaded_arr = load_csv_to_array('saved_array.csv')
    print(loaded_arr)

    """
    return np.loadtxt(filepath, delimiter=',')


def estimate_noise_statistics(original_array, noisy_array):
    """
    Estimate the noise statistics (mean and standard deviation) for each column in the array.
    
    Parameters:
    original_array (np.ndarray): The original array (1D or 2D).
    noisy_array (np.ndarray): The noisy array (same shape as original_array).

    Returns:
    dict: A dictionary containing the mean and standard deviation of the noise for each column.
          If the array is 1D, returns a single mean and standard deviation.
          
    # Example usage for 1D array
    original_1d = np.array([1, 2, 3, 4, 5]*100)
    noisy_1d = original_1d + np.random.normal(0, 0.5, original_1d.shape)
    result_1d = estimate_noise_statistics(original_1d, noisy_1d)
    print("1D array noise statistics:", result_1d)
    
    # Example usage for 2D array
    original_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    noisy_2d = original_2d + np.random.normal(0, 0.5, original_2d.shape)
    result_2d = estimate_noise_statistics(original_2d, noisy_2d)
    print("2D array noise statistics:", result_2d)
    """
    if original_array.shape != noisy_array.shape:
        raise ValueError("Original array and noisy array must have the same shape")

    noise = noisy_array - original_array

    if noise.ndim == 1:
        noise_mean = np.mean(noise)
        noise_std = np.std(noise)
        return {'mean': noise_mean, 'std': noise_std}
    else:
        noise_means = np.mean(noise, axis=0)
        noise_stds = np.std(noise, axis=0)
        return {'means': noise_means, 'stds': noise_stds}

import matplotlib.pyplot as plt
from matplotlib import rcParams

def set_publcn_matplotlib_defaults():
    """
    Set default configurations for Matplotlib plots suitable for publication.
    This includes setting the font to Times New Roman and adjusting the font size.

    # Call this function at the beginning of your notebook to apply the settings
    set_matplotlib_defaults()
    
    # Example plot to demonstrate the settings
    import numpy as np
    
    t = np.arange(0, 2, 0.01)
    s = 1 + np.sin(2 * np.pi * t)
    
    # Global settings plot
    plt.figure()
    plt.plot(t, s)
    plt.title('Global Settings Plot')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.show()
    
    # Override figure size and font size locally for this plot
    plt.figure(figsize=(10, 5))  # Override figure size
    plt.plot(t, s)
    plt.title('Custom Figure Size and Font Size Plot', fontsize=18)  # Override title font size
    plt.xlabel('Time (s)', fontsize=14)  # Override x-axis label font size
    plt.ylabel('Amplitude', fontsize=14)  # Override y-axis label font size
    plt.xticks(fontsize=12)  # Override x-axis tick labels font size
    plt.yticks(fontsize=12)  # Override y-axis tick labels font size
    plt.show()
    """
    # plt.style.use('seaborn-whitegrid')  # Optional: You can choose a different style or remove this line.
    
    rcParams.update({
        'font.family': 'serif',
        'font.serif': 'Times New Roman',
        'font.size': 12,             # General font size
        'axes.titlesize': 14,        # Font size for axes title
        'axes.labelsize': 12,        # Font size for axes labels
        'xtick.labelsize': 10,       # Font size for x-axis tick labels
        'ytick.labelsize': 10,       # Font size for y-axis tick labels
        'legend.fontsize': 10,       # Font size for legend
        'figure.titlesize': 16,      # Font size for figure title
        'lines.linewidth': 1.5,      # Line width
        'lines.markersize': 6,       # Marker size
        'figure.dpi': 300,           # Figure DPI for high-resolution images
        'savefig.dpi': 300,          # DPI for saving figures
        'savefig.format': 'png',     # Default format for saving figures
        'text.usetex': False,        # Use TeX for rendering text (False is usually better for compatibility)
    })



def calculate_noise_statistics(clean_data, noisy_data):
    """
    Calculate the mean, standard deviation, percentage of the additive noise,
    and signal-to-noise ratio (SNR) for each column in the array. If the array
    is 1D, it returns the metrics for the single dimension.

    Parameters:
    clean_data (np.ndarray): The original clean data (1D or 2D).
    noisy_data (np.ndarray): The noisy data (same shape as clean_data).

    Returns:
    dict: A dictionary containing the mean, standard deviation, percentage of
          the noise, and SNR for each column. If the array is 1D, returns the
          metrics for the single dimension.

    # Example usage for 1D array
    clean_data_1d = np.array([1, 2, 3, 4, 5]*1000)
    noisy_data_1d = clean_data_1d + np.random.normal(0, 0.1, size=clean_data_1d.shape)
    
    stats_1d = calculate_noise_statistics(clean_data_1d, noisy_data_1d)
    print("1D array noise statistics:", stats_1d)
    
    # Example usage for 2D array
    clean_data_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    noisy_data_2d = clean_data_2d + np.random.normal(0, 0.1, size=clean_data_2d.shape)
    
    stats_2d = calculate_noise_statistics(clean_data_2d, noisy_data_2d)
    print("2D array noise statistics:", stats_2d)
    """
    if clean_data.shape != noisy_data.shape:
        raise ValueError("Clean data and noisy data must have the same shape")

    # Calculate the noise
    noise = noisy_data - clean_data
    
    if noise.ndim == 1:
        noise_mean = np.mean(noise)
        noise_std = np.std(noise)
        clean_data_mean = np.mean(clean_data)
        noise_percentage = (noise_std / clean_data_mean) * 100
        snr = clean_data_mean / noise_std
        
        return {
            'noise_mean': noise_mean,
            'noise_std': noise_std,
            'noise_percentage': noise_percentage,
            'snr': snr
        }
    else:
        noise_means = np.mean(noise, axis=0)
        noise_stds = np.std(noise, axis=0)
        clean_data_means = np.mean(clean_data, axis=0)
        noise_percentages = (noise_stds / clean_data_means) * 100
        snrs = clean_data_means / noise_stds
        
        return {
            'noise_means': noise_means,
            'noise_stds': noise_stds,
            'noise_percentages': noise_percentages,
            'snrs': snrs
        }
