import numpy as np
import matplotlib.pyplot as plt

# Function to calculate sample variance
def sample_variance(data):
    n = len(data)
    if n < 2:
        raise ValueError("Sample size must be at least 2 to calculate variance")
    mean = np.mean(data)
    return np.sum((data - mean) ** 2) / (n - 1)

# Function to calculate variance using bootstrap technique
def bootstrap_variance(data, num_samples, sample_size):
    variances = []
    n = len(data)
    for _ in range(num_samples):
        # Resample with replacement
        resample = np.random.choice(data, size=sample_size, replace=True)
        variances.append(sample_variance(resample))
    return variances

# Scenario (a)
data_a = np.array([1, 3])
sample_size_a = len(data_a)
print(len (data_a))
num_samples_a = 1000  # Number of bootstrap samples

# Calculate sample variance for scenario (a)
sample_var_a = sample_variance(data_a)

# Calculate variances using bootstrap for scenario (a)
bootstrap_variances_a = bootstrap_variance(data_a, num_samples_a, sample_size_a)

print("Sample Variance (Scenario a):", sample_var_a)
print("Bootstrap Variances (Scenario a):", bootstrap_variances_a)

# Scenario (b)
data_b = np.array([5, 4, 9, 6, 21, 17, 11, 20, 7, 10, 21, 15, 13, 8, 6])
sample_size_b = len(data_b)
num_samples_b = 1000  # Number of bootstrap samples

# Calculate sample variance for scenario (b)
sample_var_b = sample_variance(data_b)

# Calculate variances using bootstrap for scenario (b)
bootstrap_variances_b = bootstrap_variance(data_b, num_samples_b, sample_size_b)

print("\nSample Variance (Scenario b):", sample_var_b)

# Plot histogram of bootstrap variances for scenario (b)
plt.hist(bootstrap_variances_b, color='b', label='Bootstrap Variances')
plt.axvline(sample_var_b, color='r', linestyle='dashed', linewidth=2, label=f'Sample Variance = {sample_var_b:.2f}')
plt.xlabel('Variances')
plt.ylabel('Frequency')
plt.legend()
plt.title('Histogram of Bootstrap Variances (Scenario b)')
plt.show()

import numpy as np
import matplotlib.pyplot as plt

# Parameters
lambda_val = 4  # Lambda for the exponential distribution
sample_size = 100  # Number of random numbers in each sample
num_repeats = 20  # Number of times to repeat the process

# Initialize lists to store means and standard deviations
means = []
std_devs = []

# Repeat the process num_repeats times
for _ in range(num_repeats):
    # Generate 100 random numbers from exponential distribution
    random_numbers = np.random.exponential(scale=1/lambda_val, size=sample_size)

    # Calculate the mean of the generated random numbers
    mean = np.mean(random_numbers)

    # Calculate the standard deviation of the generated random numbers
    std_dev = np.std(random_numbers, ddof=1)  # Use ddof=1 for sample standard deviation

    # Append the mean and standard deviation to the lists
    means.append(mean)
    std_devs.append(std_dev)

# Plot the histogram of means
plt.hist(means, bins=20, alpha=0.7, color='b', label='Means')

# Calculate and plot the sample mean and sample standard deviation
sample_mean = np.mean(means)
sample_std_dev = np.std(means, ddof=1)
plt.axvline(sample_mean, color='r', linestyle='dashed', linewidth=2, label=f'Sample Mean = {sample_mean:.2f}')
plt.axvline(sample_std_dev, color='g', linestyle='dashed', linewidth=2, label=f'Sample Std Dev = {sample_std_dev:.2f}')

# Add labels and legend
plt.xlabel('Sample Means')
plt.ylabel('Frequency')
plt.legend()


# Show the plot
plt.title('Histogram of Sample Means')
plt.show()

import numpy as np
from scipy.stats import norm

# Travel times data
travel_times = np.array([42, 28, 53, 57, 67, 39, 35, 50, 44, 39])

mean_travel_time2=np.mean(travel_times)

# Calculate the mean and standard deviation of the log of the travel times
log_travel_times = np.log(travel_times)
mean_log_time = np.mean(log_travel_times)
std_log_time = np.std(log_travel_times, ddof=1)  # Use ddof=1 for sample standard deviation

# Number of data points
n = len(travel_times)

# Confidence level
confidence_level_95 = 0.95
confidence_level_99 = 0.99

# Calculate the standard error of the mean (SEM)
sem = std_log_time / np.sqrt(n)

# Calculate the z-scores for 95% and 99% confidence intervals
z_95 = norm.ppf(1 - (1 - confidence_level_95) / 2)
print(z_95)
z_99 = norm.ppf(1 - (1 - confidence_level_99) / 2)

# Calculate the mean travel time (back-transform from log scale)
mean_travel_time = np.exp(mean_log_time)
print(mean_travel_time)

# Calculate confidence intervals
ci_95_lower = np.exp(mean_log_time - z_95 * sem)
ci_95_upper = np.exp(mean_log_time + z_95 * sem)

ci_99_lower = np.exp(mean_log_time - z_99 * sem)
ci_99_upper = np.exp(mean_log_time + z_99 * sem)

# Calculate the 95% upper confidence interval (for comparison with a specific value)
ci_95_upper_one_sided = np.exp(mean_log_time + z_95 * sem)

# Calculate the 95% lower confidence interval (for comparison with a specific value)
ci_95_lower_one_sided = np.exp(mean_log_time - z_95 * sem)

# Print results
print(f"Estimated Mean Travel Time (lognormal): {mean_travel_time:.2f} minutes")
print(f"95% Confidence Interval: ({ci_95_lower:.2f}, {ci_95_upper:.2f}) minutes")
print(f"99% Confidence Interval: ({ci_99_lower:.2f}, {ci_99_upper:.2f}) minutes")
print(f"95% Upper Confidence Interval (one-sided): {ci_95_upper_one_sided:.2f} minutes")
print(f"95%  Lower Confidence Interval (one-sided): {ci_95_lower_one_sided:.2f} minutes")


import numpy as np



# Transition probability matrix

P = np.array([

[0.8, 0.1, 0.1], # Transition probabilities for sons of professionals

[0.2, 0.6, 0.2], # Transition probabilities for sons of skilled laborers

[0.25, 0.25, 0.5] # Transition probabilities for sons of unskilled laborers
])



# Number of generations to look ahead

generations = 2 # Grandson is two generations away



# Initial state probabilities (sons of unskilled laborers)

initial_state = np.array([0, 0, 1]) # Grandfather is an unskilled laborer



# Calculate the state probabilities after 'generations' generations

final_state = np.linalg.matrix_power(P, generations).dot(initial_state)



# Probability that a randomly chosen grandson of an unskilled laborer is a professional

probability_professional = final_state[0]



print("Probability that a randomly chosen grandson of an unskilled laborer is a professional:", probability_professional)