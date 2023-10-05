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
