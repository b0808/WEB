import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Given data
data = [1079, 972, 1037, 979, 965, 979, 914, 1056, 932, 907, 904, 900, 637, 1133, 947, 982, 895, 1033, 931, 964]
sample_size = len(data)
sample_mean = np.mean(data)
population_mean = 1000  # Claimed population mean
alpha = 0.05  # Significance level

# Calculate the sample standard deviation
sample_std_dev = np.std(data, ddof=1)

# Calculate the t-statistic
t_statistic = (sample_mean - population_mean) / (sample_std_dev / np.sqrt(sample_size))

# Calculate the degrees of freedom
degrees_of_freedom = sample_size - 1

# Find the critical t-value
critical_t_value = stats.t.ppf(1 - alpha, degrees_of_freedom)

# Create a range of values for the x-axis
x = np.linspace(800, 1200, 1000)

# Create the t-distribution curve
t_distribution = stats.t.pdf(x, degrees_of_freedom)

# Plot the t-distribution curve
plt.plot(x, t_distribution, label='t-Distribution')

# Shade the critical region
plt.fill_between(x, t_distribution, where=(x < critical_t_value), color='red', alpha=0.5, label='Critical Region')

# Plot the test statistic
plt.axvline(x=t_statistic, color='blue', linestyle='--', label='Test Statistic')

# Add labels and legend
plt.xlabel('x')
plt.ylabel('Probability Density')
plt.title('One-Sample t-Test for Monthly Electricity Consumption')
plt.legend()

# Show the plot
plt.show()

# Make a decision based on the test result
if t_statistic < -critical_t_value or t_statistic > critical_t_value:
    print("Reject the null hypothesis. The data contradicts the claim that the monthly mean electricity consumption is 1000 kWh.")
else:
    print("Fail to reject the null hypothesis. The data does not contradict the claim that the monthly mean electricity consumption is 1000 kWh.")
