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
