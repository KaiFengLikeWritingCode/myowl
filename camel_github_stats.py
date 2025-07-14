import matplotlib.pyplot as plt

# GitHub statistics
stats = {
    'Stars': 13300,
    'Forks': 1400,
    'Issues': 367,
    'Pull Requests': 167,
    'Commits': 1410
}

# Extracting data for plotting
categories = list(stats.keys())
values = list(stats.values())

# Creating the bar plot
plt.figure(figsize=(10, 6))
plt.bar(categories, values, color=['blue', 'green', 'red', 'purple', 'orange'])

# Adding titles and labels
plt.title('GitHub Statistics for camel-ai\'s camel Framework')
plt.xlabel('Categories')
plt.ylabel('Counts')

# Displaying the plot
plt.tight_layout()
plt.show()