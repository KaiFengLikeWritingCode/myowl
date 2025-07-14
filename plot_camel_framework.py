import matplotlib.pyplot as plt

# Data for plotting
labels = ['Stars', 'Forks']
counts = [13300, 1400]  # Converted to integers for plotting

# Create a bar plot
plt.figure(figsize=(8, 6))
plt.bar(labels, counts, color=['blue', 'green'])

# Add title and labels
plt.title('GitHub Repository Stats for camel-ai/camel')
plt.xlabel('Category')
plt.ylabel('Count')

# Show the plot
plt.savefig('camel_framework_stats.png')
plt.show()