import matplotlib.pyplot as plt

# Data for plotting
labels = ['Stars', 'Forks']
values = [13300, 1400]  # Converted to integers for plotting

# Create a bar plot
plt.figure(figsize=(8, 6))
plt.bar(labels, values, color=['blue', 'green'])

# Add title and labels
plt.title('GitHub Repository Stats for camel-ai/camel')
plt.xlabel('Category')
plt.ylabel('Count')

# Save the plot to a file
plt.savefig('camel_ai_github_stats.png')
plt.show()