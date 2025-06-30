import matplotlib.pyplot as plt

# Data for plotting
labels = ['Stars', 'Forks']
values = [13000, 1400]

# Create a bar plot
plt.figure(figsize=(8, 6))
plt.bar(labels, values, color=['blue', 'green'])

# Add title and labels
plt.title('Camel-AI GitHub Repository')
plt.xlabel('Category')
plt.ylabel('Count')

# Save the plot as an image
plt.savefig('camel_ai_plot.png')

# Show the plot
plt.show()