import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the CSV file
df = pd.read_csv('allruns.csv')  # Replace with your CSV file path

# Create the scatter plot
plt.figure(figsize=(10, 6))

# Plot individual points
sns.scatterplot(data=df, x='kernel_size', y='score', alpha=0.5)

# Calculate median scores for each kernel size
median_scores = df.groupby('kernel_size')['score'].median().reset_index()

# Plot median points and connect them with a line
plt.plot(median_scores['kernel_size'], median_scores['score'], 
         color='red', marker='o', linewidth=2, label='Median Score')

# Customize the plot
plt.title('Scores by Kernel Size')
plt.xlabel('Kernel Size')
plt.ylabel('Score')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

# Show the plot
plt.tight_layout()
plt.show()
