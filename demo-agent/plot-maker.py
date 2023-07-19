import csv
import matplotlib.pyplot as plt

# Read data from the CSV file
def read_csv_file(filename):
    data = []
    with open(filename, 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            data.append(float(row[0]))
    return data

# Example usage
filename = '/home/jiaxinyang/Desktop/rewards_1.csv'  # Replace with your CSV file name

# Read data from the CSV file
data = read_csv_file(filename)

# Generate x-axis values (Training epochs)
epochs = range(len(data))

# Generate y-axis values (Mean of Q-function)
mean_q_values = data

# Plot the data
plt.plot(epochs, mean_q_values)
plt.xlabel('Training epochs')
plt.ylabel('Mean of Q-function')
plt.title('Q-function over Training Epochs')
plt.xlim(0, 1000)  # Set x-axis limits
plt.ylim(-10, 10)  # Set y-axis limits
plt.grid(True)
plt.show()

