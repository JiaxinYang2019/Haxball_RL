import csv
import re
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  # Use the Tkinter backend

def plot_rewards(file_path):
    rewards = []
    with open(file_path, 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            if len(row) > 0:  # Skip empty rows
                reward_string = row[0]
                # Use regular expression to match the expected format
                pattern = r'(-?\d+\.\d+)'
                match = re.search(pattern, reward_string)
                if match:
                    reward = float(match.group(1))
                    rewards.append(reward)

    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Reward Progression')
    plt.show()

# Example usage
file_path = '/home/jiaxinyang/Group-2/Jiaxin_Yang/build/rewards.csv'  # Provide the correct path to your rewards.csv file
plot_rewards(file_path)
