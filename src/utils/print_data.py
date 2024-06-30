import matplotlib.pyplot as plt
import re

# Regular expression to extract data from the text
pattern = r'Episode (\d+): Score: (-?\d+\.\d+), Average Score: (-?\d+\.\d+), Critic Loss: (-?\d+\.\d+), Actor Loss: (-?\d+\.\d+), System Loss: (-?\d+\.\d+), Reward Loss: (-?\d+\.\d+)'

# Extract data from the text
data = []
with open('data.txt', 'r') as f:
    for line in f:
        match = re.search(pattern, line)
        if match:
            data.append([float(x) for x in match.groups()[1:]])

# Separate data into different lists
scores, avg_scores, critic_losses, actor_losses, system_losses, reward_losses = zip(*data)

# Create a figure and a set of subplots
fig, axs = plt.subplots(6, figsize=(10, 20))

# Plot the data
axs[0].plot(scores)
axs[0].set_title('Scores')

axs[1].plot(avg_scores)
axs[1].set_title('Average Scores')

axs[2].plot(critic_losses)
axs[2].set_title('Critic Losses')

axs[3].plot(actor_losses)
axs[3].set_title('Actor Losses')

axs[4].plot(system_losses)
axs[4].set_title('System Losses')

axs[5].plot(reward_losses)
axs[5].set_title('Reward Losses')

# Layout so plots do not overlap
fig.tight_layout()

plt.show()