import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


# Function to generate random initial positions and headings for objects
def generate_initial_states(num_objects):
    initial_positions = np.random.rand(num_objects, 2) # * 10  Random initial positions within 10x10 grid
    headings = np.random.rand(num_objects) * 2 * np.pi  # Random headings in radians
    return initial_positions, headings


# Function to update positions of objects based on constant velocity
def update_positions(positions, headings, velocity):
    delta_x = velocity * np.cos(headings)
    delta_y = velocity * np.sin(headings)
    positions[:, 0] += delta_x
    positions[:, 1] += delta_y


# Function to simulate sensor observations
def simulate_observations(true_positions, noise_std, missed_detection_prob):
    num_objects = true_positions.shape[0]
    observations = []
    for i in range(num_objects):
        if 0 < true_positions[i][0] < 1 and 0 < true_positions[i][1] < 1:
            if np.random.rand() > missed_detection_prob:
                observation = true_positions[i] + np.random.normal(scale=noise_std, size=2)
                observations.append(observation)
    return observations


# Main function to simulate the motion of objects and sensor observations
def simulate(num_objects, num_steps, velocity, noise_std, missed_detection_prob):
    initial_positions, headings = generate_initial_states(num_objects)
    observations = simulate_observations(initial_positions, noise_std, missed_detection_prob)
    true_positions_history = [initial_positions.copy()]
    observations_history = [observations]

    for _ in range(num_steps):
        # Update true positions
        update_positions(initial_positions, headings, velocity)
        true_positions_history.append(initial_positions.copy())

        # Simulate sensor observations
        observations = simulate_observations(initial_positions, noise_std, missed_detection_prob)
        observations_history.append(observations)

    return true_positions_history, observations_history


# Function to visualize the simulation
def visualize_simulation(true_positions_history, observations_history, show=True):
    fig, ax = plt.subplots()
    norm = Normalize(vmin=0, vmax=len(true_positions_history) - 1)
    cmap = plt.get_cmap('viridis')
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    for i, (true_positions, observations) in enumerate(zip(true_positions_history, observations_history)):
        # Plot true positions
        ax.scatter(true_positions[:, 0], true_positions[:, 1], color=cmap(norm(i)), label=f'True Positions (Step {i})')

        # Plot sensor observations
        for j, observation in enumerate(observations):
            if observation is not None:
                ax.scatter(observation[0], observation[1], color='red', alpha=0.5, label=f'Observations (Step {i})')

    plt.colorbar(sm, ax=ax, label='Time Step')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend(['Truth Data', 'Observations'])
    plt.title('Multi-target Tracking Simulation')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    plt.savefig('Images/truthPlot.png', format='png')
    if show: plt.show()
    else: plt.close()


# Parameters
num_objects = 3
num_steps = 10
velocity = 0.065
noise_std = 0.0075  # Standard deviation of Gaussian noise
missed_detection_prob = 0.02  # Probability of missed detection

if __name__ == '__main__':
    # Run simulation
    true_positions_history, observations_history = simulate(num_objects, num_steps, velocity, noise_std, missed_detection_prob)

    # Visualize simulation
    visualize_simulation(true_positions_history, observations_history)
