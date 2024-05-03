import csv

import matplotlib.pyplot as plt

from MyMHT import MyMHT
from openmht.cli import write_uv_csv, read_uv_csv
from openmht.mht import MHT
from truth_data import simulate, visualize_simulation


def generate_data():
    num_objects = 4
    num_steps = 7
    velocity = 0.075
    noise_std = 0.005  # Standard deviation of Gaussian noise
    missed_detection_prob = 0.02  # Probability of missed detection

    true_positions_history, observations_history = simulate(num_objects, num_steps, velocity, noise_std, missed_detection_prob)
    visualize_simulation(true_positions_history, observations_history, show=False)
    data = []
    for i, frame in enumerate(observations_history):
        for obs in frame:
            data.append([i, obs[0], obs[1]])

    with open('data.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['frame', 'u', 'v'])  # Header row
        for row in data:
            writer.writerow(row)


def plot_MTT(tracks):
    for track in tracks:
        x = [point[0] for point in track if point]
        y = [point[1] for point in track if point]
        plt.scatter(x, y, color='black')
        plt.plot(x, y)
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title('MHT Best Hypothesis')
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    plt.show()


parameters = {
    'v': 307200,
    'dth': 0.5,
    'k': 0,
    'q': 0.00001,
    'r': 0.01,
    'n': 2,
    'bth': 150,
    'nmiss': 2,
    'pd': 0.98,
    'ck': True
}

generate_data()
sample_file = "SampleData/SampleInput.csv"
file = "data.csv"
detections = read_uv_csv(file)

# mht = MHT(detections, parameters)
# mhtSol = mht.run()
# write_uv_csv("mhtSol.csv", mhtSol)
# plot_MTT(mhtSol)


my = MyMHT(detections, parameters)
mySol = my.run()
write_uv_csv("mySol.csv", mySol)
plot_MTT(mySol)

