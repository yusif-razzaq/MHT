import csv

from openmht.cli import write_uv_csv, read_uv_csv
from openmht.mht import MHT
from openmht.plot_tracks import plot_2d_tracks
from truth_data import simulate, visualize_simulation


def generate_data():
    num_objects = 3
    num_steps = 10
    velocity = 0.05
    noise_std = 0.0005  # Standard deviation of Gaussian noise
    missed_detection_prob = 0.02  # Probability of missed detection

    true_positions_history, observations_history = simulate(num_objects, num_steps, velocity, noise_std, missed_detection_prob)
    visualize_simulation(true_positions_history, observations_history)
    data = []
    for i, frame in enumerate(observations_history):
        for obs in frame:
            data.append([i, obs[0], obs[1]])

    with open('data.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['frame', 'u', 'v'])  # Header row
        for row in data:
            writer.writerow(row)


parameters = {
    'v': 307200,
    'dth': 5000,
    'k': 0,
    'q': 0.00001,
    'r': 0.01,
    'n': 3,
    'bth': 150,
    'nmiss': 10,
    'pd': 0.90,
    'ck': False
}

# generate_data()
detections = read_uv_csv('data.csv')

mht = MHT(detections, parameters)
mhtSol = mht.run()
write_uv_csv("mhtSol.csv", mhtSol)
plot_2d_tracks("mhtSol.csv")

# my = MyMHT(detections, parameters)
# mySol = my.run()
# write_uv_csv("mySol.csv", mySol)
