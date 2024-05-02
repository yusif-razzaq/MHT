import logging
from copy import deepcopy

from openmht.kalman_filter import KalmanFilter
from openmht.mht import MHT
from openmht.weighted_graph import WeightedGraph
# from truth_data import observations_history


class MyMHT:
    """Main class for the MHT algorithm."""

    def __init__(self, observation_history, params):
        self.observation_history = list(observation_history)
        self.params = params
        self.track_leaves = []
        self.detections_map = []
        self.track_hists = []
        self.track_count = 0
        self.prune_ids = set()

    def run(self):
        for index, frame in enumerate(self.observation_history):
            print(f"Processing frame {index}...")
            self.track_count = len(self.track_leaves)  # Existing tracks before frame
            self.generate_tracks(index, frame)
            self.add_missed_detections()
            solution_coordinates = self.prune_trees(index)
        return solution_coordinates

    def generate_tracks(self, k, frame):
        self.detections_map.append({})
        for detection_index, detection in enumerate(frame):
            detection_id = str(detection_index)
            self.detections_map[k][detection_id] = detection

            # Update all existing branches according to one detection in one frame
            for i in range(self.track_count):
                track_tree = self.track_leaves[i]
                extended_branch = deepcopy(track_tree)
                extended_branch.update(detection)
                self.track_leaves.append(extended_branch)
                self.track_hists.append(self.track_hists[i] + [detection_id])

            # Create new branches for each detection
            self.track_leaves.append(KalmanFilter(detection, v=self.params.get('v'), dth=self.params.get('dth'),
                                                  k=self.params.get('k'), q=self.params.get('q'), r=self.params.get('r'),
                                                  nmiss=self.params.get('nmiss'), ck=self.params.get('ck')))
            track_id = [''] * k + [detection_id]
            self.track_hists.append(track_id)

    def prune_trees(self, k):
        prune_index = max(0, k - self.params.get('n'))
        conflicting_tracks = self.get_conflicting_tracks()
        best_hypothesis = self.global_hypothesis(conflicting_tracks)  # Returns ids of tracks in best hypothesis
        non_solution_ids = list(set(range(len(self.track_leaves))) - set(best_hypothesis))  # Complement of tracks in best hypothesis
        solution_coordinates = []
        for track in best_hypothesis:
            track_hist = self.track_hists[track]
            track_coordinates = []
            for i, detection in enumerate(track_hist):
                if detection == '': track_coordinates.append(None)
                else: track_coordinates.append(self.detections_map[i][detection])
            solution_coordinates.append(track_coordinates)

            # Prune branches that diverge from the solution track tree at frame k-N
            d_id = track_hist[prune_index]
            if d_id != '':
                for non_solution_id in non_solution_ids:
                    if d_id == self.track_hists[non_solution_id][prune_index]:
                        self.prune_ids.add(non_solution_id)

        for i in sorted(self.prune_ids, reverse=True):
            del self.track_hists[i]
            del self.track_leaves[i]
        return solution_coordinates

    def add_missed_detections(self):
        # Update the previous filter with a missed detection
        self.prune_ids.clear()
        for j in range(self.track_count):
            update_success = self.track_leaves[j].update(None)
            self.track_hists[j].append('')
            if not update_success:
                self.prune_ids.add(j)

    def get_conflicting_tracks(self):
        # Create a conflict matrix for each frame. Each row is a pair of conflicting tracks by index.
        conflicting_tracks = []
        for detections_a in self.track_hists:
            for detections_b in self.track_hists:
                if detections_a != detections_b:
                    conflicting = False
                    for k, detection in enumerate(detections_a):
                        if detection != '' and detection == detections_b[k]:
                            conflicting = True
                            break

                    if conflicting:
                        conflicting_tracks.append((self.track_hists.index(detections_a), self.track_hists.index(detections_b)))
        return conflicting_tracks

    def global_hypothesis(self, conflicting_tracks):
        """
        Generate a global hypothesis by finding the maximum weighted independent
        set of a graph with tracks as vertices, and edges between conflicting tracks.
        """
        gh_graph = WeightedGraph()
        for index, track in enumerate(self.track_leaves):
            gh_graph.add_weighted_vertex(str(index), track.get_track_score())

        gh_graph.set_edges(conflicting_tracks)

        mwis_ids = gh_graph.mwis()

        return mwis_ids


if __name__ == '__main__':
    pass
