import pandas as pd
import numpy as np

class Expert:
    "Expert base class"
    rng = None
    def __init__(self, conf) -> None:
        "Initialize expert accuracy and setup configuration"
        self.accuracy = conf.accuracy if conf.accuracy else None
        self.n_labels = conf.n_labels
        self.conf = conf
        Expert.rng = conf.rng

class ExpertReal(Expert):
    """Expert for real data experiments"""
    def __init__(self, conf) -> None:
        super().__init__(conf)
        # Directory of expert predictions
        self.human_probs = conf.human_probs
        self.ground_truth = conf.ground_truth
        self.confusion_matrix = self.create_confusion_matrix()
        self.w_matrix = self.get_w_from_confusion_matrix() 

    def get_w_from_confusion_matrix(self):
        return np.log(self.confusion_matrix)

    def create_confusion_matrix(self):
        with open(self.human_probs, "rb") as f:
            cm_per_sample = np.load(f)

        with open(self.ground_truth, "r") as f:
            csv = np.loadtxt(f, delimiter=',')
            y = csv[:,-1].astype(int) - 1
            assert(all([y_val >= 0 for y_val in y]))

        cm = np.zeros(shape=(self.n_labels,self.n_labels))
        for i in range(self.n_labels):
            idx = np.argwhere(y == i).flatten()
            cm[i] = cm_per_sample[idx].mean(axis=0)
        return cm

def read_classification_data(file_path: str, classifier_data: bool= False, n_features: int = 10):
    """Read the classification data files and extract the probabilities."""

    expert_choices = []
    label_softmax = []
    ground_truths = []

    with open(file_path, 'r') as datafile:
        for line in datafile:
            single_line_data = line.strip().split(',')

            if not classifier_data:
                human_expert_choices = [float(x) for x in single_line_data[0:n_features]]
                human_expert_probabilities = [float(x) for x in single_line_data[n_features:n_features*2]]
                ground_truth_class = float(single_line_data[n_features*2])
            else:
                human_expert_choices = [float(x) for x in single_line_data[1:n_features+1]]
                human_expert_probabilities = [float(x) for x in single_line_data[n_features+1:n_features*2+1]]
                ground_truth_class = float(single_line_data[0])
                
            expert_choices.append(human_expert_choices)
            label_softmax.append(human_expert_probabilities)
            ground_truths.append(ground_truth_class)

    return expert_choices, np.array(label_softmax), np.array(ground_truths)

