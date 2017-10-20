from pgmpy.models import DynamicBayesianNetwork
from pgmpy.inference import DBNInference

import networkx as nx
import numpy as np


class DbnCnnInterface(object):
    def __init__(self, model_file='../DBN/network.nx'):
        nx_model = nx.read_gpickle(model_file)
        self.dbn = DynamicBayesianNetwork(nx_model.edges())
        self.dbn.add_cpds(*nx_model.cpds)
        self.dbn.initialize_initial_state()
        self.dbn_infer = DBNInference(self.dbn)

    def filter_q_values(self, q_values, evidence=0, method='binary'):
        inferred = np.ndarray(shape=(len(q_values),), dtype=float)
        inferred.fill(0)
        variables = self.dbn.get_slice_nodes(1)
        ev = {node: 0 for node in self.dbn.get_slice_nodes(0)}
        if evidence != 0:
            self.set_evidence(ev, evidence)
        q = self.dbn_infer.query(variables=variables, evidence=ev)
        for variable in q.values():
            action = self.get_action_id(variable.variables[0])
            if method == 'binary':
                inferred[action] = 1 if variable.values[1] > 0 else 0
            else:
                inferred[action] = variable.values[1]
        return q_values * inferred

    def get_action_id(self, action):
        if action[0] == 'Prompt':
            return 0
        elif action[0] == 'Reward':
            return 1
        elif action[0] == 'Abort':
            return 2
        return 3

    def set_evidence(self, evidence, id):
        if id == 1:
            evidence[("Prompt", 0)] = 1
