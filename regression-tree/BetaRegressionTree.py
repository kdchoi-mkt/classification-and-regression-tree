import numpy as np
from RegressionTree import RegressionDecisionTree


class BetaRegressionTree(RegressionDecisionTree):
    def __init__(self, max_depth, min_data=1, report=False):
        self.max_depth = max_depth
        self.min_data = min_data
        self.report = report

    def predict(self, X):
        """Parameter
        ============
        X: explain vector (N vector)

        Return
        ======
        Predicted mean value
        """
        mean_value = list()
        for value in X:
            mean_value.append(value.T @ self.tree.find_minimal_node(value).beta)
        return np.array(mean_value)

    def _finalize_tree(self):
        """Finalize Tree"""

        to_visit_node = [self.tree]
        will_visit_node = list()

        while len(to_visit_node) > 0:
            has_child = False
            visit_node = to_visit_node.pop(0)

            if visit_node.to_left() != None:
                has_child = True
                will_visit_node.append(visit_node.to_left())

            if visit_node.to_left() != None:
                has_child = True
                will_visit_node.append(visit_node.to_right())

            if has_child:
                del visit_node.data
            else:
                exp_data = visit_node.data[:, :-1]
                dep_data = visit_node.data[:, -1]
                visit_node.beta = (
                    np.linalg.inv(exp_data.T @ exp_data) @ exp_data.T @ dep_data
                )

            if len(to_visit_node) == 0:
                to_visit_node = will_visit_node
