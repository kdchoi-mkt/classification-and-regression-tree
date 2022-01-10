import numpy as np
from DecisionTree import DecisionTree


class RegressionDecisionTree(object):
    def __init__(self, max_depth, min_data=1, report=False):
        self.max_depth = max_depth
        self.min_data = min_data
        self.report = report

    def fit(self, X, y):
        """Parameter
        ===========
        X: explain variable (Nxk matrix)
        y: dependent variable (N vector)

        Return
        ======
        Regression Tree (itself)
        """
        index = 0
        self.now_depth = 1

        data = np.column_stack([X, y])
        self.tree = DecisionTree(data=data)
        to_visit_node = [self.tree]
        will_visit_node = list()

        while True:
            if self.criterion() == True:
                break

            now_visit = to_visit_node.pop(0)

            exp_var = now_visit.get_data()[:, :-1]
            dep_var = now_visit.get_data()[:, -1]

            if len(dep_var) > self.min_data:
                min_variance_cand = np.zeros(shape=exp_var.shape[1])
                min_variance_val = np.zeros(shape=exp_var.shape[1])
                for index in range(0, exp_var.shape[1]):
                    exp_vector = exp_var.T[index]
                    split = np.vectorize(
                        lambda x: self._split_center(exp_vector, dep_var, x)
                    )
                    split_variance = split(exp_vector)
                    min_variance_cand[index] = exp_vector[split_variance.argmin()]
                    min_variance_val[index] = split_variance.min()

                now_visit.split_data(
                    min_variance_val.argmin(),
                    min_variance_cand[min_variance_val.argmin()],
                )

                if now_visit.to_left() != None:
                    will_visit_node.append(now_visit.to_left())
                if now_visit.to_right() != None:
                    will_visit_node.append(now_visit.to_right())

                if self.report:
                    print(self.now_depth, index, min_variance_cand)

            if len(to_visit_node) == 0:
                to_visit_node = will_visit_node
                will_visit_node = list()
                self.now_depth += 1

        self._finalize_tree()

        return self

    def _split_center(self, exp_vector, dep_var, value):
        """Parameter
        ===========
        X: explain vector (N vector)
        y: dependent variable (N vector)

        Return
        ======
        Total Variance for Split
        """
        split_upper = dep_var[exp_vector > value]
        split_lower = dep_var[exp_vector <= value]

        if len(split_upper) == 0 or len(split_lower) == 0:
            return np.inf

        return np.var(split_upper) * len(split_upper) + np.var(split_lower) * len(
            split_lower
        )

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

            if len(to_visit_node) == 0:
                to_visit_node = will_visit_node

    def criterion(self):
        if self.now_depth > self.max_depth:
            return True

        return False

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
            mean_value.append(
                self.tree.find_minimal_node(value).get_data()[:, -1].mean()
            )
        return np.array(mean_value)
