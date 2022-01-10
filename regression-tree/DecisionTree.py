class DecisionTree(object):
    def __init__(self, data=None):
        self.left = None  # Less than value
        self.right = None  # Greater than value
        self.criterion_value = None
        self.criterion_index = None
        self.data = data

    def split_data(self, criterion_index, criterion_value):
        self.criterion_index = criterion_index
        self.criterion_value = criterion_value

        left_data = self.data[self.data.T[criterion_index] <= criterion_value]
        right_data = self.data[self.data.T[criterion_index] > criterion_value]

        self.left = DecisionTree(left_data)
        self.right = DecisionTree(right_data)

        # del self.data  # Split node does not require to hold all data

    def get_criterion(self):
        return self.criterion_index, self.criterion_value

    def get_data(self):
        return self.data

    def to_left(self):
        return self.left

    def to_right(self):
        return self.right

    def description(self):
        return f"Length of Data: {len(self.data)}, Criterion Value: {self.criterion_value}, Criterion Index: {self.criterion_index}"

    def find_minimal_node(self, value, descr=False):
        node = self

        while True:
            crit_index, crit_value = node.get_criterion()
            if crit_value == None:
                return node

            if value[crit_index] <= crit_value:
                if descr:
                    print("left")
                node = node.left
            else:
                if descr:
                    print("right")
                node = node.right
