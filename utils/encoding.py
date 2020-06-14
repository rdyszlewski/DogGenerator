
class OneHotEncoder:

    def __init__(self):
        self.__labels_map = {}

    def encode(self, labels):
        self.__labels_map = self.__prepare_labels_map(labels)
        encoded_labels = self.__encode(labels, self.__labels_map)
        return encoded_labels


    def __prepare_labels_map(self, labels):
        counter = 0
        labels_map = {}
        for label in labels:
            if label not in labels_map:
                labels_map[label] = counter
                counter += 1
        return labels_map

    def __encode(self, labels, labels_map):
        encoded_labels = []
        labels_size = len(labels_map)
        for label in labels:
            values = [0] * labels_size
            label_index = labels_map[label]
            values[label_index] = 1
            encoded_labels.append(values)

        return encoded_labels
