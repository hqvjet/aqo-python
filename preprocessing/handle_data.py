import pandas as pd

class HandleData:

    def __init__(self, data: pd.DataFrame):
        self.data = self.normalize_data(data)

    def normalize_data(self, data):
        '''
            Function to normalize data
        '''
        data = data[data['nfeatures'] != 0]
        data = data.values.tolist()

        for row_index in range(len(data)):
            features = data[row_index][3][2:-2].split('},{')
            new_features = []
            for point in features:
                new_features.append([float(i) for i in point.split(',')])

            data[row_index][3] = new_features

            targets = data[row_index][4][1:-1].split(',')
            data[row_index][4] = [float(i) for i in targets]

        return data
