import pandas as pd

class HandleData:

    def __init__(self, data: pd.DataFrame):
        self.data = self.normalize_data(data)

    def normalize_data(self, data):
        '''
            Function to normalize data
        '''
        data = data[data['nfeatures'] != 0]
        for col in data:
            if col == 'features':
                data.loc[:, col] = data[col].str.replace(r'[{}]', '', regex=True)
                data.loc[:, col] = data[col].str.split('},').apply(
                    lambda x: [list(map(float, item.split(','))) for item in x]
                )
            elif col == 'targets':
                data.loc[:, col] = data[col].str.replace(r'[{}]', '', regex=True)
                data.loc[:, col] = data[col].str.split(',').apply(
                    lambda x: [float(i) for i in x]
                )

        return data
