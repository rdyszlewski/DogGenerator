class DataSplitter:

    @staticmethod
    def split(x_train, iteration, batch_size):
        start_index = batch_size * iteration
        end_index = batch_size * iteration + batch_size
        if end_index > x_train.shape[0]:
            end_index = x_train.shape[0] - 1
        train_data = x_train[start_index: end_index]
        if len(train_data) > batch_size:
            remains = train_data.shape[0] - len(train_data)
            train_data.concatenate(train_data[:remains - 1])
        return train_data
