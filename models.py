class Model():
    def __init__(self):
        self.theta0 = 0
        self.theta1 = 0

    def __call__(self, value):
        return self.theta0 + (self.theta1 * value)


class Dataset():
    def __init__(self, data):
        self.data = data
        self.index = 0

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index < len(self.data):
            element = self.data[self.index]
            self.index += 1
            return element
        else:
            raise StopIteration
    
    def __len__(self):
        return len(self.data)


class Trainer():

    def train(self, learning_rate, model, dataset):
        loss = 0
        gradient0 = 0
        gradient1 = 0

        for value, y in dataset:
            x = model(value)
            error = x - y
            loss += error ** 2

            gradient0 += error
            gradient1 += error * value
        
        gradient0 /= len(dataset)
        gradient1 /= len(dataset)

        model.theta0 -= gradient0 * learning_rate
        model.theta1 -= gradient1 * learning_rate

        return loss
