class Model():
    def __init__(self, n_x = 1, n_y = 1):
        self.theta0 = 0
        self.theta1 = 0
        self.n_x = n_x
        self.n_y = n_y
        self.train = False

    def __call__(self, value):
        if self.train:
            return self.theta0 + (self.theta1 * value)
        else:
            value = value * self.n_x
            y = self.theta0 + (self.theta1 * value)
            return y / self.n_y
    
    def save(self, path, n_x = 1.0, n_y = 1.0):
        try:
            with open(path, 'w') as file:
                file.write(str(self.theta0))
                file.write('\n')
                file.write(str(self.theta1))
                file.write('\n')
                file.write(str(n_x))
                file.write('\n')
                file.write(str(n_y))
        except FileNotFoundError:
            print("Error: file not found")
    
    def load(self, path):
        try:
            with open(path, 'r') as file:
                first_line = file.readline().strip()
                second_line = file.readline().strip()
                third_line = file.readline().strip()
                fourth_line = file.readline().strip()
                
                self.theta0 = float(first_line)
                self.theta1 = float(second_line)
                self.n_x = float(third_line)
                self.n_y = float(fourth_line)

        except Exception as e:
            print("Error: ", e)



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

    def _step(self, learning_rate, model, dataset):
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

    def train(self, learning_rate, epochs, model, dataset):
        for i in range(epochs):
            model.train = True
            loss = self._step(learning_rate, model, dataset)
            print(f"[Epoch {i}] Loss: {loss}")
