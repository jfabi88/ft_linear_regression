import csv
import sys
import matplotlib.pyplot as plt
from models import Model, Trainer, Dataset

def read_file(filename = "./data.csv"):
    data = []
    try:
        with open(filename, newline="") as filecsv:
            reader = csv.reader(filecsv, delimiter=",")
            header = next(reader)
            for row in reader:
                km, price = map(int, row)
                data.append((km, price))
            return data
    except (
            FileNotFoundError,
            PermissionError,
            IsADirectoryError,
            OSError,
            csv.Error,
            ValueError,
            TypeError,
            StopIteration
    ) as e:
        print("Error: {e}")
        input("\nPress Enter to exit...")
        exit(-1)


def parse(args):
    learning_rate = 0.1

    if len(args) < 2:
        print("Default learning rate(pass an argument to set): 0.1")
    else:
        try:
            learning_rate = float(args[1])
            print(f"Learnging rate: {learning_rate}")
        except ValueError:
            print("Error: Wrong learning rate")
            input("\nPremi Invio per chiudere...")
            exit(-1)

    return learning_rate


def plot(data, model):
    x = [km for km, price in data]
    y = [price for km, price in data]

    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, color='blue', label='Dati originali')

    plt.title("Chilometraggio vs Prezzo", fontsize=16)
    plt.xlabel("Chilometraggio (km)", fontsize=14)
    plt.ylabel("Prezzo (€)", fontsize=14)

    y_pred = [model(km) for km in x]
    plt.plot(x, y_pred, color='red', label='Linea di regressione')

    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.show()


def main(args):
    data = []
    epochs = int(args[2])
    learning_rate = parse(args)

    data = read_file("./data.csv")
    max_km = 240000
    max_price = 8290
    data = [(km / max_km, price / max_price) for km, price in data]
    train = Trainer()
    model = Model()
    dataset = Dataset(data)
    for _ in range(epochs):
        loss = train.train(learning_rate, model, dataset)
        print(f"Loss: {loss}")
    plot(dataset, model)


if __name__ == "__main__":
    main(sys.argv)
    input("\nPremi Invio per chiudere...")
