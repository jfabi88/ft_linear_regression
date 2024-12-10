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
    learning_rate = 0.9
    epochs = 100

    if len(args) < 2:
        print("Default learning rate(pass an argument to set): 0.9")
    else:
        try:
            learning_rate = float(args[1])
            print(f"Learnging rate: {learning_rate}")
        except ValueError:
            print("Error: Wrong learning rate")
            input("\nPremi Invio per chiudere...")
            exit(-1)
    
    if len(args) < 3:
         print("Default epochs(pass an argument to set): 100")
    else:
        try:
            epochs = int(args[2])
            print(f"Epochs: {epochs}")
        except ValueError:
            print("Error: Wrong learning rate")
            input("\nPremi Invio per chiudere...")
            exit(-1)

    return learning_rate, epochs


def normalize_data(data):
    max_x = 0
    max_y = 0

    for x, y in data:
        if x > max_x:
            max_x = x
        if y > max_y:
            max_y = y
    
    data = [(x / max_x, y / max_y) for x, y in data]

    return data, max_x, max_y


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
    learning_rate, epochs = parse(args)

    data = read_file("./data.csv")

    data, max_km, max_price = normalize_data(data)

    trainer = Trainer()
    model = Model(1.0 / max_km, 1.0 / max_price)
    dataset = Dataset(data)

    trainer.train(learning_rate, epochs, model, dataset)
    model.save("parameters.txt", 1.0 / max_km, 1.0 / max_price)

    plot(dataset, model)


if __name__ == "__main__":
    main(sys.argv)
    input("\nPremi Invio per chiudere...")
