import sys
from models import Model

def main(args):
    model = Model()
    if len(args) < 2:
        return
    if len(args) > 2:
        model.load(args[2])
    
    mileage = float(args[1])
    print(model(mileage))


if __name__ == "__main__":
    main(sys.argv)
    input("\nPremi Invio per chiudere...")