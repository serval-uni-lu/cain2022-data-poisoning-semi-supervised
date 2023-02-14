import yaml
import datasets as ds
from types import FunctionType
import label_propagation_cached
import sklearn.semi_supervised
import sys


class Config:
    def __init__(self, path: str):
        file = open(f"{path}", "r")
        c = yaml.load(file, Loader=yaml.FullLoader)
        self.datasets: dict = c.get("datasets")
        self.commons = c.get("common")
        self.seed: int = self.commons.get("random_seed")
        self.ssl_algo = {}
        self.p_labelled = self.commons.get("labelled_proportion")
        self.flip_budgets = self.commons.get("flip_budget_proportion")
        # Assign every loader function name to its implementation
        for name, attr in self.datasets.items():
            loader_function_name: str = attr.get("loader_function")
            attr["loader_function"]: FunctionType = getattr(ds, loader_function_name)

        # Assign the different ssl algorithms
        ssl: str
        for ssl_name, ssl in self.commons.get("ssl_algo").items():
            string = ssl.split(".")
            class_name = string.pop(-1)
            if len(string) > 1:
                module = sys.modules[".".join(string)]
            else:
                module = sys.modules[string[0]]
            self.ssl_algo[ssl_name] = getattr(module, class_name)


if __name__ == "__main__":
    config = Config("config.yaml")
