import json
import os
from typing import Any, Dict


class ResultsDatabase:
    def __init__(self, path: str, name: str = "database"):
        self._path = f"{path}/{name}.json"
        if os.path.exists(self._path):
            with open(self._path, "r") as f:
                self._data = json.load(f)
            self.check_duplicates()
        else:
            self._data = []

    def lookup(self, params: Dict[str, Any]):
        for database_item in self._data:
            database_params = database_item["params"]
            if database_params == params:
                return True
        return False

    def check_duplicates(self):
        for i in range(len(self._data)):
            for j in range(i + 1, len(self._data)):
                params_i = self._data[i]["params"]
                params_j = self._data[j]["params"]
                if params_i == params_j:
                    raise ValueError("Duplicate params found in database")

    def add_result(self, params: Dict[str, Any], result: Dict[str, Any]):
        new_item = {"params": params, "result": result}
        self._data.append(new_item)

    def modify_result(self, params: Dict[str, Any], result: Dict[str, Any]):
        for database_item in self._data:
            if database_item["params"] == params:
                database_item["result"] = result
                return
        self.add_result(params, result)

    def save(self):
        with open(self._path, "w") as f:
            json.dump(self._data, f)
