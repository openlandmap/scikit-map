import json
import sqlite3
from osgeo import ogr
import os
import csv

from .exceptions import LucasDataError, LucasLoadError


class LucasClassAggr:
    """Perform LC class aggregation.

    :param str gpkg_path: path to GPKG file created by :class:`.io.LucasIO.to_gpkg()`.
    """
    def __init__(self, gpkg_path):
        self._gpkg_path = gpkg_path

    def _load_classes(self, json_path):
        """Load aggregation rules from JSON file.

        :param str json_path: path to JSON file with defined aggregation rules

        :return dictionary: dictionary of original classes and names of aggregated classes
        """
        csv_lc1 = os.path.join(os.path.dirname(__file__), "lc1_codes.csv")
        with open(csv_lc1, newline='') as csv_f:
            layer_reader = csv.DictReader(csv_f, delimiter=";")
            # collect possible lc1 codes
            possible_codes = []
            for row in layer_reader:
                possible_codes.append(row["code"])

        try:
            with open(json_path) as json_file:
                try:
                    classes = json.load(json_file)
                    values = list(classes.values())
                    values_list = []
                    for i in values:
                        for j in i:
                            values_list.append(j)
                            if j not in possible_codes:
                                raise LucasDataError(f"Code {j} is not Land Cover code!")

                    if len(values_list) != len(set(values_list)):
                        raise LucasDataError("Some code is used repeatedly!")

                except ValueError as e:
                    raise LucasDataError(f"Invalid json file: {e}")
        except FileNotFoundError as e:
            raise LucasLoadError(f"Invalid json file path: {e}")

        return classes

    def apply(self, json_path):
        """Apply aggregation rules defined in JSON file on GPKG file

        :param str JSON_path: path to JSON file with defined aggregation rules
        """
        driver = ogr.GetDriverByName("GPKG")
        if os.path.exists(self._gpkg_path):
            gpkg = driver.Open(self._gpkg_path)
            layer = gpkg.GetLayer()
            layer_name = layer.GetName()

            if layer_name[6:8] == "st":
                columns_h = []
                columns_a = []
                layer_definition = layer.GetLayerDefn()
                for i in range(layer_definition.GetFieldCount()):
                    attr = layer_definition.GetFieldDefn(i).GetName()
                    if attr in ["lc1_h_2006", "lc1_h_2009", "lc1_h_2012", "lc1_h_2015", "lc1_h_2018"]:
                        columns_h.append(attr)
                        columns_a.append(attr.replace("h", "a"))
                if not columns_h:
                    raise LucasDataError(f"There is no lc1_h column in gpkg file!")
            else:
                columns_h = ["lc1_h"]
                columns_a = ["lc1_a"]


            classes = self._load_classes(json_path)
            with sqlite3.connect(self._gpkg_path) as con:
                con.enable_load_extension(True)
                cur = con.cursor()
                cur.execute('SELECT load_extension("mod_spatialite");')
                for h_column, new_column in zip(columns_h, columns_a):
                    try:
                        cur.execute(f"CREATE INDEX IF NOT EXISTS {h_column}_idx ON {layer_name}({h_column})")
                        cur.execute(f"ALTER TABLE {layer_name} ADD COLUMN {new_column} TEXT")
                        for key in classes:
                            q_marks = "?" * len(classes[key])
                            sql_query = f"UPDATE {layer_name} SET {new_column} =? WHERE {h_column} IN ({','.join(q_marks)})"
                            val = tuple([key] + classes[key])
                            cur.execute(sql_query, val)
                    except sqlite3.OperationalError as e:
                        raise LucasDataError(f"Not possible to aggregate: {e}")
            con.close()

        else:
            raise LucasLoadError("GPKG file doesn't exist")
