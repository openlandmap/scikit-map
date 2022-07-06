'''
Analyze downloaded LUCAS samples.
'''

import json
import sqlite3
from osgeo import ogr
import os
import csv

from .exceptions import LucasDataError, LucasLoadError, LucasConfigError


class LucasClassAggregate:
    """Perform LC class aggregation.

    :param str gpkg_path: path to GPKG file created by :class:`.io.LucasIO.to_gpkg()`.
    """
    def __init__(self, gpkg_path, mappings=None, mappings_file=None):
        if mappings is not None and mappings_file is not None:
            raise LucasConfigError('Only one of the parameters "data" and '
                                   '"json_path" should be defined')
        if mappings is None and mappings_file is None:
            raise LucasConfigError('One of the parameters "data" and '
                                   '"json_path" should be defined')

        self._gpkg_path = gpkg_path
        self.mappings = mappings
        self.mappings_file = mappings_file

    def _load_classes(self, classes):
        """Load aggregation rules from JSON file.

        :param dict classes: defined aggregation rules

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

        return classes

    def apply(self):
        """Apply aggregation rules on GPKG file

        :param dict data: defined aggregation rules
        """
        if self.mappings is not None:
            self._apply_from_data()
        else:
            self._apply_from_file()

    def _apply_from_file(self):
        """Apply aggregation rules defined in JSON file on GPKG file

        :param str json_path: path to JSON file with defined aggregation rules
        """
        try:
            with open(self.mappings_file) as json_file:
                self.mappings = json.load(json_file)
                self._apply_from_data()
        except FileNotFoundError as e:
            raise LucasLoadError(f"Invalid json file path: {e}")

    def _apply_from_data(self):
        """Apply aggregation rules on GPKG file

        :param dict data: defined aggregation rules
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


            classes = self._load_classes(self.mappings)
            with sqlite3.connect(self._gpkg_path) as con:
                con.enable_load_extension(True)
                cur = con.cursor()
                cur.execute('SELECT load_extension("mod_spatialite");')
                for h_column, new_column in zip(columns_h, columns_a):
                    try:
                        cur.execute(
                            f"ALTER TABLE {layer_name} ADD COLUMN {new_column} "
                            f"TEXT")
                    except sqlite3.OperationalError:
                        sql_query = f"UPDATE {layer_name} SET {new_column} = " \
                                    f"null"
                        cur.execute(sql_query)
                        print(f"Column {new_column} already exists in the table "
                              f"{layer_name} - it will be rewritten with the new "
                              f"data")
                    try:
                        cur.execute(f"CREATE INDEX IF NOT EXISTS {h_column}_idx ON {layer_name}({h_column})")
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


class LucasClassTranslate:
    """Perform LC class translation.

    :param str gpkg_path: path to GPKG file created by :class:`.io.LucasIO.to_gpkg()`.
    """
    def __init__(self, gpkg_path, csvpath=None):
        if csvpath is None:
            csvpath = os.path.join(os.path.split(__file__)[0],
                                   'LUCAS_unique_lc1_lu1_combinations_ALL.csv')

        self.gpkg_path = gpkg_path
        self.csv_path = csvpath
        self.source_col = 'CLC3'

        if not os.path.exists(self.gpkg_path):
            raise LucasLoadError("GPKG file doesn't exist")
        if not os.path.exists(self.csv_path):
            raise LucasLoadError("CSV file doesn't exist")

    def get_translations(self):
        """Get supported translations.

        :return list: list of translation tables
        """
        with open(self.csv_path) as trans_data:
            return trans_data.readline().split(',')

    def set_translations(self, source_col):
        """Get supported translations.

        :return list: list of translation tables
        """
        self.source_col = source_col

    def apply(self):
        """Apply translation rules on GPKG file.
        """
        driver = ogr.GetDriverByName("GPKG")

        gpkg = driver.Open(self.gpkg_path)
        layer = gpkg.GetLayer()
        layer_name = layer.GetName()

        with open(self.csv_path) as trans_data:
            header = trans_data.readline()
            header_list = header.split(',')

            new_col = 'clc3'
            lc1_col = 'lc1'
            lu1_col = 'lu1'

            if self.source_col not in header_list:
                raise LucasConfigError(
                    f'Column {self.source_col} not found in the header. '
                    f'Only columns {header} found')

            new_col_ind = header_list.index(self.source_col)
            lc1_ind = header_list.index('LC1')
            lu1_ind = header_list.index('LU1')
            repre_ind = header_list.index('Representativeness')

            new_col = self.source_col.lower()

            with sqlite3.connect(self.gpkg_path) as con:
                con.enable_load_extension(True)
                cur = con.cursor()
                cur.execute('SELECT load_extension("mod_spatialite");')
                try:
                    cur.execute(
                        f"ALTER TABLE {layer_name} ADD COLUMN {new_col} TEXT")
                except sqlite3.OperationalError:
                    sql_query = f"UPDATE {layer_name} SET {new_col} = null"
                    cur.execute(sql_query)
                    print(f"Column {new_col} already exists in the table "
                          f"{layer_name} - it will be rewritten with the new "
                          f"data")

                sql_query = f"UPDATE {layer_name} SET {new_col} = ? WHERE " \
                            f"{lc1_col} = ? AND {lu1_col} = ?"

                for line in trans_data.readlines():
                    line_split = line.split(',')
                    if line_split[repre_ind] == '1':
                        cur.execute(sql_query,
                                    (line_split[new_col_ind],
                                     line_split[lc1_ind],
                                     line_split[lu1_ind]))
