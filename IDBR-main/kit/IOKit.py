import json
import csv

class IOKit:

    # 读取txt 文本
    @staticmethod
    def read_txt(path):
        with open(path, "r") as f:
            data = f.readlines()
            f.close()
            return data

    @staticmethod
    def write_json(path, json_str):
        f = open(path, 'w')
        f.write(json_str)
        f.close()

    @staticmethod
    def read_json(path):
        with open(path, "r") as f:
            data = f.read()
            f.close()
            return data


    @staticmethod
    def read_csv(path):
        with open(path, "r") as f:
            data = f.read()
            f.close()
            return data