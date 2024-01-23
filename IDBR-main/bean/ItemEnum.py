import json
from kit.IOKit import IOKit

class ItemEnum:

    def __init__(self) -> None:       
        self.enum = {}

    def addEnum(self, b):
        if b not in self.enum.keys():
            self.enum[b] = len(self.enum)

    def get_item_size(self):
        return len(self.enum)

