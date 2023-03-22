import json


class Log():
    # initialization of class
    def __init__(self, opt):
        self.opt = opt
        self.add_opt()
        self.data = {'config': {}, 'results': {}}

    # Make json file 
    def make_json(self):
        with open('./result/data.json', 'w', encoding='utf-8') as f:
            json.dump(self.data, f, ensure_ascii=False, indent=10)

    # Add data from Opt
    def add_opt(self):
        print(self.opt)
