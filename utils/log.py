import json


class Log():
    # initialization of class
    def __init__(self, opt):
        self.data = {'config': {}, 'results': []}
        self.opt = opt
        self.add_opt()

    # Make json file 
    def make_json(self):
        with open('./results/data.json', 'w', encoding='utf-8') as f:
            json.dump(self.data, f, ensure_ascii=False, indent=4)

    # Add data from Opt
    def add_opt(self):
        self.data['config']['batchSize'] = self.opt.batchSize
        self.data['config']['num_points'] = self.opt.num_points
        self.data['config']['learning_type'] = self.opt.learning_type
        self.data['config']['loss_type'] = self.opt.loss_type
        self.data['config']['start_num_class'] = self.opt.start_num_class
        self.data['config']['step_num_class'] = self.opt.step_num_class
        self.data['config']['number_of_sample_old_classes'] = self.opt.n_cands
        self.data['config']['dist_temperature'] = self.opt.dist_temperature
        self.data['config']['dist_factor'] = self.opt.dist_factor
        self.data['config']['cands_path'] = self.opt.cands_path
        self.data['config']['few_shots'] = self.opt.few_shots
        self.data['config']['model'] = self.opt.model
        self.data['config']['dataset'] = self.opt.dataset
        self.data['config']['dataset_type'] = self.opt.dataset_type
        self.data['config']['feature_transform'] = self.opt.feature_transform
        self.data['config']['input_transform'] = self.opt.input_transform
        self.data['config']['KD'] = self.opt.KD
    


