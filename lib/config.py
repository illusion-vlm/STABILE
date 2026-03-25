from argparse import ArgumentParser

BATCHNORM_MOMENTUM = 0.01

class Config(object):
    """Wrapper class for model hyperparameters."""

    def __init__(self):
        """
        Defaults
        """
        self.mode = None
        self.base_path = None
        self.save_path = None
        self.save_folder = None
        self.model_path = None
        self.data_path = None
        self.datasize = None
        self.ckpt = None
        self.optimizer = None
        self.lr = 1e-5
        self.factor = 0.5
        self.obj_att_layer = 3
        self.enc_layer = 1
        self.dec_layer = 3

        self.no_logging = None
        
        self.obj_retriever = None
        
        self.losses_alpha = None
        self.losses_belta = None
        self.losses_temp = None
        
        self.moco_s_k = None
        self.moco_c_k = None
        self.moco_base_k = None

        self.nepoch = 20
        self.parser = self.setup_parser()
        self.args = vars(self.parser.parse_args())
        self.__dict__.update(self.args)

    def setup_parser(self):
        """
        Sets up an argument parser
        :return:
        """
        parser = ArgumentParser(description='training code')
        parser.add_argument('-mode', dest='mode', help='predcls/sgcls/sgdet', default='predcls', type=str)
        parser.add_argument('-base_path', default='/root/workspace/', type=str)
        parser.add_argument('-save_path', default='checkpoints/stabile/', type=str)
        parser.add_argument('-save_folder', default='datetime', type=str)
        parser.add_argument('-model_path', default=None, type=str)
        parser.add_argument('-data_path', default='datasets/ag/', type=str)
        parser.add_argument('-datasize', dest='datasize', help='mini dataset or whole', default='large', type=str)
        parser.add_argument('-ckpt', dest='ckpt', help='checkpoint', default=None, type=str)
        parser.add_argument('-optimizer', help='adamw/adam/sgd', default='adamw', type=str)
        parser.add_argument('-lr', dest='lr', help='learning rate', default=1e-5, type=float)
        parser.add_argument('-nepoch', help='epoch number', default=20, type=float)
        parser.add_argument('-factor', dest='factor', help='factor', default=0.5, type=float)
        parser.add_argument('-obj_att_layer', dest='obj_att_layer', help='object retriever module attention layer numbers', default=3, type=int)
        parser.add_argument('-enc_layer', dest='enc_layer', help='spatial encoder layer', default=1, type=int)
        parser.add_argument('-dec_layer', dest='dec_layer', help='temporal decoder layer', default=3, type=int)

        parser.add_argument('-log_iter', dest='log_iter', help='log_iter', default=1000, type=int)
        parser.add_argument('-no_logging', action='store_true')
        parser.add_argument('-scheduler_step', dest='scheduler_step', help='recall/mrecall', default='recall', type=str)
        
        parser.add_argument('-obj_retriever', action='store_true')
        
        parser.add_argument('-s_k', dest='s_k', help='s_k', default=1024, type=int)
        parser.add_argument('-c_k', dest='c_k', help='c_k', default=1024, type=int)
        parser.add_argument('-base_k', dest='base_k', help='base_k', default=2, type=int)
        parser.add_argument('-contrastive_type', dest='contrastive_type', help='uml/linear', default='uml', type=str)
        parser.add_argument('-losses_alpha', dest='losses_alpha', help='losses_alpha', default=1.7, type=float)
        parser.add_argument('-losses_beta', dest='losses_beta', help='losses_beta', default=0.5, type=float)
        parser.add_argument('-losses_t', dest='losses_t', help='losses_t', default=0.07, type=float)
        
        return parser
