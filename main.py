from Trainer import STsim_Trainer
import torch
import os

if __name__ == '__main__':
    print(torch.__version__)
    print(torch.cuda.device_count())
    print(torch.cuda.is_available())

    # train and test

    STsim = STsim_Trainer()

    load_model_name = None
    load_optimizer_name = None 

    STsim.ST_train(load_model=load_model_name, load_optimizer=load_optimizer_name)

    # STsim.ST_eval(load_model=load_model_name)

