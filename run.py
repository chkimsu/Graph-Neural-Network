import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from train_eval import run

from custom_datasets import CustomDataset
from models.original_model import Net as Net_ori
from models.normalized import Net as Net_imp
from models.final_model import Net as Net_fin


if __name__ == '__main__':
    dataset = CustomDataset()

    kwargs = {
        'dataset': dataset,
        'str_optimizer': 'Adam',
        'runs': 1,
        'epochs': 200, 
        'lr': 0.01,
        'lr_decay': None,
        'weight_decay': 0.0005,
        'early_stopping': 0,
        'gamma': None,
        'model_path': None
    }

    """
    step 1: experiment of original model from SSP
    
    """
    kwargs['logger'] = 'run_original'
    kwargs['model'] = Net_ori(dataset)
    kwargs['epochs'] = 200
    kwargs['model_path'] = './save/model_ori.pth'
    run(**kwargs)

    """
    step 2: experiment of original model with 2 times widened hidden size for 10 times longer training epochs
    
    """
    kwargs['logger'] = 'run_original_longer'
    kwargs['epochs'] = 2000
    kwargs['model'] = Net_ori(dataset, hidden=32)
    kwargs['model_path'] = './save/model_widen.pth'
    run(**kwargs)

    """
    step 3: experiment of improved model
    
    """
    kwargs['logger'] = 'run_improved'
    kwargs['epochs'] = 2000
    kwargs['lr'] = 4e-2
    kwargs['lr_decay'] = 0.999
    kwargs['weight_decay'] = 1e-4
    kwargs['logger'] = 'run_improved'
    kwargs['model'] = Net_imp(dataset, hidden=16)
    kwargs['model_path'] = './save/model_imp.pth'
    run(**kwargs)

    """
    step 4: experiment of final model
    
    """
    kwargs['logger'] = 'run_final'
    kwargs['model'] = Net_fin(dataset, hidden=32, blocks=5)
    kwargs['model_path'] = './save/model_fin.pth'
    run(**kwargs)





