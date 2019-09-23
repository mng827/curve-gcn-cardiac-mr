import os

from data_loader import load_pytorch
from misc.config import process_config
from misc.utils import get_logger, get_args, makedirs
from models.poly_gnn import PolyGNN
from train import Trainer


def main():

    config = None
    try:
        args = get_args()
        config = process_config(args.config)

        if config is None:
            raise Exception()
    except:
        print("Add a config file using \'--config file_name.json\'")
        exit(1)

    makedirs(config.summary_dir)
    makedirs(config.checkpoint_dir)

    logger = get_logger('log', logpath=config.summary_dir, filepath=os.path.abspath(__file__))

    train_loader, test_loader = load_pytorch(config)

    model = PolyGNN(state_dim=128,
                    n_adj=4,
                    coarse_to_fine_steps=config.coarse_to_fine_steps,
                    get_point_annotation=False)

    trainer = Trainer(model, train_loader, test_loader, config, logger)

    if config.train:
        trainer.train()

    if config.validation:
        trainer.resume(os.path.join(config.checkpoint_dir, 'model.pth'))
        trainer.test_epoch(cur_epoch=999, plot=True)


if __name__ == "__main__":
    main()

