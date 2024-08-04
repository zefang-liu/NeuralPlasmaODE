"""
Main Procedure
"""
from src.trainer import TrainerITER

if __name__ == '__main__':
    trainer = TrainerITER(is_inductive=True, start_up=False)
    trainer.init_net_opt()
    trainer.train_net()
