import argparse


def get_train_parser():
    parser = argparse.ArgumentParser(
        description='arguments for training')
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--target', type=str)
    parser.add_argument('--model', type=str)
    parser.add_argument('--save_path', type=str)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--gpus', type=int, default=0)
    parser.add_argument('--num_epoch', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--criterion', type=str)
    parser.add_argument('--m_per_class', type=int, default=2)
    parser.add_argument('--optimizer', type=str, default="adam")
    parser.add_argument('--use_ddp', type=int, default=1)
    parser.add_argument('--loader', type=str, default="default")
    parser.add_argument('--temperature', type=float, default=0.07)
    parser.add_argument('--seed', type=int, default=42)
    return parser
