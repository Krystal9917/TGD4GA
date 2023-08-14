import argparse
import os


class Args:
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--ds', type=str, default="")
        parser.add_argument('--room_seqs_path', type=str, default=os.path.abspath(
            os.path.join(os.path.dirname(__file__), os.path.pardir, "data",
                         "room_seqs" + ".csv")))
        parser.add_argument('--room_seqs_parquet_path', type=str, default=os.path.abspath(
            os.path.join(os.path.dirname(__file__), os.path.pardir, "data",
                         "room_seqs_infer" + ".parquet")))
        parser.add_argument('--model_states_path', type=str, default=os.path.abspath(
            os.path.join(os.path.dirname(__file__), os.path.pardir, "data",
                         "model_states", "optimal_sequence_model" + ".bin")))
        parser.add_argument('--vocab_size', type=int, default=57)
        parser.add_argument('--max_len', type=int, default=128)
        parser.add_argument('--padding_idx', type=int, default=0)
        parser.add_argument('--output_size', type=int, default=1)
        parser.add_argument('--n_layers', type=int, default=1)
        parser.add_argument('--emb_dim', type=int, default=32)
        parser.add_argument('--n_heads', type=int, default=2)
        parser.add_argument('--batch_size', type=int, default=512)
        parser.add_argument('--lr', type=float, default=0.001)
        parser.add_argument('--n_epochs', type=int, default=100)
        parser.add_argument('--drop_rate', type=float, default=0.5)
        parser.add_argument('--device', type=str, default="gpu")
        parser.add_argument('--infer_device', type=str, default="cpu")
        parser.add_argument('--seed', type=int, default=20)
        parser.add_argument('--best_auc', type=float, default=0.91)
        args = parser.parse_args()
        args_dict = vars(args)
        self.args_dict = args_dict
