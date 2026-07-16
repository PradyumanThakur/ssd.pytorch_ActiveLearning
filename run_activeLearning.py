"""
Run the complete Active Learning experiment.

Pipeline
--------
Round-00
    ↓
Round-01
    ↓
Round-02
    ↓
...
Round-N
"""

import argparse
import subprocess
from pathlib import Path


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--method", choices=["random", "cdal"], required=True)
    parser.add_argument("--num-rounds", type=int, default=9)
    parser.add_argument("--dataset", default="VOC")
    parser.add_argument("--dataset_root", default="datasets/PASCAL_VOC/VOCdevkit")
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--initial-budget", default=1000, type=int)
    parser.add_argument("--acquisition-budget", default=1000, type=int)
    parser.add_argument("--save-freq", default=9999, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--cuda", default=True, type=str2bool)
    parser.add_argument("--start-round", type=int, default=0, 
                    help="""
                    0 : Run Round-00 then all AL rounds.
                    >0: Resume from an existing AL round.
                    Example:
                    --start-round 4
                    """)
    return parser.parse_args()


def main():
    args = parse_args()

    if args.start_round == 0:
        print("=" * 70)
        print("Running Round-00")
        print("=" * 70)

        subprocess.run(
            [
                "python",
                "run_round_00.py",
                "--experiment-name", args.experiment_name,
                "--dataset", args.dataset,
                "--dataset_root", args.dataset_root,
                "--epochs", str(args.epochs),
                "--batch_size", str(args.batch_size),
                "--num_workers", str(args.num_workers),
                "--lr", str(args.lr),
                "--initial-budget", str(args.initial_budget),
                "--acquisition-budget", str(args.acquisition_budget),
                "--save-freq", str(args.save_freq),
                "--seed", str(args.seed),
                "--cuda", str(args.cuda),
            ],
            check=True,
        )

        start_round = 1
    else:
        print("=" * 70)
        print(f"Resuming from Round-{args.start_round:02d}")
        print("=" * 70)

        start_round = args.start_round

    for round_no in range(start_round, args.num_rounds + 1):

        print("=" * 70)
        print(f"Running {args.method.upper()} Round-{round_no:02d}")
        print("=" * 70)

        cmd = [
                "python",
                "al_round.py",

                "--method", args.method,
                "--round", str(round_no),

                "--experiment-name", args.experiment_name,

                "--dataset", args.dataset,
                "--dataset_root", args.dataset_root,

                "--epochs", str(args.epochs),
                "--batch_size", str(args.batch_size),
                "--num_workers", str(args.num_workers),

                "--lr", str(args.lr),

                "--acquisition-budget", str(args.acquisition_budget),

                "--save-freq", str(args.save_freq),

                "--seed", str(args.seed),

                "--cuda", str(args.cuda),
        ]

        if round_no == start_round and start_round > 0:
            resume_ckpt = Path("experiments") / args.experiment_name / args.method / f"round_{round_no:02d}" / "checkpoints" / "checkpoint_latest.pth"

            if resume_ckpt.exists():
                print(f"Resuming training from {resume_ckpt}")
                cmd.extend(["--resume", str(resume_ckpt)])
            else:
                raise FileNotFoundError(
                    f"Checkpoint not found: {resume_ckpt}"
                )
            
            subprocess.run(cmd, check=True)

    print("\n")
    print("=" * 70)
    print("Active Learning Finished")
    print("=" * 70)


if __name__ == "__main__":
    main()