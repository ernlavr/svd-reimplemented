import argparse

from svdllm.main import add_eval_args, run_eval_from_args


def main():
    parser = argparse.ArgumentParser(description="Evaluate perplexity of a HF causal LM")
    add_eval_args(parser)
    args = parser.parse_args()
    run_eval_from_args(args)


if __name__ == "__main__":
    main()

