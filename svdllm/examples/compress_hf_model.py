import argparse

from svdllm.main import add_compress_args, run_compress_from_args


def main():
    parser = argparse.ArgumentParser(description="Compress a HF causal LM with SVD-LLM")
    add_compress_args(parser)
    args = parser.parse_args()
    run_compress_from_args(args)


if __name__ == "__main__":
    main()

