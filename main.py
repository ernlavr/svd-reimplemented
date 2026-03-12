from __future__ import annotations

"""
Top-level entrypoint for SVD-LLM utilities.

This forwards to the CLI implemented in `svdllm.main`, so you can run:

    python main.py compress ...
    python main.py eval ...
"""

from svdllm.main import main


if __name__ == "__main__":
    main()

