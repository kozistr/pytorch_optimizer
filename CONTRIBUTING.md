*This guideline is very much a WIP*

Contributions to `pytorch-optimizer` for code, documentation, and tests are always welcome!

# Coding Style

Currently, `black` and `ruff` are used to format & lint the code. Here are the [lint options](https://github.com/kozistr/pytorch_optimizer/blob/main/pyproject.toml#L69)
Or you just simply run `make format` and `make check` on the project root.

You can just install the pip packages to your computer or use `uv` to create an `venv`.

A few differences from the default `black` (or another style guide) are

1. line-length is **119** characters.
2. **single quote** is preferred instead of a double quote.

But, maybe, if you feel or think that it's too much or takes much time, then feel free to ask the maintainer to fix the lint stuff!

# Documentation

Docstring style is `Google`, and documentation will be built & deployed automatically via `readthedocs`. You can find an example from [here](https://github.com/kozistr/pytorch_optimizer/blob/main/pytorch_optimizer/optimizer/adamp.py#L14).

# Test

You can run a test by `make test` on the project root!

# Question

If you have any questions about contribution, please ask in the Issues, Discussions, or just in PR :)

Thank you!
