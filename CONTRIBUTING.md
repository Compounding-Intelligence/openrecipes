# Contributing to OpenRecipes

OpenRecipes grows with every recipe people measure. If you have tuned a model on
H100, H200, B200 or another GPU, we would like your recipe: a faster one for a
pair already listed, or one for a model or GPU the catalog does not list yet.

## Add a recipe

1. Measure it under the protocol: https://openrecipes.ai/protocol
2. Check it against its reference recipe with the convergence check.
3. Write one YAML file from the template at https://openrecipes.ai/submit,
   rebuild the site with `python3 tools/build.py`, and open a pull request.

We review every recipe for its measurement, its correctness evidence, and
anything that should not be public, such as internal hostnames, account names or
data you may not cite.

## Other contributions

Fixes and improvements to the tools, the row builder and the site are welcome.
For a larger change, open an issue first so we can agree on the approach.

## Questions

Open an issue on GitHub.

## License

Contributions are licensed under the Apache License, Version 2.0, the license of
this repository.
