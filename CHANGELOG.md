# Changelog

<!--

Changelog follow the https://keepachangelog.com/ standard (at least the headers)

This allow to:

* auto-parsing release notes during the automated releases from github-action:
  https://github.com/marketplace/actions/pypi-github-auto-release
* Have clickable headers in the rendered markdown

To release a new version (e.g. from `1.0.0` -> `2.0.0`):

* Create a new `# [2.0.0] - YYYY-MM-DD` header and add the current
  `[Unreleased]` notes.
* At the end of the file:
  * Define the new link url:
  `[2.0.0]: https://github.com/google-research/e3x/compare/v1.0.0...v2.0.0`
  * Update the `[Unreleased]` url: `v1.0.0...HEAD` -> `v2.0.0...HEAD`

-->

## [Unreleased]



## [1.0.2] - 2024-04-05

* Added e3x.ops.normalize_and_return_norm
* Added option to include mapping as weighting in mapped functions
* Added option to return vector norms to e3x.nn.basis
* Added e3x.nn.ExponentialBasis (wraps e3x.nn.basis, injecting learnable gamma)
* Added e3x.ops.inverse_softplus helper function
* Added e3x.so3.delley_quadrature for computing Delley quadratures of S2

## [1.0.1] - 2024-01-17

* Increased minimum required Python version to 3.9

## [1.0.0] - 2023-12-31

* Initial release

[Unreleased]: https://github.com/google-research/e3x/compare/v1.0.2...HEAD
[1.0.2]: https://github.com/google-research/e3x/releases/tag/v1.0.2
[1.0.1]: https://github.com/google-research/e3x/releases/tag/v1.0.1
[1.0.0]: https://github.com/google-research/e3x/releases/tag/v1.0.0
