# Security Policy

## Supported versions

`birdnet` is under active development and ships a single line of releases. Security fixes are
made against the **latest release on [PyPI](https://pypi.python.org/pypi/birdnet)**. Please
upgrade to the latest version before reporting a problem. We do not backport fixes to older releases.

## Reporting a vulnerability

**Please do not open a public issue for security problems.**

Report privately through GitHub's
[**Report a vulnerability**](https://github.com/birdnet-team/birdnet/security/advisories/new)
button (Security tab → Report a vulnerability). If you'd rather use email, contact
**josefhaupt@gmx.net**.

Please include enough to reproduce: the affected version, backend/model, platform, and a minimal
proof of concept. You can expect an initial response within a few days — this is a small,
maintainer-run project, so please allow reasonable time for a fix before any public disclosure.

## Scope

The parts of this library most relevant to security are less about the inference math and more
about **what gets downloaded and loaded from disk**:

- **Model and taxonomy auto-download.** Official models and the V3.0 taxonomy are fetched over the
  network (~3 GB) on first use. Issues in how those artifacts are retrieved, verified, or cached
  are in scope.
- **Loading model files.** Loading a model can deserialize third-party files — notably `torch.load`
  (pickle) for `.pt` backends, and SavedModel/TFLite artifacts. Loading a **custom or untrusted
  model file** (`load_custom`) executes whatever that file contains; treat model files like code
  and only load ones you trust. Vulnerabilities in *our* loading path are in scope.
- **Path and file handling** for audio inputs and result exports.

## Out of scope

- crashes on malformed audio (please file those as a regular bug), model prediction accuracy, and
vulnerabilities in optional dependencies themselves (report those upstream, though we're glad to bump a pin).
