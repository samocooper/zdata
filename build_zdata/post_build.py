"""Optional post-build steps shared by the zarr and MTX+CSV build paths.

Both builders finish by running the same three optional steps over the freshly
written zdata directory. Keeping them here avoids the two copies drifting apart
and gives them one consistent failure policy.

Failure policy
--------------
These steps are genuinely optional -- a zdata object is valid without them --
but several are *load-bearing for downstream ML*: ``sample_id`` is the batch key
that ``feature_presence`` is indexed by, so a silent ``sample_id`` failure makes
the feature-presence step fail too, and an atlas that looks built is missing the
key training artefacts.

``strict=True`` (recommended for pipelines) raises on the first failure.
``strict=False`` preserves the historical behaviour of warning and continuing,
but reports a summary at the end so failures cannot pass unnoticed, and skips
steps whose prerequisites failed rather than emitting a second confusing error.
"""
from __future__ import annotations

import os


class PostBuildError(RuntimeError):
    """A required post-build step failed."""


def run_post_build_steps(
    zdata_dir,
    obs_filename: str = "obs.parquet",
    *,
    generate_sample_id: bool = True,
    optimize_obs: bool = True,
    generate_feature_presence: bool = True,
    feature_presence_sample_col: str = "sample_id",
    feature_presence_var_dirs=None,
    strict: bool = False,
    verbose: bool = True,
) -> dict:
    """Run the optional post-build steps. Returns a {step: status} summary.

    Status is one of ``"ok"``, ``"skipped: <reason>"`` or ``"failed: <error>"``.
    """
    zdata_dir = str(zdata_dir)
    results: dict[str, str] = {}

    def _say(msg):
        if verbose:
            print(msg, flush=True)

    def _fail(step, exc):
        results[step] = f"failed: {exc}"
        if strict:
            raise PostBuildError(f"post-build step '{step}' failed: {exc}") from exc
        _say(f"  ! {step} FAILED: {exc}")

    # --- 1. globally-unique monotonic sample_id (+ sample_uid) ---------------
    if generate_sample_id:
        try:
            from zdata.build_zdata.sample_id import assign_global_sample_id
            n = assign_global_sample_id(zdata_dir, obs_filename=obs_filename,
                                        verbose=verbose)
            results["sample_id"] = "ok"
            _say(f"  ok sample_id / sample_uid ({n:,} samples)")
        except Exception as e:
            _fail("sample_id", e)
    else:
        results["sample_id"] = "skipped: disabled"

    # --- 2. compact obs dtypes ----------------------------------------------
    if optimize_obs:
        try:
            from zdata.build_zdata.optimize_obs import optimize_obs_parquet
            optimize_obs_parquet(os.path.join(zdata_dir, obs_filename),
                                 verbose=verbose)
            results["optimize_obs"] = "ok"
            _say("  ok obs dtypes optimised")
        except Exception as e:
            _fail("optimize_obs", e)
    else:
        results["optimize_obs"] = "skipped: disabled"

    # --- 3. per-sample gene-presence matrix ---------------------------------
    # Depends on sample_id: skip rather than emit a confusing second error.
    if not generate_feature_presence:
        results["feature_presence"] = "skipped: disabled"
    elif not feature_presence_var_dirs:
        results["feature_presence"] = "skipped: no feature_presence_var_dirs"
        _say("  - feature_presence skipped: pass feature_presence_var_dirs "
             "(per-study var.csv source) to enable")
    elif results.get("sample_id", "").startswith("failed") \
            and feature_presence_sample_col == "sample_id":
        results["feature_presence"] = "skipped: sample_id step failed"
        _say("  - feature_presence skipped: depends on sample_id, which failed")
    else:
        try:
            from zdata.build_zdata.feature_presence import build_feature_presence_matrix
            build_feature_presence_matrix(
                zdata_dir,
                var_source_dirs=feature_presence_var_dirs,
                sample_col=feature_presence_sample_col,
                obs_filename=obs_filename,
                verbose=verbose,
            )
            results["feature_presence"] = "ok"
            _say("  ok feature_presence_matrix.npz")
        except Exception as e:
            _fail("feature_presence", e)

    failed = [k for k, v in results.items() if v.startswith("failed")]
    if failed and verbose:
        print(f"\n  WARNING: {len(failed)} post-build step(s) failed: "
              f"{', '.join(failed)}", flush=True)
        print("  The zdata object is readable but is missing artefacts that "
              "downstream ML training expects.", flush=True)
    return results
