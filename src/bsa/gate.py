"""Quality gate and promotion for a trained run.

    python -m bsa.gate --run-dir outputs/<run_id>          # check, register, promote to @champion
    python -m bsa.gate --run-dir outputs/<run_id> --push   # ...and publish to the Hugging Face Hub

A candidate passes when it clears the absolute floors in ``gate.min_test`` and does
not fall more than ``gate.max_regression`` below the current champion. A champion
comparison only counts if both models were scored on the identical test set.
Exits non-zero when the gate fails, so it can block a CI job.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from bsa.config import load_config

CHAMPION_ALIAS = "champion"


@dataclass
class Check:
    name: str
    passed: bool
    detail: str


@dataclass
class GateResult:
    checks: list[Check]

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)

    def to_dict(self) -> dict:
        return {"passed": self.passed, "checks": [asdict(check) for check in self.checks]}


def check_gate(candidate: dict, gate_cfg: dict, champion: dict | None) -> GateResult:
    test = candidate["test"]
    checks = [
        Check(f"min_{metric}", test[metric] >= floor, f"{test[metric]:.4f} >= {floor:.4f}")
        for metric, floor in gate_cfg["min_test"].items()
    ]
    if champion is None:
        checks.append(Check("no_regression", True, "no champion yet; skipped"))
    elif champion["test_fingerprint"] != candidate["dataset"]["fingerprints"]["test"]:
        checks.append(Check(
            "comparable_test_set", False,
            f"champion v{champion['version']} was scored on a different test set",
        ))
    else:
        metric, tolerance = gate_cfg["compare_metric"], gate_cfg["max_regression"]
        best = champion["metrics"][metric]
        checks.append(Check(
            f"no_regression_{metric}", test[metric] >= best - tolerance,
            f"{test[metric]:.4f} >= champion v{champion['version']} {best:.4f} - {tolerance}",
        ))
    return GateResult(checks)


def current_champion(client: MlflowClient, name: str) -> dict | None:
    try:
        version = client.get_model_version_by_alias(name, CHAMPION_ALIAS)
    except MlflowException:
        return None
    run = client.get_run(version.run_id)
    return {
        "version": version.version,
        "run_id": version.run_id,
        "test_fingerprint": run.data.tags.get("test_fingerprint"),
        "metrics": {
            k.removeprefix("test_"): v for k, v in run.data.metrics.items() if k.startswith("test_")
        },
    }


def promote(client: MlflowClient, run_id: str, name: str) -> str:
    if "model" not in {artifact.path for artifact in client.list_artifacts(run_id)}:
        raise SystemExit("Run has no logged model artifact (log_model_artifact was false); cannot register.")
    version = mlflow.register_model(f"runs:/{run_id}/model", name)
    client.set_registered_model_alias(name, CHAMPION_ALIAS, version.version)
    print(f"Registered {name} v{version.version} as @{CHAMPION_ALIAS}")
    return version.version


def push_to_hub(client: MlflowClient, run_dir: Path, candidate: dict, repo_id: str) -> None:
    from huggingface_hub import CommitOperationAdd, HfApi

    files = [p for p in (run_dir / "model").iterdir() if p.is_file()]
    operations = [CommitOperationAdd(path_in_repo=p.name, path_or_fileobj=str(p)) for p in files]
    for report in ("metrics.json", "gate_report.json"):
        operations.append(CommitOperationAdd(path_in_repo=report, path_or_fileobj=str(run_dir / report)))

    run_id, test = candidate["run_id"], candidate["test"]
    api = HfApi()
    commit = api.create_commit(
        repo_id=repo_id,
        operations=operations,
        commit_message=(
            f"Promote run {run_id[:8]}: test accuracy {test['accuracy']:.4f}, "
            f"macro-F1 {test['f1_macro']:.4f}"
        ),
    )
    tag = f"run-{run_id[:8]}"
    api.create_tag(repo_id, tag=tag, revision=commit.oid)
    client.set_tag(run_id, "hf_revision", commit.oid)
    print(f"Pushed to https://huggingface.co/{repo_id} (commit {commit.oid[:8]}, tag {tag})")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Gate, register and optionally publish a run.")
    parser.add_argument("--run-dir", required=True, type=Path, help="outputs/<run_id>")
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("-o", "--override", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--push", action="store_true", help="publish to the Hugging Face Hub if passed")
    args = parser.parse_args(argv)

    cfg = load_config(args.config, args.override)
    gate_cfg = cfg["gate"]
    candidate = json.loads((args.run_dir / "metrics.json").read_text(encoding="utf-8"))
    run_id = candidate["run_id"]

    mlflow.set_tracking_uri(cfg["tracking_uri"])
    client = MlflowClient()
    champion = current_champion(client, gate_cfg["registered_model"])
    already_champion = champion is not None and champion["run_id"] == run_id

    if already_champion:
        result = GateResult([Check("already_champion", True, f"run is champion v{champion['version']}")])
    else:
        result = check_gate(candidate, gate_cfg, champion)

    report_path = args.run_dir / "gate_report.json"
    report_path.write_text(
        json.dumps({"run_id": run_id, "champion": champion, **result.to_dict()}, indent=2),
        encoding="utf-8",
    )
    client.set_tag(run_id, "gate", "passed" if result.passed else "failed")
    client.log_artifact(run_id, str(report_path))

    for check in result.checks:
        print(f"  [{'PASS' if check.passed else 'FAIL'}] {check.name}: {check.detail}")
    if not result.passed:
        print("Gate FAILED: model not promoted.")
        return 1

    if not already_champion:
        promote(client, run_id, gate_cfg["registered_model"])
    if args.push:
        push_to_hub(client, args.run_dir, candidate, gate_cfg["hf_repo"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
