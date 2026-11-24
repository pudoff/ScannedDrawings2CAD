"""Versioned development splits: verified pairs only, grouped before augmentation."""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re

import ezdxf

SYNTHETIC = re.compile(r"^data/dxf/drawing_\d+\.dxf$", re.IGNORECASE)
FAMILIES = {"плита": "plate", "фланец": "flange", "вал": "shaft",
            "кронштейн": "bracket", "крышка": "cover"}


def digest_json(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def _read(value):
    return json.loads(Path(value).read_text(encoding="utf-8")) if isinstance(value, (str, Path)) else value


def build_family_catalog(dataset_root: Path, manifest: dict | Path) -> dict:
    """Read actual synthetic CAD labels. Pool real data until its families are reviewed.

    A family denotes a conservative leakage group, not semantic equivalence of
    every member. Different geometric variants of a synthetic template stay
    together. The real pool is deliberately ineligible for independent testing.
    """
    manifest = _read(manifest)
    root = Path(dataset_root).resolve()
    records = []
    for item in sorted(manifest["files"], key=lambda f: f["path"]):
        if item["kind"] != "dxf":
            continue
        relative = item["path"]
        source = (root / relative).resolve()
        entry = {"target_path": relative, "target_sha256": item["sha256"],
                 "family_id": None, "train_only": True, "evidence": []}
        try:
            if not source.is_relative_to(root):
                raise ValueError("Path outside dataset")
            with source.open("rb") as stream:
                current = hashlib.file_digest(stream, "sha256").hexdigest()
            if current != item["sha256"]:
                raise ValueError("File differs from audit manifest")
            if SYNTHETIC.fullmatch(relative):
                doc = ezdxf.readfile(source)
                labels = [entity.dxf.text for entity in doc.modelspace() if entity.dxftype() == "TEXT"]
                matched = {(FAMILIES[match.group(1)], label) for label in labels
                           if (match := re.match(r"^(плита|фланец|вал|кронштейн|крышка)\s+\d", label.casefold()))}
                families = {family for family, _ in matched}
                if len(families) != 1:
                    raise ValueError("Missing or conflicting synthetic CAD family label")
                family = next(iter(families))
                entry.update(family_id=f"synthetic:{family}", train_only=family == "plate",
                             evidence=sorted({label for _, label in matched}), method="cad_title_label")
            else:
                entry.update(family_id="real:unreviewed-pool", method="conservative_real_pool",
                             evidence=["All customer-real variants remain development-only until family review."])
            entry["status"] = "grouped"
        except Exception as exc:
            entry.update(status="quarantine", error=str(exc))
        records.append(entry)
    return {"schema_version": "1.0", "manifest_sha256": digest_json(manifest), "families": records,
            "policy": {"synthetic": "CAD title label, all template variants together",
                       "real": "Single train-only pool, semantic families not yet reviewed",
                       "debugged": "drawing_001 was inspected previously; whole plate family is train-only",
                       "acceptance": "The ten external acceptance scans are not supplied or assigned here"}}


def build_splits(manifest: dict | Path, verification: dict | Path,
                 catalog: dict | Path, *, seed: str = "44labs-development-v1") -> dict:
    manifest, verification, catalog = map(_read, (manifest, verification, catalog))
    manifest_digest = digest_json(manifest)
    if catalog.get("manifest_sha256") != manifest_digest:
        raise ValueError("Family catalog is stale or belongs to another manifest")
    files = {item["path"]: item for item in manifest["files"]}
    families = {item["target_path"]: item for item in catalog["families"]}
    grouped, quarantine = [], []
    seen = set()
    for pair in verification["pairs"]:
        source, target = pair["input_path"], pair["target_path"]
        key = (source, target)
        if key in seen:
            continue
        seen.add(key)
        record = {"input_path": source, "target_path": target}
        reason = None
        if pair["status"] != "verified":
            reason = pair.get("release_rejection", "pair_not_verified")
        elif source not in files or target not in files:
            reason = "file_missing_from_manifest"
        elif (not pair.get("input_sha256") or not pair.get("target_sha256")
              or pair["input_sha256"] != files[source]["sha256"]
              or pair["target_sha256"] != files[target]["sha256"]):
            reason = "verification_hash_mismatch"
        elif not families.get(target, {}).get("family_id"):
            reason = "family_unresolved"
        elif families[target]["target_sha256"] != files[target]["sha256"]:
            reason = "family_hash_mismatch"
        if reason:
            quarantine.append({**record, "reason": reason})
            continue
        family = families[target]
        grouped.append({**record, "input_sha256": files[source]["sha256"],
                        "target_sha256": files[target]["sha256"], "family_id": family["family_id"],
                        "train_only": bool(family["train_only"])})

    # Union both declared families and exact duplicates, including source/target identity.
    parent = {item["family_id"]: item["family_id"] for item in grouped}

    def find(key):
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[max(a, b)] = min(a, b)

    by_hash = {}
    for item in grouped:
        for digest in (item["input_sha256"], item["target_sha256"]):
            if digest in by_hash:
                union(item["family_id"], by_hash[digest])
            by_hash[digest] = item["family_id"]
    components = defaultdict(list)
    for item in grouped:
        components[find(item["family_id"])].append(item)
    eligible = [key for key, items in components.items() if not any(item["train_only"] for item in items)]
    eligible.sort(key=lambda key: hashlib.sha256(f"{seed}:{key}".encode()).hexdigest())
    assignment = {key: "train" for key in components}
    # Hold out whole families. Fractions are intentionally not forced at sample level.
    if len(eligible) >= 2:
        assignment[eligible[0]] = "test"
        assignment[eligible[1]] = "validation"
    elif eligible:
        assignment[eligible[0]] = "validation"
    splits = {"train": [], "validation": [], "test": []}
    groups = []
    for key, items in sorted(components.items()):
        split = assignment[key]
        families_in_group = sorted({item["family_id"] for item in items})
        group_id = "group-" + digest_json(families_in_group)[:16]
        groups.append({"group_id": group_id, "families": families_in_group, "split": split,
                       "train_only": any(item["train_only"] for item in items), "pair_count": len(items)})
        for item in sorted(items, key=lambda x: (x["input_path"], x["target_path"])):
            splits[split].append({**item, "group_id": group_id})
    # All unverified input files remain visible; absent verification is not an acceptance.
    covered = {item["input_path"] for item in grouped} | {item["input_path"] for item in quarantine}
    excluded_inputs = [{"input_path": p, "reason": "no_verified_correspondence"}
                       for p, item in sorted(files.items())
                       if item["kind"] in {"pdf", "image"} and item.get("role") != "ancillary" and p not in covered]
    validate_splits(splits)
    warnings = ["Development split only; not the ten-scan external acceptance set.",
                "Customer-real semantic families are unreviewed and all remain train-only.",
                "Automated visual correspondence does not certify geometry or dimensional accuracy."]
    if not splits["validation"] or not splits["test"]:
        warnings.append("Insufficient verified independent families for nonempty validation and test sets.")
    return {"schema_version": "1.0", "seed": seed, "status": "development_only",
            "manifest_sha256": manifest_digest, "verification_sha256": digest_json(verification),
            "catalog_sha256": digest_json(catalog), "splits": splits, "groups": groups,
            "quarantine": sorted(quarantine, key=lambda x: (x["input_path"], x["target_path"])),
            "excluded_inputs": excluded_inputs, "warnings": warnings,
            "summary": {"pairs_by_split": {key: len(items) for key, items in splits.items()},
                        "group_count": len(groups), "quarantine_pair_count": len(quarantine),
                        "excluded_input_count": len(excluded_inputs)}}


def validate_splits(splits: dict) -> None:
    owners = {}
    for split, items in splits.items():
        for item in items:
            if item.get("train_only") and split != "train":
                raise ValueError("Train-only family leaked into evaluation")
            keys = [("family", item["family_id"]), ("group", item["group_id"]),
                    ("file", item["input_path"]), ("file", item["target_path"]),
                    ("hash", item["input_sha256"]), ("hash", item["target_sha256"])]
            for key in keys:
                if key in owners and owners[key] != split:
                    raise ValueError(f"Leakage across {owners[key]} / {split}: {key}")
                owners[key] = split


def write_splits(dataset_root: Path, manifest: dict | Path, verification: dict | Path,
                 output: Path, *, seed: str = "44labs-development-v1") -> dict:
    output, root = Path(output), Path(dataset_root).resolve()
    if output.resolve().is_relative_to(root):
        raise ValueError("Split artifacts must be outside dataset")
    output.mkdir(parents=True, exist_ok=False)
    manifest, verification = _read(manifest), _read(verification)
    catalog = build_family_catalog(root, manifest)
    current_verification = deepcopy(verification)
    input_checks = {}
    for pair in current_verification["pairs"]:
        if pair["status"] != "verified":
            continue
        relative = pair["input_path"]
        if relative not in input_checks:
            source = (root / relative).resolve()
            try:
                if not source.is_relative_to(root):
                    raise ValueError("Input path outside dataset")
                with source.open("rb") as stream:
                    input_checks[relative] = {"sha256": hashlib.file_digest(stream, "sha256").hexdigest()}
            except (OSError, ValueError) as exc:
                input_checks[relative] = {"sha256": None, "error": str(exc)}
        if input_checks[relative]["sha256"] != pair.get("input_sha256"):
            pair.update(status="quarantine", release_rejection="input_changed_since_verification")
    result = build_splits(manifest, current_verification, catalog, seed=seed)
    result["verification_sha256"] = digest_json(verification)
    result["release_validation"] = {"input_checks": input_checks,
                                    "accepted_pair_files_rehashed": True}
    for name, value in (("families.json", catalog), ("splits.json", result)):
        (output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = ["# Рабочее разбиение", "", "Это разбиение для разработки, не итоговая приёмочная выборка.", "",
             "| Выборка | Пар |", "|---|---:|"]
    lines.extend(f"| {name} | {len(items)} |" for name, items in result["splits"].items())
    lines.extend(["", f"Неподтверждённых пар: {len(result['quarantine'])}; входов без проверки: {len(result['excluded_inputs'])}.",
                  "", "Все варианты семейства и точные дубли остаются в одной выборке.",
                  "Плита (ранее разобран drawing_001) и все реальные данные используются только в train.",
                  "Соответствие по изображению не является проверкой точности размеров."])
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result
