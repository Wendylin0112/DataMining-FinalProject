import argparse
from pathlib import Path

import pandas as pd


def parse_override(value):
    if "=" not in value:
        raise argparse.ArgumentTypeError("Override must use route=threshold format.")
    route, threshold = value.split("=", 1)
    route = route.strip()
    if not route:
        raise argparse.ArgumentTypeError("Route name cannot be empty.")
    try:
        threshold = float(threshold)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Threshold must be a number.") from exc
    if not 0 <= threshold <= 1:
        raise argparse.ArgumentTypeError("Threshold must be between 0 and 1.")
    return route, threshold


def parse_args():
    parser = argparse.ArgumentParser(
        description="Regenerate a submission CSV from detailed prediction scores with route-specific thresholds."
    )
    parser.add_argument("--detailed", required=True, help="Detailed prediction CSV with defect_route and defect_score.")
    parser.add_argument("--template", default="test_submission_template.csv", help="Submission template CSV.")
    parser.add_argument("--output", required=True, help="Output submission CSV.")
    parser.add_argument("--detailed-output", default=None, help="Optional adjusted detailed output CSV.")
    parser.add_argument(
        "--override",
        action="append",
        type=parse_override,
        default=[],
        help="Route threshold override, e.g. --override global=0.50 --override yoke-suspension=0.50",
    )
    parser.add_argument(
        "--component-override",
        action="append",
        type=parse_override,
        default=[],
        help="Predicted-component threshold override, applied after route overrides.",
    )
    parser.add_argument(
        "--default-threshold",
        type=float,
        default=None,
        help="Optional threshold for routes that do not have an override. If omitted, keep existing pred_label.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    detailed = pd.read_csv(args.detailed)
    template = pd.read_csv(args.template)

    required_columns = {"filename", "defect_route", "defect_score", "pred_label"}
    missing = required_columns.difference(detailed.columns)
    if missing:
        raise ValueError(f"{args.detailed} is missing columns: {sorted(missing)}")
    if "filename" not in template.columns:
        raise ValueError(f"{args.template} must contain a filename column.")

    thresholds = dict(args.override)
    adjusted = detailed.copy()

    if args.default_threshold is not None:
        adjusted["pred_label"] = (adjusted["defect_score"] >= args.default_threshold).astype(int)

    for route, threshold in thresholds.items():
        mask = adjusted["defect_route"] == route
        if not mask.any():
            print(f"warning: route not found in detailed predictions: {route}")
            continue
        adjusted.loc[mask, "pred_label"] = (adjusted.loc[mask, "defect_score"] >= threshold).astype(int)

    component_thresholds = dict(args.component_override)
    for component, threshold in component_thresholds.items():
        if "predicted_component" not in adjusted.columns:
            raise ValueError("component overrides require a predicted_component column.")
        mask = adjusted["predicted_component"] == component
        if not mask.any():
            print(f"warning: component not found in detailed predictions: {component}")
            continue
        adjusted.loc[mask, "pred_label"] = (adjusted.loc[mask, "defect_score"] >= threshold).astype(int)

    submission = template[["filename"]].merge(
        adjusted[["filename", "pred_label"]],
        on="filename",
        how="left",
        validate="one_to_one",
    )
    if submission["pred_label"].isna().any():
        missing_file = submission.loc[submission["pred_label"].isna(), "filename"].iloc[0]
        raise ValueError(f"Missing prediction for template filename: {missing_file}")
    submission["pred_label"] = submission["pred_label"].astype(int)
    submission.to_csv(args.output, index=False)

    if args.detailed_output:
        adjusted.to_csv(args.detailed_output, index=False)

    print("Applied thresholds")
    if args.default_threshold is not None:
        print(f"  default={args.default_threshold:.4f}")
    for route, threshold in thresholds.items():
        print(f"  {route}={threshold:.4f}")
    for component, threshold in component_thresholds.items():
        print(f"  component:{component}={threshold:.4f}")
    print("\nPrediction counts")
    print(adjusted.groupby(["predicted_component", "pred_label"]).size().unstack(fill_value=0))
    print(f"\nSaved submission: {Path(args.output)}")
    if args.detailed_output:
        print(f"Saved detailed predictions: {Path(args.detailed_output)}")


if __name__ == "__main__":
    main()
