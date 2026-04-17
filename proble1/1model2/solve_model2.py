from __future__ import annotations

import json
import math
import random
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


CURRENT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = CURRENT_DIR / "output"
PARENT_DIR = CURRENT_DIR.parent
MODEL1_DIR = PARENT_DIR / "1model1"
if not (MODEL1_DIR / "solve_model1.py").exists():
    MODEL1_DIR = PARENT_DIR

if str(MODEL1_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL1_DIR))

import solve_model1 as model1


@dataclass(frozen=True)
class Pattern:
    pattern_id: str
    vehicle_key: str
    vehicle_label: str
    source: str
    counts: tuple[int, int, int, int, int]
    used_volume_cm3: int
    used_weight_kg: int
    space_utilization: float
    weight_utilization: float
    lambda_score: float
    layers: list[model1.LayerCandidate]
    item_rows: pd.DataFrame
    layer_rows: pd.DataFrame


ITEM_ORDER = model1.ITEM_ORDER
ITEM_INDEX = model1.ITEM_INDEX


def counts_to_dict(counts: tuple[int, ...]) -> dict[str, int]:
    return {item_key: counts[ITEM_INDEX[item_key]] for item_key in ITEM_ORDER}


def clone_item_types(
    base_item_types: list[model1.ItemType], counts: tuple[int, int, int, int, int]
) -> list[model1.ItemType]:
    cloned: list[model1.ItemType] = []
    for idx, item in enumerate(base_item_types):
        cloned.append(
            model1.ItemType(
                key=item.key,
                category=item.category,
                dims=item.dims,
                weight=item.weight,
                quantity=counts[idx],
                stackable=item.stackable,
            )
        )
    return cloned


def subtract_counts(lhs: tuple[int, ...], rhs: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(max(0, a - b) for a, b in zip(lhs, rhs))


def fit_counts(lhs: tuple[int, ...], rhs: tuple[int, ...]) -> bool:
    return all(r <= l for l, r in zip(lhs, rhs))


def total_item_volume(counts: tuple[int, ...], item_types: list[model1.ItemType]) -> int:
    return sum(
        counts[idx] * item.dims[0] * item.dims[1] * item.dims[2] for idx, item in enumerate(item_types)
    )


def total_item_weight(counts: tuple[int, ...], item_types: list[model1.ItemType]) -> int:
    return sum(counts[idx] * item.weight for idx, item in enumerate(item_types))


def build_pattern(
    vehicle: model1.VehicleType,
    base_item_types: list[model1.ItemType],
    counts: tuple[int, int, int, int, int],
    source: str,
    pattern_serial: int,
    extra_random_layers: int,
    alns_iterations: int,
) -> Pattern | None:
    if sum(counts) == 0:
        return None
    item_types = clone_item_types(base_item_types, counts)
    available_counter = Counter(counts_to_dict(counts))
    candidate_layers = model1.generate_candidate_layers(
        vehicle,
        item_types,
        available_counter,
        extra_random_layers=extra_random_layers,
    )
    if not candidate_layers:
        return None
    base_solution = model1.solve_exact_layer_stack(vehicle, item_types, candidate_layers, counts)
    improved = model1.alns_improve(
        vehicle,
        item_types,
        base_solution,
        counts,
        iterations=alns_iterations,
    )
    used_counts = tuple(counts[idx] - improved.remaining_counts[idx] for idx in range(len(counts)))
    if sum(used_counts) == 0:
        return None
    item_rows = model1.expand_solution_rows(vehicle, improved.layers, item_types)
    summary = model1.evaluate_solution(vehicle, item_types, improved)
    summary["validation"] = model1.validate_items(item_rows, vehicle)
    layer_rows = pd.DataFrame(model1.summarize_layers(improved.layers))
    return Pattern(
        pattern_id=f"{vehicle.key}_pattern_{pattern_serial:03d}",
        vehicle_key=vehicle.key,
        vehicle_label=vehicle.label,
        source=source,
        counts=used_counts,
        used_volume_cm3=summary["used_volume_cm3"],
        used_weight_kg=summary["used_weight_kg"],
        space_utilization=summary["space_utilization"],
        weight_utilization=summary["weight_utilization"],
        lambda_score=summary["lambda"],
        layers=improved.layers,
        item_rows=item_rows,
        layer_rows=layer_rows,
    )


def build_manual_singleton_pattern(
    vehicle: model1.VehicleType,
    base_item_types: list[model1.ItemType],
    item_key: str,
    pattern_serial: int,
) -> Pattern | None:
    item = next((entry for entry in base_item_types if entry.key == item_key), None)
    if item is None:
        return None
    orientation_code, (dx, dy, dz) = model1.orientation_map(item)[0]
    if dx > vehicle.length or dy > vehicle.width or dz > vehicle.effective_height:
        return None

    counts = tuple(1 if key == item_key else 0 for key in ITEM_ORDER)
    placement = model1.BlockPlacement(
        block_key=f"singleton_block_{item_key}",
        item_key=item_key,
        category=item.category,
        orientation_code=orientation_code,
        x=0,
        y=0,
        dx=dx,
        dy=dy,
        dz=dz,
        nx=1,
        ny=1,
        count=1,
        weight=item.weight,
        supportable=item.stackable,
        offsets=((0, 0, 0),),
    )
    layer = model1.LayerCandidate(
        key=f"singleton_layer_{item_key}",
        height=dz,
        volume=dx * dy * dz,
        weight=item.weight,
        placements=(placement,),
        counts=counts,
        score_hint=0.0,
        coverage_ratio=(dx * dy) / vehicle.floor_area,
        has_fragile=item.category == model1.FRAGILE,
    )
    item_types = clone_item_types(base_item_types, counts)
    solution = model1.SearchResult(
        score=min((dx * dy * dz) / vehicle.volume, item.weight / vehicle.max_weight),
        used_volume=dx * dy * dz,
        used_weight=item.weight,
        remaining_counts=(0, 0, 0, 0, 0),
        layers=[layer],
    )
    item_rows = model1.expand_solution_rows(vehicle, [layer], item_types)
    summary = model1.evaluate_solution(vehicle, item_types, solution)
    summary["validation"] = model1.validate_items(item_rows, vehicle)
    layer_rows = pd.DataFrame(model1.summarize_layers([layer]))
    return Pattern(
        pattern_id=f"{vehicle.key}_pattern_{pattern_serial:03d}",
        vehicle_key=vehicle.key,
        vehicle_label=vehicle.label,
        source=f"singleton_{item_key}",
        counts=counts,
        used_volume_cm3=summary["used_volume_cm3"],
        used_weight_kg=summary["used_weight_kg"],
        space_utilization=summary["space_utilization"],
        weight_utilization=summary["weight_utilization"],
        lambda_score=summary["lambda"],
        layers=[layer],
        item_rows=item_rows,
        layer_rows=layer_rows,
    )


def apply_caps(
    base_counts: tuple[int, ...],
    multipliers: tuple[float, float, float, float, float],
) -> tuple[int, int, int, int, int]:
    capped = []
    for count, multiplier in zip(base_counts, multipliers):
        if count == 0 or multiplier <= 0:
            capped.append(0)
            continue
        value = max(1, int(math.floor(count * multiplier)))
        capped.append(min(count, value))
    return tuple(capped)


def add_pattern(patterns: dict[tuple[int, ...], Pattern], pattern: Pattern | None) -> None:
    if pattern is None:
        return
    kept = patterns.get(pattern.counts)
    if kept is None:
        patterns[pattern.counts] = pattern
        return
    current_score = (pattern.lambda_score, pattern.used_weight_kg, pattern.used_volume_cm3)
    kept_score = (kept.lambda_score, kept.used_weight_kg, kept.used_volume_cm3)
    if current_score > kept_score:
        patterns[pattern.counts] = pattern


def generate_pattern_library(
    vehicle: model1.VehicleType,
    base_item_types: list[model1.ItemType],
) -> tuple[list[Pattern], list[str]]:
    base_counts = tuple(item.quantity for item in base_item_types)
    patterns: dict[tuple[int, ...], Pattern] = {}
    pattern_serial = 1

    scenario_specs = [
        ("full", (1.0, 1.0, 1.0, 1.0, 1.0), 2, 40),
        ("heavy_bias", (1.0, 1.0, 0.4, 0.4, 1.0), 3, 35),
        ("bulky_bias", (0.6, 0.6, 1.0, 1.0, 0.6), 3, 35),
        ("standard_bias", (1.0, 1.0, 0.5, 0.5, 0.5), 2, 32),
        ("directed_bias", (0.6, 0.6, 0.5, 1.0, 1.0), 3, 36),
        ("fragile_bias", (0.7, 0.7, 1.0, 0.6, 0.6), 4, 36),
    ]

    for source, multipliers, extra_random_layers, alns_iterations in scenario_specs:
        counts = apply_caps(base_counts, multipliers)
        pattern = build_pattern(
            vehicle,
            base_item_types,
            counts,
            source,
            pattern_serial,
            extra_random_layers,
            alns_iterations,
        )
        pattern_serial += 1
        add_pattern(patterns, pattern)

    rng = random.Random(20260417 + vehicle.length)
    for idx in range(8):
        multipliers = tuple(round(rng.uniform(0.35, 1.0), 2) for _ in ITEM_ORDER)
        counts = apply_caps(base_counts, multipliers)
        pattern = build_pattern(
            vehicle,
            base_item_types,
            counts,
            f"randomized_{idx + 1}",
            pattern_serial,
            4,
            28,
        )
        pattern_serial += 1
        add_pattern(patterns, pattern)

    for idx, item_key in enumerate(ITEM_ORDER):
        counts = [0] * len(ITEM_ORDER)
        counts[idx] = base_counts[idx]
        pattern = build_pattern(
            vehicle,
            base_item_types,
            tuple(counts),
            f"single_type_{item_key}",
            pattern_serial,
            1,
            16,
        )
        pattern_serial += 1
        add_pattern(patterns, pattern)

    for idx, item_key in enumerate(ITEM_ORDER):
        counts = [0] * len(ITEM_ORDER)
        counts[idx] = 1
        pattern = build_pattern(
            vehicle,
            base_item_types,
            tuple(counts),
            f"singleton_{item_key}",
            pattern_serial,
            0,
            0,
        )
        if pattern is None:
            pattern = build_manual_singleton_pattern(vehicle, base_item_types, item_key, pattern_serial)
        pattern_serial += 1
        add_pattern(patterns, pattern)

    greedy_pattern_counts: list[tuple[int, ...]] = []
    remaining = base_counts
    greedy_round = 0
    while sum(remaining) > 0 and greedy_round < 12:
        pattern = build_pattern(
            vehicle,
            base_item_types,
            remaining,
            f"greedy_residual_{greedy_round + 1}",
            pattern_serial,
            3 + greedy_round % 3,
            30,
        )
        pattern_serial += 1
        if pattern is None or sum(pattern.counts) == 0:
            first_nonzero = next(idx for idx, value in enumerate(remaining) if value > 0)
            fallback_counts = [0] * len(ITEM_ORDER)
            fallback_counts[first_nonzero] = 1
            pattern = build_pattern(
                vehicle,
                base_item_types,
                tuple(fallback_counts),
                f"fallback_singleton_{ITEM_ORDER[first_nonzero]}",
                pattern_serial,
                0,
                0,
            )
            pattern_serial += 1
        if pattern is None:
            break
        add_pattern(patterns, pattern)
        greedy_pattern_counts.append(pattern.counts)
        remaining = subtract_counts(remaining, pattern.counts)
        greedy_round += 1

    pattern_list = list(patterns.values())
    pattern_list.sort(
        key=lambda p: (
            p.lambda_score,
            p.used_weight_kg,
            p.used_volume_cm3,
            sum(p.counts),
        ),
        reverse=True,
    )
    count_to_pattern_id = {pattern.counts: pattern.pattern_id for pattern in pattern_list}
    greedy_pattern_ids = [
        count_to_pattern_id[counts] for counts in greedy_pattern_counts if counts in count_to_pattern_id
    ]
    return pattern_list, greedy_pattern_ids


def greedy_cover(
    patterns: list[Pattern],
    demand_counts: tuple[int, ...],
) -> list[str]:
    remaining = demand_counts
    chosen: list[str] = []
    while sum(remaining) > 0:
        feasible = [p for p in patterns if fit_counts(remaining, p.counts) and sum(p.counts) > 0]
        if not feasible:
            break
        feasible.sort(
            key=lambda p: (
                sum(min(remaining[idx], p.counts[idx]) for idx in range(len(ITEM_ORDER))),
                p.lambda_score,
                p.used_volume_cm3,
            ),
            reverse=True,
        )
        selected = feasible[0]
        chosen.append(selected.pattern_id)
        remaining = subtract_counts(remaining, selected.counts)
    if sum(remaining) > 0:
        raise RuntimeError("Failed to build a feasible greedy cover from the generated pattern library.")
    return chosen


def lower_bound(
    remaining: tuple[int, ...],
    patterns: list[Pattern],
    item_types: list[model1.ItemType],
) -> int:
    if sum(remaining) == 0:
        return 0
    item_bounds = []
    for idx, rem in enumerate(remaining):
        if rem == 0:
            continue
        max_item_cover = max((pattern.counts[idx] for pattern in patterns), default=0)
        if max_item_cover == 0:
            return 10**9
        item_bounds.append(math.ceil(rem / max_item_cover))
    volume_rem = total_item_volume(remaining, item_types)
    weight_rem = total_item_weight(remaining, item_types)
    max_pattern_volume = max((pattern.used_volume_cm3 for pattern in patterns), default=1)
    max_pattern_weight = max((pattern.used_weight_kg for pattern in patterns), default=1)
    return max(
        max(item_bounds, default=0),
        math.ceil(volume_rem / max_pattern_volume),
        math.ceil(weight_rem / max_pattern_weight),
    )


def solve_master_problem(
    patterns: list[Pattern],
    demand_counts: tuple[int, ...],
    item_types: list[model1.ItemType],
    greedy_seed_solution: list[str],
) -> list[Pattern]:
    pattern_by_id = {pattern.pattern_id: pattern for pattern in patterns}
    incumbent_ids = greedy_seed_solution or greedy_cover(patterns, demand_counts)
    incumbent_patterns = [pattern_by_id[pattern_id] for pattern_id in incumbent_ids if pattern_id in pattern_by_id]
    delivered = [0] * len(ITEM_ORDER)
    for pattern in incumbent_patterns:
        for idx, value in enumerate(pattern.counts):
            delivered[idx] += value
    incumbent_exact = tuple(delivered) == demand_counts
    if not incumbent_exact:
        incumbent_ids = greedy_cover(patterns, demand_counts)
        incumbent_patterns = [pattern_by_id[pattern_id] for pattern_id in incumbent_ids]
        delivered = [0] * len(ITEM_ORDER)
        for pattern in incumbent_patterns:
            for idx, value in enumerate(pattern.counts):
                delivered[idx] += value
        incumbent_exact = tuple(delivered) == demand_counts

    best_ids = list(incumbent_ids) if incumbent_exact else []
    best_count = len(best_ids) if incumbent_exact else 10**9
    memo: dict[tuple[int, ...], int] = {}

    def dfs(remaining: tuple[int, ...], chosen_ids: list[str]) -> None:
        nonlocal best_ids, best_count
        if sum(remaining) == 0:
            if len(chosen_ids) < best_count:
                best_ids = list(chosen_ids)
                best_count = len(chosen_ids)
            return

        bound = lower_bound(remaining, patterns, item_types)
        if len(chosen_ids) + bound >= best_count:
            return

        cached = memo.get(remaining)
        if cached is not None and cached <= len(chosen_ids):
            return
        memo[remaining] = len(chosen_ids)

        candidate_item_indices = [idx for idx, value in enumerate(remaining) if value > 0]
        target_idx = min(
            candidate_item_indices,
            key=lambda idx: sum(
                1
                for pattern in patterns
                if pattern.counts[idx] > 0 and fit_counts(remaining, pattern.counts)
            ),
        )
        candidates = [
            pattern
            for pattern in patterns
            if pattern.counts[target_idx] > 0 and fit_counts(remaining, pattern.counts)
        ]
        candidates.sort(
            key=lambda pattern: (
                pattern.counts[target_idx],
                sum(pattern.counts),
                pattern.lambda_score,
                pattern.used_volume_cm3,
            ),
            reverse=True,
        )

        for pattern in candidates:
            new_remaining = subtract_counts(remaining, pattern.counts)
            chosen_ids.append(pattern.pattern_id)
            dfs(new_remaining, chosen_ids)
            chosen_ids.pop()

    dfs(demand_counts, [])
    return [pattern_by_id[pattern_id] for pattern_id in best_ids]


def build_pattern_library_df(patterns: list[Pattern]) -> pd.DataFrame:
    rows = []
    for pattern in patterns:
        row = {
            "pattern_id": pattern.pattern_id,
            "vehicle_key": pattern.vehicle_key,
            "vehicle_label": pattern.vehicle_label,
            "source": pattern.source,
            "used_volume_cm3": pattern.used_volume_cm3,
            "used_weight_kg": pattern.used_weight_kg,
            "space_utilization": round(pattern.space_utilization, 6),
            "weight_utilization": round(pattern.weight_utilization, 6),
            "lambda": round(pattern.lambda_score, 6),
            "layer_count": len(pattern.layers),
        }
        for item_key in ITEM_ORDER:
            row[f"count_{item_key}"] = pattern.counts[ITEM_INDEX[item_key]]
        rows.append(row)
    return pd.DataFrame(rows)


def renumber_pattern_rows(
    pattern: Pattern,
    vehicle_id: int,
) -> pd.DataFrame:
    df = pattern.item_rows.copy()
    vehicle_prefix = f"V{vehicle_id:02d}"
    df["vehicle_id"] = vehicle_id
    df["pattern_id"] = pattern.pattern_id
    df["pattern_source"] = pattern.source
    df["item_uid"] = df["item_uid"].map(lambda uid: f"{vehicle_prefix}_{uid}")
    if "block_id" in df.columns:
        df["block_id"] = df["block_id"].map(lambda block_id: f"{vehicle_prefix}_{block_id}")
    if "support_id" in df.columns:
        df["support_id"] = df["support_id"].map(
            lambda support_id: support_id
            if support_id == "floor"
            else "|".join(f"{vehicle_prefix}_{token}" for token in str(support_id).split("|"))
        )
    return df


def build_vehicle_summary_df(selected_patterns: list[Pattern]) -> pd.DataFrame:
    rows = []
    for vehicle_id, pattern in enumerate(selected_patterns, 1):
        row = {
            "vehicle_id": vehicle_id,
            "pattern_id": pattern.pattern_id,
            "pattern_source": pattern.source,
            "used_volume_cm3": pattern.used_volume_cm3,
            "used_weight_kg": pattern.used_weight_kg,
            "space_utilization": round(pattern.space_utilization, 6),
            "weight_utilization": round(pattern.weight_utilization, 6),
            "lambda": round(pattern.lambda_score, 6),
        }
        for item_key in ITEM_ORDER:
            row[f"count_{item_key}"] = pattern.counts[ITEM_INDEX[item_key]]
        rows.append(row)
    return pd.DataFrame(rows)


def build_fleet_summary(
    vehicle: model1.VehicleType,
    item_types: list[model1.ItemType],
    selected_patterns: list[Pattern],
) -> dict[str, object]:
    demand_counts = tuple(item.quantity for item in item_types)
    delivered_counts = [0] * len(ITEM_ORDER)
    total_used_volume = 0
    total_used_weight = 0
    for pattern in selected_patterns:
        total_used_volume += pattern.used_volume_cm3
        total_used_weight += pattern.used_weight_kg
        for idx, value in enumerate(pattern.counts):
            delivered_counts[idx] += value
    exact_cover = tuple(delivered_counts) == demand_counts
    return {
        "vehicle_key": vehicle.key,
        "vehicle_label": vehicle.label,
        "vehicle_trip_cost": vehicle.trip_cost,
        "vehicle_count": len(selected_patterns),
        "total_transport_cost": len(selected_patterns) * vehicle.trip_cost,
        "demand_counts": counts_to_dict(demand_counts),
        "delivered_counts": counts_to_dict(tuple(delivered_counts)),
        "vehicle_volume_cm3": vehicle.volume,
        "vehicle_weight_capacity_kg": vehicle.max_weight,
        "avg_space_utilization": round(
            sum(pattern.space_utilization for pattern in selected_patterns) / max(1, len(selected_patterns)),
            6,
        ),
        "avg_weight_utilization": round(
            sum(pattern.weight_utilization for pattern in selected_patterns) / max(1, len(selected_patterns)),
            6,
        ),
        "total_used_volume_cm3": total_used_volume,
        "total_used_weight_kg": total_used_weight,
        "exact_cover": exact_cover,
        "selected_pattern_ids": [pattern.pattern_id for pattern in selected_patterns],
    }


def solve_vehicle_type(
    vehicle: model1.VehicleType,
    item_types: list[model1.ItemType],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    demand_counts = tuple(item.quantity for item in item_types)
    patterns, greedy_pattern_ids = generate_pattern_library(vehicle, item_types)
    pattern_library_df = build_pattern_library_df(patterns)
    selected_patterns = solve_master_problem(patterns, demand_counts, item_types, greedy_pattern_ids)
    vehicle_summary_df = build_vehicle_summary_df(selected_patterns)
    fleet_items_df = pd.concat(
        [renumber_pattern_rows(pattern, vehicle_id) for vehicle_id, pattern in enumerate(selected_patterns, 1)],
        ignore_index=True,
    )
    fleet_summary = build_fleet_summary(vehicle, item_types, selected_patterns)
    return pattern_library_df, vehicle_summary_df, fleet_items_df, fleet_summary


def write_outputs(
    pattern_outputs: dict[str, pd.DataFrame],
    vehicle_outputs: dict[str, pd.DataFrame],
    fleet_outputs: dict[str, pd.DataFrame],
    summary_outputs: dict[str, dict[str, object]],
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    workbook_path = OUTPUT_DIR / "model2_results.xlsx"
    with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
        for vehicle_key, df in pattern_outputs.items():
            df.to_excel(writer, sheet_name=f"{vehicle_key}_patterns", index=False)
        for vehicle_key, df in vehicle_outputs.items():
            df.to_excel(writer, sheet_name=f"{vehicle_key}_fleet", index=False)
        for vehicle_key, df in fleet_outputs.items():
            df.to_excel(writer, sheet_name=f"{vehicle_key}_items", index=False)
        summary_rows = pd.DataFrame.from_dict(summary_outputs, orient="index")
        summary_rows.index.name = "vehicle_key_index"
        summary_rows = summary_rows.reset_index()
        summary_rows.to_excel(writer, sheet_name="summary", index=False)

    for vehicle_key, df in pattern_outputs.items():
        df.to_csv(OUTPUT_DIR / f"{vehicle_key}_pattern_library.csv", index=False, encoding="utf-8-sig")
    for vehicle_key, df in vehicle_outputs.items():
        df.to_csv(OUTPUT_DIR / f"{vehicle_key}_fleet_vehicles.csv", index=False, encoding="utf-8-sig")
    for vehicle_key, df in fleet_outputs.items():
        df.to_csv(OUTPUT_DIR / f"{vehicle_key}_fleet_items.csv", index=False, encoding="utf-8-sig")
    for vehicle_key, summary in summary_outputs.items():
        (OUTPUT_DIR / f"{vehicle_key}_fleet_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def main() -> None:
    vehicles, item_types = model1.build_inputs()
    pattern_outputs: dict[str, pd.DataFrame] = {}
    vehicle_outputs: dict[str, pd.DataFrame] = {}
    fleet_outputs: dict[str, pd.DataFrame] = {}
    summary_outputs: dict[str, dict[str, object]] = {}

    for vehicle in vehicles.values():
        pattern_library_df, vehicle_summary_df, fleet_items_df, fleet_summary = solve_vehicle_type(
            vehicle,
            item_types,
        )
        pattern_outputs[vehicle.key] = pattern_library_df
        vehicle_outputs[vehicle.key] = vehicle_summary_df
        fleet_outputs[vehicle.key] = fleet_items_df
        summary_outputs[vehicle.key] = fleet_summary

    write_outputs(pattern_outputs, vehicle_outputs, fleet_outputs, summary_outputs)
    console_summary = {
        vehicle_key: {
            "vehicle_count": summary["vehicle_count"],
            "total_transport_cost": summary["total_transport_cost"],
            "avg_space_utilization": summary["avg_space_utilization"],
            "avg_weight_utilization": summary["avg_weight_utilization"],
            "exact_cover": summary["exact_cover"],
        }
        for vehicle_key, summary in summary_outputs.items()
    }
    print(json.dumps(console_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
