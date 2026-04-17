from __future__ import annotations

import itertools
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path(__file__).resolve().parent / "output"


STANDARD = "standard"
FRAGILE = "fragile"
DIRECTED = "directed"


@dataclass(frozen=True)
class VehicleType:
    key: str
    label: str
    length: int
    width: int
    height: int
    max_weight: int
    trip_cost: int

    @property
    def effective_height(self) -> int:
        return self.height - 3

    @property
    def floor_area(self) -> int:
        return self.length * self.width

    @property
    def volume(self) -> int:
        return self.length * self.width * self.effective_height


@dataclass(frozen=True)
class ItemType:
    key: str
    category: str
    dims: tuple[int, int, int]
    weight: int
    quantity: int
    stackable: bool


@dataclass(frozen=True)
class BlockTemplate:
    key: str
    item_key: str
    category: str
    orientation_code: str
    dx: int
    dy: int
    dz: int
    nx: int
    ny: int
    count: int
    weight: int
    volume: int
    supportable: bool
    offsets: tuple[tuple[int, int, int], ...]


@dataclass(frozen=True)
class Rect:
    x: int
    y: int
    w: int
    h: int

    @property
    def area(self) -> int:
        return self.w * self.h


@dataclass(frozen=True)
class BlockPlacement:
    block_key: str
    item_key: str
    category: str
    orientation_code: str
    x: int
    y: int
    dx: int
    dy: int
    dz: int
    nx: int
    ny: int
    count: int
    weight: int
    supportable: bool
    offsets: tuple[tuple[int, int, int], ...]


@dataclass(frozen=True)
class LayerCandidate:
    key: str
    height: int
    volume: int
    weight: int
    placements: tuple[BlockPlacement, ...]
    counts: tuple[int, int, int, int, int]
    score_hint: float
    coverage_ratio: float
    has_fragile: bool


@dataclass
class SearchResult:
    score: float
    used_volume: int
    used_weight: int
    remaining_counts: tuple[int, int, int, int, int]
    layers: list[LayerCandidate]


ITEM_ORDER = ("G1", "G2", "G3", "G4", "G5")
ITEM_INDEX = {item_key: idx for idx, item_key in enumerate(ITEM_ORDER)}


def build_inputs() -> tuple[dict[str, VehicleType], list[ItemType]]:
    vehicles = {
        "vehicle_1": VehicleType(
            key="vehicle_1",
            label="车型1",
            length=420,
            width=210,
            height=220,
            max_weight=6000,
            trip_cost=450,
        ),
        "vehicle_2": VehicleType(
            key="vehicle_2",
            label="车型2",
            length=680,
            width=245,
            height=250,
            max_weight=10000,
            trip_cost=700,
        ),
    }
    items = [
        ItemType("G1", STANDARD, (60, 40, 30), 12, 80, True),
        ItemType("G2", STANDARD, (50, 35, 25), 8, 100, True),
        ItemType("G3", FRAGILE, (70, 50, 40), 15, 30, False),
        ItemType("G4", DIRECTED, (80, 60, 50), 25, 40, True),
        ItemType("G5", DIRECTED, (40, 40, 60), 18, 50, True),
    ]
    return vehicles, items


def orientation_map(item: ItemType) -> list[tuple[str, tuple[int, int, int]]]:
    l, w, h = item.dims
    ordered = [
        ("O1", (l, w, h)),
        ("O2", (l, h, w)),
        ("O3", (w, l, h)),
        ("O4", (w, h, l)),
        ("O5", (h, l, w)),
        ("O6", (h, w, l)),
    ]
    seen: set[tuple[int, int, int]] = set()
    result: list[tuple[str, tuple[int, int, int]]] = []
    if item.category == DIRECTED:
        return [("O1", (l, w, h))]
    for code, dims in ordered:
        if dims not in seen:
            seen.add(dims)
            result.append((code, dims))
    return result


def generate_block_templates(vehicle: VehicleType, item_types: list[ItemType]) -> list[BlockTemplate]:
    patterns = ((1, 1), (2, 1), (1, 2), (2, 2), (3, 1))
    templates: list[BlockTemplate] = []
    for item in item_types:
        for orientation_code, (dx, dy, dz) in orientation_map(item):
            if dz > vehicle.effective_height:
                continue
            for nx, ny in patterns:
                count = nx * ny
                if count > item.quantity:
                    continue
                length = dx * nx
                width = dy * ny
                if length > vehicle.length or width > vehicle.width:
                    continue
                offsets = []
                for ix in range(nx):
                    for iy in range(ny):
                        offsets.append((ix * dx, iy * dy, 0))
                templates.append(
                    BlockTemplate(
                        key=f"{item.key}_{orientation_code}_{nx}x{ny}",
                        item_key=item.key,
                        category=item.category,
                        orientation_code=orientation_code,
                        dx=length,
                        dy=width,
                        dz=dz,
                        nx=nx,
                        ny=ny,
                        count=count,
                        weight=item.weight * count,
                        volume=length * width * dz,
                        supportable=item.stackable and item.category != FRAGILE,
                        offsets=tuple(offsets),
                    )
                )
    return templates


def rank_templates(strategy: str, templates: list[BlockTemplate], seed: int) -> list[BlockTemplate]:
    rng = random.Random(seed)
    ordered = list(templates)
    if strategy == "heavy":
        ordered.sort(key=lambda t: (t.weight, t.volume, t.dx * t.dy), reverse=True)
    elif strategy == "dense":
        ordered.sort(key=lambda t: (t.weight / max(1, t.dx * t.dy), t.weight, t.count), reverse=True)
    elif strategy == "large_area":
        ordered.sort(key=lambda t: (t.dx * t.dy, t.weight, t.count), reverse=True)
    elif strategy == "high_count":
        ordered.sort(key=lambda t: (t.count, t.weight, t.volume), reverse=True)
    elif strategy == "support_first":
        ordered.sort(
            key=lambda t: (
                1 if t.supportable else 0,
                t.dx * t.dy,
                t.weight,
            ),
            reverse=True,
        )
    elif strategy == "randomized":
        rng.shuffle(ordered)
        ordered.sort(key=lambda t: (t.supportable, rng.random()), reverse=True)
    else:
        ordered.sort(key=lambda t: (t.volume, t.weight), reverse=True)
    return ordered


def prune_free_rects(rects: list[Rect]) -> list[Rect]:
    pruned: list[Rect] = []
    for idx, rect in enumerate(rects):
        if rect.w <= 0 or rect.h <= 0:
            continue
        contained = False
        for jdx, other in enumerate(rects):
            if idx == jdx:
                continue
            if (
                rect.x >= other.x
                and rect.y >= other.y
                and rect.x + rect.w <= other.x + other.w
                and rect.y + rect.h <= other.y + other.h
            ):
                contained = True
                break
        if not contained:
            pruned.append(rect)
    unique: dict[tuple[int, int, int, int], Rect] = {}
    for rect in pruned:
        unique[(rect.x, rect.y, rect.w, rect.h)] = rect
    return list(unique.values())


def split_rect(rect: Rect, used_w: int, used_h: int) -> list[Rect]:
    result: list[Rect] = []
    if rect.w - used_w > 0:
        result.append(Rect(rect.x + used_w, rect.y, rect.w - used_w, rect.h))
    if rect.h - used_h > 0:
        result.append(Rect(rect.x, rect.y + used_h, used_w, rect.h - used_h))
    return result


def counts_fit(used_counts: Counter[str], available_counts: Counter[str], block: BlockTemplate) -> bool:
    return used_counts[block.item_key] + block.count <= available_counts[block.item_key]


def build_layer(
    vehicle: VehicleType,
    height: int,
    templates: list[BlockTemplate],
    available_counts: Counter[str],
    strategy: str,
    seed: int,
) -> LayerCandidate | None:
    layer_templates = [t for t in templates if t.dz == height]
    if not layer_templates:
        return None
    ordered = rank_templates(strategy, layer_templates, seed)
    free_rects = [Rect(0, 0, vehicle.length, vehicle.width)]
    used_counts: Counter[str] = Counter()
    placements: list[BlockPlacement] = []
    while True:
        free_rects = sorted(prune_free_rects(free_rects), key=lambda r: (r.y, r.x, -r.area))
        placed = False
        for rect_idx, rect in enumerate(free_rects):
            for template in ordered:
                if not counts_fit(used_counts, available_counts, template):
                    continue
                if template.dx <= rect.w and template.dy <= rect.h:
                    placements.append(
                        BlockPlacement(
                            block_key=template.key,
                            item_key=template.item_key,
                            category=template.category,
                            orientation_code=template.orientation_code,
                            x=rect.x,
                            y=rect.y,
                            dx=template.dx,
                            dy=template.dy,
                            dz=template.dz,
                            nx=template.nx,
                            ny=template.ny,
                            count=template.count,
                            weight=template.weight,
                            supportable=template.supportable,
                            offsets=template.offsets,
                        )
                    )
                    used_counts[template.item_key] += template.count
                    selected = free_rects.pop(rect_idx)
                    free_rects.extend(split_rect(selected, template.dx, template.dy))
                    placed = True
                    break
            if placed:
                break
        if not placed:
            break
    if not placements:
        return None
    counts_tuple = [0] * len(ITEM_ORDER)
    total_volume = 0
    total_weight = 0
    fragile = False
    covered_area = 0
    for placement in placements:
        counts_tuple[ITEM_INDEX[placement.item_key]] += placement.count
        total_volume += placement.dx * placement.dy * placement.dz
        total_weight += placement.weight
        covered_area += placement.dx * placement.dy
        fragile = fragile or placement.category == FRAGILE
    coverage_ratio = covered_area / vehicle.floor_area
    if coverage_ratio < 0.10:
        return None
    score_hint = min(total_volume / vehicle.volume, total_weight / vehicle.max_weight)
    return LayerCandidate(
        key=f"{vehicle.key}_h{height}_{strategy}_{seed}_{len(placements)}",
        height=height,
        volume=total_volume,
        weight=total_weight,
        placements=tuple(placements),
        counts=tuple(counts_tuple),
        score_hint=score_hint,
        coverage_ratio=coverage_ratio,
        has_fragile=fragile,
    )


def generate_candidate_layers(
    vehicle: VehicleType,
    item_types: list[ItemType],
    available_counts: Counter[str],
    extra_random_layers: int = 0,
) -> list[LayerCandidate]:
    templates = generate_block_templates(vehicle, item_types)
    heights = sorted({template.dz for template in templates})
    strategies = [
        "heavy",
        "dense",
        "large_area",
        "high_count",
        "support_first",
    ]
    layers: list[LayerCandidate] = []
    for height in heights:
        for idx, strategy in enumerate(strategies):
            layer = build_layer(vehicle, height, templates, available_counts, strategy, idx + height)
            if layer is not None:
                layers.append(layer)
        for rand_idx in range(extra_random_layers):
            layer = build_layer(
                vehicle,
                height,
                templates,
                available_counts,
                "randomized",
                height * 100 + rand_idx,
            )
            if layer is not None:
                layers.append(layer)
    unique: dict[tuple[tuple[int, int, int, int], ...], LayerCandidate] = {}
    for layer in layers:
        signature = tuple(
            sorted(
                (
                    placement.x,
                    placement.y,
                    placement.dx,
                    placement.dy,
                )
                for placement in layer.placements
            )
        )
        kept = unique.get(signature)
        if kept is None or (layer.score_hint, layer.weight, layer.volume) > (
            kept.score_hint,
            kept.weight,
            kept.volume,
        ):
            unique[signature] = layer
    deduped = list(unique.values())
    deduped.sort(
        key=lambda layer: (
            layer.score_hint,
            layer.coverage_ratio,
            layer.weight,
            layer.volume,
        ),
        reverse=True,
    )
    return deduped[:28]


def subtract_counts(lhs: tuple[int, ...], rhs: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(a - b for a, b in zip(lhs, rhs))


def can_apply_layer(remaining_counts: tuple[int, ...], layer: LayerCandidate) -> bool:
    return all(rem >= need for rem, need in zip(remaining_counts, layer.counts))


def remaining_volume(remaining_counts: tuple[int, ...], item_types: list[ItemType]) -> int:
    total = 0
    for idx, item in enumerate(item_types):
        total += remaining_counts[idx] * item.dims[0] * item.dims[1] * item.dims[2]
    return total


def remaining_weight(remaining_counts: tuple[int, ...], item_types: list[ItemType]) -> int:
    total = 0
    for idx, item in enumerate(item_types):
        total += remaining_counts[idx] * item.weight
    return total


def subtract_overlap(source: Rect, cutter: Rect) -> list[Rect]:
    x1 = max(source.x, cutter.x)
    y1 = max(source.y, cutter.y)
    x2 = min(source.x + source.w, cutter.x + cutter.w)
    y2 = min(source.y + source.h, cutter.y + cutter.h)
    if x1 >= x2 or y1 >= y2:
        return [source]
    pieces: list[Rect] = []
    if source.x < x1:
        pieces.append(Rect(source.x, source.y, x1 - source.x, source.h))
    if x2 < source.x + source.w:
        pieces.append(Rect(x2, source.y, source.x + source.w - x2, source.h))
    if source.y < y1:
        pieces.append(Rect(x1, source.y, x2 - x1, y1 - source.y))
    if y2 < source.y + source.h:
        pieces.append(Rect(x1, y2, x2 - x1, source.y + source.h - y2))
    return [piece for piece in pieces if piece.w > 0 and piece.h > 0]


def rect_fully_covered(target: Rect, supports: list[Rect]) -> bool:
    remaining = [target]
    for support in supports:
        next_remaining: list[Rect] = []
        for piece in remaining:
            next_remaining.extend(subtract_overlap(piece, support))
        remaining = next_remaining
        if not remaining:
            return True
    return not remaining


def layer_supports(lower: LayerCandidate | None, upper: LayerCandidate) -> bool:
    if lower is None:
        return True
    if lower.has_fragile:
        return False
    support_rects = [
        Rect(placement.x, placement.y, placement.dx, placement.dy)
        for placement in lower.placements
        if placement.supportable
    ]
    if not support_rects:
        return False
    for placement in upper.placements:
        upper_rect = Rect(placement.x, placement.y, placement.dx, placement.dy)
        if not rect_fully_covered(upper_rect, support_rects):
            return False
    return True


def score_from_remaining(
    vehicle: VehicleType,
    item_types: list[ItemType],
    initial_counts: tuple[int, ...],
    remaining_counts_state: tuple[int, ...],
) -> tuple[float, int, int]:
    used_volume = remaining_volume(initial_counts, item_types) - remaining_volume(remaining_counts_state, item_types)
    used_weight = remaining_weight(initial_counts, item_types) - remaining_weight(remaining_counts_state, item_types)
    score = min(used_volume / vehicle.volume, used_weight / vehicle.max_weight)
    return score, used_volume, used_weight


def solve_exact_layer_stack(
    vehicle: VehicleType,
    item_types: list[ItemType],
    candidate_layers: list[LayerCandidate],
    initial_counts: tuple[int, ...],
    height_limit: int | None = None,
    starting_lower: LayerCandidate | None = None,
) -> SearchResult:
    memo: dict[tuple[int, int, tuple[int, ...]], SearchResult] = {}
    start_height = vehicle.effective_height if height_limit is None else height_limit
    root_lower_token = -2 if starting_lower is not None else -1

    def dfs(lower_idx: int, remaining_h: int, remaining_counts_state: tuple[int, ...]) -> SearchResult:
        key = (lower_idx, remaining_h, remaining_counts_state)
        cached = memo.get(key)
        if cached is not None:
            return cached

        base_score, base_volume, base_weight = score_from_remaining(
            vehicle, item_types, initial_counts, remaining_counts_state
        )
        best = SearchResult(
            score=base_score,
            used_volume=base_volume,
            used_weight=base_weight,
            remaining_counts=remaining_counts_state,
            layers=[],
        )

        remaining_total_volume = remaining_volume(remaining_counts_state, item_types)
        remaining_total_weight = remaining_weight(remaining_counts_state, item_types)
        optimistic_volume = base_volume + min(remaining_h * vehicle.floor_area, remaining_total_volume)
        optimistic_weight = min(vehicle.max_weight, base_weight + remaining_total_weight)
        optimistic_score = min(optimistic_volume / vehicle.volume, optimistic_weight / vehicle.max_weight)
        if optimistic_score <= best.score + 1e-9:
            memo[key] = best
            return best

        if lower_idx == -2:
            lower_layer = starting_lower
        elif lower_idx < 0:
            lower_layer = None
        else:
            lower_layer = candidate_layers[lower_idx]
        remaining_capacity = vehicle.max_weight - base_weight
        feasible_indices = []
        for idx, layer in enumerate(candidate_layers):
            if layer.height > remaining_h:
                continue
            if layer.weight > remaining_capacity:
                continue
            if not can_apply_layer(remaining_counts_state, layer):
                continue
            if not layer_supports(lower_layer, layer):
                continue
            feasible_indices.append(idx)
        feasible_indices.sort(
            key=lambda idx: (
                candidate_layers[idx].score_hint,
                candidate_layers[idx].weight,
                candidate_layers[idx].coverage_ratio,
            ),
            reverse=True,
        )
        for idx in feasible_indices:
            layer = candidate_layers[idx]
            new_counts_state = subtract_counts(remaining_counts_state, layer.counts)
            sub = dfs(idx, remaining_h - layer.height, new_counts_state)
            if (
                sub.score > best.score + 1e-9
                or (
                    math.isclose(sub.score, best.score, rel_tol=1e-9, abs_tol=1e-9)
                    and (sub.used_weight, sub.used_volume) > (best.used_weight, best.used_volume)
                )
            ):
                best = SearchResult(
                    score=sub.score,
                    used_volume=sub.used_volume,
                    used_weight=sub.used_weight,
                    remaining_counts=sub.remaining_counts,
                    layers=[layer, *sub.layers],
                )
        memo[key] = best
        return best

    return dfs(root_lower_token, start_height, initial_counts)


def summarize_layers(layers: list[LayerCandidate]) -> list[dict[str, object]]:
    rows = []
    z_cursor = 0
    for idx, layer in enumerate(layers, 1):
        rows.append(
            {
                "layer_id": idx,
                "layer_key": layer.key,
                "z_start": z_cursor,
                "height": layer.height,
                "weight": layer.weight,
                "volume_cm3": layer.volume,
                "coverage_ratio": round(layer.coverage_ratio, 4),
                "has_fragile": layer.has_fragile,
            }
        )
        z_cursor += layer.height
    return rows


def evaluate_solution(
    vehicle: VehicleType,
    item_types: list[ItemType],
    solution: SearchResult,
) -> dict[str, object]:
    item_map = {item.key: item for item in item_types}
    used_counts = {
        item_key: item_map[item_key].quantity - solution.remaining_counts[ITEM_INDEX[item_key]]
        for item_key in ITEM_ORDER
    }
    total_item_volume = sum(item.dims[0] * item.dims[1] * item.dims[2] * item.quantity for item in item_types)
    total_item_weight = sum(item.weight * item.quantity for item in item_types)
    return {
        "vehicle_key": vehicle.key,
        "vehicle_label": vehicle.label,
        "vehicle_volume_cm3": vehicle.volume,
        "vehicle_weight_capacity_kg": vehicle.max_weight,
        "used_volume_cm3": solution.used_volume,
        "used_weight_kg": solution.used_weight,
        "space_utilization": round(solution.used_volume / vehicle.volume, 6),
        "weight_utilization": round(solution.used_weight / vehicle.max_weight, 6),
        "lambda": round(solution.score, 6),
        "loaded_item_counts": used_counts,
        "all_item_volume_cm3": total_item_volume,
        "all_item_weight_kg": total_item_weight,
    }


def alns_improve(
    vehicle: VehicleType,
    item_types: list[ItemType],
    base_solution: SearchResult,
    initial_counts: tuple[int, ...],
    iterations: int = 45,
) -> SearchResult:
    current = base_solution
    best = base_solution
    rng = random.Random(20260417 + len(base_solution.layers) + vehicle.length)
    temperature = 0.08

    for iteration in range(iterations):
        if not current.layers:
            break
        remove_k = 1 if len(current.layers) < 3 else rng.choice((1, 1, 2))
        remove_indices = set(rng.sample(range(len(current.layers)), k=remove_k))
        kept_layers = [layer for idx, layer in enumerate(current.layers) if idx not in remove_indices]

        remaining_counts_state = list(initial_counts)
        for layer in kept_layers:
            remaining_counts_state = list(subtract_counts(tuple(remaining_counts_state), layer.counts))
        remaining_counts_tuple = tuple(remaining_counts_state)
        remaining_h = vehicle.effective_height - sum(layer.height for layer in kept_layers)
        if remaining_h <= 0:
            continue

        available_counter = Counter(
            {item_key: remaining_counts_tuple[ITEM_INDEX[item_key]] for item_key in ITEM_ORDER}
        )
        extra_layers = 3 + (iteration % 4)
        candidate_layers = generate_candidate_layers(
            vehicle,
            item_types,
            available_counter,
            extra_random_layers=extra_layers,
        )
        repaired = solve_exact_layer_stack(
            vehicle,
            item_types,
            candidate_layers,
            remaining_counts_tuple,
            height_limit=remaining_h,
            starting_lower=kept_layers[-1] if kept_layers else None,
        )
        candidate = SearchResult(
            score=repaired.score,
            used_volume=repaired.used_volume,
            used_weight=repaired.used_weight,
            remaining_counts=repaired.remaining_counts,
            layers=[*kept_layers, *repaired.layers],
        )
        candidate_metrics = score_from_remaining(
            vehicle,
            item_types,
            initial_counts,
            candidate.remaining_counts,
        )
        candidate.score = candidate_metrics[0]
        candidate.used_volume = candidate_metrics[1]
        candidate.used_weight = candidate_metrics[2]

        delta = candidate.score - current.score
        if delta >= 0 or rng.random() < math.exp(delta / max(temperature, 1e-6)):
            current = candidate
        if (
            current.score > best.score + 1e-9
            or (
                math.isclose(current.score, best.score, rel_tol=1e-9, abs_tol=1e-9)
                and (current.used_weight, current.used_volume) > (best.used_weight, best.used_volume)
            )
        ):
            best = current
        temperature *= 0.965
    return best


def expand_solution_rows(
    vehicle: VehicleType,
    layers: list[LayerCandidate],
    item_types: list[ItemType],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    serial_by_item: Counter[str] = Counter()
    block_serial = 0
    z_cursor = 0
    for layer_id, layer in enumerate(layers, 1):
        for placement in layer.placements:
            block_serial += 1
            support_id = "floor" if layer_id == 1 else f"L{layer_id - 1}_block"
            base_item = next(item for item in item_types if item.key == placement.item_key)
            single_dx = placement.dx // placement.nx
            single_dy = placement.dy // placement.ny
            for offset_x, offset_y, offset_z in placement.offsets:
                serial_by_item[placement.item_key] += 1
                item_uid = f"{placement.item_key}_{serial_by_item[placement.item_key]:03d}"
                rows.append(
                    {
                        "vehicle_id": 1,
                        "vehicle_key": vehicle.key,
                        "vehicle_label": vehicle.label,
                        "item_uid": item_uid,
                        "item_type": placement.item_key,
                        "x": placement.x + offset_x,
                        "y": placement.y + offset_y,
                        "z": z_cursor + offset_z,
                        "dx": single_dx,
                        "dy": single_dy,
                        "dz": placement.dz,
                        "orientation_code": placement.orientation_code,
                        "layer_id": layer_id,
                        "support_id": support_id,
                        "block_id": f"block_{block_serial}",
                        "category": placement.category,
                        "unit_weight": base_item.weight,
                    }
                )
        z_cursor += layer.height
    return pd.DataFrame(rows)


def validate_items(df: pd.DataFrame, vehicle: VehicleType) -> dict[str, object]:
    records = df.to_dict("records")
    boundary_violations = []
    for record in records:
        if (
            record["x"] < 0
            or record["y"] < 0
            or record["z"] < 0
            or record["x"] + record["dx"] > vehicle.length
            or record["y"] + record["dy"] > vehicle.width
            or record["z"] + record["dz"] > vehicle.effective_height
        ):
            boundary_violations.append(record["item_uid"])
    overlap_pairs = 0
    for idx in range(len(records)):
        a = records[idx]
        for jdx in range(idx + 1, len(records)):
            b = records[jdx]
            separated = (
                a["x"] + a["dx"] <= b["x"]
                or b["x"] + b["dx"] <= a["x"]
                or a["y"] + a["dy"] <= b["y"]
                or b["y"] + b["dy"] <= a["y"]
                or a["z"] + a["dz"] <= b["z"]
                or b["z"] + b["dz"] <= a["z"]
            )
            if not separated:
                overlap_pairs += 1
    return {
        "boundary_violations": boundary_violations,
        "overlap_pairs": overlap_pairs,
    }


def solve_vehicle(vehicle: VehicleType, item_types: list[ItemType]) -> tuple[pd.DataFrame, dict[str, object], pd.DataFrame]:
    initial_counter = Counter({item.key: item.quantity for item in item_types})
    initial_counts = tuple(initial_counter[item_key] for item_key in ITEM_ORDER)
    candidate_layers = generate_candidate_layers(vehicle, item_types, initial_counter, extra_random_layers=2)
    base_solution = solve_exact_layer_stack(vehicle, item_types, candidate_layers, initial_counts)
    improved = alns_improve(vehicle, item_types, base_solution, initial_counts, iterations=50)
    item_rows = expand_solution_rows(vehicle, improved.layers, item_types)
    validation = validate_items(item_rows, vehicle)
    summary = evaluate_solution(vehicle, item_types, improved)
    summary["validation"] = validation
    layer_rows = pd.DataFrame(summarize_layers(improved.layers))
    return item_rows, summary, layer_rows


def write_outputs(
    all_item_outputs: dict[str, pd.DataFrame],
    all_layer_outputs: dict[str, pd.DataFrame],
    all_summaries: dict[str, dict[str, object]],
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    workbook_path = OUTPUT_DIR / "model1_results.xlsx"
    with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
        for vehicle_key, df in all_item_outputs.items():
            df.to_excel(writer, sheet_name=f"{vehicle_key}_items", index=False)
        for vehicle_key, df in all_layer_outputs.items():
            df.to_excel(writer, sheet_name=f"{vehicle_key}_layers", index=False)
        summary_rows = pd.DataFrame.from_dict(all_summaries, orient="index")
        summary_rows.index.name = "vehicle_key_index"
        summary_rows = summary_rows.reset_index()
        summary_rows.to_excel(writer, sheet_name="summary", index=False)
    for vehicle_key, df in all_item_outputs.items():
        df.to_csv(OUTPUT_DIR / f"{vehicle_key}_items.csv", index=False, encoding="utf-8-sig")
    for vehicle_key, df in all_layer_outputs.items():
        df.to_csv(OUTPUT_DIR / f"{vehicle_key}_layers.csv", index=False, encoding="utf-8-sig")
    for vehicle_key, summary in all_summaries.items():
        (OUTPUT_DIR / f"{vehicle_key}_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def main() -> None:
    vehicles, item_types = build_inputs()
    all_item_outputs: dict[str, pd.DataFrame] = {}
    all_layer_outputs: dict[str, pd.DataFrame] = {}
    all_summaries: dict[str, dict[str, object]] = {}

    for vehicle in vehicles.values():
        item_rows, summary, layer_rows = solve_vehicle(vehicle, item_types)
        all_item_outputs[vehicle.key] = item_rows
        all_layer_outputs[vehicle.key] = layer_rows
        all_summaries[vehicle.key] = summary

    write_outputs(all_item_outputs, all_layer_outputs, all_summaries)
    console_summary = {
        vehicle_key: {
            "lambda": summary["lambda"],
            "space_utilization": summary["space_utilization"],
            "weight_utilization": summary["weight_utilization"],
            "loaded_item_counts": summary["loaded_item_counts"],
        }
        for vehicle_key, summary in all_summaries.items()
    }
    print(json.dumps(console_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
