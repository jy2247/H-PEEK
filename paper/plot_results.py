#!/usr/bin/env python3
"""Generate compact, dependency-light figures for the H-PEEK report.

The script deliberately uses Pillow instead of a plotting stack so that the
figures can be regenerated inside the locked project environment. Every point
is an arithmetic mean over the three recorded seeds; whiskers show sample
standard deviation.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import mean, stdev

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "figures"
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
COLORS = {"PEEK": "#2166ac", "H-PEEK": "#b2182b", "H-PEEK-m2": "#4d9221"}


def load(name: str) -> dict:
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def fonts() -> tuple[ImageFont.FreeTypeFont, ...]:
    return tuple(ImageFont.truetype(path, size) for path, size in ((FONT, 24), (FONT, 19), (FONT, 16), (BOLD, 27), (BOLD, 20)))


TITLE, LABEL, SMALL, TITLE_BOLD, LABEL_BOLD = fonts()


def stat(values: list[float]) -> tuple[float, float]:
    return mean(values), stdev(values) if len(values) > 1 else 0.0


def metric(run: dict, section: str, key: str) -> float:
    return float(run[section][key])


def line(draw: ImageDraw.ImageDraw, points: list[tuple[float, float]], color: str, width: int = 5) -> None:
    if len(points) > 1:
        draw.line(points, fill=color, width=width, joint="curve")
    for x, y in points:
        draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill=color, outline="white", width=2)


def panel(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], title: str, ylabel: str, ymax: float) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    draw.text((x0, y0 - 42), title, font=LABEL_BOLD, fill="#1b1b1b")
    left, top, right, bottom = x0 + 86, y0 + 12, x1 - 24, y1 - 58
    draw.rectangle((left, top, right, bottom), outline="#555555", width=2)
    for tick in range(5):
        value = ymax * tick / 4
        y = bottom - (bottom - top) * tick / 4
        draw.line((left, y, right, y), fill="#dddddd", width=1)
        label = f"{value:.1f}" if ymax < 10 else f"{value:.0f}"
        draw.text((left - 70, y - 12), label, font=SMALL, fill="#444444")
    draw.text((x0 + 6, top + 4), ylabel, font=SMALL, fill="#444444")
    return left, top, right, bottom


def map_xy(x: float, y: float, xmin: float, xmax: float, ymax: float, axes: tuple[int, int, int, int]) -> tuple[float, float]:
    left, top, right, bottom = axes
    px = left + (x - xmin) / (xmax - xmin) * (right - left) if xmax > xmin else (left + right) / 2
    py = bottom - y / ymax * (bottom - top) if ymax else bottom
    return px, py


def legend(draw: ImageDraw.ImageDraw, x: int, y: int, labels: list[str]) -> None:
    for index, label in enumerate(labels):
        px = x + index * 190
        draw.line((px, y + 11, px + 34, y + 11), fill=COLORS[label], width=6)
        draw.ellipse((px + 12, y + 4, px + 24, y + 16), fill=COLORS[label])
        draw.text((px + 44, y - 2), label, font=SMALL, fill="#222222")


def open_loop_figure() -> None:
    peek = load("results/matrix/peek_cluster_offered_load_4096blocks.json")["runs"]
    hpeek = load("results/matrix/hpeek_history_offered_load_4096blocks.json")["runs"]
    m2 = load("results/matrix/hpeek_history_minfreq2_offered_load_4096blocks.json")["runs"]
    grouped = {"PEEK": peek, "H-PEEK": hpeek, "H-PEEK-m2": m2}
    loads = [4, 8, 12, 16]
    image = Image.new("RGB", (1800, 1120), "white")
    draw = ImageDraw.Draw(image)
    draw.text((70, 28), "Open-loop load: throughput, TTFT tail, and fairness", font=TITLE_BOLD, fill="#111111")
    legend(draw, 1040, 35, ["PEEK", "H-PEEK", "H-PEEK-m2"])
    panels = [
        ((40, 120, 880, 535), "Throughput (requests/s)", "requests/s", 14),
        ((920, 120, 1760, 535), "Tail latency (TTFT P95)", "seconds", 2.4),
        ((40, 610, 880, 1025), "Class fairness (Jain index)", "Jain", 1.0),
        ((920, 610, 1760, 1025), "Hot/cold mean TTFT gap", "seconds", 1.3),
    ]
    for box, title, ylabel, ymax in panels:
        axes = panel(draw, box, title, ylabel, ymax)
        left, top, right, bottom = axes
        for index, offered in enumerate(loads):
            x = left + index * (right - left) / (len(loads) - 1)
            draw.text((x - 11, bottom + 15), str(offered), font=SMALL, fill="#444444")
        for label, rows in grouped.items():
            points = []
            for offered in loads:
                subset = [r for r in rows if r["offered_qps"] == offered]
                if title.startswith("Throughput"):
                    vals = [r["overall"]["requests_per_s"] for r in subset]
                elif title.startswith("Tail"):
                    vals = [r["overall"]["ttft_p95_s"] for r in subset]
                elif title.startswith("Class"):
                    vals = [r["latency_equity_jain"] for r in subset]
                else:
                    vals = [r["cold"]["ttft_mean_s"] - r["hot"]["ttft_mean_s"] for r in subset]
                value, error = stat(vals)
                x = left + loads.index(offered) * (right - left) / (len(loads) - 1)
                px, py = map_xy(offered, value, min(loads), max(loads), ymax, axes)
                ey = error / ymax * (bottom - top)
                draw.line((px, py - ey, px, py + ey), fill=COLORS[label], width=2)
                points.append((px, py))
            line(draw, points, COLORS[label])
    draw.text((700, 1072), "Offered load (requests/s)", font=LABEL, fill="#222222")
    image.save(OUT / "open_loop_performance.png", optimize=True)


def main_workload_figure() -> None:
    files = {
        "PEEK": "results/matrix/peek_cluster_4096blocks.json",
        "H-PEEK": "results/matrix/hpeek_history_joint_4096blocks.json",
    }
    workloads = [("periodic_reuse", "Periodic"), ("stale_history_pollution", "Drift"), ("singleton_overhead", "No sharing"), ("queue_share", "Queue share")]
    image = Image.new("RGB", (1800, 1100), "white")
    draw = ImageDraw.Draw(image)
    draw.text((70, 28), "Paired workloads: end-to-end performance", font=TITLE_BOLD, fill="#111111")
    legend(draw, 1270, 35, ["PEEK", "H-PEEK"])
    panels = [((40, 120, 880, 1025), "Throughput", "requests/s", 70), ((920, 120, 1760, 1025), "TTFT P95", "seconds", 0.85)]
    for box, title, ylabel, ymax in panels:
        axes = panel(draw, box, title, ylabel, ymax)
        left, top, right, bottom = axes
        step = (right - left) / len(workloads)
        for index, (_, short) in enumerate(workloads):
            x = left + step * (index + 0.5)
            draw.text((x - 45, bottom + 15), short, font=SMALL, fill="#444444")
            for offset, label in enumerate(("PEEK", "H-PEEK")):
                rows = [r for r in load(files[label])["runs"] if r["scenario"] == workloads[index][0]]
                if title == "Throughput":
                    vals = [r["summary"]["requests_per_s"] for r in rows]
                else:
                    vals = [r["summary"]["ttft_p95_s"] for r in rows]
                value, error = stat(vals)
                bar_w = min(72, step / 4)
                bx = x + (offset - 0.5) * bar_w * 1.15
                _, py = map_xy(0, value, 0, 1, ymax, axes)
                by = bottom
                draw.rectangle((bx - bar_w / 2, py, bx + bar_w / 2, by), fill=COLORS[label])
                ey = error / ymax * (bottom - top)
                draw.line((bx, py - ey, bx, py + ey), fill="#111111", width=3)
                draw.text((bx - 27, py - 34), f"{value:.2f}", font=SMALL, fill="#222222")
    draw.text((720, 1060), "Workload (three-seed mean ± s.d.)", font=LABEL, fill="#222222")
    image.save(OUT / "paired_workloads.png", optimize=True)


def cache_overhead_figure() -> None:
    peek = load("results/matrix/peek_cluster_4096blocks.json")["runs"]
    hpeek = load("results/matrix/hpeek_history_joint_4096blocks.json")["runs"]
    rows = {"PEEK": peek, "H-PEEK": hpeek}
    return_ttft = {label: stat([r["return_request"]["ttft_s"] * 1000 for r in data if r["scenario"] == "periodic_reuse"]) for label, data in rows.items()}
    return_hits = {label: stat([r["metrics"].get('vllm:prompt_tokens_by_source_total{engine="0",model_name="qwen2.5-3b",source="local_cache_hit"}', 0) for r in data if r["scenario"] == "periodic_reuse"]) for label, data in rows.items()}
    profiles = {
        "PEEK": load("results/matrix/peek_cluster_offered_load_profile.json"),
        "H-PEEK": load("results/matrix/hpeek_history_offered_load_profile.json"),
        "H-PEEK-m2": load("results/matrix/hpeek_history_minfreq2_offered_load_profile.json"),
    }
    image = Image.new("RGB", (1800, 1100), "white")
    draw = ImageDraw.Draw(image)
    draw.text((70, 28), "Cache benefit versus CPU-path overhead", font=TITLE_BOLD, fill="#111111")
    legend(draw, 1160, 35, ["PEEK", "H-PEEK", "H-PEEK-m2"])
    panels = [((40, 120, 880, 530), "Periodic return TTFT", "milliseconds", 300), ((920, 120, 1760, 530), "Periodic cache-hit tokens", "tokens", 13000), ((40, 610, 880, 1025), "Blocks scanned (open-loop)", "blocks", 80000), ((920, 610, 1760, 1025), "Eviction time (open-loop)", "milliseconds", 240)]
    values = {
        "Periodic return TTFT": {"PEEK": return_ttft["PEEK"], "H-PEEK": return_ttft["H-PEEK"]},
        "Periodic cache-hit tokens": {"PEEK": return_hits["PEEK"], "H-PEEK": return_hits["H-PEEK"]},
        "Blocks scanned (open-loop)": {k: (float(v["evict_scanned"]), 0.0) for k, v in profiles.items()},
        "Eviction time (open-loop)": {k: (float(v["evict_ns"]) / 1e6, 0.0) for k, v in profiles.items()},
    }
    for box, title, ylabel, ymax in panels:
        axes = panel(draw, box, title, ylabel, ymax)
        left, top, right, bottom = axes
        labels = list(values[title])
        step = (right - left) / len(labels)
        for index, label in enumerate(labels):
            x = left + step * (index + 0.5)
            value, error = values[title][label]
            _, py = map_xy(0, value, 0, 1, ymax, axes)
            bar_w = min(100, step / 2)
            draw.rectangle((x - bar_w / 2, py, x + bar_w / 2, bottom), fill=COLORS[label])
            ey = error / ymax * (bottom - top)
            if error:
                draw.line((x, py - ey, x, py + ey), fill="#111111", width=3)
            draw.text((x - 38, bottom + 15), label, font=SMALL, fill="#444444")
            draw.text((x - 36, py - 34), f"{value:.0f}", font=SMALL, fill="#222222")
    draw.text((710, 1060), "Policy (profiled runs; lower is better for overhead)", font=LABEL, fill="#222222")
    image.save(OUT / "cache_overhead.png", optimize=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    open_loop_figure()
    main_workload_figure()
    cache_overhead_figure()
    print(f"wrote figures to {OUT}")


if __name__ == "__main__":
    main()
