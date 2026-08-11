from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageChops, ImageStat


ROOT = Path(__file__).resolve().parent
REPLAY_ROOT = ROOT / "evals" / "vision_replay"
HISTORY_INDEX = REPLAY_ROOT / "history" / "history_index.json"
QUEUE_JSON = REPLAY_ROOT / "history" / "review_queue.json"
QUEUE_HTML = REPLAY_ROOT / "history" / "review_queue.html"
SIGNATURE_SIZE = (48, 48)
DEFAULT_THRESHOLD = 1.0
HIGH_RISK_ACTIONS = {
    "tap",
    "type_text",
    "type_symbol",
    "type_pinyin",
    "clear_text",
    "android_back",
    "android_home",
}


def image_signature(image: Image.Image) -> Image.Image:
    result = image.convert("L")
    top = min(30, max(0, result.height - 1))
    bottom = max(top + 1, result.height - 8)
    return result.crop((0, top, result.width, bottom)).resize(SIGNATURE_SIZE)


def signature_distance(first: Image.Image, second: Image.Image) -> float:
    if first.size != second.size:
        raise ValueError("图像签名尺寸不一致。")
    return float(ImageStat.Stat(ImageChops.difference(first, second)).mean[0])


def cluster_items(
    items: list[dict[str, Any]],
    *,
    threshold: float,
) -> list[dict[str, Any]]:
    clusters: list[dict[str, Any]] = []
    for item in items:
        image_path = REPLAY_ROOT / item["image"]
        with Image.open(image_path) as image:
            signature = image_signature(image)
        best_cluster: dict[str, Any] | None = None
        best_distance = float("inf")
        for cluster in clusters:
            distance = signature_distance(signature, cluster["_signature"])
            if distance <= threshold and distance < best_distance:
                best_cluster = cluster
                best_distance = distance
        if best_cluster is None:
            clusters.append(
                {
                    "_signature": signature,
                    "representative": item,
                    "members": [item],
                    "max_distance_from_representative": 0.0,
                }
            )
        else:
            best_cluster["members"].append(item)
            best_cluster["max_distance_from_representative"] = max(
                best_cluster["max_distance_from_representative"],
                round(best_distance, 3),
            )
    return clusters


def _cluster_priority(cluster: dict[str, Any]) -> int:
    members = cluster["members"]
    actions = {
        str((item.get("recorded_decision") or {}).get("action") or "")
        for item in members
    }
    failure_count = sum(
        item["review_status"] == "failure_candidate" for item in members
    )
    success_count = sum(
        item["review_status"] == "success_candidate" for item in members
    )
    mixed_bonus = 80 if len(actions) > 1 else 0
    risk_bonus = 40 if actions.intersection(HIGH_RISK_ACTIONS) else 0
    return (
        len(members) * 10
        + failure_count * 3
        + success_count
        + mixed_bonus
        + risk_bonus
    )


def build_review_queue(threshold: float = DEFAULT_THRESHOLD) -> dict[str, Any]:
    history = json.loads(HISTORY_INDEX.read_text(encoding="utf-8"))
    pending = [
        item
        for item in history["items"]
        if item["review_status"] != "golden"
    ]
    raw_clusters = cluster_items(pending, threshold=threshold)
    raw_clusters.sort(
        key=lambda cluster: (
            -_cluster_priority(cluster),
            -len(cluster["members"]),
            cluster["representative"]["id"],
        )
    )
    clusters: list[dict[str, Any]] = []
    for number, cluster in enumerate(raw_clusters, start=1):
        members = cluster["members"]
        actions = sorted(
            {
                str((item.get("recorded_decision") or {}).get("action") or "missing")
                for item in members
            }
        )
        screen_types = sorted(
            {
                str(
                    (item.get("recorded_decision") or {}).get("screen_type")
                    or "missing"
                )
                for item in members
            }
        )
        review_statuses = sorted({item["review_status"] for item in members})
        clusters.append(
            {
                "id": f"cluster_{number:03d}",
                "priority": _cluster_priority(cluster),
                "representative": {
                    key: cluster["representative"].get(key)
                    for key in (
                        "id",
                        "run",
                        "step",
                        "goal",
                        "outcome",
                        "image",
                        "recorded_decision",
                    )
                },
                "member_count": len(members),
                "max_distance_from_representative": cluster[
                    "max_distance_from_representative"
                ],
                "recorded_actions": actions,
                "recorded_screen_types": screen_types,
                "review_statuses": review_statuses,
                "needs_conflict_review": len(actions) > 1 or len(screen_types) > 1,
                "members": [
                    {
                        key: item.get(key)
                        for key in (
                            "id",
                            "run",
                            "step",
                            "goal",
                            "outcome",
                            "image",
                            "review_status",
                            "recorded_decision",
                        )
                    }
                    for item in members
                ],
            }
        )
    payload = {
        "schema_version": 1,
        "description": (
            "Near-duplicate review queue. Similarity only chooses a representative; "
            "it never propagates a golden label automatically."
        ),
        "threshold": threshold,
        "stats": {
            "pending_screenshots": len(pending),
            "clusters": len(clusters),
            "review_reduction": len(pending) - len(clusters),
            "singletons": sum(cluster["member_count"] == 1 for cluster in clusters),
            "multi_member_clusters": sum(
                cluster["member_count"] > 1 for cluster in clusters
            ),
            "conflict_clusters": sum(
                cluster["needs_conflict_review"] for cluster in clusters
            ),
        },
        "clusters": clusters,
    }
    QUEUE_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    QUEUE_HTML.write_text(render_review_html(payload), encoding="utf-8")
    return payload


def render_review_html(payload: dict[str, Any]) -> str:
    rows = []
    for cluster in payload["clusters"]:
        representative = cluster["representative"]
        image_path = _html_image_path(representative["image"])
        goal = html.escape(str(representative.get("goal") or ""))
        actions = html.escape(", ".join(cluster["recorded_actions"]))
        screens = html.escape(", ".join(cluster["recorded_screen_types"]))
        conflict = "需要冲突复核" if cluster["needs_conflict_review"] else "动作一致"
        member_thumbnails = []
        for member in cluster["members"]:
            decision = member.get("recorded_decision") or {}
            action = html.escape(str(decision.get("action") or "missing"))
            screen_type = html.escape(
                str(decision.get("screen_type") or "missing")
            )
            member_thumbnails.append(
                f"""
                <figure>
                  <img src="{_html_image_path(member['image'])}"
                       alt="{html.escape(member['id'])}">
                  <figcaption>{action} · {screen_type}<br>
                    <span>{html.escape(member['id'])}</span>
                  </figcaption>
                </figure>
                """
            )
        rows.append(
            f"""
            <article>
              <div class="summary">
                <img src="{image_path}" alt="{html.escape(cluster['id'])}">
              </div>
              <div class="details">
                <h2>{html.escape(cluster['id'])} · {cluster['member_count']} 张</h2>
                <p>{goal}</p>
                <dl>
                  <dt>历史动作</dt><dd>{actions}</dd>
                  <dt>页面类型</dt><dd>{screens}</dd>
                  <dt>状态</dt><dd>{conflict}</dd>
                  <dt>代表样本</dt><dd>{html.escape(representative['id'])}</dd>
                </dl>
                <details{' open' if cluster['needs_conflict_review'] else ''}>
                  <summary>查看全部 {cluster['member_count']} 张成员图</summary>
                  <div class="members">{''.join(member_thumbnails)}</div>
                </details>
              </div>
            </article>
            """
        )
    stats = payload["stats"]
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>手机视觉题库审核队列</title>
  <style>
    body {{ margin: 0; font-family: "Microsoft YaHei", sans-serif; color: #182026; background: #f4f6f7; }}
    header {{ padding: 24px; background: #12372f; color: white; }}
    header h1 {{ margin: 0 0 8px; font-size: 24px; }}
    header p {{ margin: 0; }}
    main {{ max-width: 1120px; margin: 0 auto; padding: 20px; }}
    article {{ display: grid; grid-template-columns: 220px 1fr; gap: 20px; margin-bottom: 16px; padding: 16px; background: white; border: 1px solid #d8dee2; border-radius: 6px; }}
    .summary img {{ width: 220px; height: 300px; object-fit: contain; background: #080b0c; }}
    h2 {{ margin: 0 0 10px; font-size: 18px; }}
    p {{ line-height: 1.6; }}
    dl {{ display: grid; grid-template-columns: 92px 1fr; gap: 6px 12px; }}
    dt {{ font-weight: 700; }}
    dd {{ margin: 0; overflow-wrap: anywhere; }}
    details {{ margin-top: 16px; }}
    summary {{ cursor: pointer; font-weight: 700; }}
    .members {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: 12px; margin-top: 12px; }}
    figure {{ margin: 0; min-width: 0; }}
    figure img {{ width: 100%; height: 220px; object-fit: contain; background: #080b0c; }}
    figcaption {{ margin-top: 5px; font-size: 12px; line-height: 1.4; overflow-wrap: anywhere; }}
    figcaption span {{ color: #59636b; }}
    @media (max-width: 680px) {{
      article {{ grid-template-columns: 1fr; }}
      .summary img {{ width: 100%; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>手机视觉题库审核队列</h1>
    <p>待审核 {stats['pending_screenshots']} 张 · {stats['clusters']} 个视觉簇 · 减少 {stats['review_reduction']} 次重复查看 · 冲突簇 {stats['conflict_clusters']}</p>
  </header>
  <main>{''.join(rows)}</main>
</body>
</html>
"""


def _html_image_path(image: str) -> str:
    path = Path(image.replace("\\", "/"))
    if path.parts and path.parts[0] == "history":
        path = Path(*path.parts[1:])
    return html.escape(path.as_posix())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="生成历史视觉截图去重审核队列。")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not 0.0 <= args.threshold <= 30.0:
        raise ValueError("threshold 必须在 0～30 之间。")
    payload = build_review_queue(args.threshold)
    stats = payload["stats"]
    print(
        f"审核队列完成：{stats['pending_screenshots']} 张待审核图，"
        f"聚为 {stats['clusters']} 组，减少 {stats['review_reduction']} 次重复查看；"
        f"冲突组 {stats['conflict_clusters']}。"
    )
    print(f"JSON：{QUEUE_JSON}")
    print(f"HTML：{QUEUE_HTML}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
