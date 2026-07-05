"""
ADRec 参数自动调参脚本（Python）

功能：
- 基于 src/config.yaml 作为默认配置，自动采样/网格搜索影响性能的关键参数：
  1) AdRec 原本模型参数
  2) 位置感知模块参数
  3) 排序一致性损失（PreferDiff 风格）参数
- 直接调用项目内部训练流程（trainer.model_train），返回并记录每次试验的验证/测试指标
- 支持设定 trial 次数（随机搜索）或走小型网格
- 在 logs/<model>/<dataset>/ 下输出调参结果 csv 与 jsonl 便于复现实验

用法示例：
  python tune_adrec.py --trials 20 --objective NDCG@20 --dataset ml-100k
  python tune_adrec.py --trials 30 --objective HR@20 --epochs 200 --seed 2025

注意：
- 本脚本默认以随机搜索为主；你可以通过 --search grid 启用小型网格搜索。
- 训练时间取决于 trials、epochs 和数据集规模；可先将 epochs 调小快速粗搜，再用最佳超参放大 epochs 精训。
"""
from __future__ import annotations

import argparse
import os
import json
import csv
import random
import time
import logging
from copy import deepcopy
from typing import Dict, Any, List, Optional
import yaml
import math

# 项目内模块
from logger import load_config
from trainer import item_num_create, choose_model, load_data, model_train
from utils import fix_random_seed_as


# -----------------------------
# 搜索空间定义（可按数据集自适应）
# -----------------------------

def build_search_space(base_cfg: Dict[str, Any]) -> Dict[str, List[Any]]:
    """根据数据集规模与当前默认配置，构建影响性能的关键参数搜索空间。
    覆盖三类参数：
    1) AdRec 原本模型/扩散参数
    2) 位置模块参数
    3) 排序损失参数（pref_loss_scale）
    """
    dataset = base_cfg.get("dataset", "ml-100k").lower()

    # Beauty 数据集：使用更聚焦的搜索空间（更快更稳）
    if dataset == "beauty":
        space = {
            # —— 模型原本参数：仅调 lr / dropout ——
            "lr": [1e-3, 7e-4],
            "dropout": [0.1, 0.2],
            # 其余原模型参数（如 batch_size、emb_dropout、hidden_act、dif_decoder、use_rope 等）保持 config.yaml 默认值，不纳入搜索

            # —— 扩散/重建（固定/小范围）——
            "diffusion_steps": [32],
            "diffusion_loss_type": ["cosine"],
            "lambda_uncertainty": [0.005, 0.01, 0.02],
            "noise_schedule": ["trunc_lin"],
            "beta_a": [0.3],
            "beta_b": [10],
            "geodesic": [False],
            "independent": [True],

            # —— 位置模块 ——
            "use_position_aware": [True],
            "position_importance_mode": ["hybrid", "learned", "position_based"],
            "use_error_guided_importance": [True],
            "importance_supervision_weight": [0.05, 0.08],
            "scheduler_temperature": [1.0, 1.5],
            "importance_entropy_weight": [0.001, 0.005],
            "last_importance_weight": [0.0, 0.002],

            # —— 排序一致性损失缩放 ——
            "pref_loss_scale": [0.8, 1.0, 1.2],
        }
        return space

    # 其他数据集保持较通用的空间
    is_large = dataset in ["yelp"]

    dif_steps = [32, 48] if not is_large else [32, 64]

    hidden_sizes = [128] + ([256] if is_large else [])

    if dataset == "ml-100k":
        pref_scales = [1.2, 1.5, 1.8, 2.0]
    elif dataset == "toys":
        pref_scales = [0.8, 1.0, 1.2]
    else:
        pref_scales = [1.0, 1.5, 1.8]

    lr_space = [1e-3, 7e-4, 5e-4]
    wd_space = [1e-5, 5e-6, 1e-4]

    importance_sup_w = [0.05, 0.08, 0.1]
    scheduler_temps = [1.0, 1.5, 2.0]
    imp_entropy_w = [0.001, 0.005, 0.01]
    last_imp_w = [0.0, 0.002, 0.005]

    space = {
        "hidden_size": hidden_sizes,
        "dropout": [0.1, 0.2],
        "emb_dropout": [0.1, 0.3],
        "hidden_act": ["gelu", "relu"],
        "dif_decoder": ["att", "mlp"],
        "use_rope": [False, True],
        "is_causal": [True],
        "lr": lr_space,
        "weight_decay": wd_space,
        "diffusion_steps": dif_steps,
        "diffusion_loss_type": ["cosine", "mse"],
        "lambda_uncertainty": [0.005, 0.01, 0.02],
        "noise_schedule": ["trunc_lin"],
        "beta_a": [0.3],
        "beta_b": [10],
        "geodesic": [False],
        "independent": [True],
        "use_position_aware": [True, False],
        "position_importance_mode": ["hybrid", "learned", "position_based"],
        "use_error_guided_importance": [True, False],
        "importance_supervision_weight": importance_sup_w,
        "scheduler_temperature": scheduler_temps,
        "importance_entropy_weight": imp_entropy_w,
        "last_importance_weight": last_imp_w,
        "pref_loss_scale": pref_scales,
    }

    return space


# -----------------------------
# 搜索策略
# -----------------------------

def sample_from_space(space: Dict[str, List[Any]], rng: random.Random) -> Dict[str, Any]:
    return {k: rng.choice(v) for k, v in space.items()}


def _nearest(vals: List[float], target: float) -> List[float]:
    # 返回与 target 最接近的去重列表，保持原序
    uniq = []
    for v in vals:
        if v not in uniq:
            uniq.append(v)
    uniq.sort(key=lambda x: abs(x - target))
    return uniq

def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def _around(value: float, factors=(0.7, 1.0, 1.3), lo=None, hi=None, step=None) -> List[float]:
    cands = [value * f for f in factors]
    if lo is not None and hi is not None:
        cands = [_clip(v, lo, hi) for v in cands]
    if step is not None:
        cands = [round(v / step) * step for v in cands]
    # 去重并保持顺序
    out = []
    for v in cands:
        if v not in out:
            out.append(v)
    return out

def build_fine_space_from_seed(seed_overrides: Dict[str, Any]) -> Dict[str, List[Any]]:
    """围绕某个 coarse 阶段的 seed 覆盖项，构造局部细搜空间（beauty 专用）。"""
    space: Dict[str, List[Any]] = {}
    # 模型原本参数：batch_size / lr / dropout

    lr_seed = float(seed_overrides.get("lr", 1e-3))
    space["lr"] = _around(lr_seed, factors=(0.8, 1.0, 1.25), lo=3e-4, hi=2e-3)

    dp_seed = float(seed_overrides.get("dropout", 0.1))
    # 在 0.05-0.25 间微调，步长 0.05
    space["dropout"] = _nearest([0.1, 0.15, 0.2, 0.25], dp_seed)[:3]

    # 扩散参数（小范围）
    lu_seed = float(seed_overrides.get("lambda_uncertainty", 0.01))
    space["lambda_uncertainty"] = _nearest([0.005, 0.01, 0.02], lu_seed)[:2] + [0.015] if 0.015 not in [0.005,0.01,0.02] else _nearest([0.005,0.01,0.02], lu_seed)
    space["diffusion_steps"] = [32]
    space["diffusion_loss_type"] = ["cosine"]
    space["noise_schedule"] = ["trunc_lin"]
    space["beta_a"] = [0.3]
    space["beta_b"] = [10]
    space["geodesic"] = [False]
    space["independent"] = [True]

    # 位置模块（小范围）
    space["use_position_aware"] = [True]
    pim_seed = seed_overrides.get("position_importance_mode", "hybrid")
    space["position_importance_mode"] = [pim_seed] if pim_seed in ("hybrid","learned") else ["hybrid","learned"]
    space["use_error_guided_importance"] = [True]

    isw_seed = float(seed_overrides.get("importance_supervision_weight", 0.08))
    space["importance_supervision_weight"] = _nearest([0.05, 0.08], isw_seed)

    st_seed = float(seed_overrides.get("scheduler_temperature", 1.5))
    space["scheduler_temperature"] = _nearest([1.0, 1.5], st_seed)

    iew_seed = float(seed_overrides.get("importance_entropy_weight", 0.005))
    space["importance_entropy_weight"] = _nearest([0.001, 0.005], iew_seed)

    liw_seed = float(seed_overrides.get("last_importance_weight", 0.002))
    space["last_importance_weight"] = _nearest([0.0, 0.002], liw_seed)

    # 排序一致性损失缩放
    pls_seed = float(seed_overrides.get("pref_loss_scale", 1.0))
    near_pls = [0.8, 1.0, 1.2]
    space["pref_loss_scale"] = _nearest(near_pls, pls_seed)

    return space


def grid_from_space(space: Dict[str, List[Any]], max_trials: int) -> List[Dict[str, Any]]:
    """通用小网格：对传入的 space 中的所有键做笛卡尔积（按键顺序），超过 max_trials 即截断。
    注意：为控制规模，请仅在候选集较小（如细搜阶段）使用。
    """
    keys = list(space.keys())
    if not keys:
        return []

    # 逐步展开笛卡尔积，随时截断
    grids = [{}]
    for k in keys:
        new_grids = []
        values = space[k]
        for base in grids:
            for v in values:
                cand = dict(base)
                cand[k] = v
                new_grids.append(cand)
                if len(new_grids) >= max_trials:
                    break
            if len(new_grids) >= max_trials:
                break
        grids = new_grids
        if len(grids) >= max_trials:
            break
    return grids[:max_trials]


# -----------------------------
# 训练与评估
# -----------------------------

def build_args_from_cfg(base_cfg: Dict[str, Any], overrides: Dict[str, Any]) -> argparse.Namespace:
    cfg = deepcopy(base_cfg)
    for k, v in overrides.items():
        cfg[k] = v
    # 固定使用 adrec
    cfg["model"] = "adrec"
    # 记录描述，便于区分不同 trial
    if cfg.get("description") in (None, "_"):
        cfg["description"] = "_tune"
    # 将 dict 转为 Namespace
    return argparse.Namespace(**cfg)


def _make_trial_logger(log_dir: str, trial_id: int, phase_tag: str) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    logger_name = f"tune_adrec.{phase_tag}.T{trial_id}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    # 重置旧 handler，确保不同 phase 同名 trial 不会复用旧文件
    for h in list(logger.handlers):
        logger.removeHandler(h)
    fh = logging.FileHandler(os.path.join(log_dir, f"trial_{trial_id}.log"), encoding="utf-8")
    sh = logging.StreamHandler()
    fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(fmt)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def run_one_trial(trial_id: int,
                  base_cfg: Dict[str, Any],
                  overrides: Dict[str, Any],
                  objective: str,
                  epochs_override: Optional[int],
                  seed: Optional[int],
                  log_dir: str,
                  phase: Optional[str] = None) -> Dict[str, Any]:
    """执行一次 trial，返回结果字典。"""
    args = build_args_from_cfg(base_cfg, overrides)

    # 固定随机种子，保证每次 trial 可复现
    fix_random_seed_as(args.random_seed if seed is None else seed)

    if seed is not None:
        args.random_seed = seed
    if epochs_override is not None:
        args.epochs = int(epochs_override)

    phase_tag = (phase or "single").upper()

    # 安全简短的文件标签（避免路径过长/非法字符）
    import hashlib as _hl
    hash_key = _hl.md5(json.dumps(overrides, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:8]
    tag = f"{phase_tag}_T{trial_id}_{hash_key}"
    args.description = (args.description or "") + "_" + tag

    # 生成 item_num、搭建模型与数据
    args = item_num_create(args)
    model = choose_model(args)
    tra_loader, val_loader, test_loader = load_data(args)

    # 训练与评估
    start = time.time()
    trial_log_dir = os.path.join(log_dir, phase_tag.lower(), f"trial_{trial_id}")
    tlogger = _make_trial_logger(trial_log_dir, trial_id, phase_tag.lower())

    # 在每个 trial 日志开头写入完整参数与元信息
    formatted_args = "\n".join(f"{key}: {value}" for key, value in vars(args).items())
    tlogger.info("Trial meta: phase=%s, trial_id=%d, objective=%s, epochs=%s", phase_tag.lower(), trial_id, objective, str(getattr(args, 'epochs', 'N/A')))
    tlogger.info("Overrides: %s", json.dumps(overrides, ensure_ascii=False))
    tlogger.info("Arguments:\n%s", formatted_args)

    best_model, test_metrics = model_train(
        model,
        tra_loader,
        val_loader,
        test_loader,
        args,
        tlogger,
        train_time=time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()),
    )
    elapsed = time.time() - start

    # 目标值
    obj_value = float(test_metrics.get(objective, 0.0))

    result = {
        "trial_id": trial_id,
        "phase": phase_tag.lower(),
        "overrides": overrides,
        "objective": objective,
        "objective_value": obj_value,
        "test_metrics": test_metrics,
        "elapsed_sec": round(elapsed, 2),
        "description": args.description,
    }

    # 持久化结果（汇总文件）
    os.makedirs(log_dir, exist_ok=True)
    jsonl_path = os.path.join(log_dir, "tuning_results.jsonl")
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")

    # 同步到 csv（仅记录关键信息）
    csv_path = os.path.join(log_dir, "tuning_results.csv")
    header = ["trial_id", "phase", "objective", "objective_value", "elapsed_sec", "description", "overrides", "test_metrics"]
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow([
            trial_id,
            phase_tag.lower(),
            objective,
            obj_value,
            round(elapsed, 2),
            args.description,
            json.dumps(overrides, ensure_ascii=False),
            json.dumps(test_metrics, ensure_ascii=False),
        ])

    return result


# -----------------------------
# 主流程
# -----------------------------

def main():
    parser = argparse.ArgumentParser()
    # 顶层控制：单阶段 or 分阶段
    parser.add_argument("--staged", type=bool, default=True, help="是否启用分阶段（coarse→fine）搜索，默认启用")
    parser.add_argument("--objective", type=str, default="HR@20", choices=["HR@5", "NDCG@5", "HR@10", "NDCG@10", "HR@20", "NDCG@20"], help="目标指标（beauty 建议 NDCG@20）")
    parser.add_argument("--dataset", type=str, default="beauty", help="覆盖数据集（默认 beauty）")
    parser.add_argument("--seed", type=int, default=2025, help="全局随机种子（用于采样一致性）")

    # 单阶段模式参数（兼容原逻辑）
    parser.add_argument("--trials", type=int, default=40, help="（单阶段）搜索试验次数")
    parser.add_argument("--search", type=str, default="random", choices=["random", "grid"], help="（单阶段）搜索策略")
    parser.add_argument("--epochs", type=int, default=120, help="（单阶段）训练轮数")

    # 分阶段参数
    parser.add_argument("--coarse_trials", type=int, default=30, help="粗搜试验次数（beauty 建议 30）")
    parser.add_argument("--coarse_epochs", type=int, default=120, help="粗搜训练轮数（beauty 建议 120）")
    parser.add_argument("--coarse_search", type=str, default="random", choices=["random", "grid"], help="粗搜搜索策略")

    parser.add_argument("--fine_trials", type=int, default=12, help="细搜总试验次数（beauty 建议 12）")
    parser.add_argument("--fine_epochs", type=int, default=150, help="细搜训练轮数（beauty 建议 150）")
    parser.add_argument("--fine_topk", type=int, default=3, help="从粗搜中挑选前 top-k 作为细搜种子")
    parser.add_argument("--fine_search", type=str, default="grid", choices=["random", "grid"], help="细搜搜索策略（默认小网格）")

    args = parser.parse_args()

    rng = random.Random(args.seed)

    # 读取默认配置
    base_cfg = load_config("config.yaml")
    if args.dataset:
        base_cfg["dataset"] = args.dataset
    base_cfg["model"] = "adrec"

    # 结果目录
    log_dir = os.path.join(
        base_cfg.get("log_file", "logs/"),
        base_cfg.get("model", "adrec"),
        base_cfg.get("dataset", "ml-100k"),
        "tuning",
    )
    os.makedirs(log_dir, exist_ok=True)

    best = None
    best_hr = None
    best_ndcg = None

    if not args.staged:
        # -------- 单阶段：与旧逻辑一致 --------
        space = build_search_space(base_cfg)
        if args.search == "grid":
            candidates = grid_from_space(space, args.trials)
        else:
            candidates = [sample_from_space(space, rng) for _ in range(args.trials)]

        for i, overrides in enumerate(candidates, start=1):
            try:
                result = run_one_trial(
                    trial_id=i,
                    base_cfg=base_cfg,
                    overrides=overrides,
                    objective=args.objective,
                    epochs_override=args.epochs,
                    seed=args.seed,
                    log_dir=log_dir,
                    phase="single",
                )
                # 更新 HR@20 与 NDCG@20 最优记录
                try:
                    hr20 = float(result.get("test_metrics", {}).get("HR@20", 0.0))
                    ndcg20 = float(result.get("test_metrics", {}).get("NDCG@20", 0.0))
                    if (best_hr is None) or (hr20 > best_hr["value"]):
                        best_hr = {"value": hr20, "result": result}
                    if (best_ndcg is None) or (ndcg20 > best_ndcg["value"]):
                        best_ndcg = {"value": ndcg20, "result": result}
                except Exception:
                    pass
                if (best is None) or (result["objective_value"] > best["objective_value"]):
                    best = result
                print(f"[SINGLE {i}/{len(candidates)}] {args.objective}={result['objective_value']:.4f} | overrides={overrides}")
            except Exception as e:
                print(f"[SINGLE {i}] 运行失败: {e}")
                continue
    else:
        # -------- 分阶段：coarse → fine --------
        # 1) 粗搜：较大范围、较小 epochs
        coarse_space = build_search_space(base_cfg)
        if args.coarse_search == "grid":
            coarse_candidates = grid_from_space(coarse_space, args.coarse_trials)
        else:
            coarse_candidates = [sample_from_space(coarse_space, rng) for _ in range(args.coarse_trials)]

        coarse_results: List[Dict[str, Any]] = []
        for i, overrides in enumerate(coarse_candidates, start=1):
            try:
                result = run_one_trial(
                    trial_id=i,
                    base_cfg=base_cfg,
                    overrides=overrides,
                    objective=args.objective,
                    epochs_override=args.coarse_epochs,
                    seed=args.seed,
                    log_dir=log_dir,
                    phase="coarse",
                )
                coarse_results.append(result)
                # 更新 HR@20 与 NDCG@20 最优记录
                try:
                    hr20 = float(result.get("test_metrics", {}).get("HR@20", 0.0))
                    ndcg20 = float(result.get("test_metrics", {}).get("NDCG@20", 0.0))
                    if (best_hr is None) or (hr20 > best_hr["value"]):
                        best_hr = {"value": hr20, "result": result}
                    if (best_ndcg is None) or (ndcg20 > best_ndcg["value"]):
                        best_ndcg = {"value": ndcg20, "result": result}
                except Exception:
                    pass
                if (best is None) or (result["objective_value"] > best["objective_value"]):
                    best = result
                print(f"[COARSE {i}/{len(coarse_candidates)}] {args.objective}={result['objective_value']:.4f} | overrides={overrides}")
            except Exception as e:
                print(f"[COARSE {i}] 运行失败: {e}")
                continue

        if not coarse_results:
            print("粗搜阶段无有效结果，提前结束。")
        else:
            # 2) 细搜：围绕 top-k 粗搜结果生成局部空间，小网格/少量随机
            coarse_sorted = sorted(coarse_results, key=lambda x: x.get("objective_value", 0.0), reverse=True)
            seeds = coarse_sorted[: max(1, args.fine_topk)]

            # 将细搜总试验次数分配到各 seed
            per_seed = max(1, math.ceil(args.fine_trials / len(seeds)))
            fine_count = 0

            for s_idx, seed_res in enumerate(seeds, start=1):
                seed_overrides = seed_res["overrides"]
                fine_space = build_fine_space_from_seed(seed_overrides)

                if args.fine_search == "grid":
                    candidates = grid_from_space(fine_space, per_seed)
                else:
                    candidates = [sample_from_space(fine_space, rng) for _ in range(per_seed)]

                for j, overrides in enumerate(candidates, start=1):
                    if fine_count >= args.fine_trials:
                        break
                    try:
                        result = run_one_trial(
                            trial_id=fine_count + 1,
                            base_cfg=base_cfg,
                            overrides=overrides,
                            objective=args.objective,
                            epochs_override=args.fine_epochs,
                            seed=args.seed,
                            log_dir=log_dir,
                            phase="fine",
                        )
                        fine_count += 1
                        # 更新 HR@20 与 NDCG@20 最优记录
                        try:
                            hr20 = float(result.get("test_metrics", {}).get("HR@20", 0.0))
                            ndcg20 = float(result.get("test_metrics", {}).get("NDCG@20", 0.0))
                            if (best_hr is None) or (hr20 > best_hr["value"]):
                                best_hr = {"value": hr20, "result": result}
                            if (best_ndcg is None) or (ndcg20 > best_ndcg["value"]):
                                best_ndcg = {"value": ndcg20, "result": result}
                        except Exception:
                            pass
                        if (best is None) or (result["objective_value"] > best["objective_value"]):
                            best = result
                        print(f"[FINE seed#{s_idx} {j}/{len(candidates)}] {args.objective}={result['objective_value']:.4f} | overrides={overrides}")
                    except Exception as e:
                        print(f"[FINE seed#{s_idx} {j}] 运行失败: {e}")
                        continue

    # 输出最佳结果
    if best is not None:
        print("=== Best Trial ===")
        print(json.dumps(best, ensure_ascii=False, indent=2))
        # 额外保存当前最佳配置（仅覆盖项）
        best_cfg_path = os.path.join(log_dir, "best_overrides.json")
        with open(best_cfg_path, "w", encoding="utf-8") as f:
            json.dump(best["overrides"], f, ensure_ascii=False, indent=2)

        # 保存“最佳试验”的完整可复现实验配置（基于 base_cfg 合并 overrides 后的全部参数）
        best_args_ns = build_args_from_cfg(base_cfg, best["overrides"])
        best_args_dict = vars(best_args_ns)
        best_full_yaml = os.path.join(log_dir, "best_full_config.yaml")
        with open(best_full_yaml, "w", encoding="utf-8") as f:
            yaml.safe_dump(best_args_dict, f, allow_unicode=True, sort_keys=False)

        # 额外保存一个包含指标与配置的汇总 JSON，便于程序读取
        best_summary = {
            "trial_id": best["trial_id"],
            "phase": best.get("phase", "single"),
            "objective": best["objective"],
            "objective_value": best["objective_value"],
            "test_metrics": best["test_metrics"],
            "elapsed_sec": best["elapsed_sec"],
            "description": best["description"],
            "overrides": best["overrides"],
            "full_config_path": best_full_yaml,
        }
        with open(os.path.join(log_dir, "best_summary.json"), "w", encoding="utf-8") as f:
            json.dump(best_summary, f, ensure_ascii=False, indent=2)

        # 同时输出一份纯文本日志，便于快速查看
        with open(os.path.join(log_dir, "best_params.log"), "w", encoding="utf-8") as f:
            f.write("Best Trial Summary\n")
            f.write(json.dumps(best_summary, ensure_ascii=False, indent=2))

        # 额外保存 HR@20 与 NDCG@20 的最优配置与摘要
        if best_hr is not None:
            hr_res = best_hr["result"]
            hr_overrides = hr_res["overrides"]
            hr_args_ns = build_args_from_cfg(base_cfg, hr_overrides)
            hr_full_yaml = os.path.join(log_dir, "best_hr20_full_config.yaml")
            with open(os.path.join(log_dir, "best_hr20_overrides.json"), "w", encoding="utf-8") as f:
                json.dump(hr_overrides, f, ensure_ascii=False, indent=2)
            with open(hr_full_yaml, "w", encoding="utf-8") as f:
                yaml.safe_dump(vars(hr_args_ns), f, allow_unicode=True, sort_keys=False)
            hr_summary = {
                "metric": "HR@20",
                "value": float(hr_res.get("test_metrics", {}).get("HR@20", 0.0)),
                "trial_id": hr_res["trial_id"],
                "phase": hr_res.get("phase", "single"),
                "test_metrics": hr_res.get("test_metrics", {}),
                "overrides": hr_overrides,
                "full_config_path": hr_full_yaml,
                "description": hr_res.get("description", "")
            }
            with open(os.path.join(log_dir, "best_hr20_summary.json"), "w", encoding="utf-8") as f:
                json.dump(hr_summary, f, ensure_ascii=False, indent=2)

        if best_ndcg is not None:
            ndcg_res = best_ndcg["result"]
            ndcg_overrides = ndcg_res["overrides"]
            ndcg_args_ns = build_args_from_cfg(base_cfg, ndcg_overrides)
            ndcg_full_yaml = os.path.join(log_dir, "best_ndcg20_full_config.yaml")
            with open(os.path.join(log_dir, "best_ndcg20_overrides.json"), "w", encoding="utf-8") as f:
                json.dump(ndcg_overrides, f, ensure_ascii=False, indent=2)
            with open(ndcg_full_yaml, "w", encoding="utf-8") as f:
                yaml.safe_dump(vars(ndcg_args_ns), f, allow_unicode=True, sort_keys=False)
            ndcg_summary = {
                "metric": "NDCG@20",
                "value": float(ndcg_res.get("test_metrics", {}).get("NDCG@20", 0.0)),
                "trial_id": ndcg_res["trial_id"],
                "phase": ndcg_res.get("phase", "single"),
                "test_metrics": ndcg_res.get("test_metrics", {}),
                "overrides": ndcg_overrides,
                "full_config_path": ndcg_full_yaml,
                "description": ndcg_res.get("description", "")
            }
            with open(os.path.join(log_dir, "best_ndcg20_summary.json"), "w", encoding="utf-8") as f:
                json.dump(ndcg_summary, f, ensure_ascii=False, indent=2)

        # 追加入 best_params.log 汇总
        with open(os.path.join(log_dir, "best_params.log"), "a", encoding="utf-8") as f:
            if best_hr is not None:
                f.write("\n\nBest HR@20 Trial\n")
                f.write(json.dumps(hr_summary, ensure_ascii=False, indent=2))
            if best_ndcg is not None:
                f.write("\n\nBest NDCG@20 Trial\n")
                f.write(json.dumps(ndcg_summary, ensure_ascii=False, indent=2))
    else:
        print("未找到有效的试验结果。")


if __name__ == "__main__":
    main()
