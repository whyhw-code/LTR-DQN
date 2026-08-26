"""Generate T3/T4/T5/T7 and all non-T6 empirical figures."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
CODE_DIR = Path(__file__).resolve().parent

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Export T3/T4/T5/T7 and the empirical paper figures')
    parser.add_argument('--run_dir', type=Path, default=CODE_DIR)
    parser.add_argument('--tables', default='T3,T4,T5,T7')
    parser.add_argument('--export_csvs', action='store_true')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--seed_config', type=Path, default=None)
    parser.add_argument('--ranker_tree_method', choices=['gpu_hist'], default='gpu_hist')
    parser.add_argument('--no_baselines', action='store_true')
    parser.add_argument('--dqn_eval_mode', choices=['dqn', 'fixed'], default='dqn')
    parser.add_argument('--figures', default='3,4,5,6,7')
    parser.add_argument('--appendix_figures', default='C1,C2,C3,C5')
    parser.add_argument('--force_figures', action='store_true')
    parser.add_argument('--skip_figures', action='store_true', help='Generate tables only; useful for a quick workflow check')
    return parser.parse_args()

def run_stage(label: str, script: str, arguments: list[str]) -> None:
    command = [sys.executable, str(CODE_DIR / script), *arguments]
    print(f'\n== {label} ==', flush=True)
    subprocess.run(command, cwd=CODE_DIR, check=True)

def run_internal_stage(label: str, entrypoint, arguments: list[str]) -> None:
    previous = sys.argv
    try:
        sys.argv = [label, *arguments]
        print(f'\n== {label} ==', flush=True)
        entrypoint()
    finally:
        sys.argv = previous

def shared_arguments(args: argparse.Namespace) -> list[str]:
    result = ['--run_dir', str(args.run_dir.resolve())]
    if args.seed is not None:
        result.extend(['--seed', str(args.seed)])
    if args.seed_config is not None:
        result.extend(['--seed_config', str(args.seed_config.resolve())])
    return result

def main() -> None:
    args = parse_args()
    common = shared_arguments(args)
    table_args = [*common, '--tables', args.tables, '--dqn_eval_mode', args.dqn_eval_mode]
    if args.export_csvs:
        table_args.append('--export_csvs')
    if args.no_baselines:
        table_args.append('--no_baselines')
    run_stage('Tables T3/T4/T5/T7', 'model.py', table_args)
    if args.skip_figures:
        return
    figure_args = [*common, '--figures', args.figures, '--ranker_tree_method', args.ranker_tree_method]
    if args.force_figures:
        figure_args.append('--force')
    run_internal_stage('Main-text Figures 3-7', _fig_main, figure_args)
    appendix_args = ['--run_dir', str(args.run_dir.resolve()), '--figures', args.appendix_figures]
    if args.force_figures:
        appendix_args.append('--force')
    run_internal_stage('Appendix Figures C1/C2/C3/C5', _appendix_main, appendix_args)
import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable
import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import Lasso
from sklearn.preprocessing import MinMaxScaler
from model import CODE_DIR, DATA_DIR, DQN_RANKER, FEATURES, MARKETS, PAPER_HYPERPARAMETERS, TEST_END, TEST_START, artifact_dir, backtest_predictions, evaluate_dqn, fit_baseline, load_stock_data, runtime_versions, sha256, stage_seed, train_dqn, validate_runtime, baseline_ranking, load_stage_seed_config
_fig_MARKET_ORDER = ('Main', 'ChiNext')
_fig_MARKET_TITLES = {'Main': 'Main board market', 'ChiNext': 'ChiNext market'}
_fig_INDEX_NAMES = {'Main': 'CSI 300 Index', 'ChiNext': 'ChiNext Index'}
_fig_BASELINES = ('LR', 'MLP_R', 'SVM_R', 'XGB_R', 'SVM_C', 'MLP_C', 'XGB_C')
_fig_LEARNING_RATES = (0.0001, 0.001, 0.002, 0.01, 0.1, 0.2)
_fig_MART_ESTIMATORS = (800, 900, 1000, 1100, 1200)
_fig_MART_DEPTHS = (4, 5, 6, 7, 8)
_fig_COLORS = {'Main': '#2F5597', 'ChiNext': '#D28E00', 'LTR-DQN': '#C00000', 'LambdaMART': '#4472C4', 'LambdaRank': '#70AD47', 'index': '#666666'}

def _fig_parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Recompute and export empirical main-text Figures 3-7')
    parser.add_argument('--run_dir', type=Path, default=CODE_DIR, help='Artifacts created by train.py; default uses code_1_final/temp and model')
    parser.add_argument('--output_dir', type=Path, default=None, help='Default: <run results>/figures')
    parser.add_argument('--figures', default='3,4,5,6,7', help='Comma-separated subset of 3,4,5,6,7')
    parser.add_argument('--seed_config', type=Path, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--ranker_tree_method', choices=['gpu_hist'], default='gpu_hist', help='GPU-only XGBoost tree builder')
    parser.add_argument('--n_games', type=int, default=31, help='DQN episodes for Figure 3(b), matching train.py by default')
    parser.add_argument('--force', action='store_true', help='Recompute cached figure data instead of reusing data/*.csv')
    return parser.parse_args()

def _fig_selected_figures(value: str) -> list[int]:
    figures = sorted({int(item.strip()) for item in value.split(',') if item.strip()})
    invalid = sorted(set(figures) - {3, 4, 5, 6, 7})
    if not figures or invalid:
        raise ValueError(f'figures must be a subset of 3,4,5,6,7; invalid={invalid}')
    return figures

def _fig_require_file(path: Path, purpose: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f'{purpose} not found: {path}. Run train.py first.')
    return path

def _fig_cached_csv(path: Path, force: bool, expected: dict[str, object] | None=None) -> pd.DataFrame | None:
    if path.is_file() and (not force):
        frame = pd.read_csv(path)
        if expected and any((key not in frame.columns or frame.empty or (not (frame[key].astype(str) == str(value)).all()) for (key, value) in expected.items())):
            print(f'Ignoring cache generated with different settings: {path}')
            return None
        print(f'Using cached figure data: {path}')
        return frame
    return None

def _fig_save_csv(frame: pd.DataFrame, path: Path) -> pd.DataFrame:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding='utf-8-sig')
    return frame

def _fig_digest_text(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(',', ':'), default=str)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()

def _fig_seed_signature(seed_config: dict, seed_override: int | None) -> str:
    return _fig_digest_text({'config': seed_config, 'override': seed_override})

def _fig_file_signature(paths: Iterable[Path]) -> str:
    resolved = [_fig_require_file(path, 'figure source artifact') for path in paths]
    return _fig_digest_text({str(path.resolve()): sha256(path) for path in resolved})

def _fig_implementation_paths() -> list[Path]:
    return [CODE_DIR / 'main.py', CODE_DIR / 'model.py', CODE_DIR / 'dl_dqn2.py']

def _fig_style_axis(ax, *, grid_axis: str='y') -> None:
    ax.set_facecolor('white')
    ax.grid(axis=grid_axis, color='#D9D9D9', linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color('#A6A6A6')
        spine.set_linewidth(0.8)

def _fig_save_figure(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'Saved: {path}')

def _fig_rank_groups(frame: pd.DataFrame) -> list[int]:
    return frame.groupby('qid_date', sort=True).size().tolist()

def _fig_fit_ranker_variant(market: str, model_name: str, *, learning_rate: float, max_depth: int, n_estimators: int, seed: int, tree_method: str) -> tuple[xgb.XGBRanker, pd.DataFrame]:
    """Fit the paper ranker while varying only the requested hyperparameters."""
    code = MARKETS[market]
    if tree_method != 'gpu_hist':
        raise ValueError(f'GPU execution is required for figures; got tree_method={tree_method!r}')
    resolved_tree_method = 'gpu_hist'
    (train, test) = load_stock_data(market, 3)
    combined = pd.concat([train, test], ignore_index=True)
    x_scaler = MinMaxScaler(feature_range=(-1, 1)).fit(combined[FEATURES])
    y_scaler = MinMaxScaler(feature_range=(-1, 1)).fit(combined[['real_return']])
    params = {'objective': 'rank:pairwise' if model_name == 'LambdaRank' else 'rank:map' if code == '0060' else 'rank:ndcg', 'tree_method': resolved_tree_method, 'booster': 'gbtree', 'eval_metric': 'ndcg', 'learning_rate': learning_rate, 'max_depth': max_depth, 'n_estimators': n_estimators, 'lambdarank_num_pair_per_sample': 8, 'lambdarank_pair_method': 'topk', 'random_state': seed, 'n_jobs': 1}
    model = xgb.XGBRanker(**params)
    model.fit(x_scaler.transform(train[FEATURES]), y_scaler.transform(train[['real_return']]).ravel(), group=_fig_rank_groups(train))
    predictions = pd.Series(index=test.index, dtype=float)
    for (_, group) in test.groupby('qid_date', sort=True):
        predictions.loc[group.index] = model.predict(x_scaler.transform(group[FEATURES]))
    ranked = test[['qid_date', 'stock_code', 'real_return', 'close', 'pclose']].copy()
    ranked['prediction'] = predictions.loc[ranked.index].to_numpy()
    return (model, ranked)

def _fig_compute_rank_sensitivity(data_path: Path, seed_config: dict, seed_override: int | None, tree_method: str, force: bool) -> pd.DataFrame:
    signature = _fig_seed_signature(seed_config, seed_override)
    cached = _fig_cached_csv(data_path, force, {'tree_method': tree_method, 'seed_signature': signature})
    if cached is not None:
        return cached
    rows = []
    for market in _fig_MARKET_ORDER:
        code = MARKETS[market]
        seed = stage_seed(code, 3, 'rank', seed_config, seed_override)
        for lr in _fig_LEARNING_RATES:
            (_, ranked) = _fig_fit_ranker_variant(market, 'LambdaRank', learning_rate=lr, max_depth=6, n_estimators=100, seed=seed, tree_method=tree_method)
            (metrics, _) = backtest_predictions(ranked)
            rows.append({'market': market, 'learning_rate': lr, 'ARR': metrics['ARR'], 'tree_method': tree_method, 'seed_signature': signature})
            print(f"Figure 3(a): {market} lr={lr:g} ARR={metrics['ARR']:.6f}")
    return _fig_save_csv(pd.DataFrame(rows), data_path)

def _fig_compute_dqn_lr_sensitivity(run_dir: Path, data_path: Path, work_dir: Path, seed_config: dict, seed_override: int | None, n_games: int, force: bool) -> pd.DataFrame:
    sources = []
    for market in _fig_MARKET_ORDER:
        sources.extend([artifact_dir(run_dir, 'rankings') / f'{market}_{DQN_RANKER}_train3.csv', artifact_dir(run_dir, 'rankings') / f'{market}_{DQN_RANKER}_test3.csv'])
    source_signature = _fig_file_signature([*sources, *_fig_implementation_paths()])
    signature = _fig_seed_signature(seed_config, seed_override)
    cached = _fig_cached_csv(data_path, force, {'n_games': n_games, 'source_signature': source_signature, 'seed_signature': signature})
    if cached is not None:
        return cached
    rows = []
    work_dir.mkdir(parents=True, exist_ok=True)
    for market in _fig_MARKET_ORDER:
        code = MARKETS[market]
        ranking_train = _fig_require_file(artifact_dir(run_dir, 'rankings') / f'{market}_{DQN_RANKER}_train3.csv', 'three-year LambdaRank training output')
        ranking_test = _fig_require_file(artifact_dir(run_dir, 'rankings') / f'{market}_{DQN_RANKER}_test3.csv', 'three-year LambdaRank test output')
        train_seed = stage_seed(code, 3, 'dqn', seed_config, seed_override)
        eval_seed = stage_seed(code, 3, 'evaluation', seed_config, seed_override)
        for lr in _fig_LEARNING_RATES:
            checkpoint = work_dir / f'{market}_DQN_lr_{lr:g}.pt'
            train_dqn(market, 3, ranking_train, checkpoint, lr=lr, n_games=n_games, seed=train_seed)
            metrics = evaluate_dqn(market, 3, ranking_test, checkpoint, lr=lr, seed=eval_seed, fixed_actions=False)
            rows.append({'market': market, 'learning_rate': lr, 'ARR': metrics['ARR'], 'n_games': n_games, 'source_signature': source_signature, 'seed_signature': signature})
            print(f"Figure 3(b): {market} lr={lr:g} ARR={metrics['ARR']:.6f}")
    return _fig_save_csv(pd.DataFrame(rows), data_path)

def _fig_compute_mart_sensitivity(data_path: Path, seed_config: dict, seed_override: int | None, tree_method: str, force: bool) -> pd.DataFrame:
    signature = _fig_seed_signature(seed_config, seed_override)
    cached = _fig_cached_csv(data_path, force, {'tree_method': tree_method, 'seed_signature': signature})
    if cached is not None:
        return cached
    rows = []
    for market in _fig_MARKET_ORDER:
        code = MARKETS[market]
        base = PAPER_HYPERPARAMETERS['LambdaMART'][code]
        seed = stage_seed(code, 3, 'mart', seed_config, seed_override)
        grids: Iterable[tuple[str, Iterable[float | int]]] = (('learning_rate', _fig_LEARNING_RATES), ('n_estimators', _fig_MART_ESTIMATORS), ('max_depth', _fig_MART_DEPTHS))
        for (parameter, values) in grids:
            for value in values:
                params = dict(base)
                params[parameter] = value
                (_, ranked) = _fig_fit_ranker_variant(market, 'LambdaMART', learning_rate=float(params['learning_rate']), max_depth=int(params['max_depth']), n_estimators=int(params['n_estimators']), seed=seed, tree_method=tree_method)
                (metrics, _) = backtest_predictions(ranked)
                rows.append({'market': market, 'parameter': parameter, 'value': value, 'ARR': metrics['ARR'], 'tree_method': tree_method, 'seed_signature': signature})
                print(f"Figure 4: {market} {parameter}={value} ARR={metrics['ARR']:.6f}")
    return _fig_save_csv(pd.DataFrame(rows), data_path)

def _fig_plot_two_market_lines(frame: pd.DataFrame, x_column: str, xlabel: str, output: Path, *, x_order: Iterable | None=None) -> None:
    (fig, ax) = plt.subplots(figsize=(5.8, 3.5))
    order = list(x_order) if x_order is not None else None
    for (market, marker) in (('Main', '^'), ('ChiNext', 's')):
        subset = frame[frame.market == market].copy()
        if order is not None:
            subset['_order'] = subset[x_column].map({value: i for (i, value) in enumerate(order)})
            subset = subset.sort_values('_order')
        else:
            subset = subset.sort_values(x_column)
        labels = [f'{value:g}' if isinstance(value, float) else str(value) for value in subset[x_column]]
        ax.plot(labels, subset.ARR, marker=marker, markersize=5, linewidth=1.5, color=_fig_COLORS[market], label=_fig_MARKET_TITLES[market])
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Annualized Return')
    _fig_style_axis(ax)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    _fig_save_figure(fig, output)

def _fig_draw_market_lines(ax, frame: pd.DataFrame, x_column: str, xlabel: str, *, x_order: Iterable | None=None, panel: str | None=None) -> None:
    order = list(x_order) if x_order is not None else None
    for (market, marker) in (('Main', '^'), ('ChiNext', 's')):
        subset = frame[frame.market == market].copy()
        if order is not None:
            subset['_order'] = subset[x_column].map({value: i for (i, value) in enumerate(order)})
            subset = subset.sort_values('_order')
        else:
            subset = subset.sort_values(x_column)
        labels = [f'{value:g}' if isinstance(value, float) else str(value) for value in subset[x_column]]
        ax.plot(labels, subset.ARR, marker=marker, markersize=5, linewidth=1.5, color=_fig_COLORS[market], label=_fig_MARKET_TITLES[market])
    if panel:
        ax.set_title(panel, loc='left', fontsize=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Annualized Return')
    _fig_style_axis(ax)
    ax.legend(frameon=False, fontsize=8)

def _fig_figure3(run_dir: Path, output_dir: Path, seed_config: dict, seed_override: int | None, tree_method: str, n_games: int, force: bool) -> list[Path]:
    data_dir = output_dir / 'data'
    rank = _fig_compute_rank_sensitivity(data_dir / 'Fig3a_LambdaRank_learning_rate.csv', seed_config, seed_override, tree_method, force)
    dqn = _fig_compute_dqn_lr_sensitivity(run_dir, data_dir / 'Fig3b_DQN_learning_rate.csv', output_dir / 'cache' / 'fig3_dqn_models', seed_config, seed_override, n_games, force)
    paths = [output_dir / 'Fig3a_LambdaRank_learning_rate.png', output_dir / 'Fig3b_DQN_learning_rate.png']
    _fig_plot_two_market_lines(rank, 'learning_rate', 'Learning rate', paths[0], x_order=_fig_LEARNING_RATES)
    _fig_plot_two_market_lines(dqn, 'learning_rate', 'Learning rate', paths[1], x_order=_fig_LEARNING_RATES)
    combined = output_dir / 'Fig3_hyperparameter_comparison.png'
    (fig, axes) = plt.subplots(1, 2, figsize=(11.5, 3.7))
    _fig_draw_market_lines(axes[0], rank, 'learning_rate', 'Learning rate', x_order=_fig_LEARNING_RATES, panel='(a) LambdaRank')
    _fig_draw_market_lines(axes[1], dqn, 'learning_rate', 'Learning rate', x_order=_fig_LEARNING_RATES, panel='(b) LTR-DQN')
    fig.tight_layout()
    _fig_save_figure(fig, combined)
    return paths + [combined]

def _fig_figure4(output_dir: Path, seed_config: dict, seed_override: int | None, tree_method: str, force: bool) -> list[Path]:
    frame = _fig_compute_mart_sensitivity(output_dir / 'data' / 'Fig4_LambdaMART_hyperparameters.csv', seed_config, seed_override, tree_method, force)
    specs = (('learning_rate', 'Learning rate', _fig_LEARNING_RATES, 'Fig4a_LambdaMART_learning_rate.png'), ('n_estimators', 'Number of weak learners', _fig_MART_ESTIMATORS, 'Fig4b_LambdaMART_weak_learners.png'), ('max_depth', 'Maximum tree depth', _fig_MART_DEPTHS, 'Fig4c_LambdaMART_max_depth.png'))
    paths = []
    for (parameter, xlabel, order, filename) in specs:
        path = output_dir / filename
        _fig_plot_two_market_lines(frame[frame.parameter == parameter], 'value', xlabel, path, x_order=order)
        paths.append(path)
    combined = output_dir / 'Fig4_LambdaMART_hyperparameters.png'
    (fig, axes) = plt.subplots(1, 3, figsize=(16.5, 3.8))
    for (ax, (parameter, xlabel, order, _), panel) in zip(axes, specs, ('(a) Learning rate', '(b) Weak learners', '(c) Tree depth')):
        _fig_draw_market_lines(ax, frame[frame.parameter == parameter], 'value', xlabel, x_order=order, panel=panel)
    fig.tight_layout()
    _fig_save_figure(fig, combined)
    return paths + [combined]

def _fig_index_curve(market: str) -> pd.DataFrame:
    code = MARKETS[market]
    frame = pd.read_csv(DATA_DIR / f'{code}merge_T4.csv')
    raw_dates = frame['qid_date'].astype('string').str.replace('\\.0$', '', regex=True)
    if raw_dates.str.match('^\\d{8}$').mean() > 0.8:
        dates = pd.to_numeric(raw_dates, errors='coerce')
    else:
        dates = pd.to_numeric(pd.to_datetime(raw_dates, errors='coerce').dt.strftime('%Y%m%d'), errors='coerce')
    result = pd.DataFrame({'qid_date': dates, 'funds': frame['total_profit']})
    return result.dropna().astype({'qid_date': int})

def _fig_baseline_portfolio_curve(market: str) -> pd.DataFrame:
    code = MARKETS[market]
    frame = pd.read_csv(DATA_DIR / f'{code}merge_open_close_final.csv', usecols=['qid_date', 'stock_code', 'real_return', 'close', 'pclose'])
    frame = frame[(frame.qid_date >= TEST_START) & (frame.qid_date <= TEST_END)].copy()
    frame['prediction'] = 0.0
    return backtest_predictions(frame, all_stocks=True, initial_capital=1000000)[1]

def _fig_normalize_curve(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame[['qid_date', 'funds']].copy()
    result['funds'] = pd.to_numeric(result['funds'], errors='coerce')
    result = result.dropna().sort_values('qid_date', kind='mergesort')
    if result.empty:
        raise ValueError('Cannot normalize an empty daily fund curve')
    result['wealth'] = result.funds / result.funds.iloc[0]
    return result[['qid_date', 'wealth']]

def _fig_compute_daily_curves(run_dir: Path, data_path: Path, seed_config: dict, seed_override: int | None, force: bool) -> pd.DataFrame:
    source_paths = []
    for market in _fig_MARKET_ORDER:
        source_paths.extend([DATA_DIR / f'{MARKETS[market]}merge_T4.csv', DATA_DIR / f'{MARKETS[market]}merge_open_close_final.csv', DATA_DIR / 'dapan' / f'{MARKETS[market]}merge.csv', artifact_dir(run_dir, 'rankings') / f'{market}_LambdaRank_test3.csv', artifact_dir(run_dir, 'rankings') / f'{market}_LambdaMART_test3.csv', artifact_dir(run_dir, 'models') / f'{market}_DQN_train3.pt'])
    source_signature = _fig_file_signature([*source_paths, *_fig_implementation_paths()])
    signature = _fig_seed_signature(seed_config, seed_override)
    cached = _fig_cached_csv(data_path, force, {'source_signature': source_signature, 'seed_signature': signature})
    action_paths = [data_path.parent / f'Fig6_{market}_daily_actions.csv' for market in _fig_MARKET_ORDER]
    if cached is not None and all((path.is_file() for path in action_paths)):
        return cached
    if cached is not None:
        print('Daily curve cache is incomplete; regenerating the missing Figure 6 actions.')
    rows = []
    for market in _fig_MARKET_ORDER:
        curves: dict[str, pd.DataFrame] = {_fig_INDEX_NAMES[market]: _fig_index_curve(market), 'Baseline portfolio': _fig_baseline_portfolio_curve(market)}
        code = MARKETS[market]
        for model in _fig_BASELINES:
            ranked = baseline_ranking(run_dir, market, 3, model, stage_seed(code, 3, 'baseline', seed_config, seed_override))
            curves[model] = backtest_predictions(ranked, classifier=model.endswith('_C'))[1]
        for model in ('LambdaRank', 'LambdaMART'):
            ranked_path = _fig_require_file(artifact_dir(run_dir, 'rankings') / f'{market}_{model}_test3.csv', f'{model} test ranking')
            curves[model] = backtest_predictions(pd.read_csv(ranked_path))[1]
        dqn_ranking = _fig_require_file(artifact_dir(run_dir, 'rankings') / f'{market}_{DQN_RANKER}_test3.csv', 'DQN ranking input')
        checkpoint = _fig_require_file(artifact_dir(run_dir, 'models') / f'{market}_DQN_train3.pt', 'DQN checkpoint')
        (_, dqn_daily) = evaluate_dqn(market, 3, dqn_ranking, checkpoint, seed=stage_seed(code, 3, 'evaluation', seed_config, seed_override), return_daily=True, fixed_actions=False)
        curves['LTR-DQN'] = dqn_daily
        for (model, curve) in curves.items():
            normalized = _fig_normalize_curve(curve)
            normalized['market'] = market
            normalized['model'] = model
            normalized['source_signature'] = source_signature
            normalized['seed_signature'] = signature
            rows.extend(normalized.to_dict(orient='records'))
        action_rows = dqn_daily[['qid_date', 'real_action']].copy()
        action_rows['market'] = market
        action_rows.rename(columns={'real_action': 'number_of_stocks'}).to_csv(data_path.parent / f'Fig6_{market}_daily_actions.csv', index=False, encoding='utf-8-sig')
    return _fig_save_csv(pd.DataFrame(rows), data_path)

def _fig_date_values(values: pd.Series) -> pd.Series:
    raw = pd.to_numeric(values, errors='coerce').astype('Int64').astype(str)
    return pd.to_datetime(raw, format='%Y%m%d', errors='coerce')

def _fig_figure5(curves: pd.DataFrame, output_dir: Path) -> list[Path]:
    (fig, axes) = plt.subplots(2, 1, figsize=(11.5, 8.0), sharex=False)
    for (ax, market, panel) in zip(axes, _fig_MARKET_ORDER, ('(a)', '(b)')):
        subset = curves[curves.market == market]
        for (model, group) in subset.groupby('model', sort=False):
            dates = _fig_date_values(group.qid_date)
            linewidth = 2.4 if model == 'LTR-DQN' else 1.15
            color = _fig_COLORS.get(model)
            ax.plot(dates, group.wealth, label=model, linewidth=linewidth, color=color)
        ax.set_title(f'{panel} {_fig_MARKET_TITLES[market]}', loc='left', fontsize=11)
        ax.set_ylabel('Cumulative wealth (initial = 1)')
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.tick_params(axis='x', rotation=30)
        _fig_style_axis(ax)
        ax.legend(ncol=4, fontsize=7, frameon=False, loc='upper left')
    axes[-1].set_xlabel('Trading day')
    fig.tight_layout()
    path = output_dir / 'Fig5_return_curves_all_methods.png'
    _fig_save_figure(fig, path)
    return [path]

def _fig_figure6(curves: pd.DataFrame, output_dir: Path) -> list[Path]:
    (fig, axes) = plt.subplots(4, 1, figsize=(11.0, 8.2), sharex=False, gridspec_kw={'height_ratios': [3, 1, 3, 1]})
    for (index, market) in enumerate(_fig_MARKET_ORDER):
        curve_ax = axes[index * 2]
        action_ax = axes[index * 2 + 1]
        subset = curves[(curves.market == market) & curves.model.isin([_fig_INDEX_NAMES[market], 'LambdaMART', 'LTR-DQN'])]
        for (model, group) in subset.groupby('model', sort=False):
            curve_ax.plot(_fig_date_values(group.qid_date), group.wealth, label=model, linewidth=2.1 if model == 'LTR-DQN' else 1.4, color=_fig_COLORS.get(model, _fig_COLORS['index']))
        curve_ax.set_title(f"({('a' if index == 0 else 'b')}) {_fig_MARKET_TITLES[market]}", loc='left', fontsize=11)
        curve_ax.set_ylabel('Cumulative wealth')
        curve_ax.legend(frameon=False, fontsize=8, ncol=3, loc='upper left')
        _fig_style_axis(curve_ax)
        actions = pd.read_csv(output_dir / 'data' / f'Fig6_{market}_daily_actions.csv')
        dates = _fig_date_values(actions.qid_date)
        action_ax.bar(dates, actions.number_of_stocks, width=1.0, color='#A5A5A5')
        action_ax.set_ylim(0, 4.5)
        action_ax.set_yticks([0, 1, 2, 3, 4])
        action_ax.set_ylabel('Stocks')
        action_ax.set_xlabel('Trading day')
        action_ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        action_ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        action_ax.tick_params(axis='x', rotation=30)
        _fig_style_axis(action_ax)
    fig.tight_layout()
    path = output_dir / 'Fig6_DQN_actions_and_return_curves.png'
    _fig_save_figure(fig, path)
    return [path]

def _fig_minmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    spread = np.nanmax(values) - np.nanmin(values)
    return values - np.nanmin(values) if spread == 0 else (values - np.nanmin(values)) / spread

def _fig_compute_feature_importance(data_path: Path, seed_config: dict, seed_override: int | None, tree_method: str, force: bool) -> pd.DataFrame:
    signature = _fig_seed_signature(seed_config, seed_override)
    cached = _fig_cached_csv(data_path, force, {'tree_method': tree_method, 'seed_signature': signature})
    if cached is not None:
        return cached
    rows = []
    for market in _fig_MARKET_ORDER:
        code = MARKETS[market]
        (train, _) = load_stock_data(market, 3)
        scaler = MinMaxScaler()
        x_train = scaler.fit_transform(train[FEATURES])
        y_scaler = MinMaxScaler()
        y_train = y_scaler.fit_transform(train[['real_return']]).ravel()
        lasso = Lasso(alpha=0.0001)
        lasso.fit(x_train, y_train)
        if tree_method != 'gpu_hist':
            raise ValueError(f'GPU execution is required for feature importance; got tree_method={tree_method!r}')
        resolved_tree_method = 'gpu_hist'
        xgb_model = xgb.XGBRegressor(objective='reg:squarederror', booster='gbtree', tree_method=resolved_tree_method, n_estimators=100, max_depth=4, learning_rate=0.1, subsample=1.0, colsample_bytree=0.8, random_state=stage_seed(code, 3, 'baseline', seed_config, seed_override), n_jobs=1)
        xgb_model.fit(x_train, y_train)
        (rank_model, _) = _fig_fit_ranker_variant(market, 'LambdaRank', learning_rate=PAPER_HYPERPARAMETERS['LambdaRank'][code]['learning_rate'], max_depth=6, n_estimators=100, seed=stage_seed(code, 3, 'rank', seed_config, seed_override), tree_method=tree_method)
        importance = {'LTR-DQN': _fig_minmax(rank_model.feature_importances_), 'LR': _fig_minmax(np.abs(lasso.coef_)), 'XGB_R': _fig_minmax(xgb_model.feature_importances_)}
        for (model, values) in importance.items():
            for (feature, value) in zip(FEATURES, values):
                rows.append({'market': market, 'model': model, 'feature': feature, 'importance': float(value), 'tree_method': tree_method, 'seed_signature': signature})
    return _fig_save_csv(pd.DataFrame(rows), data_path)

def _fig_figure7(output_dir: Path, seed_config: dict, seed_override: int | None, tree_method: str, force: bool) -> list[Path]:
    frame = _fig_compute_feature_importance(output_dir / 'data' / 'Fig7_feature_importance.csv', seed_config, seed_override, tree_method, force)
    (fig, axes) = plt.subplots(1, 2, figsize=(13.0, 8.2), sharex=True)
    model_colors = {'LTR-DQN': '#4472C4', 'LR': '#ED7D31', 'XGB_R': '#70AD47'}
    for (ax, market, panel) in zip(axes, _fig_MARKET_ORDER, ('(a)', '(b)')):
        market_data = frame[frame.market == market]
        y_offset = 0
        ticks = []
        labels = []
        for model in ('LTR-DQN', 'LR', 'XGB_R'):
            top = market_data[market_data.model == model].nlargest(5, 'importance')
            top = top.sort_values('importance', ascending=True)
            positions = np.arange(5) + y_offset
            ax.barh(positions, top.importance, color=model_colors[model], label=model)
            ticks.extend(positions)
            labels.extend(top.feature)
            y_offset += 6
        ax.set_yticks(ticks)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_title(f'{panel} {_fig_MARKET_TITLES[market]}', loc='left', fontsize=11)
        ax.set_xlabel('Normalized feature importance')
        _fig_style_axis(ax, grid_axis='x')
        ax.legend(frameon=False, fontsize=8, loc='lower right')
    fig.tight_layout()
    path = output_dir / 'Fig7_feature_importance.png'
    _fig_save_figure(fig, path)
    return [path]

def _fig_main() -> None:
    args = _fig_parse_args()
    validate_runtime()
    figures = _fig_selected_figures(args.figures)
    run_dir = args.run_dir.resolve()
    output_dir = (args.output_dir or artifact_dir(run_dir, 'results') / 'figures').resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'data').mkdir(parents=True, exist_ok=True)
    seed_config = load_stage_seed_config(args.seed_config)
    generated: list[Path] = []
    if 3 in figures:
        generated.extend(_fig_figure3(run_dir, output_dir, seed_config, args.seed, args.ranker_tree_method, args.n_games, args.force))
    if 4 in figures:
        generated.extend(_fig_figure4(output_dir, seed_config, args.seed, args.ranker_tree_method, args.force))
    curves = None
    if any((number in figures for number in (5, 6))):
        curves = _fig_compute_daily_curves(run_dir, output_dir / 'data' / 'Fig5_Fig6_daily_curves.csv', seed_config, args.seed, args.force)
    if 5 in figures:
        generated.extend(_fig_figure5(curves, output_dir))
    if 6 in figures:
        generated.extend(_fig_figure6(curves, output_dir))
    if 7 in figures:
        generated.extend(_fig_figure7(output_dir, seed_config, args.seed, args.ranker_tree_method, args.force))
    manifest = {'scope': 'main-text empirical Figures 3-7; excludes Figures 1-2 and Appendix C', 'run_dir': str(run_dir), 'output_dir': str(output_dir), 'figures': figures, 'seed': args.seed, 'seed_config': str(args.seed_config.resolve()) if args.seed_config else 'built-in', 'ranker_tree_method': args.ranker_tree_method, 'n_games': args.n_games, 'runtime': runtime_versions(), 'outputs': {path.name: sha256(path) for path in generated}, 'data': {path.name: sha256(path) for path in sorted((output_dir / 'data').glob('*.csv'))}}
    manifest_path = output_dir / 'figures_manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(manifest, indent=2))
import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable
import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from model import CODE_DIR, DATA_DIR, FEES, MARKETS, STAMP_TAX, TEST_END, TEST_START, artifact_dir, runtime_versions, sha256, validate_runtime
_appendix_MARKET_ORDER = ('Main', 'ChiNext')
_appendix_MARKET_TITLES = {'Main': 'Main board market', 'ChiNext': 'ChiNext market'}
_appendix_INDEX_NAMES = {'Main': 'CSI 300 Index', 'ChiNext': 'ChiNext Index'}
_appendix_MARKET_CODES = {'Main': '0060', 'ChiNext': '3068'}
_appendix_INITIAL_CAPITAL = 1000000.0
_appendix_LONG_START = 20171206
_appendix_MODELS = ('LambdaRank', 'LambdaMART', 'LTR-DQN')
_appendix_RATES = (0.5, 0.6, 0.7, 0.8, 0.9)
_appendix_COLORS = {'Main': '#2F5597', 'ChiNext': '#D28E00', 'index': '#777777', 'Baseline portfolio': '#4472C4', 'No ESG': '#C00000', 'NS 25%': '#E6A700', 'NS 50%': '#ED7D31', 'PI 25%': '#70AD47', 'PI 50%': '#264478', 'LambdaRank': '#2AA6C8', 'LambdaMART': '#ED7D31', 'LTR-DQN': '#A6A6A6'}

def _appendix_parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Recompute and export appendix Figures C1-C5')
    parser.add_argument('--run_dir', type=Path, default=CODE_DIR, help='Artifacts created by train.py; default uses code_1_final/temp and model')
    parser.add_argument('--output_dir', type=Path, default=CODE_DIR / 'results' / 'appendix_figures')
    parser.add_argument('--figures', default='C1,C2,C3,C4,C5', help='Comma-separated subset of C1,C2,C3,C4,C5')
    parser.add_argument('--t6_csv', type=Path, default=CODE_DIR / 'temp' / 't6_runs' / 't6_raw.csv', help='500-seed selected raw results used by Figure C4')
    parser.add_argument('--broker_file', type=Path, default=None, help='Optional report-level CSV containing a true brokerage identifier')
    parser.add_argument('--broker_column', default=None, help='Brokerage identifier column; default is broker_id when present, else broker_size proxy')
    parser.add_argument('--min_broker_reports', type=int, default=100, help='Minimum report observations required for one Figure C2 group')
    parser.add_argument('--force', action='store_true', help='Ignore cached appendix data')
    return parser.parse_args()

def _appendix_selected_figures(value: str) -> list[str]:
    result = []
    for item in value.split(','):
        label = item.strip().upper()
        if label and (not label.startswith('C')):
            label = f'C{label}'
        if label and label not in result:
            result.append(label)
    invalid = sorted(set(result) - {'C1', 'C2', 'C3', 'C4', 'C5'})
    if not result or invalid:
        raise ValueError(f'figures must be a subset of C1,C2,C3,C4,C5; invalid={invalid}')
    return result

def _appendix_require_file(path: Path, purpose: str) -> Path:
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f'{purpose} not found: {path}')
    return path

def _appendix_digest_text(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(',', ':'), default=str)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()

def _appendix_file_signature(paths: Iterable[Path]) -> str:
    resolved = [_appendix_require_file(path, 'appendix figure source') for path in paths]
    return _appendix_digest_text({str(path): sha256(path) for path in resolved})

def _appendix_implementation_paths() -> list[Path]:
    return [CODE_DIR / 'main.py', CODE_DIR / 'model.py']

def _appendix_cached_csv(path: Path, source_signature: str, force: bool) -> pd.DataFrame | None:
    if not path.is_file() or force:
        return None
    frame = pd.read_csv(path)
    if frame.empty or 'source_signature' not in frame.columns:
        return None
    if not (frame.source_signature.astype(str) == source_signature).all():
        print(f'Ignoring appendix cache generated from different sources: {path}')
        return None
    print(f'Using cached appendix data: {path}')
    return frame

def _appendix_save_csv(frame: pd.DataFrame, path: Path, source_signature: str) -> pd.DataFrame:
    result = frame.copy()
    result['source_signature'] = source_signature
    path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(path, index=False, encoding='utf-8-sig')
    return result

def _appendix_save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'Saved: {path.resolve()}')

def _appendix_style_axis(ax, grid_axis: str='y') -> None:
    ax.set_facecolor('white')
    ax.grid(axis=grid_axis, color='#D9D9D9', linewidth=0.7, alpha=0.85)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color('#A6A6A6')
        spine.set_linewidth(0.8)

def _appendix_to_int_dates(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_numeric(series.dt.strftime('%Y%m%d'), errors='coerce')
    raw = series.astype('string').str.strip().str.replace('\\.0$', '', regex=True)
    numeric = pd.to_numeric(raw, errors='coerce')
    serial = numeric.dropna()
    if not serial.empty and serial.between(30000, 60000).mean() > 0.8:
        dates = pd.to_datetime(numeric, unit='D', origin='1899-12-30', errors='coerce')
        return pd.to_numeric(dates.dt.strftime('%Y%m%d'), errors='coerce')
    compact = raw.str.replace('-', '', regex=False).str.replace('/', '', regex=False)
    eight_digit = compact.str.fullmatch('\\d{8}', na=False)
    result = pd.Series(np.nan, index=series.index, dtype='float64')
    result.loc[eight_digit] = pd.to_numeric(compact.loc[eight_digit], errors='coerce')
    remaining = ~eight_digit
    parsed = pd.to_datetime(raw.loc[remaining], errors='coerce')
    result.loc[remaining] = pd.to_numeric(parsed.dt.strftime('%Y%m%d'), errors='coerce')
    return result

def _appendix_as_datetime(series: pd.Series) -> pd.Series:
    raw = pd.to_numeric(series, errors='coerce').astype('Int64').astype(str)
    return pd.to_datetime(raw, format='%Y%m%d', errors='coerce')

def _appendix_normalize_funds(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame[['qid_date', 'funds']].dropna().sort_values('qid_date', kind='mergesort')
    if result.empty:
        return result
    first = float(result.funds.iloc[0])
    if first != 0:
        result['funds'] = result.funds / first * _appendix_INITIAL_CAPITAL
    return result

def _appendix_trade_selected(selected: pd.DataFrame, capital: float, commission: float=FEES, stamp_tax: float=STAMP_TAX) -> tuple[float, int, int]:
    if selected.empty:
        return (capital, 0, 0)
    allocation = capital / len(selected)
    total = 0.0
    wins = 0
    traded = 0
    for (_, row) in selected.iterrows():
        pclose = pd.to_numeric(row.get('pclose'), errors='coerce')
        close = pd.to_numeric(row.get('close'), errors='coerce')
        if pd.isna(pclose) or pd.isna(close) or pclose <= 0:
            total += allocation
            continue
        lots = int(allocation / (100 * pclose))
        purchase_fee = lots * 100 * pclose * commission
        shares = int((allocation - purchase_fee) / (100 * pclose)) * 100
        cash = allocation - purchase_fee - shares * pclose
        sell = shares * close - shares * close * (commission + stamp_tax) + cash
        total += sell
        traded += 1
        wins += int(sell > allocation)
    return (total, wins, traded)

def _appendix_backtest(frame: pd.DataFrame, *, selection: str, actions: dict[int, int] | None=None, commission: float=FEES, stamp_tax: float=STAMP_TAX) -> tuple[pd.DataFrame, int, int]:
    capital = _appendix_INITIAL_CAPITAL
    rows = []
    total_wins = total_trades = 0
    ordered = frame.sort_values('qid_date', kind='mergesort')
    for (date, group) in ordered.groupby('qid_date', sort=True):
        if selection == 'all':
            selected = group
        elif selection == 'top4':
            selected = group.nlargest(min(4, len(group)), 'prediction')
        elif selection == 'actions':
            top_n = int((actions or {}).get(int(date), 0))
            selected = group.iloc[0:0] if top_n <= 0 else group.nlargest(min(top_n, len(group)), 'prediction')
        else:
            raise ValueError(f'Unknown selection mode: {selection}')
        before = capital
        (capital, wins, trades) = _appendix_trade_selected(selected, capital, commission, stamp_tax)
        total_wins += wins
        total_trades += trades
        rows.append({'qid_date': int(date), 'funds': capital, 'day_return': (capital - before) / before if before else np.nan, 'number_of_stocks': len(selected)})
    return (pd.DataFrame(rows), total_wins, total_trades)

def _appendix_curve_metrics(curve: pd.DataFrame, wins: int, trades: int) -> dict[str, float]:
    if curve.empty:
        return {name: np.nan for name in ('ARR', 'MDR', 'CR', 'SR', 'WR')}
    arr = (curve.funds.iloc[-1] / _appendix_INITIAL_CAPITAL) ** (242 / len(curve)) - 1
    drawdown = (curve.funds - curve.funds.cummax()) / curve.funds.cummax()
    mdr = -float(drawdown.min())
    cr = arr / mdr if mdr else np.nan
    std = curve.day_return.std()
    sr = ((1 + curve.day_return.mean()) ** 242 - 1 - 0.025) / (std * 242 ** 0.5) if std else np.nan
    wr = wins / trades if trades else np.nan
    return {'ARR': float(arr), 'MDR': mdr, 'CR': float(cr), 'SR': float(sr), 'WR': float(wr)}

def _appendix_load_actions(run_dir: Path, market: str) -> tuple[dict[int, int], Path]:
    path = _appendix_require_file(artifact_dir(run_dir, 'actions') / f'{market}_DQN_actions3.csv', f'{market} DQN actions (run main.py/T7main.py first if missing)')
    frame = pd.read_csv(path)
    action_col = 'action' if 'action' in frame.columns else 'real_action'
    if action_col not in frame.columns:
        raise ValueError(f'No action column in {path}: {frame.columns.tolist()}')
    frame['qid_date'] = _appendix_to_int_dates(frame.qid_date)
    frame[action_col] = pd.to_numeric(frame[action_col], errors='coerce').fillna(0).astype(int)
    return (dict(zip(frame.qid_date.dropna().astype(int), frame.loc[frame.qid_date.notna(), action_col])), path)

def _appendix_dqn_ranking_path(run_dir: Path, market: str) -> Path:
    return _appendix_require_file(artifact_dir(run_dir, 'rankings') / f'{market}_LambdaRank_test3.csv', f'{market} LambdaRank test ranking used by DQN')

def _appendix_index_curve(market: str, start: int, end: int) -> tuple[pd.DataFrame, Path]:
    code = _appendix_MARKET_CODES[market]
    candidates = [DATA_DIR / f'{code}merge.csv', DATA_DIR / 'dapan' / f'{code}merge.csv']
    path = next((p for p in candidates if p.is_file()), None)
    if path is None:
        raise FileNotFoundError(f'Index curve source not found for {market}: {candidates}')
    frame = pd.read_csv(path)
    date_col = 'qid_date' if 'qid_date' in frame.columns else 'trade_date'
    frame['qid_date'] = _appendix_to_int_dates(frame[date_col])
    fund_col = 'total_profit' if 'total_profit' in frame.columns else None
    if fund_col is None:
        raise ValueError(f'Index source lacks total_profit: {path}')
    frame['funds'] = pd.to_numeric(frame[fund_col], errors='coerce')
    frame = frame[(frame.qid_date >= start) & (frame.qid_date <= end)]
    return (_appendix_normalize_funds(frame), path)

def _appendix_baseline_curve(market: str, start: int, end: int) -> tuple[pd.DataFrame, Path]:
    path = _appendix_require_file(DATA_DIR / f'{_appendix_MARKET_CODES[market]}merge_open_close_final.csv', 'stock data')
    frame = pd.read_csv(path, usecols=['qid_date', 'stock_code', 'close', 'pclose'])
    frame['qid_date'] = _appendix_to_int_dates(frame.qid_date)
    frame = frame[(frame.qid_date >= start) & (frame.qid_date <= end)]
    (curve, _, _) = _appendix_backtest(frame, selection='all')
    return (_appendix_normalize_funds(curve), path)

def _appendix_compute_c1(data_path: Path, force: bool) -> pd.DataFrame:
    sources = [DATA_DIR / f'{code}merge.csv' for code in _appendix_MARKET_CODES.values()] + [DATA_DIR / f'{code}merge_open_close_final.csv' for code in _appendix_MARKET_CODES.values()] + _appendix_implementation_paths()
    signature = _appendix_file_signature(sources)
    cached = _appendix_cached_csv(data_path, signature, force)
    if cached is not None:
        return cached
    rows = []
    for market in _appendix_MARKET_ORDER:
        (index, _) = _appendix_index_curve(market, _appendix_LONG_START, TEST_END)
        (baseline, _) = _appendix_baseline_curve(market, _appendix_LONG_START, TEST_END)
        for (model, curve) in ((_appendix_INDEX_NAMES[market], index), ('Baseline portfolio', baseline)):
            part = curve.copy()
            part['market'] = market
            part['model'] = model
            rows.extend(part.to_dict(orient='records'))
    return _appendix_save_csv(pd.DataFrame(rows), data_path, signature)

def _appendix_plot_c1(frame: pd.DataFrame, path: Path) -> None:
    (fig, axes) = plt.subplots(2, 1, figsize=(11.2, 6.8), sharex=False)
    for (ax, market, panel) in zip(axes, _appendix_MARKET_ORDER, ('(a)', '(b)')):
        subset = frame[frame.market == market]
        for (model, group) in subset.groupby('model', sort=False):
            color = _appendix_COLORS['Baseline portfolio'] if model == 'Baseline portfolio' else _appendix_COLORS['index']
            ax.plot(_appendix_as_datetime(group.qid_date), group.funds / 1000000, label=model, linewidth=1.7, color=color)
        ax.set_title(f'{panel} {_appendix_MARKET_TITLES[market]}', loc='left', fontsize=11)
        ax.set_ylabel('Total fund (million)')
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.tick_params(axis='x', rotation=30)
        ax.legend(frameon=False, fontsize=8, loc='upper left')
        _appendix_style_axis(ax)
    axes[-1].set_xlabel('Trading day')
    fig.tight_layout()
    _appendix_save_figure(fig, path)

def _appendix_broker_source(args: argparse.Namespace, market: str) -> tuple[pd.DataFrame, Path, str, str]:
    path = args.broker_file or DATA_DIR / f'{_appendix_MARKET_CODES[market]}merge_open_close_final.csv'
    path = _appendix_require_file(path, 'brokerage report data')
    frame = pd.read_csv(path)
    if 'market' in frame.columns and args.broker_file:
        frame = frame[frame.market.astype(str).str.lower().str.contains(market.lower())]
    requested = args.broker_column
    if requested:
        if requested not in frame.columns:
            raise ValueError(f'Broker column {requested!r} not found in {path}')
        column = requested
        mode = 'true_identifier' if requested not in {'broker_size', 'broker_status'} else 'proxy'
    elif 'broker_id' in frame.columns:
        (column, mode) = ('broker_id', 'true_identifier')
    elif 'brokerage_id' in frame.columns:
        (column, mode) = ('brokerage_id', 'true_identifier')
    elif 'broker_size' in frame.columns:
        (column, mode) = ('broker_size', 'proxy')
    else:
        raise ValueError(f'No brokerage identifier in {path}. Pass --broker_file and --broker_column.')
    return (frame, path, column, mode)

def _appendix_compute_c2(data_path: Path, args: argparse.Namespace) -> pd.DataFrame:
    source_paths = []
    source_meta = []
    loaded = {}
    for market in _appendix_MARKET_ORDER:
        (frame, path, column, mode) = _appendix_broker_source(args, market)
        loaded[market] = (frame, column, mode)
        source_paths.append(path)
        source_meta.append((market, column, mode, args.min_broker_reports))
    signature = _appendix_digest_text({'files': _appendix_file_signature([*source_paths, *_appendix_implementation_paths()]), 'settings': source_meta})
    cached = _appendix_cached_csv(data_path, signature, args.force)
    if cached is not None:
        return cached
    rows = []
    required = {'qid_date', 'stock_code', 'close', 'pclose'}
    for market in _appendix_MARKET_ORDER:
        (frame, broker_column, mode) = loaded[market]
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f'C2 source for {market} is missing {missing}')
        frame = frame.copy()
        frame['qid_date'] = _appendix_to_int_dates(frame.qid_date)
        frame = frame[(frame.qid_date >= _appendix_LONG_START) & (frame.qid_date <= TEST_END)]
        frame = frame.dropna(subset=[broker_column, 'qid_date', 'pclose', 'close'])
        counts = frame[broker_column].value_counts()
        keep = counts[counts >= args.min_broker_reports].index
        for (broker, group) in frame[frame[broker_column].isin(keep)].groupby(broker_column, sort=True):
            (curve, wins, trades) = _appendix_backtest(group, selection='all')
            metrics = _appendix_curve_metrics(curve, wins, trades)
            rows.append({'market': market, 'broker_group': str(broker), 'broker_column': broker_column, 'broker_grouping_mode': mode, 'n_reports': len(group), **metrics})
    result = pd.DataFrame(rows)
    if result.empty:
        raise ValueError('No brokerage group satisfies --min_broker_reports')
    return _appendix_save_csv(result, data_path, signature)

def _appendix_plot_c2(frame: pd.DataFrame, path: Path) -> None:
    metrics = [('ARR', 1.0), ('MDR', 10.0), ('CR', 1.0), ('SR', 1.0), ('WR', 10.0)]
    positions = np.arange(len(metrics), dtype=float)
    width = 0.28
    (fig, ax) = plt.subplots(figsize=(10.5, 5.2))
    for (offset, market, color) in ((-width / 1.5, 'Main', '#8EC5BD'), (width / 1.5, 'ChiNext', '#F2EFA6')):
        values = [pd.to_numeric(frame.loc[frame.market == market, name], errors='coerce').dropna() * scale for (name, scale) in metrics]
        bp = ax.boxplot(values, positions=positions + offset, widths=width, patch_artist=True, manage_ticks=False, showfliers=True)
        for box in bp['boxes']:
            box.set_facecolor(color)
            box.set_edgecolor('#666666')
        for element in ('whiskers', 'caps', 'medians'):
            for artist in bp[element]:
                artist.set_color('#666666')
        bp['boxes'][0].set_label(_appendix_MARKET_TITLES[market])
    ax.set_xticks(positions, ['ARR', 'MDRx10', 'CR', 'SR', 'WRx10'])
    ax.set_xlabel('Evaluation metrics')
    ax.set_ylabel('Value')
    mode = ', '.join(sorted(frame.broker_grouping_mode.unique()))
    if mode != 'true_identifier':
        ax.set_title('Brokerage-performance proxy groups (broker_size)', loc='left', fontsize=10)
    ax.legend(frameon=True, fontsize=8, loc='upper left')
    _appendix_style_axis(ax)
    fig.tight_layout()
    _appendix_save_figure(fig, path)

def _appendix_compute_c3(run_dir: Path, data_path: Path, force: bool) -> pd.DataFrame:
    sources = []
    for market in _appendix_MARKET_ORDER:
        (_, action_path) = _appendix_load_actions(run_dir, market)
        sources.extend([action_path, _appendix_dqn_ranking_path(run_dir, market)])
    signature = _appendix_file_signature([*sources, *_appendix_implementation_paths()])
    cached = _appendix_cached_csv(data_path, signature, force)
    if cached is not None:
        return cached
    settings = (('fee=0.00%, tax=0.00%', 0.0, 0.0), ('fee=0.01%, tax=0.10%', 0.0001, 0.001), ('fee=0.03%, tax=0.10%', 0.0003, 0.001), ('fee=0.05%, tax=0.10%', 0.0005, 0.001))
    rows = []
    for market in _appendix_MARKET_ORDER:
        (actions, _) = _appendix_load_actions(run_dir, market)
        ranked = pd.read_csv(_appendix_dqn_ranking_path(run_dir, market))
        ranked['qid_date'] = _appendix_to_int_dates(ranked.qid_date)
        ranked = ranked[(ranked.qid_date >= TEST_START) & (ranked.qid_date <= TEST_END)]
        for (label, commission, tax) in settings:
            (curve, _, _) = _appendix_backtest(ranked, selection='actions', actions=actions, commission=commission, stamp_tax=tax)
            curve['market'] = market
            curve['scenario'] = label
            curve['commission'] = commission
            curve['stamp_tax'] = tax
            rows.extend(curve.to_dict(orient='records'))
    return _appendix_save_csv(pd.DataFrame(rows), data_path, signature)

def _appendix_plot_c3(frame: pd.DataFrame, path: Path) -> None:
    scenario_colors = ('#FFC000', '#A5A5A5', '#4472C4', '#ED7D31')
    (fig, axes) = plt.subplots(2, 1, figsize=(10.8, 6.8), sharex=False)
    for (ax, market, panel) in zip(axes, _appendix_MARKET_ORDER, ('(a)', '(b)')):
        subset = frame[frame.market == market]
        for (color, (scenario, group)) in zip(scenario_colors, subset.groupby('scenario', sort=False)):
            ax.plot(_appendix_as_datetime(group.qid_date), group.funds / 1000000, label=scenario, color=color, linewidth=1.5)
        ax.set_title(f'{panel} {_appendix_MARKET_TITLES[market]}', loc='left', fontsize=11)
        ax.set_ylabel('Total fund (million)')
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.tick_params(axis='x', rotation=30)
        ax.legend(frameon=False, fontsize=7, loc='upper left')
        _appendix_style_axis(ax)
    axes[-1].set_xlabel('Trading day')
    fig.tight_layout()
    _appendix_save_figure(fig, path)

def _appendix_compute_c4(t6_csv: Path, data_path: Path, force: bool) -> pd.DataFrame:
    t6_csv = _appendix_require_file(t6_csv, 'T6 500-seed selected results')
    signature = _appendix_file_signature([t6_csv, *_appendix_implementation_paths()])
    cached = _appendix_cached_csv(data_path, signature, force)
    if cached is not None:
        return cached
    frame = pd.read_csv(t6_csv)
    if 'sampling_rate' not in frame.columns and 'rate' in frame.columns:
        frame['sampling_rate'] = pd.to_numeric(frame.rate, errors='coerce')
    required = {'market', 'sampling_rate', 'model', 'seed', 'ARR'}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f'T6 input is missing {missing}: {t6_csv}')
    frame = frame[frame.market.isin(_appendix_MARKET_ORDER) & frame.model.isin(_appendix_MODELS) & frame.sampling_rate.isin(_appendix_RATES)].copy()
    counts = frame.groupby(['market', 'sampling_rate', 'model']).size()
    incomplete = counts[counts < 500]
    if not incomplete.empty:
        raise ValueError(f'Figure C4 requires 500 results per cell; incomplete:\n{incomplete}')
    return _appendix_save_csv(frame, data_path, signature)

def _appendix_plot_c4(frame: pd.DataFrame, path: Path) -> None:
    (fig, axes) = plt.subplots(2, 1, figsize=(10.6, 7.2), sharex=False)
    positions = np.arange(len(_appendix_RATES), dtype=float)
    width = 0.22
    offsets = (-width, 0.0, width)
    for (ax, market, panel) in zip(axes, _appendix_MARKET_ORDER, ('(a)', '(b)')):
        subset = frame[frame.market == market]
        for (model, offset) in zip(_appendix_MODELS, offsets):
            values = [pd.to_numeric(subset[(subset.model == model) & (subset.sampling_rate == rate)].ARR, errors='coerce').dropna() for rate in _appendix_RATES]
            bp = ax.boxplot(values, positions=positions + offset, widths=width * 0.9, patch_artist=True, manage_ticks=False, showfliers=True, flierprops={'markersize': 2.2, 'markerfacecolor': '#555555', 'markeredgecolor': '#555555'})
            for box in bp['boxes']:
                box.set_facecolor(_appendix_COLORS[model])
                box.set_edgecolor('#555555')
            for element in ('whiskers', 'caps', 'medians'):
                for artist in bp[element]:
                    artist.set_color('#555555')
            bp['boxes'][0].set_label(model)
        ax.set_xticks(positions, [f'{int(rate * 100)}%' for rate in _appendix_RATES])
        ax.set_title(f'{panel} {_appendix_MARKET_TITLES[market]}', loc='left', fontsize=11)
        ax.set_ylabel('Annualized return')
        ax.legend(frameon=True, fontsize=8, loc='upper right')
        _appendix_style_axis(ax)
    axes[-1].set_xlabel('Sampling rate')
    fig.tight_layout()
    _appendix_save_figure(fig, path)

def _appendix_esg_curve(frame: pd.DataFrame, actions: dict[int, int], *, threshold: float | None, prefilter: bool) -> pd.DataFrame:
    capital = _appendix_INITIAL_CAPITAL
    rows = []
    for (date, group) in frame.sort_values('qid_date', kind='mergesort').groupby('qid_date', sort=True):
        top_n = int(actions.get(int(date), 0))
        if top_n <= 0:
            selected = group.iloc[0:0]
        elif threshold is None:
            selected = group.nlargest(min(top_n, len(group)), 'prediction')
        elif prefilter:
            eligible = group[group.ESG >= threshold]
            selected = eligible.nlargest(min(top_n, len(eligible)), 'prediction')
        else:
            selected = group.nlargest(min(top_n, len(group)), 'prediction')
            selected = selected[selected.ESG >= threshold]
        before = capital
        (capital, _, _) = _appendix_trade_selected(selected, capital)
        rows.append({'qid_date': int(date), 'funds': capital, 'day_return': (capital - before) / before if before else np.nan, 'number_of_stocks': len(selected)})
    return pd.DataFrame(rows)

def _appendix_compute_c5(run_dir: Path, data_path: Path, force: bool) -> pd.DataFrame:
    sources = []
    for market in _appendix_MARKET_ORDER:
        (_, action_path) = _appendix_load_actions(run_dir, market)
        sources.extend([action_path, DATA_DIR / 'ESG' / f'{_appendix_MARKET_CODES[market]}temp_test_ndcg_train3_esg.csv', DATA_DIR / f'{_appendix_MARKET_CODES[market]}merge.csv', DATA_DIR / f'{_appendix_MARKET_CODES[market]}merge_open_close_final.csv'])
    signature = _appendix_file_signature([*sources, *_appendix_implementation_paths()])
    cached = _appendix_cached_csv(data_path, signature, force)
    if cached is not None:
        return cached
    rows = []
    for market in _appendix_MARKET_ORDER:
        (actions, _) = _appendix_load_actions(run_dir, market)
        esg_path = _appendix_require_file(DATA_DIR / 'ESG' / f'{_appendix_MARKET_CODES[market]}temp_test_ndcg_train3_esg.csv', f'{market} ESG ranking data')
        esg = pd.read_csv(esg_path)
        esg['qid_date'] = _appendix_to_int_dates(esg.qid_date)
        esg = esg[(esg.qid_date >= TEST_START) & (esg.qid_date <= TEST_END)].copy()
        (index, _) = _appendix_index_curve(market, TEST_START, TEST_END)
        (baseline, _) = _appendix_baseline_curve(market, TEST_START, TEST_END)
        curves = {_appendix_INDEX_NAMES[market]: index, 'Baseline portfolio': baseline, 'No ESG': _appendix_esg_curve(esg, actions, threshold=None, prefilter=False), 'NS 25%': _appendix_esg_curve(esg, actions, threshold=5.52, prefilter=False), 'NS 50%': _appendix_esg_curve(esg, actions, threshold=6.02, prefilter=False), 'PI 25%': _appendix_esg_curve(esg, actions, threshold=5.52, prefilter=True), 'PI 50%': _appendix_esg_curve(esg, actions, threshold=6.02, prefilter=True)}
        for (strategy, curve) in curves.items():
            part = _appendix_normalize_funds(curve)
            part['market'] = market
            part['strategy'] = strategy
            rows.extend(part.to_dict(orient='records'))
    return _appendix_save_csv(pd.DataFrame(rows), data_path, signature)

def _appendix_plot_c5(frame: pd.DataFrame, path: Path) -> None:
    (fig, axes) = plt.subplots(2, 1, figsize=(11.0, 7.0), sharex=False)
    for (ax, market, panel) in zip(axes, _appendix_MARKET_ORDER, ('(a)', '(b)')):
        subset = frame[frame.market == market]
        for (strategy, group) in subset.groupby('strategy', sort=False):
            color = _appendix_COLORS.get(strategy, _appendix_COLORS['index'])
            ax.plot(_appendix_as_datetime(group.qid_date), group.funds / 1000000, label=strategy, color=color, linewidth=1.5)
        ax.set_title(f'{panel} {_appendix_MARKET_TITLES[market]}', loc='left', fontsize=11)
        ax.set_ylabel('Total fund (million)')
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.tick_params(axis='x', rotation=30)
        ax.legend(frameon=False, fontsize=7, ncol=4, loc='upper left')
        _appendix_style_axis(ax)
    axes[-1].set_xlabel('Trading day')
    fig.tight_layout()
    _appendix_save_figure(fig, path)

def _appendix_main() -> None:
    args = _appendix_parse_args()
    validate_runtime()
    figures = _appendix_selected_figures(args.figures)
    run_dir = args.run_dir.resolve()
    output_dir = args.output_dir.resolve()
    data_dir = output_dir / 'data'
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    data_outputs: list[Path] = []
    notes = {'C1': 'Long-horizon index and all-report baseline portfolio curves.', 'C2': 'Uses a true brokerage identifier when supplied; otherwise broker_size is an explicitly labelled proxy.', 'C3': 'Current DQN daily stock counts and current LambdaRank ranking, re-backtested under four fee settings.', 'C4': 'Uses the fixed 500-seed-per-cell T6 selected-result ledger.', 'C5': 'Current DQN daily stock counts applied to the supplied ESG ranking data.'}
    if 'C1' in figures:
        data_path = data_dir / 'FigC1_long_horizon_curves.csv'
        frame = _appendix_compute_c1(data_path, args.force)
        path = output_dir / 'FigC1_baseline_portfolio_and_indices.png'
        _appendix_plot_c1(frame, path)
        outputs.append(path)
        data_outputs.append(data_path)
    if 'C2' in figures:
        data_path = data_dir / 'FigC2_brokerage_performance.csv'
        frame = _appendix_compute_c2(data_path, args)
        path = output_dir / 'FigC2_brokerage_performance_boxplots.png'
        _appendix_plot_c2(frame, path)
        outputs.append(path)
        data_outputs.append(data_path)
        notes['C2_grouping_mode'] = sorted(frame.broker_grouping_mode.unique().tolist())
        notes['C2_grouping_column'] = sorted(frame.broker_column.unique().tolist())
    if 'C3' in figures:
        data_path = data_dir / 'FigC3_transaction_cost_curves.csv'
        frame = _appendix_compute_c3(run_dir, data_path, args.force)
        path = output_dir / 'FigC3_transaction_cost_sensitivity.png'
        _appendix_plot_c3(frame, path)
        outputs.append(path)
        data_outputs.append(data_path)
    if 'C4' in figures:
        data_path = data_dir / 'FigC4_sampling_ARR.csv'
        frame = _appendix_compute_c4(args.t6_csv, data_path, args.force)
        path = output_dir / 'FigC4_sampling_robustness_boxplots.png'
        _appendix_plot_c4(frame, path)
        outputs.append(path)
        data_outputs.append(data_path)
    if 'C5' in figures:
        data_path = data_dir / 'FigC5_ESG_curves.csv'
        frame = _appendix_compute_c5(run_dir, data_path, args.force)
        path = output_dir / 'FigC5_ESG_strategy_curves.png'
        _appendix_plot_c5(frame, path)
        outputs.append(path)
        data_outputs.append(data_path)
    manifest = {'scope': 'appendix empirical Figures C1-C5', 'run_dir': str(run_dir), 'output_dir': str(output_dir), 'figures': figures, 'runtime': runtime_versions(), 'notes': {key: value for (key, value) in notes.items() if key in figures or key.startswith('C2_')}, 'outputs': {path.name: sha256(path) for path in outputs}, 'data': {path.name: sha256(path) for path in data_outputs}}
    manifest_path = output_dir / 'appendix_figures_manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True), encoding='utf-8')
    print(json.dumps(manifest, indent=2, ensure_ascii=True))
if __name__ == '__main__':
    main()
