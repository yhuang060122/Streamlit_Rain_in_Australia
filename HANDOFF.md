# 项目交接文档 (HANDOFF.md)

> **项目名称**：Rain in Australia — XGBoost 多目标预测与 Streamlit 交互式应用  
> **代码仓库**：`C:\Users\yanhu\source\Streamlit_Rain_in_Australia`  
> **最后更新时间**：2026-09-30  
> **运行环境**：Windows, Python 3.13（项目内 `.venv`）, pandas 3.0.6, xgboost 3.4.1, scikit-learn 1.9.1, streamlit 1.64.0

---

## 1. 我们在做什么任务 (Project Overview)

本项目基于 Kaggle 经典数据集 **「Rain in Australia」**（`weatherAUS.csv`，来源 BOM 澳洲气象局），构建一套可运行的机器学习预测应用。当前数据取自官方 Rattle 镜像 `https://rattle.togaware.com/weatherAUS.csv`，共 **49 个气象站、275,410 行、2007-11 至 2026-01** 的逐日观测（比 Kaggle 冻结版 14.5 万行更新，覆盖到 2026 年）。

不同于常见单目标版本，本项目的预测范围扩展到 **4 个目标**：

| 目标 | 类型 | 含义 |
|---|---|---|
| `RainTomorrow` | 二分类 | 明天是否下雨 (J+1) |
| `RainInTwoDays` | 二分类 | 后天是否下雨 (J+2) |
| `MaxTempTomorrow` | 回归 | 明天最高温 (J+1) |
| `MaxTempInTwoDays` | 回归 | 后天最高温 (J+2) |

项目来源：`docs/` 目录是一份**法语团队作业**（主报告、7 个 notebook、重采样分析、决策邮件），本版本将其定案后的方案落地为：**一条 XGBoost 训练管道 + 一个 Streamlit 交互应用**。

---

## 2. 已经完成了什么 (Completed Work)

### 2.1 需求梳理与方案定案
- 通读 `docs/` 下法语报告（`Rapport_Projet_Meteo_Australie.docx`）、重采样分析（`Analyse_Reechantillonnage.docx`）与团队邮件（`Discussion project.txt`），锁定最终决策：
  1. 范围扩展为 **4 个目标**（雨 J+1/J+2 + 最高温 J+1/J+2）。
  2. **保留高缺失列**（Evaporation/Sunshine/Cloud9am/Cloud3pm，缺失 38%~48%），不做删除——交给 XGBoost 原生处理 NaN。
  3. 模型只用 **XGBoost**；J+1 与 J+2 因删行不同需用**各自的数据集**，不可混用。

### 2.2 数据管线（[`src/pipeline.py`](file:///c:/Users/yanhu/source/Streamlit_Rain_in_Australia/src/pipeline.py)）
- `load_raw()`：读取 CSV，用 `na_values=["NA"]` 正确识别缺失标记。
- `engineer()`：二进制化 `RainToday/RainTomorrow`；按站点排序后派生季节特征；用 `groupby('Location').shift(-1/-2)` 并结合日期间隔判断，构造 `RainInTwoDays`、`MaxTempTomorrow`、`MaxTempInTwoDays` 四个目标。
- **排除 `RISK_MM` 列**（它是 `RainTomorrow` 的泄漏版：`RISK_MM>0 ⟺ RainTomorrow=Yes`）。
- `cast_categories()`：用固定 `pd.CategoricalDtype` 锁定风向(16)、季节(4)、站点(49)的类别顺序，保证训练/预测 category code 一致。
- 训练/预测均**不手动填补缺失值**，依赖 XGBoost `enable_categorical=True` + `tree_method='hist'` 原生处理 NaN 与分类变量。

### 2.3 模型训练（4 个 XGBoost）
- 80/20 分层切分（分类目标 `stratify=y`），固定 `random_state=42`。
- 分类模型带 `scale_pos_weight`（按训练集正负比自动计算），回归模型无。
- 固定超参（未寻优）：`n_estimators=400, learning_rate=0.05, max_depth=6, subsample=0.85, colsample_bytree=0.85, min_child_weight=5, reg_lambda=5, reg_alpha=0.1`。
- **测试集指标**（22 个特征）：

| 目标 | 类型 | 样本(train/test) | 指标 |
|---|---|---|---|
| RainTomorrow | 分类 | 213,853 / 53,464 | ROC-AUC **0.904** · F1 0.672 · Recall 0.80 · Acc 0.828 |
| RainInTwoDays | 分类 | 213,480 / 53,371 | ROC-AUC 0.780 · F1 0.510 · Recall 0.71 |
| MaxTempTomorrow | 回归 | 216,435 / 54,109 | MAE **1.95 °C** · RMSE 2.70 · R² 0.857 |
| MaxTempInTwoDays | 回归 | 216,246 / 54,062 | MAE 2.49 °C · RMSE 3.37 · R² 0.777 |

- 产物保存到 `models/`：4 个 `.joblib` + `meta.json`（含特征列、分类类别、站点列表、指标、明日降雨模型特征重要度）。

### 2.4 Streamlit 交互应用（[`app.py`](file:///c:/Users/yanhu/source/Streamlit_Rain_in_Australia/app.py)）
- **📅 历史数据回放**：选站点 + 日期 → 展示当日实况（温度/降雨/湿度/气压/是否已下雨）→ 预测 4 目标（降雨概率进度条 + 最高温）→ 与实际值对照（降雨 ✅/❌、气温误差）。
- **✍️ 手动输入**：填写当日 20 项气象指标 → 实时预测（分类字段下拉、数值字段带默认值）。
- **特征重要度**：展示明日降雨模型 Top 15 特征。
- 侧栏「模型与指标」面板汇总 4 目标测试指标。

### 2.5 修复原始 notebook 的 bug
- 原始 `Australia_meteo_model_step4.ipynb` 在 `MaxTempTomorrow`/`MaxTempInTwoDays` 的 imputation 里误把 `X1_train` 用于 `X3_train`/`X4_train`（跨数据集取数，索引错乱）。本版本移除了手动 imputation，从根上规避了该问题。

### 2.6 自动化验证
- 使用 `streamlit.testing.v1.AppTest` 对 `app.py` 做无头渲染测试：历史模式与手动模式（含点击预测）均 **0 异常**通过。

---

## 3. 当前状态与未竟事宜 (Current Status & Blockers)

* **代码质量与运行状态**：管线与 App 均可运行，模型已训练完成并落盘，AppTest 端到端 0 异常。Streamlit 服务为本地启动、**非持久运行**（会话结束后需手动重启）。
* **已知问题 / 待改进**：
  1. **切分方式为随机分层而非时序切分**：当前 `train_test_split`（`stratify=y`，随机洗牌）会引入轻微时序泄漏——模型可能用到"未来的气象"预测"过去的降雨"。离线指标可能偏高，真实业务需改为按 `Date` 时序切分。
  2. **未做超参寻优**：当前 4 个模型使用固定经验参数，尚未引入 `RandomizedSearchCV`/Optuna。
  3. **未做概率校准**：`scale_pos_weight` 会使输出概率发生偏移，`predict_proba` 的绝对概率不宜直接当真实降雨概率解读，需 Isotonic/Platt 校准。
  4. **无可解释性（SHAP）**：仅用 XGBoost 内置 `feature_importances_`，无 TreeSHAP 全局/局部归因。
  5. **未构造时序滞后特征**：特征仅含当天值 + 季节 + J+1/J+2 目标，未引入 `Rainfall_Lag1`、近 3 日累计降水等滞后/滚动特征。
  6. **手动输入默认值为全局固定值**：未按站点取典型值，输入体验与合理性一般。
  7. **原始材料仍为法语**：`docs/` 报告与 notebook 未翻译，可读性受限（本 HANDOFF 已提炼关键结论）。
  8. **数据版本漂移**：训练用 Rattle 官方镜像（至 2026-01），与 Kaggle 冻结版（至 2017）行数/范围不同，复现原报告数字时需注意。

---

## 4. 下一步计划 (Next Steps & Roadmap)

1. **改为时序切分 (Temporal Split)**：
   - 按 `Location` 分组、`Date` 排序，用时间顺序切分 Train / Val / Test（如 70/15/15），替代随机 `train_test_split`，消除时序泄漏。
2. **引入自动化超参寻优**：
   - 对两个分类模型基于 `RandomizedSearchCV`（PR-AUC 评分）做快速采样寻优；回归模型用负 MAE 评分。
3. **概率校准 (Probability Calibration)**：
   - 对分类模型引入 `CalibratedClassifierCV`（Isotonic/Platt），使 `predict_proba` 可作为真实降雨概率；App 阈值与文案据此调整。
4. **SHAP 可解释性**：
   - 引入 `shap` 库，新增全局 Beeswarm 与单样本 Waterfall，替换/补充当前的简单特征重要度条形图。
5. **时序滞后与滚动特征**：
   - 构建 `Rainfall_Lag1`、`MaxTemp_Lag1`、`Rainfall_Roll3d_Sum`、`Pressure_Trend` 等，进一步提升 J+1/J+2 预测力。
6. **按站点默认值与更友好的输入**：
   - 手动输入模式按站点预填典型值；考虑将 20 个字段分组折叠，减少视觉负担。
7. **模型评估深化页**：
   - 增加混淆矩阵、ROC/PR 曲线、按站点/季节聚合的评估视图。

---

## 5. 踩过的坑与前车之鉴 (Lessons Learned & Pitfalls to Avoid)

> [!CAUTION]
> **以下为本次开发中实测踩到的坑，后续迭代务必注意！**

1. **pandas 3.0 下 `pd.DataFrame([Series])` 不再保留 category dtype**：
   - *现象*：把单行 `Series` 用 `pd.DataFrame([row])` 重建后，分类列塌缩成 `str`/`float64`，传给 XGBoost 报错 `DataFrame.dtypes ... must be int,float,bool or category`。
   - *防范*：预测时直接传 DataFrame 切片（`df[mask]`，保留 dtype），**不要** `iloc[0]` 后再 `pd.DataFrame([...])` 重建；并用固定 `pd.CategoricalDtype(categories=...)` 显式锁定类别顺序。
2. **xgboost 3.4 `enable_categorical` 要求训练/预测 category code 一致**：
   - *现象*：若不固定类别顺序，单行预测时类别 code 与训练时不一致，预测结果错乱。
   - *防范*：风向/季节/站点统一用 `pd.CategoricalDtype(categories=[固定列表])` 构造 dtype，训练与预测共用同一套 categories。
3. **泄漏列 `RISK_MM` 必须排除**：
   - *现象*：Rattle 镜像含 `RISK_MM`（明日降雨量 mm），与 `RainTomorrow` 完全等价，若当特征会得到虚假满分。
   - *防范*：特征列表中显式剔除 `RISK_MM`。
4. **原始 notebook 的跨数据集 imputation bug**：
   - *现象*：`Australia_meteo_model_step4.ipynb` 中温度回归目标的 imputation 误用 `X1_train` 取数（应为 `X3_train`/`X4_train`），索引错乱。
   - *防范*：copy-paste 后务必核对变量名；本版本直接去掉手动 imputation，交给 XGBoost 处理 NaN。
5. **新版本库的弃用 API**：
   - pandas 3.0：`pd.api.types.is_categorical_dtype` 弃用 → 改 `isinstance(dtype, pd.CategoricalDtype)`。
   - scikit-learn 1.9：`mean_squared_error(squared=False)` 弃用 → 改 `root_mean_squared_error`。
   - streamlit 1.64：`use_container_width` 弃用 → 改 `width="stretch"`。
6. **Windows 环境与虚拟环境路径**：
   - 运行脚本务必用 `.venv\Scripts\python.exe`（本机项目内 `.venv`），直接用 `python` 会落到系统 Python、缺三方库报错。
   - 下载数据时 `curl` 在 Git Bash 下正常，注意 CSV 里缺失标记是字符串 `"NA"`（非空），读取必须 `na_values=["NA"]`。

---

## 6. 常用命令速查 (Quick Reference Commands)

```bash
# 1. 安装依赖（首次）
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt

# 2. （重新）训练并保存 4 个模型到 models/（约 1.5 分钟）
.venv/Scripts/python.exe -m src.pipeline

# 3. 启动 Streamlit 本地服务
.venv/Scripts/python.exe -m streamlit run app.py

# 4. 无头渲染冒烟测试（验证 App 无异常）
.venv/Scripts/python.exe -c "from streamlit.testing.v1 import AppTest; at=AppTest.from_file('app.py'); at.run(); print('exceptions:', len(at.exception))"
```

---

## 附：目录结构

```
Streamlit_Rain_in_Australia/
├── app.py                  # Streamlit 交互界面（历史回放 + 手动输入）
├── src/
│   ├── pipeline.py         # 预处理 + 特征工程 + 训练 + 保存模型
│   └── __init__.py
├── data/weatherAUS.csv     # 原始数据（Rattle 官方镜像，49 站 2007–2026）
├── models/                 # 4 个 .joblib + meta.json
├── docs/                   # 原始法语项目材料（报告 + 7 notebooks + 邮件）
├── requirements.txt
└── README.md
```
