# 澳大利亚降雨与气温预测 (Streamlit)

基于 Kaggle 数据集 [weather-dataset-rattle-package](https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package)（BOM 澳洲气象局，49 站点、2007–2026 逐日观测）的交互式预测应用。

## 预测目标（4 个）
| 目标 | 类型 | 含义 |
|---|---|---|
| `RainTomorrow` | 二分类 | 明天是否下雨 (J+1) |
| `RainInTwoDays` | 二分类 | 后天是否下雨 (J+2) |
| `MaxTempTomorrow` | 回归 | 明天最高温 (J+1) |
| `MaxTempInTwoDays` | 回归 | 后天最高温 (J+2) |

## 方法
- 模型：XGBoost（`enable_categorical` 原生处理分类变量与 NaN）。
- **不手动填补缺失值**，保留高缺失列（Evaporation/Sunshine/Cloud 等），交给 XGBoost 处理。
- 已修复原 step4 notebook 的跨数据集 imputation bug（`X1_train` 误用于 `X3_train`/`X4_train`）。
- 排除了 `RISK_MM`（`RainTomorrow` 的泄漏版）。

## 快速开始
```bash
# 1. 安装依赖（首次）
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt

# 2. 训练并保存模型（约几分钟）
.venv/Scripts/python.exe -m src.pipeline

# 3. 启动应用
.venv/Scripts/python.exe -m streamlit run app.py
```

## 目录结构
```
data/weatherAUS.csv     # 原始数据（rattle.togaware.com 官方镜像）
src/pipeline.py         # 预处理 + 训练 + 保存模型
models/                 # 训练产物（4 个 .joblib + meta.json）
app.py                  # Streamlit 界面
docs/                   # 原始项目材料（法语报告 + notebooks）
```

## 应用功能
- **历史数据回放**：选站点 + 日期，展示当日实况、预测明日/后日降雨概率与最高温，并与实际值对照。
- **手动输入**：填写当日天气，实时预测。
- 特征重要度可视化（明日降雨模型 Top 15）。
