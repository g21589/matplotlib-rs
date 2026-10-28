# 官方 Matplotlib 的 Python 建圖優化實驗

本實驗直接在官方 **Matplotlib 3.11.2** 上套用 Python／NumPy 補丁，
檢驗大量箱形圖的建圖成本能否在沿用官方繪圖核心時明顯降低。
補丁只影響啟用它的 Python 行程；一般回傳型別仍是官方 `Line2D` 與 `PathPatch`。
這是針對規則數值資料的實驗補丁，尚未合併至 Matplotlib 上游。

## 比較方法

五個版本使用完全相同的工作負載與資料：

- `official`：未修改的官方 Matplotlib。
- `statistics_only`：僅加入規則 float64 矩陣的批次統計。
- `artists_only`：僅加入樣式 prototype、獨立 artist、callback 延後配置與座標範圍批次更新。
- `python_optimized`：上述兩項同時啟用。
- `rust_0_0_2`：matplotlib-rs 0.0.2，作為同批量測的對照。

所有官方版本均保留官方 Agg 繪製與 Pillow 編碼；執行時檢查沒有載入 `matplotlib_rs._mplrs`。
每次建立 30 組 × 180 箱、每箱 100 個常態隨機樣本（共 540,000 個值），
seed=20261026，每組不同顏色，保留離群點、whiskers、caps 與 median。
單一 Axes、36 × 12 吋、100 DPI、3,600 × 1,200 像素。
每次必須保留 **5,400 個 PathPatch 與 32,400 個 Line2D**。

每版預熱一次、不納入統計，再各量測五次，全部使用全新 Python 行程依序執行。
五輪以輪轉順序安排，使每個版本在五個順序位置各出現一次。
資料產生不計入建圖時間，沒有關閉垃圾回收，也沒有將多個箱形圖合併成 collection。
官方優化版會在建圖階段完成所有線段的座標陣列與 Path，
並在第一次 draw 前驗證 32,400 條線的 Path 已準備完成，避免將成本延後至繪製階段。
這項驗證的時間另存為 `pre_draw_check_seconds`，不計入建圖或繪製時間。
Rust 對照的幾何已在建圖階段寫入原生 handle，不以 Python `_path` 快取作此項檢查。

全部版本明確使用 **Pillow `compress_level=6`** 儲存 PNG，
因此存檔時間不受 matplotlib-rs 原生 `Fastest` 壓縮設定影響。
同時核對資料 SHA256、RGBA SHA256、物件數與儲存檔案 SHA256。

環境：CPython 3.12.14、NumPy 2.3.5、Pillow 13.0.0、Linux x86_64、
Intel Xeon Platinum 8573C、容器 CPU quota 2 核、記憶體上限 8 GiB。
使用專案 no-hinting matplotlibrc 與 Agg。正式量測期間未平行執行其他測試或編譯。

## 結果

以下為五輪牆鐘時間中位數，建圖欄括號內為最小～最大值。
「建立＋首次繪製」由各輪兩階段相加後取中位數，不能直接相加其他欄位的中位數。

| 版本 | 建圖（秒） | 首次繪製（秒） | 建圖＋首次繪製（秒） | 最高 RSS（MiB） | 建圖倍率 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 官方原版 | 17.063（15.961～21.523） | 3.732 | 20.795 | 513.1 | 1.00× |
| 官方＋僅批次統計 | 18.204（15.462～31.878） | 3.787 | 22.683 | 513.1 | 0.94× |
| 官方＋僅物件流程優化 | 2.612（2.318～3.392） | 3.168 | 5.802 | 294.2 | 6.53× |
| 官方＋兩項 Python／NumPy 優化 | 2.225（1.849～3.421） | 3.785 | 5.887 | 294.3 | **7.67×** |
| matplotlib-rs 0.0.2 | 0.973（0.948～2.106） | 0.237 | 1.196 | 251.0 | 17.54× |

官方版合併優化的五輪個別建圖倍率為 **10.74×、7.67×、6.02×、8.76×、6.75×**。
建圖＋首次繪製中位數加速 **3.53×**，最高 RSS 中位數減少 **42.7%**。
首次繪製的中位數仍約 3.7～3.8 秒，與已完成 Path 的檢查一致，沒有把建圖成本藏到 draw 階段。

單獨批次統計的整體建圖中位數未加速，且有一次 31.878 秒的明顯波動。
這五輪不足以推論該變更造成穩定退步，也不能宣稱它單獨明顯改善整體建圖；
目前最明確的主要收益是反覆樣式／artist 初始化、裁切與座標範圍處理的合併。
在先優化物件流程後，合併統計優化的建圖中位數比僅物件優化再少約 14.8%。
各版本存在波動，這項細部差異不宜解讀為精確的成本占比。

30 次執行（含預熱）的資料與 RGBA 指紋全部一致，
PNG 不僅像素相同，**儲存位元組亦完全一致，每張 492,545 bytes**。
所有官方行程均未載入本專案的 Rust extension。

結果顯示，這個規律負載的建圖收益可以由官方實作的 Python／NumPy 流程優化取得。
Rust 對照仍在建圖與繪製管線較快；例如與官方合併優化版相比，首次繪製中位數約快 16 倍。
這是不同實作管線的實測差距，不能直接當作 Rust 與 C++ 的語言速度比。

[完整五輪 JSON](official-boxplot-optimization.json) 保留所有正式輪次、預熱與指紋。
本地原始 PNG、CSV 與 stderr 位於 `tests/output/official-boxplot-comparison/`。

## 正確性與範圍

`tests/bench/test_official_boxplot_fastpath.py` 包含四項測試方法，均通過：

- 186 組統計輸入／選項組合逐值比較，含 float32／整數、NaN、masked、空資料、
  ragged input、percentile whiskers、autorange 與 bootstrap；不符合條件時沿用原流程。
- 24 組圖像／物件行為情境，檢查垂直／水平、notch、mean、隱藏元件、不同寬度、
  labels、callback 獨立性、sticky edges、資料／樣式修改、刪除、relim、縮放／移動 Axes、
  log scale 與 property cycle，並比較像素、資料範圍、labels 與 dtype。
- 自訂 Axes 的 `plot` hook、自訂 transform 會使用一般流程；另檢查 callback 與整張 Figure 的 pickle。
- 相同案例的 PNG／SVG／PDF／EPS 全部內容一致。

補丁在特定數值條件下共用已驗證的樣式與變換，仍建立獨立的 artist、資料、Path、
clip bounds、callback 與 sticky edges。自訂 Axes、單位、transform、複雜樣式等使用原始流程。
這些測試不代表 Matplotlib 上游完整測試集合均已執行，也不代表所有圖形都會獲得相同倍率。

## 重跑方式

僅比較官方版本，無需安裝本專案的 Rust extension：

```bash
python -m pip install 'matplotlib==3.11.2' 'numpy==2.3.5' 'pillow==13.0.0'
python tests/bench/test_official_boxplot_fastpath.py
python tests/bench/official_boxplot_compare.py \
    --output tests/output/official-boxplots-python-only \
    --rounds 5 \
    --variants official statistics_only artists_only python_optimized
```

包含本次 Rust 對照的五版本比較：

```bash
sh scripts/dev_build.sh --locked
python tests/bench/official_boxplot_compare.py \
    --output tests/output/official-boxplots-five-variants \
    --rounds 5
```

每次使用新的輸出目錄。結果包含 `results.json`、`timings.csv`、`summary.md`、
各輪 PNG 與 stderr；JSON 保留五輪原始值、最小／最大值、程式碼 SHA256 與環境設定。

在自己的官方 Matplotlib 繪圖行程試用補丁：

```python
import sys
sys.path.insert(0, "tests/bench")
import official_boxplot_fastpath

official_boxplot_fastpath.install()
import matplotlib.pyplot as plt
# 接著沿用一般 ax.boxplot(...)。

# 不再使用時可還原：
official_boxplot_fastpath.uninstall()
```
