# Python 3.10／Matplotlib 3.8.2 箱形圖優化

本版將官方 Matplotlib 的 Python／NumPy 建圖實驗補丁移植至
**CPython 3.10／Matplotlib 3.8.2**，沿用官方 Agg 與 Pillow 繪製及編碼。
同一個 `official_boxplot_fastpath.py` 同時支援原先的 3.11.2 實驗。
補丁在呼叫 `install()` 的行程內生效，不需要本專案 Rust extension。

## 安裝與使用

在儲存庫根目錄建立獨立環境，依賴完整固定於
[`requirements-official-3.8.2.txt`](../../tests/bench/requirements-official-3.8.2.txt)：

```bash
python3.10 -m venv .venv-mpl382
. .venv-mpl382/bin/activate
python -m pip install -r tests/bench/requirements-official-3.8.2.txt
python tests/bench/test_official_boxplot_fastpath.py
```

在自己的程式中啟用：

```python
import sys
sys.path.insert(0, "tests/bench")  # 儲存庫根目錄執行；亦可使用此目錄的絕對路徑。
import official_boxplot_fastpath
import matplotlib.pyplot as plt
import numpy as np

official_boxplot_fastpath.install()
fig, ax = plt.subplots()
ax.boxplot(np.random.default_rng(17).normal(size=(100, 180)),
           patch_artist=True, vert=False)
fig.savefig("boxplots.png")
official_boxplot_fastpath.uninstall()
```

也可使用 `with official_boxplot_fastpath.enabled():` 限定生效區段。
僅支援明確驗證的 Matplotlib 3.8.2／3.11.2；其他版本會拒絕啟用。
此相容版本是官方 Matplotlib 的實驗補丁；matplotlib-rs 0.0.2 的發布套件仍要求 Python >=3.11。

## 3.8.2 的相容調整

- 保留原版 `Axes.boxplot`／`Axes.bxp` 簽名，水平圖使用 `vert=False`，資料標籤使用 `labels`。
- 沿用 3.8.2 的 artist `_childN` labels，不加入新版 bxp 的圖例預設值。
- 使用 3.8.2 的 property cycle 結構，保留尚未指定樣式時的原始循環行為。
- 每個 artist 分別配置 3.8.2 所需的 sticky edges pair；資料、Path、callbacks 及 clip bounds 仍獨立。
- 標籤數量錯誤時保留 3.8.2 原本的錯誤訊息；自訂 Axes、transform、單位與複雜樣式沿用原流程。
- 可重複切換僅統計／僅物件／合併優化，各模式不殘留前一模式的設定。

## 效能量測方法

與前次實驗相同：30 組 × 180 箱、每箱 100 個常態隨機樣本，
seed=20261026，各組不同顏色。單一 Axes、36 × 12 吋、100 DPI、3,600 × 1,200 像素。
每次保留 5,400 個 PathPatch 與 32,400 個 Line2D，共 **37,800 個 artists**。
不合併 collection、不停用垃圾回收；資料產生不計入建圖時間。
所有線段 Path 在建圖階段完成，並在首次 draw 前驗證。

比較原版、僅批次統計、僅物件流程優化及合併優化四種官方實作。
各版預熱一次不計入統計，再量測五次；全部使用全新行程依序執行，逐輪輪轉順序。
全部使用相同 Pillow `compress_level=6`；核對資料、RGBA、PNG 指紋與物件數。
這次未加入 Rust 對照，四種實作都使用 Python 3.10 的官方 Matplotlib 3.8.2。

量測環境：CPython 3.10.22、Matplotlib 3.8.2、NumPy 1.26.4、Pillow 10.2.0、
Linux x86_64、Intel Xeon Platinum 8573C、容器 CPU quota 2 核、記憶體上限 8 GiB。
使用 Agg 與專案 no-hinting 設定；量測期間沒有平行執行測試或編譯。

## 五輪結果

下表為牆鐘時間中位數，建圖欄括號內為最小～最大值。
「建圖＋首次繪製」由每輪兩階段相加後取中位數。

| 官方 3.8.2 版本 | 建圖（秒） | 首次繪製（秒） | 建圖＋首次繪製（秒） | 最高 RSS（MiB） | 建圖倍率 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 原版 | 23.240（21.042～31.020） | 4.957 | 28.197 | 596.8 | 1.00× |
| 僅批次統計 | 21.598（18.813～27.415） | 4.836 | 26.361 | 596.4 | 1.08× |
| 僅物件流程優化 | 4.171（3.551～4.620） | 4.784 | 8.909 | 356.7 | 5.57× |
| 合併優化 | 3.033（2.402～3.212） | 5.142 | 8.353 | 356.3 | **7.66×** |

合併優化的建圖＋首次繪製加速 **3.38×**，最高 RSS 中位數減少 **40.3%**。
五輪各自的建圖倍率為 **7.33×、9.68×、9.66×、9.95×、6.69×**。
本次未達建圖中位數 10×，但物件流程優化在 3.8.2 仍有明確改善。
僅統計優化的改善較小，與原版建圖時間的波動範圍重疊。

首次繪製、重繪與 PNG 存檔中位數接近原版，沒有相同程度的加速。
優化沿用官方繪製核心；建圖結束時 32,400 條線的 Path 均已準備完成。
僅物件流程優化有一次首次繪製 10.185 秒的波動，完整原始時間均保留。

24 次執行（四次預熱、20 次正式量測）的資料與 RGBA 指紋完全一致；
PNG 位元組也全部一致，每張 **481,338 bytes**。
所有行程均未載入 `matplotlib_rs._mplrs`。

[完整 JSON](official-boxplot-optimization-3.8.2.json) 包含所有預熱、五輪數據、
檔案指紋、程式碼 SHA256 與環境資訊。`git_commit` 是量測時的基底 commit，
量測中的工作區程式以 `source_sha256` 識別，已核對與本次提交內容一致。
本機 PNG、CSV 與 stderr 位於 `tests/output/official-boxplot-comparison-3.8.2/`。
此倍率以本次 3.8.2 原版為基準；與先前 3.11.2 實驗的 Python、NumPy、Pillow 版本不同，
不可直接將兩次結果解讀為單一套件版本或語言的速度差異。

## 正確性與 CI

兩個環境的 `test_official_boxplot_fastpath.py` 均通過五項測試方法：

- 186 組統計逐值比較，另檢查 labels 與例外訊息；非規則資料沿用原流程。
- 24 組像素／可修改物件情境，含水平、notch、mean、隱藏元件、寬度、labels、
  callback、sticky edges、獨立裁切、修改資料／樣式、刪除、relim、log 與 property cycle。
- API 簽名與優化模式切換保持正確，自訂 Axes／transform 使用原流程。
- 相同案例的 PNG／SVG／PDF／EPS 位元組完全一致，callbacks 與 Figure pickle 後仍可正確繪製。

[`official-boxplot.yml`](../../.github/workflows/official-boxplot.yml) 會在相關變更時測試
Python 3.10／Matplotlib 3.8.2 與 Python 3.12／Matplotlib 3.11.2。
這些是補丁的針對性檢查，尚未執行 Matplotlib 3.8.2 的完整上游測試集合。

## 重跑效能比較

在上述 Python 3.10 環境中執行：

```bash
python tests/bench/official_boxplot_compare.py \
    --output tests/output/official-boxplots-mpl382-new \
    --rounds 5 \
    --variants official statistics_only artists_only python_optimized
```

每次使用新的輸出目錄。輸出包含各輪 PNG、stderr、`results.json`、`timings.csv` 與 `summary.md`，
保留環境版本、來源程式碼 SHA256、完整預熱及正式量測資料。
