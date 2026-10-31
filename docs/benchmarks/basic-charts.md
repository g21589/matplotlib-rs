# 大量資料的基本圖表效能

以 [`tests/bench/basic_charts.py`](../../tests/bench/basic_charts.py) 對 24 種常見基本圖表
各放入大量資料，比較官方 **Matplotlib 3.11.2**、上一版 **matplotlib-rs 0.0.3**（commit `4a2b72c`）
與本分支，找出效能問題並修正。所有修正都必須與官方版本的繪圖結果相同：
每一輪都比對 matplotlib-rs 與 Matplotlib 的 RGBA 畫布，**24 個案例全部逐像素一致**。

## 測試方法

- 每個案例在全新 Python 行程中執行，資料以固定 seed（20261031）產生且不計入時間。
- 圖為 8 × 6 吋、100 DPI（800 × 600 像素），使用專案的 no-hinting matplotlibrc 與 Agg。
- `build`：`plt.subplots()` 加上繪圖呼叫；`draw`：第一次 `canvas.draw()`；
  `build_and_draw` 為同一行程內兩者相加；另記錄第二次 `draw` 與 `savefig` PNG。
- 三個實作各跑 3 輪，每輪交替執行順序，表中為中位數（秒）；倍率為其他實作耗時 ÷ 本版耗時。
- 環境：CPython 3.12.3、NumPy 2.3.5、Pillow 13.0.0、Linux x86_64、
  Intel Xeon Processor @ 2.10GHz、4 vCPU；量測期間未同時進行編譯或其他測試。
  本表量測 commit `57b2a8c`（第 1～8 項修正；第 3 項的查表與第 9 項之後加入）。
  原始數據與每輪時間見 [`basic-charts.json`](basic-charts.json)。

## 結果：建立＋首次繪製

| 圖表與資料量 | Matplotlib | 0.0.3 | 本版 | 對 Matplotlib | 對 0.0.3 |
| --- | ---: | ---: | ---: | ---: | ---: |
| plot，1,000,000 點 | 0.215 | 0.218 | 0.199 | 1.1× | 1.1× |
| plot，20 條 × 100,000 點 | 0.348 | 0.410 | 0.355 | 1.0× | 1.2× |
| plot 'o-'，200,000 點 | 0.173 | 0.102 | 0.087 | 2.0× | 1.2× |
| plot '--'，1,000,000 點 | 0.205 | 0.254 | 0.136 | 1.5× | 1.9× |
| scatter，200,000 點（大小、顏色各異，alpha） | 2.347 | 2.161 | 1.145 | 2.1× | 1.9× |
| scatter，1,000,000 點（同大小、同色） | 0.450 | 0.311 | 0.199 | 2.3× | 1.6× |
| bar，20,000 根 | 10.279 | 13.717 | 0.131 | 78× | **105×** |
| barh，20,000 根（黑色邊框） | 11.798 | 15.332 | 0.195 | 60× | 79× |
| hist，10,000,000 筆、200 bins | 0.274 | 0.258 | 0.049 | 5.6× | 5.2× |
| hist，2,000,000 筆、10,000 bins | 5.488 | 6.389 | 0.076 | 73× | 85× |
| hist step，3 組 × 2,000,000 筆、2,000 bins | 0.295 | 0.216 | 0.083 | 3.6× | 2.6× |
| hist2d，2,000,000 點、500 × 500 | 0.446 | 0.434 | 0.438 | 1.0× | 1.0× |
| hexbin，1,000,000 點、gridsize 150 | 0.374 | 0.190 | 0.170 | 2.2× | 1.1× |
| pie，500 片 | 0.289 | 0.388 | 0.398 | 0.7× | 1.0× |
| imshow，4000 × 4000 | 1.874 | 1.289 | 1.012 | 1.9× | 1.3× |
| pcolormesh，1000 × 1000 | 0.538 | 0.607 | 0.590 | 0.9× | 1.0× |
| fill_between，1,000,000 點 | 0.324 | 0.232 | 0.324 | 1.0× | 0.7× |
| step，1,000,000 點 | 0.232 | 0.288 | 0.254 | 0.9× | 1.1× |
| errorbar，20,000 點 | 0.301 | 0.822 | 0.260 | 1.2× | 3.2× |
| stem，20,000 點 | 0.358 | 0.865 | 0.323 | 1.1× | 2.7× |
| stackplot，8 層 × 200,000 點 | 0.341 | 0.255 | 0.269 | 1.3× | 0.9× |
| contourf，1000 × 1000、20 levels | 1.148 | 1.184 | 1.216 | 0.9× | 1.0× |
| boxplot，1,000 箱 × 200 筆 | 4.901 | 0.210 | 0.198 | 25× | 1.1× |
| violinplot，300 組 × 2,000 筆 | 1.071 | 1.102 | 1.292 | 0.8× | 0.9× |

粗體為 100 倍以上。官方 Matplotlib 與 0.0.3 在同一個案例的時間各輪相差可達一倍
（例如物件很多時的垃圾回收與記憶體配置），倍率以同一批交錯量測的中位數計算。

物件數量造成的問題改善最多：2 萬根長條建立＋繪製從 0.0.3 的 13.7 秒降至 0.131 秒（105 倍，
比官方快 78 倍），1 萬個 bins 的直方圖 6.4 → 0.076 秒（85 倍，比官方快 73 倍）；
只看建圖階段，bar、barh、hist（1 萬 bins）比 0.0.3 快 108～131 倍。
errorbar／stem 的繪製在 0.0.3 比官方慢 3～6 倍，現在比 0.0.3 快 14 倍與 5.6 倍、比官方快 2.2 倍與 1.5 倍。

## 找到的效能問題與修正

### 1. 每根長條都走完整的物件流程（bar、barh、hist）

`Axes.bar` 對每根長條呼叫 `Rectangle` 建構子、`add_patch`（設定 artist 屬性、裁切路徑、
以 `iter_bezier` 逐段計算資料範圍）並完整同步一次 Rust handle，約 **0.8 毫秒／根**；
2 萬根長條建圖 16 秒，比官方還慢。

[`axes/_bar.py`](../../python/matplotlib_rs/axes/_bar.py) 對每種樣式只用一般建構子建立一個
prototype，再以 `Patch.copy_many_rects` 一次複製全部 Rust handle（換上每根長條的
patch transform 與 sticky edge），並複製 prototype 的屬性字典、填入 `_x0`、`_y0`、`_width`、
`_height` 與 `_sticky_edges`。資料範圍一次加入所有長條的四個角點
（min／max／minpos 與加入順序無關，零寬零高的長條照原規則略過）。
回傳的仍是可各自修改的 `Rectangle`；有單位、逐根標籤、自訂 transform 或每根樣式都不同時，沿用原迴圈。
`_convert_dx` 對一般數值陣列也改為一次以 NumPy 做與原迴圈相同的逐元素運算，不再逐一呼叫單位轉換。

### 2. 矩形掃描：每列每邊一個 cell

Agg 的 rasterizer 對每條邊在每個像素列產生一個 cell，再排序後掃描；
2 萬根高而細的長條每次繪製產生約 800 萬個 cell。
rasterizer 現在先暫存路徑最初的 16 條整數線段：若全部水平或垂直（矩形、矩形的描邊等），
就逐列直接由垂直線段算出該列的 cell（水平線段不產生 cell），以同一段掃描程式輸出；
否則依序重送給原本的 `line`。左右邊互相抵消（比次像素還窄的長條）時不再逐列計算。
渲染器只在 rasterizer「有 cell」時清空掃描線儲存，因此是否有 cell 以 Agg 相同的 cell 累計規則重算。

### 3. 混色的三次 64 位元除法

Agg 的 `fixed_blender_rgba_plain` 每個像素做三次整數除法。分子小於 2²⁶、除數不超過 2¹⁶，
改以 Granlund–Montgomery 的 `⌈2⁴²/d⌉` 乘法與位移求得相同的商，
並快取最近一組（alpha、底色 alpha）的乘數；不透明底色在同一段內只需算一次。
每像素 6.5 → 2.2 ns；以全部 2³² 種輸入窮舉驗證結果相同。
64 像素以上、底色不透明的水平段，再依（顏色、alpha）快取三個 256 項的結果表
（以同一算式計算），每個像素只需查表；大量帶邊框長條（barh）的重繪約 0.078 → 0.047 秒。

### 4. LineCollection 的 NumPy borrow 為平方複雜度（errorbar、stem）

`draw_path_collection` 一次取得所有路徑陣列的 `PyReadonlyArray`；這些路徑是同一個陣列的 view，
rust-numpy 對每個新 borrow 都要和所有存活的 borrow 比對，2 萬條線段約 4 億次比較，
**繪製比官方慢 3～6 倍**。現在逐條取得並轉換，同時只有一個 borrow。

### 5. `_path` 模組以 `tolist()` 轉換陣列（plot 建圖）

`is_sorted_and_has_non_nan`、路徑 codes、transform 矩陣與 `get_path_collection_extents`
的 offsets 都先轉成 Python list；100 萬點的 `plot` 建圖因此比官方慢 2～3 倍。
現在直接讀 NumPy 陣列（`is_sorted_and_has_non_nan` 依官方對 int32／int64／float32／float64 分派）。

### 6. 直方圖計數

`hist` 的計數改由 Rust 多執行緒執行 NumPy 的兩種演算法：等寬 bins 的
`((x - first) / (last - first)) * n` 截斷後依邊界 ±1 修正（NumPy 1.25～2.3 相同），
以及任意邊界的比較計數；bin 邊界仍由 NumPy 自己的 `_get_bin_edges` 計算。
只用於 65,536 筆以上、無權重的 float64 資料，結果與 `numpy.histogram` 相同。

### 7. 直線段 patch 的資料範圍

`_update_patch_limits` 對每段路徑建立 `BezierSegment`；只有直線段時，
這些點就是路徑頂點（CLOSEPOLY 的頂點除外），改為直接使用。
曲線、非有限值、格式不正確或 CLOSEPOLY 之後直接接線段的路徑仍用原迴圈。

### 8. 大量 path collection 平行點陣化（scatter、pcolormesh 等）

沒有裁切路徑、hatch 與 sketch 的 `draw_path_collection`（256 條路徑以上，含 `draw_quad_mesh`）：
每次呼叫啟動一組執行緒，各自以設定相同的 rasterizer 執行與原迴圈相同的轉換與點陣化，
記錄原本會送給渲染器的掃描線；主執行緒再依原順序渲染每批 512 × 執行緒數條路徑，
因此每個像素的混色順序不變。執行緒數為可用核心數（最多 8），可用環境變數
`MPLRS_THREADS` 指定（`1` 為單執行緒）。本機 4 vCPU 上點陣化約加速 2.4 倍。

### 9. 其他

渲染器重複使用掃描線緩衝區；由 Python 給定路徑的 patch（包括每根長條）共用同一份 Rust 路徑，
繪製時也不再複製；長條的常數欄位（例如預設的 `bottom=0`）共用同一個 NumPy 純量。

## 正確性驗證

| 項目 | 結果 |
| --- | --- |
| 本測試 24 個案例 × 3 輪 | matplotlib-rs 畫布與 Matplotlib 逐像素一致 |
| `tests/compare/compare.py`（58 個案例，PNG／SVG／PDF／EPS） | 58／58 完全一致 |
| `tests/lowlevel/diff_renderer.py`（12 個 seed × 1,000 次隨機繪製） | 與官方 C++ Agg 逐位元組一致 |
| `tests/lowlevel/diff_path_collection.py`（大量隨機 collection，4 個 seed） | 一致；故意反序渲染時會偵測到差異 |
| 混色窮舉（Rust 測試，2³² 種輸入） | 與 Agg 的除法結果相同 |
| 矩形掃描（Rust 測試，3 個 seed × 50 萬個隨機形狀，含裁切、描邊、兩種填色規則、三種掃描線） | 與通用掃描相同；舊版的「有無 cell」錯誤會被偵測 |
| `tests/unit/test_bar_batch.py` | 15 種 bar／hist 設定與原迴圈及 Matplotlib 相同；長條可各自修改 |
| `tests/unit/test_histogram.py` | 124 種資料與 bins 組合與 `numpy.histogram` 相同 |
| 其他既有單元與底層測試 | 全部通過 |

## 限制

- 100 倍以上只出現在長條數量造成的問題（與 0.0.3 相比）；與官方相比，長條與直方圖 bins 為 60～78 倍。
  剩下的成本主要是建立 2 萬個與 Matplotlib 相同的 `Rectangle` 物件（每個約 1.6 KB 的屬性字典、
  首次配置記憶體的分頁成本與之後的垃圾回收），以及大量帶邊框長條的混色。
  官方 Matplotlib 原本就以 C++ Agg 處理的工作（折線的路徑簡化、每個散點各自點陣化、
  影像重採樣、等高線計算）兩者成本相近，這些案例只有約 1～3 倍；
  每張圖的固定成本約 10 毫秒，也限制了 0.3 秒以下案例的倍率。
- pie、violinplot、hist2d、contourf 等主要成本在 Python／NumPy 的案例與官方相近，
  部分中位數略慢（0.7～0.9 倍，各輪波動大）；pie 每個 wedge 的 Python 建立流程仍比官方慢，尚未處理。
- 平行化只用於 path collection，效果取決於可用的 CPU 核心數。
- 時間會隨 CPU、記憶體配置與垃圾回收時機變動；官方 Matplotlib 在物件很多的案例各輪差異尤其大。

## 重跑

```bash
python tests/bench/basic_charts.py --rounds 3 --output tests/output/basic-charts
# 同時量測上一版（其 python 目錄需已放入編譯好的 extension）：
python tests/bench/basic_charts.py --impls mpl base rs --base-source /path/to/0.0.3/python \
    --rounds 3 --output tests/output/basic-charts-3way
```

`--cases` 可只跑部分案例，`--scale` 可等比例放大或縮小資料量。
輸出目錄包含每個實作的 PNG、RGBA 畫布（`.npy`）與 `results.json`。
