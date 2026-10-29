# 繪圖效能量測

## 官方 Matplotlib 的 Python 優化對照

`official_boxplot_fastpath.py` 支援官方 Matplotlib **3.8.2 與 3.11.2**，
套用行程內的 Python／NumPy 實驗補丁，
不載入本專案的 Rust extension；`official_boxplot_compare.py` 使用相同 5,400 箱工作負載，
比較原版、僅統計優化、僅物件流程優化、兩者合併與 matplotlib-rs 0.0.2。
全部使用 Pillow 壓縮等級 6，各預熱一次、各量測五輪，保留全部 37,800 個 artists。

官方版合併優化後建圖中位數 **17.063 → 2.225 秒（7.67×）**，
建圖＋首次繪製 **20.795 → 5.887 秒（3.53×）**；30 次執行的 PNG 位元組全部相同。
這是輔助分析建圖成本的實驗補丁，尚未合併至上游。
方法、分項對照、限制、完整數據及重跑命令見
[官方版優化報告](../../docs/benchmarks/official-boxplot-optimization.md)。

```bash
python tests/bench/test_official_boxplot_fastpath.py
python tests/bench/official_boxplot_compare.py \
    --output tests/output/official-boxplots-python-only --rounds 5 \
    --variants official statistics_only artists_only python_optimized
```

### Python 3.10／Matplotlib 3.8.2

使用獨立環境及固定依賴，保留 3.8.2 的 `vert=False`、`labels` 與原始 artist labels 行為：

```bash
python3.10 -m venv .venv-mpl382
. .venv-mpl382/bin/activate
python -m pip install -r tests/bench/requirements-official-3.8.2.txt
python tests/bench/test_official_boxplot_fastpath.py
python tests/bench/official_boxplot_compare.py \
    --output tests/output/official-boxplots-mpl382-new --rounds 5 \
    --variants official statistics_only artists_only python_optimized
```

在此環境，量測工具預設只比較四個官方版本，不需要建置或匯入 Rust extension。
matplotlib-rs 0.0.2 發布套件的 Python >=3.11 要求仍適用。
3.8.2 的實測結果與使用方式見 [相容版本報告](../../docs/benchmarks/official-boxplot-optimization-3.8.2.md)。
五輪建圖中位數 **23.240 → 3.033 秒（7.66×）**，建圖＋首次繪製 **28.197 → 8.353 秒（3.38×）**；
24 次執行（含預熱）的 PNG 位元組全部一致。

## 5,400 個箱形圖

`boxplots.py` 比較同一張圖上的 30 組箱形圖，每組 180 個；每個箱形圖有
100 個隨機常態樣本，共 540,000 個值。各組使用不同顏色，並保留離群點、
whiskers、caps 與 median。只有一個 Axes，組與組之間有間隔，以 G01～G30 標示。

在專案根目錄、已安裝 Matplotlib 3.11.2 與本專案 release extension 的環境執行：

```bash
python tests/bench/boxplots.py --output tests/output/boxplots-5400
```

可調整 `--groups`、`--boxes`、`--samples`、`--seed`、`--rounds`、`--redraws`、
`--dpi`、`--width`、`--height`。預設 seed=20261026、36 × 12 吋、100 DPI，
輸出為 3,600 × 1,200 像素。重跑時請指定新的輸出目錄，保留先前結果。

兩套實作各預熱一次，再各執行五個全新 Python 行程；行程依序執行，
每輪交替誰先執行。使用相同 seed 與 NumPy 版本，並檢查每次資料 SHA-256 一致。
資料產生與色盤準備不計入繪圖時間。

量測欄位：

- `build`：建立 Figure、Axes、箱形圖統計、所有 artists 與標籤。
- `first_draw`：第一次 `canvas.draw()`。
- `build_and_first_draw`：同一行程內 build 加 first_draw。
- `redraw`：不改變圖形的兩次 `canvas.draw()` 耗時中位數。
- `save_png`：完成上述繪製後呼叫 `savefig()`，包含再次繪製、PNG 壓縮與磁碟寫入。
- `peak_rss_mib`：該行程的最高 RSS，包含匯入、資料、圖形物件與輸出階段。

整體比較以五輪中位數為主，同時保留最小值與最大值。建立＋首次繪製的中位數
從各輪總時間計算，不能直接相加其他欄位的中位數。倍率為 Matplotlib／matplotlib-rs
耗時，大於 1 代表 matplotlib-rs 較快。

輸出目錄包含逐輪 PNG、`matplotlib.png`、`matplotlib-rs.png`、`results.json` 與 stderr 紀錄。
JSON 包含牆鐘時間、CPU 時間、資料與畫布指紋，以及兩張 PNG 的逐像素比較。
會驗證箱體數量為 5,400；另有 32,400 個 Line2D，共約 37,800 個箱形圖 artists。
測試對象的 Git commit、Python／NumPy／Matplotlib API 版本與參數皆保留於輸出，
合成資料只用於此壓測。

### 2026-10-26 實測（0.0.1 初版）

測試 matplotlib-rs 0.0.1（核心 commit `151b46d`）與 Matplotlib 3.11.2，
CPython 3.12.14、NumPy 2.3.5、Pillow 13.0.0、Linux x86_64。
Rust extension 使用 release、fat LTO 編譯。主機為 Intel Xeon Platinum 8573C，
3 個可見 CPU、容器 CPU quota 2 核、記憶體限制 8 GiB；Agg 與專案 no-hinting rc 設定。

下表以五輪牆鐘時間的中位數為主，括號為最小～最大值，單位皆為秒。

| 階段 | Matplotlib | matplotlib-rs | 中位數倍率 |
| --- | --- | --- | --- |
| 建立圖形物件 | 28.775（16.822～30.802） | 32.863（31.249～62.134） | 0.88× |
| 首次繪製 | 4.299（3.626～4.688） | 0.312（0.303～0.382） | 13.76× |
| 建立＋首次繪製 | 33.074（20.447～35.244） | 33.175（31.597～62.440） | 1.00× |
| 相同圖形重繪 | 2.006（1.836～2.451） | 0.260（0.250～0.284） | 7.72× |
| PNG 存檔（含再次繪製） | 2.382（1.648～3.165） | 0.300（0.281～0.365） | 7.94× |

在這個案例中，Rust 的繪製、重繪與存檔階段較快，但圖形物件建立占據大部分時間，
兩套實作的建立＋首次繪製中位數接近。matplotlib-rs 建圖耗時的範圍尤其大，
尚未進一步判定成因；這組數值不代表所有圖形或環境的速度。

行程最高 RSS 的五輪中位數為 Matplotlib **513.0 MiB**、matplotlib-rs **593.8 MiB**，
後者高約 **15.8%**。此數值包含整個行程，不是單獨的 renderer 記憶體。

兩次預熱與十次正式執行的資料與首次畫布指紋皆一致。
3,600 × 1,200 PNG 的比較為 **0 個差異像素**、最大通道差 **0**。
每次皆有 5,400 個箱體與 32,400 個 Line2D。
完整逐輪紀錄與成圖保留於本地 `tests/output/boxplots-5400/`。

### 建立物件優化

以原始 0.0.1 wheel 和修改版在同一 Python／NumPy 環境重新交錯量測，
各預熱一次、各五輪獨立行程；參數仍是 30 組 × 180 箱 × 100 樣本、
seed=20261026、3,600 × 1,200 PNG。資料產生不計入建圖時間，樣式、離群點與物件數均保留。

剖析原版發現，32,400 次通用 `plot()` 呼叫反覆解析格式與單位、建立裁切 transform、
更新資料範圍及初始化 Line2D。全腳本 cProfile 的呼叫數由約 3,417 萬降至 239 萬；
profiling 本身會增加耗時，以下速度比較使用未開 profiler 的正式量測。

實作調整：

- 對一般數值 Axes 重用已驗證的樣式 prototype，建立獨立 Line2D／PathPatch 與 Rust handle，
  每組合併更新資料範圍。回傳型別、labels、autoscale、個別修改及刪除行為仍保留。
- 有限 float64 二維樣本批次計算 percentile、mean 與 whiskers；保留原始每欄運算順序及離群點順序。
  bootstrap、masked／非有限資料等條件沿用一般統計路徑。
- CallbackRegistry 各自獨立，在首次使用時配置 maps；Rust 直接複製原生樣式並置換幾何資料。
  單位轉換、自訂 Axes／transform、複雜樣式仍使用一般物件建立路徑。

以下為五輪中位數，括號為最小～最大值：

| 階段 | 原始 0.0.1（秒） | 優化後（秒） | 中位數加速 |
| --- | --- | --- | --- |
| 建立圖形物件 | 36.601（27.074～39.121） | 1.458（1.105～3.137） | 25.10× |
| 首次繪製 | 0.385（0.255～0.447） | 0.264（0.244～0.321） | 1.46× |
| 建立＋首次繪製 | 37.047（27.486～39.506） | 1.779（1.369～3.421） | 20.83× |
| 相同圖形重繪 | 0.283（0.207～0.319） | 0.230（0.216～0.246） | 1.23× |
| PNG 存檔（含再次繪製） | 0.302（0.242～0.339） | 0.296（0.248～0.728） | 1.02× |

五輪各自的新舊建圖加速為 12.47×、24.50×、25.10×、15.47×、26.21×，每輪均超過 10×。
最高 RSS 的中位數為 **593.6 → 266.9 MiB**，減少約 **55.0%**。

驗證結果：

- 12 次執行（含預熱）資料與首次畫布指紋一致；儲存的前後 PNG 差異像素為 0。
- 每次都保留 5,400 個 PathPatch 與 32,400 個 Line2D，沒有將箱形圖改為少量 collection。
- 新增 186 組統計等價比較及 19 個圖像／物件行為情境，檢查獨立 callback、
  顏色／線寬／資料修改、回呼序列化、sticky edges、刪除、relim 與 property cycle。
- Rust 測試、全部 Python 單元／底層差異／模型測試通過。
- 原有 57 個繪圖案例加新增箱形圖變體案例，共 58 案例、74 張 PNG 全部一致。
  SVG／PDF 各 71／71 一致；EPS 70／71，一項原有 `polar_more.eps` 內容差異保留。
- 官方 gallery 的 6 個 boxplot／bxp 範例全部逐像素一致。

重跑新舊版本比較（先編譯修改後的 extension）：

```bash
sh scripts/dev_build.sh --locked
python tests/bench/boxplot_build_compare.py \
    --baseline-wheel /path/to/original-matplotlib-rs-0.0.1.whl \
    --output tests/output/boxplots-build-before-after \
    --rounds 5 --min-speedup 10
```

基準 wheel 會解開至輸出目錄，各版以獨立行程匯入各自的 Python 與 Rust extension，
共用目前 Python 與 NumPy，不修改基準套件或工作負載。
程式會檢查資料指紋、畫布指紋與物件數；建圖中位數加速未達指定門檻會以非零退出碼結束。
逐輪資料、前後 PNG 與原始碼來源在本地 `tests/output/boxplots-build-before-after/`。

## 一般四格圖

`phases.py` 與 `best_of.sh` 使用較小的 2 × 2 subplot 工作負載，詳見專案根目錄 README。
