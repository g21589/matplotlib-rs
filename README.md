# matplotlib-rs

以 Rust 實作繪圖與字型核心，提供與 Matplotlib 相容的 Python／`pyplot` 介面。
目前發行版本為 `0.0.3`：支援 **Python 3.10**，執行期**只依賴 NumPy**，並進一步加快大量箱形圖。
Python 層包含依 Matplotlib 3.11 調整的模組，原生核心經 PyO3 與 NumPy 連接，
處理 Agg 點陣化、字型、圖形物件、座標軸與影像重採樣。

## 安裝

需要 Python 3.10 以上；唯一的執行期依賴是 NumPy 1.25 以上。
Linux x86_64、glibc 2.28 以上、CPython 3.10～3.13 可直接安裝預編譯 wheel：

```bash
python -m pip install matplotlib-rs==0.0.3
python -m pip list   # 只會多出 matplotlib-rs 與 numpy
```

PyPI 與 [GitHub Release](https://github.com/g21589/matplotlib-rs/releases/tag/v0.0.3) 使用完全相同的已測試套件；
Release 另附 source distribution、效能原始紀錄與 `SHA256SUMS`。
發布流程在只有 NumPy 的乾淨環境測試每個 wheel；本機與 CI 實測 Linux x86_64，其他平台尚未驗證。

從 release 原始碼安裝：

```bash
git clone --branch v0.0.3 https://github.com/g21589/matplotlib-rs.git
cd matplotlib-rs
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install .
```

需要 Rust stable 工具鏈與本機 C 編譯器（Linux 上的 `cc`／gcc；Windows 為 MSVC），
pip 會透過 maturin 編譯 Rust extension 與內建的 qhull。
發布流程、驗證方式與首次授權設定見 [CI／CD 說明](docs/CI.md#pypi-發布)。

Matplotlib 原本依賴的套件改由本專案提供，不需另外安裝：

| Matplotlib 依賴 | matplotlib-rs 0.0.3 的作法 |
| --- | --- |
| cycler、pyparsing、python-dateutil（含 six） | 純 Python，原樣內建於 `matplotlib_rs._vendor`（`tools/vendor_deps.py` 可重現）；已安裝的 cycler 會優先使用 |
| kiwisolver（constrained layout） | Rust 移植 kiwi 1.5 求解器，運算順序與 kiwi 相同 |
| ContourPy（等高線） | Rust 移植預設的 `mpl2014` 演算法，頂點逐位元相同 |
| qhull（Delaunay 三角網格） | 內建並編譯 Matplotlib 使用的 qhull 8.0.2，三角形順序相同 |
| Pillow（PNG 讀寫、PDF／SVG 內嵌影像） | Rust 讀寫 PNG；`imread` 的數值與 Matplotlib／Pillow 相同 |
| fontTools、packaging | 只有 Type 42 字型嵌入需要 fontTools；packaging 改為內建版本解析 |

已安裝 Pillow 時仍由 Pillow 編碼內嵌影像，與 Matplotlib 的檔案逐位元組相同。
下列功能仍需選用套件（`pip install "matplotlib-rs[full]"`），缺少時會丟出說明用途的 `ImportError`：
JPEG／TIFF／WebP／GIF／AVIF 讀寫、`pil_kwargs`、`PillowWriter` 需要 Pillow；
`pdf.fonttype`／`ps.fonttype` 設為 42 需要 fontTools；`contour.algorithm` 為
`serial`／`threaded`／`mpl2005` 需要 ContourPy。usetex、GUI 後端與影片輸出需要的外部程式不變。

先前的 0.0.2 需要 Python 3.11 以上，並會安裝 ContourPy、Pillow、kiwisolver 等 Matplotlib 依賴。

## 基本使用

```python
import matplotlib_rs

# 必須在任何 matplotlib / mpl_toolkits 匯入之前呼叫。
matplotlib_rs.install()

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot([1, 2, 3], [1, 4, 9], marker="o", label="y = x²")
ax.set(xlabel="x", ylabel="y", title="matplotlib-rs")
ax.legend()
fig.savefig("example.png", dpi=150)
plt.close(fig)
```

也可以直接使用 `import matplotlib_rs.pyplot as plt`。
`install()` 會將 `matplotlib` 與 `mpl_toolkits` 匯入導向本套件；
若未安裝 cycler 或 python-dateutil，為 Matplotlib 撰寫的腳本中 `from cycler import cycler`、
`import dateutil` 也會取得內建版本。請在獨立的 Python 行程內選擇實作。
套件發行版本可透過 `importlib.metadata.version("matplotlib-rs")` 查詢；
`matplotlib_rs.__version__` 保留 `3.11.2`，供沿用的 Matplotlib 相容程式判斷 API 版本。

## 功能與相容性

| 類別 | 專案內的測試案例 |
| --- | --- |
| 基本圖表 | line、scatter、bar／barh、hist／hist2d、hexbin、pie、boxplot、violin、stem、step、stackplot |
| 場與影像 | imshow、pcolormesh、contour、tri、quiver、streamplot、影像重採樣 |
| 座標與版面 | log、logit、日期、polar、3D、subplots、subfigures、mosaic、tight／constrained layout、inset、secondary axes、twin axes |
| 文字與外觀 | annotation、箭頭、字型、mathtext、文字 bbox、legend、table、style、hatch、spines、ticks |
| 繪製與輸出 | clip paths、blitting、animated artists、rasterization、PNG、SVG、PDF、EPS、透明背景與 tight bbox |

這是測試案例覆蓋的功能清單，並不代表完整 Matplotlib API、所有參數或 GUI 後端均已相容。
本次以 Agg 離線繪圖為驗證範圍；GUI 互動、外部 TeX／影片工具及跨平台行為未完成驗證。
逐像素與向量格式比對結果、官方 gallery 的差異及略過項目會列於
[0.0.1 完整驗證報告](docs/releases/0.0.1.md)；之後新增的測試見
[0.0.2](docs/releases/0.0.2.md) 與 [0.0.3](docs/releases/0.0.3.md) 發行紀錄。

## 測試與功能覆蓋率

本次測試以 Matplotlib **3.11.2** 為參考實作，固定使用
[`tests/mplrc/matplotlibrc`](tests/mplrc/matplotlibrc) 與 Agg 後端。
PNG 以 RGBA 像素完全一致為通過條件；SVG／PDF／EPS 比對固定時間與 SVG hash salt 後的內容，
僅忽略 PostScript 的輸出檔名行。

| 測試範圍 | 0.0.1 實測結果 |
| --- | --- |
| Rust 單元測試 | 1／1 通過；另無 doctest 案例 |
| 三角網格／影像重採樣 | 1,015／1,015、6,800／6,800 完全一致 |
| 文件快取 | 124 個類別，無過期項目 |
| 底層 renderer／字型 | 4 支差異測試全通過；300 次 renderer 迭代、2,016 次 TTF 字形繪製、85,827 個 CFF 字形、480 個字型 API 比較 |
| 原生模型 | 與參考圖像完全一致 |
| 自有繪圖案例 | 57／57 案例的 PNG 完全一致，共 69／69 張（100%） |
| SVG／PDF | 各 66／66 份內容完全一致（100%） |
| EPS | 65／66 份內容完全一致（98.48%）；`polar_more.eps` 有差異 |
| 同時符合四種格式 | 56／57 案例（98.25%） |
| 官方 gallery PNG | 508／544 支出圖腳本完全一致（93.38%），另有 36 支像素差異 |
| 官方 gallery 執行範圍 | 共 581 支；8 支未留下 figure、29 支參考端略過、Rust 執行錯誤 0 支 |
| 乾淨環境安裝 | 未安裝原版 Matplotlib，wheel 可安裝並輸出 PNG／SVG／PDF／EPS |

0.0.2 另加入箱形圖統計與可變物件的回歸測試，以及 1 個包含 5 張 PNG 的繪圖案例。
本機重跑後，58／58 案例、74／74 張 PNG、SVG 71／71、PDF 71／71 完全一致；
EPS 修正裁切快取生命週期、穩定參考端後為 71／71 內容一致。
另重跑 6 個官方 boxplot gallery，全部逐像素一致。
上表 581 支完整 gallery 的數字屬於 0.0.1，0.0.2 未重新執行全部官方 gallery。

0.0.1 的 `polar_more.eps` 裁切路徑快取定義有差異，當時的 Ghostscript 轉圖像素一致，原始內容比較仍記為失敗。
0.0.2 發現此快取也可能因臨時 transform 的位址重用而裁切錯誤，已修正；
CI 比對時亦明確保留參考端 transform 的生命週期，詳見 [CI 說明](docs/CI.md)。
官方 gallery 的全量結果與未通過案例另外列於驗證報告。

完整結果與可重現命令見 [驗證報告](docs/releases/0.0.1.md)。
功能案例通過率與原始碼行／分支覆蓋率是不同指標；本版未量測原始碼覆蓋率。

### 0.0.3 實測：Python 3.10 與只有 NumPy 的環境

| 測試範圍 | 結果 |
| --- | --- |
| CPython 3.11／3.12／3.13（安裝參考套件） | Rust 5／5；全部單元、底層差異與模型測試通過；58／58 案例的 PNG／SVG／PDF／EPS 全部一致 |
| **CPython 3.10、只裝 NumPy** 對 Matplotlib 3.11.2（Python 3.12） | **58／58 案例一致**：PNG 逐像素、EPS 逐位元組；9 個含內嵌點陣影像的案例 SVG／PDF 解碼後像素相同，其餘內容逐位元組相同 |
| 內建 kiwi | 6 種 constrained layout（含 subfigures、colorbar、mosaic、`outside` legend）座標與 kiwisolver 逐位元相同；400 組隨機系統誤差 < 1e-9 |
| 內建 mpl2014 等高線 | 200 個隨機場（遮罩、NaN、chunk、corner mask）共 3,600 組線與填色結果與 ContourPy 逐位元相同 |
| 內建 qhull | 150 組點集（含共圓格點、重複點、極端長寬比）三角形與鄰接陣列和 Matplotlib `_qhull` 完全相同 |
| PNG 讀寫 | 12 種 Pillow 產生的 PNG（1／2／4／8／16 bit、調色盤與透明、interlace）讀出與 Matplotlib 的 `imread` 數值相同 |
| NumPy 1.25.2／1.26.4（Python 3.10） | 與 NumPy 2.x 輸出相同；僅 SVG clip-path id 不同（Matplotlib 以 `str()` 雜湊 NumPy 數值，NumPy 2 改變了顯示格式，官方版亦同） |
| 安裝 | cp310 wheel 與 sdist（含 Rust 與 qhull 編譯）在乾淨 Python 3.10 venv 安裝後只有 `matplotlib-rs` 與 `numpy`，煙霧測試通過 |

constrained layout 的 kiwisolver 會依變數記憶體位址排序，同一份腳本只要輸出路徑長度不同，
SVG 座標末位就可能改變；內建 kiwi 以建立順序排序，結果固定，實測常見版面與 kiwisolver 相同。

## 效能測試

使用 [`tests/bench/phases.py`](tests/bench/phases.py) 的 2 × 2 subplot 工作負載：
每個 panel 繪製兩條各 1,000 點的曲線、scatter、標題、圖例與 grid，並執行 tight layout。
分開量測 import、建立圖表、第一次 PNG 存檔及之後十次更新標題與存檔的平均時間。
以下為各階段五輪最小值；完整逐輪資料、中位數與量測環境列於 [驗證報告](docs/releases/0.0.1.md)。

| 階段 | Matplotlib 最小值（ms） | matplotlib-rs 最小值（ms） | 倍率 |
| --- | --- | --- | --- |
| import | 588.7 | 329.0 | 1.79× |
| 建立圖表與 tight layout | 188.8 | 199.9 | 0.94× |
| 第一次 PNG 存檔 | 294.0 | 103.0 | 2.85× |
| 後續 PNG 存檔（每次） | 299.6 | 86.3 | 3.47× |

倍率為 Matplotlib／matplotlib-rs 耗時。建立圖表階段略慢，首存與後續存檔在此工作負載較快；五輪數值有明顯波動。

```bash
sh tests/bench/best_of.sh mpl 5
sh tests/bench/best_of.sh rs 5
```

`best_of.sh` 報告五個全新 Python 行程中各階段的最小值；
import 包含 NumPy／pyplot，第一次存檔包含首次繪製成本，後續存檔仍包含 PNG 壓縮與磁碟寫入。
效能會隨工作負載、字型、CPU、NumPy／BLAS 與快取狀態改變。

### 大量資料的基本圖表

[`tests/bench/basic_charts.py`](tests/bench/basic_charts.py) 對 24 種常見圖表各放入大量資料
（百萬點折線、20 萬點散佈圖、2 萬根長條、千萬筆直方圖、4000 × 4000 影像等），
在全新行程中量測建立與繪製，並逐像素比對 Matplotlib 3.11.2 的畫布。
找到並修正的效能問題以長條／直方圖最嚴重：每根長條約 0.8 毫秒的 Python 物件流程。
建立＋首次繪製的三輪中位數（秒）：

| 圖表與資料量 | Matplotlib | 0.0.3 | 本版 | 對 Matplotlib | 對 0.0.3 |
| --- | ---: | ---: | ---: | ---: | ---: |
| bar，20,000 根 | 10.279 | 13.717 | 0.131 | 78× | **105×** |
| barh，20,000 根（黑色邊框） | 11.798 | 15.332 | 0.195 | 60× | 79× |
| hist，2,000,000 筆、10,000 bins | 5.488 | 6.389 | 0.076 | 73× | 85× |
| hist，10,000,000 筆、200 bins | 0.274 | 0.258 | 0.049 | 5.6× | 5.2× |
| hist step，3 組 × 2,000,000 筆、2,000 bins | 0.295 | 0.216 | 0.083 | 3.6× | 2.6× |
| errorbar，20,000 點 | 0.301 | 0.822 | 0.260 | 1.2× | 3.2× |
| stem，20,000 點 | 0.358 | 0.865 | 0.323 | 1.1× | 2.7× |
| scatter，200,000 點（大小、顏色各異，alpha） | 2.347 | 2.161 | 1.145 | 2.1× | 1.9× |
| plot '--'，1,000,000 點 | 0.205 | 0.254 | 0.136 | 1.5× | 1.9× |
| boxplot，1,000 箱 × 200 筆 | 4.901 | 0.210 | 0.198 | 25× | 1.1× |

24 個案例全部與 Matplotlib 逐像素一致。官方以 C++ Agg 處理的工作（折線簡化、逐點點陣化的散佈圖、
影像重採樣、等高線）兩者成本相近，倍率約 1～3 倍；全部結果、修正內容、驗證與限制見
[基本圖表效能報告](docs/benchmarks/basic-charts.md)。

```bash
python tests/bench/basic_charts.py --rounds 3 --output tests/output/basic-charts
```

`MPLRS_THREADS` 可指定大量 path collection（scatter、pcolormesh 等）平行點陣化的執行緒數，`1` 為單執行緒。

### 5,400 個箱形圖壓測

新增 [`tests/bench/boxplots.py`](tests/bench/boxplots.py)：單一 Axes 放入 30 組、
每組 180 個箱形圖，每箱 100 個隨機樣本，每組不同顏色。
0.0.1 初版與 Matplotlib 的五輪比較中，首次繪製為 4.299 → 0.312 秒（13.76×）；
建立圖形加首次繪製為 33.074 → 33.175 秒（約相同）。兩套 PNG 逐像素一致。
後續物件建立優化與原始 0.0.1 wheel 同批交錯重測：建圖中位數
**36.601 → 1.458 秒（25.10×）**，五輪各自加速 **12.47～26.21×**；
建立＋首次繪製為 **37.047 → 1.779 秒（20.83×）**。
保留全部 5,400 個箱體與 32,400 條線，PNG 仍逐像素一致，最高 RSS 中位數減少約 55%。
完整條件、範圍與重跑命令見 [壓測說明](tests/bench/README.md)。
五輪數據另存為 [可檢查的 JSON](docs/benchmarks/boxplots-0.0.2.json)。

另在官方 Matplotlib 3.11.2 上套用 Python／NumPy 實驗補丁，以相同負載量測五輪：
建圖中位數 **17.063 → 2.225 秒（7.67×）**，保留全部 37,800 個物件。
官方行程未載入本專案 Rust extension；全部版本改用相同 Pillow PNG 壓縮設定，
30 次執行的 PNG 位元組亦完全一致。分項優化與繪製對照見
[官方版優化實驗報告](docs/benchmarks/official-boxplot-optimization.md)。

同一實驗補丁另支援 **Python 3.10／Matplotlib 3.8.2**，沿用該版原始 API。
五輪建圖中位數 **23.240 → 3.033 秒（7.66×）**，建圖＋首次繪製加速 **3.38×**，
24 次執行的 PNG 位元組全部一致。安裝、使用與完整數據見
[3.8.2 相容版本報告](docs/benchmarks/official-boxplot-optimization-3.8.2.md)。

#### 0.0.3：批次建立箱形圖、Python 3.10 只有 NumPy

0.0.3 一次建立每個 `bxp` 呼叫的全部 artists（NumPy 計算座標、Rust 批次複製 handle 與 Python 物件），
並在建立期間暫停循環垃圾回收；每個箱體與線段仍是可獨立修改的物件。五輪中位數：

| 比較 | 建圖 | 建圖＋首次繪製 |
| --- | --- | --- |
| matplotlib-rs 0.0.2 → 0.0.3（Python 3.12，交錯量測） | 1.119 → **0.470 秒（2.38×）** | 1.374 → 0.705 秒（1.95×） |
| 官方 Matplotlib 3.8.2 → 0.0.3 只有 NumPy（**Python 3.10**） | 17.664 → **0.492 秒（35.9×）** | 21.986 → **0.712 秒（30.9×）** |
| 官方 Matplotlib 3.11.2 → 0.0.3（Python 3.12） | 15.006 → **0.451 秒（33.3×）** | 18.837 → 0.672 秒（28.0×） |

0.0.3 在 Python 3.10 只有 NumPy 時的畫布與 Matplotlib 3.11.2 逐像素相同；
與 3.8.2 的差異只在文字（3.8 與 3.11 的文字繪製不同），繪圖區內沒有差異像素。
Python 3.10 的最高 RSS 為 594.9（3.8.2）→ 295.5 MiB。
主機為 4 vCPU Intel Xeon 2.80GHz；數據見 [`docs/benchmarks/boxplots-0.0.3.json`](docs/benchmarks/boxplots-0.0.3.json)，
重跑方式見 [壓測說明](tests/bench/README.md#003批次建立與-python-310)。

## 開發與重跑測試

```bash
python -m pip install 'maturin>=1.5,<2' 'matplotlib==3.11.2'
python -m pip install -e .

# Linux：編譯並把 extension 放入 Python 原始碼目錄。
sh scripts/dev_build.sh --locked
export PYTHONPATH="$PWD/python"
export MPLBACKEND=Agg

sh scripts/test_rust.sh

# 這些測試檔是可直接執行的比較程式，請逐一執行並檢查退出碼。
for test in tests/unit/test_*.py tests/lowlevel/diff_*.py tests/model/first_plot.py; do
    python "$test" || exit 1
done
COMPARE_VECTOR=svg,pdf,eps COMPARE_STABLE_EPS=1 python tests/compare/compare.py

# Python 3.10、只有 NumPy 的環境（參考端使用上面的 Python 與套件）：
python3.10 -m venv /tmp/numpy-only && /tmp/numpy-only/bin/pip install numpy
PYTHON=/tmp/numpy-only/bin/python sh scripts/dev_build.sh --locked
NUMPY_ONLY_PYTHON=/tmp/numpy-only/bin/python python tests/unit/test_numpy_only.py
COMPARE_RS_PYTHON=/tmp/numpy-only/bin/python COMPARE_DECODE_IMAGES=1 \
    COMPARE_VECTOR=svg,pdf,eps COMPARE_STABLE_EPS=1 python tests/compare/compare.py
```

Rust 單元測試的 PyO3／libpython 連結方式、官方 gallery 的取得與執行方式，
以及測試限制見 [驗證報告](docs/releases/0.0.1.md)。
測試結果請使用新的輸出目錄，並依實際像素差異及退出碼判斷。

建置 wheel 與 source distribution：

```bash
maturin build --release --locked --out dist
maturin sdist --out dist
```

## CI 與版本發布

GitHub Actions 在 push／PR 執行 Python 3.11～3.13 的 Rust、單元、底層與四格式圖形比較，
並另以 **Python 3.10 只安裝 NumPy** 的環境對 Matplotlib 3.11.2 跑同一套比較（含 NumPy 1.25.2 最低版本）。
推送 `vX.Y.Z` 標籤後，發布流程會檢查版本一致性、執行測試、建置 CPython 3.10～3.13 的 wheel 與 sdist、
在只有 NumPy 的乾淨環境驗證 wheel，並與 0.0.1 基準版交錯量測 5,400 個箱形圖。
必須像素一致且建圖中位數至少加速 10 倍，才能發布 GitHub Release。
GitHub Release 成功後，PyPI 流程會沿用同一組已驗證的套件，以 Trusted Publishing 發布。
完整流程、首次授權設定與本機入口見 [CI／CD 說明](docs/CI.md)。

## 授權

依 [`PSF-2.0`](licenses/LICENSE_MATPLOTLIB) 授權；
本專案包含由 Matplotlib 衍生的 Python 程式碼與資料，保留其授權及版權聲明。
內建的第三方程式碼各依其授權：cycler（[BSD](licenses/LICENSE_CYCLER)）、
pyparsing（[MIT](licenses/LICENSE_PYPARSING)）、python-dateutil（[Apache-2.0／BSD](licenses/LICENSE_DATEUTIL)）、
six（[MIT](licenses/LICENSE_SIX)）、qhull（[Qhull](licenses/LICENSE_QHULL)）；
Rust 移植的 kiwi 與 ContourPy mpl2014 演算法分別依 [BSD](licenses/LICENSE_KIWISOLVER) 與
[BSD-3-Clause](licenses/LICENSE_CONTOURPY)。
