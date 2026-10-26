# matplotlib-rs

以 Rust 實作繪圖與字型核心，提供與 Matplotlib 相容的 Python／`pyplot` 介面。
`0.0.1` 是首個公開版本；Python 層包含依 Matplotlib 3.11 調整的模組，
原生核心經 PyO3 與 NumPy 連接，處理 Agg 點陣化、字型、圖形物件、座標軸與影像重採樣。
部分功能仍由 Python 與其他套件處理，例如 ContourPy 等高線及 Kiwi constrained layout。

## 安裝

需要 Python 3.11 以上、Rust stable 工具鏈與本機 C linker。
本次實測環境為 Linux x86_64、CPython 3.12；其他平台尚未驗證。

從 release 原始碼安裝：

```bash
git clone --branch v0.0.1 https://github.com/g21589/matplotlib-rs.git
cd matplotlib-rs
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install .
```

pip 會透過 maturin 編譯 Rust extension 並安裝所需 Python 依賴。
也可從 [Releases](https://github.com/g21589/matplotlib-rs/releases/tag/v0.0.1)
下載符合 Python 與平台的 wheel，再執行 `python -m pip install /path/to/matplotlib_rs-0.0.1-*.whl`。
本次提供的 wheel 適用 CPython 3.12、Linux x86_64、glibc 2.35 以上；其他環境請從原始碼建置。
此版本尚未發布至 PyPI。

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
請在獨立的 Python 行程內選擇實作。
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
[0.0.1 驗證報告](docs/releases/0.0.1.md)。

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

`polar_more.eps` 的裁切路徑快取定義不同；以 Ghostscript 10.05.1 轉為 100 DPI PNG 後，像素完全一致。
原始 EPS 內容比較仍記為失敗，不因這項額外檢查改為通過。
官方 gallery 的全量結果與未通過案例另外列於驗證報告。

完整結果與可重現命令見 [驗證報告](docs/releases/0.0.1.md)。
功能案例通過率與原始碼行／分支覆蓋率是不同指標；本版未量測原始碼覆蓋率。

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
COMPARE_VECTOR=svg,pdf,eps python tests/compare/compare.py
```

Rust 單元測試的 PyO3／libpython 連結方式、官方 gallery 的取得與執行方式，
以及測試限制見 [驗證報告](docs/releases/0.0.1.md)。
測試結果請使用新的輸出目錄，並依實際像素差異及退出碼判斷。

建置 wheel 與 source distribution：

```bash
maturin build --release --locked --out dist
maturin sdist --out dist
```

## 授權

依 [`PSF-2.0`](licenses/LICENSE_MATPLOTLIB) 授權；
本專案包含由 Matplotlib 衍生的 Python 程式碼與資料，保留其授權及版權聲明。
