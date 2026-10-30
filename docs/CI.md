# CI 與 GitHub Release

## 自動測試

[CI workflow](../.github/workflows/ci.yml) 在 branch push、pull request、手動執行與 release 呼叫時啟動。
同一分支／PR 的新版本會取消舊 CI；release 呼叫依 tag 分組，不會被 branch CI 取消。
Linux x86_64／Ubuntu 24.04，Python 3.11、3.12、3.13 分別建置 release extension，執行：

- Python／Cargo／lockfile 版本檢查。
- 以 `tools/vendor_deps.py` 重新產生內建的 cycler／pyparsing／python-dateutil／six，必須與提交內容完全相同。
- Rust 單元測試與 doctest 探索。
- 全部 `tests/unit/test_*.py`、`tests/lowlevel/diff_*.py` 與原生模型案例。
  `test_native_ports.py` 將內建的 kiwi、mpl2014 等高線、qhull 與 PNG 讀寫，
  逐位元比較 kiwisolver、ContourPy、Matplotlib `_qhull` 與 Pillow 的結果；
  `test_numpy_only.py` 以不含 site-packages、只有 NumPy 的子行程繪圖，與完整環境比較。
- 全部自有繪圖案例；PNG 逐像素一致，SVG／PDF／EPS 內容一致。

另一項工作在 **Python 3.10 只安裝 NumPy** 的虛擬環境執行本專案（參考端仍為 Python 3.12 的 Matplotlib 3.11.2）：

- 建置 3.10 extension，執行 Rust 測試與 `test_native_ports.py`（沒有參考套件時只做自我檢查）。
- `test_numpy_only.py` 以該 3.10 環境為子行程；另建立 NumPy 1.25.2 環境重跑，確認宣告的最低版本。
- 全部繪圖案例以 `COMPARE_RS_PYTHON` 跨直譯器比較。沒有 Pillow 時內嵌影像由本專案編碼，
  `COMPARE_DECODE_IMAGES=1` 會在 SVG／PDF 不同時解碼內嵌影像再比較像素，其餘內容仍須逐位元相同；
  PNG 與 EPS 仍要求完全一致。

參考版本與數值依賴固定於 [`requirements-ci.txt`](../requirements-ci.txt)。
Python 3.11 使用相容的 ContourPy 1.3.3；Python 3.12 以上使用 1.4.0。
安裝開源字型，避免 CFF 測試因找不到字型而空跑。
字型數量依 runner 套件版本而異，測試記錄會保留實際比較數量。

0.0.2 修正 EPS 快取未保留臨時 affine transform、造成 Python 物件位址重用的問題。
新增測試以同一路徑搭配 1,000 個不同裁切轉換，檢查命令與序列化幾何均獨立。
Matplotlib 3.11.2 參考端也有相同問題，因此 CI 明確設定 `COMPARE_STABLE_EPS=1`：
僅在參考比較行程保留原有 transform 的生命週期，不改動座標或繪圖命令。
此設定會印於日誌；未設定時仍使用原始參考行為。
全部 EPS 仍嚴格比較內容，沒有略過檔案或接受像素誤差。

本機使用相同入口：

```bash
python -m pip install -r requirements-ci.txt
sh scripts/dev_build.sh --locked
sh scripts/test_rust.sh
bash scripts/ci_test.sh
```

Python 3.10／只有 NumPy 的對照（參考端使用上面的環境）：

```bash
python3.10 -m venv /tmp/numpy-only && /tmp/numpy-only/bin/pip install numpy
PYTHON=/tmp/numpy-only/bin/python sh scripts/dev_build.sh --locked
NUMPY_ONLY_PYTHON=/tmp/numpy-only/bin/python python tests/unit/test_numpy_only.py
COMPARE_RS_PYTHON=/tmp/numpy-only/bin/python COMPARE_DECODE_IMAGES=1 \
    COMPARE_VECTOR=svg,pdf,eps COMPARE_STABLE_EPS=1 python tests/compare/compare.py
```

測試輸出與日誌位於 `tests/output/ci/`；CI 成功或失敗皆上傳為 Actions artifact，保留 14 天。
完整上游 gallery 不在每次 CI 執行範圍。

## 發布

[Release workflow](../.github/workflows/release.yml) 由 `vX.Y.Z` tag push 觸發，流程如下：

1. 檢查 tag、`pyproject.toml`、`Cargo.toml`、`Cargo.lock` 相符，且存在該版 release notes。
2. 呼叫完整 CI；同時建置 CPython 3.10／3.11／3.12／3.13 的 manylinux_2_28 x86_64 wheel 與 sdist。
3. 每個 wheel 安裝到乾淨 venv（只會一併安裝 NumPy），確認沒有其他套件，
   測試可變 boxplot、mathtext、constrained layout、等高線、三角網格與 PNG／SVG／PDF／EPS。
4. 從不可變的 `v0.0.1` tag 建置原始 wheel，與目前版在同一 Python／NumPy 環境交錯量測。
   每版預熱 1 次、量測 5 次，使用完全相同的 30 × 180 箱形圖、每箱 100 樣本。
   保留 5,400 個箱體與 32,400 條線；資料與 RGBA hash 必須相同，建圖中位數至少加速 10 倍。
5. 全部通過才建立 draft release、附加 wheel、sdist、效能紀錄與 SHA256SUMS，最後公開 release。
   上傳失敗時維持 draft，可重跑失敗 job；同一 tag 使用 concurrency 避免同時發布。

效能門檻比較同一 runner 的前後版本，結果會因機器而變動；GitHub Actions 實測值會附於該版 release，
本機 25.10× 數據另存於 [`docs/benchmarks/boxplots-0.0.2.json`](benchmarks/boxplots-0.0.2.json)。

發布新版本時更新三個版本欄位與 `docs/releases/X.Y.Z.md`，提交並推送分支，之後推送 annotated tag：

```bash
python scripts/check_version.py --tag vX.Y.Z
git tag -a vX.Y.Z -m 'Release X.Y.Z'
git push origin vX.Y.Z
```

僅 publish job 取得 `contents: write`；其他 job 僅 `contents: read`。
GitHub Release 流程使用 `GITHUB_TOKEN`，不需新增外部發布密鑰。
不要移動已公開版本的 tag；修正後應發布下一版。

## 官方 Matplotlib 優化補丁相容性

[Official boxplot workflow](../.github/workflows/official-boxplot.yml) 在相關程式變更時，
分別使用 Python 3.10／Matplotlib 3.8.2 與 Python 3.12／Matplotlib 3.11.2 執行
`tests/bench/test_official_boxplot_fastpath.py`，也可手動啟動。
五項測試涵蓋統計逐值比較、原始 API／啟用模式、24 組圖像與可修改物件情境、
特殊 Axes／transform 回退、PNG／SVG／PDF／EPS 內容與 Figure pickle。
兩項工作都不建置或匯入本專案 Rust extension，執行紀錄上傳為 artifact。
這個相容性工作流驗證實驗補丁，套件的主要 CI 與發布 Python 版本仍由原流程管理。

## PyPI 發布

[Publish PyPI workflow](../.github/workflows/publish-pypi.yml) 在 Release workflow 成功後自動啟動，
也可手動指定既有版本（例如 `v0.0.2`）。手動執行時設定 `dry_run=true` 可只驗證、不上傳。

發布流程會確認 GitHub Release 已公開、tag 對應 commit 曾通過完整 Release workflow，
下載該版本的 wheel 與 source distribution，核對 `SHA256SUMS`、GitHub asset digest 與套件名稱／版本，
再執行 `twine check --strict`。效能紀錄壓縮檔與校驗碼本身不會上傳 PyPI。
因此 PyPI 使用與 GitHub Release 完全相同的已測試套件，不重新建置，也不移動既有 tag。

首次發布前，PyPI 專案擁有者須在
[PyPI Publishing](https://pypi.org/manage/account/publishing/) 建立 Pending Publisher：

| 欄位 | 值 |
| --- | --- |
| PyPI Project Name | `matplotlib-rs` |
| Owner | `g21589` |
| Repository | `matplotlib-rs` |
| Workflow | `publish-pypi.yml` |
| Environment | `pypi` |

上傳 job 使用 GitHub environment `pypi` 與 `id-token: write`，透過 PyPI Trusted Publishing 取得短效授權，
不需要設定長效 API token。既有 PyPI 專案則在該專案的 Publishing 設定中新增相同發布者。

```bash
# 只驗證既有 0.0.2，不上傳。
gh workflow run publish-pypi.yml -f tag=v0.0.2 -f dry_run=true

# 上傳尚未發布的新版本；將 vX.Y.Z 替換為實際的 GitHub Release tag。
gh workflow run publish-pypi.yml -f tag=vX.Y.Z -f dry_run=false
```

0.0.2 已發布至 [PyPI](https://pypi.org/project/matplotlib-rs/0.0.2/)，
可執行 `python -m pip install matplotlib-rs==0.0.2`。
PyPI 不允許覆寫已使用的檔名；請勿以重新建置的內容重傳相同版本。
