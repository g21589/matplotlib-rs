# CI 與 GitHub Release

## 自動測試

[CI workflow](../.github/workflows/ci.yml) 在 branch push、pull request、手動執行與 release 呼叫時啟動。
Linux x86_64／Ubuntu 24.04，Python 3.11、3.12、3.13 分別建置 release extension，執行：

- Python／Cargo／lockfile 版本檢查。
- Rust 單元測試與 doctest 探索。
- 全部 `tests/unit/test_*.py`、`tests/lowlevel/diff_*.py` 與原生模型案例。
- 全部自有繪圖案例；PNG 逐像素一致，SVG／PDF／EPS 內容一致。

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

測試輸出與日誌位於 `tests/output/ci/`；CI 成功或失敗皆上傳為 Actions artifact，保留 14 天。
完整上游 gallery 不在每次 CI 執行範圍。

## 發布

[Release workflow](../.github/workflows/release.yml) 由 `vX.Y.Z` tag push 觸發，流程如下：

1. 檢查 tag、`pyproject.toml`、`Cargo.toml`、`Cargo.lock` 相符，且存在該版 release notes。
2. 呼叫完整 CI；同時建置 CPython 3.11／3.12／3.13 的 manylinux_2_28 x86_64 wheel 與 sdist。
3. 每個 wheel 安裝到沒有參考 Matplotlib 的乾淨 venv，測試可變 boxplot、mathtext、layout 與 PNG／SVG／PDF／EPS。
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
流程使用 GitHub 的 `GITHUB_TOKEN`，不需新增外部發布密鑰，不會自動發布 PyPI。
不要移動已公開版本的 tag；修正後應發布下一版。
