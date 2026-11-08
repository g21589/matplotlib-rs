# PNG 存檔：Rust 編碼器

0.0.4 的 `savefig` 以 `png` crate 最快模式壓縮（檔案約為 Pillow 的 2 倍），
`imsave`、`AxesImage.write_png`、色彩對應表的 `_repr_png_` 與帶 `pil_kwargs` 的 `savefig`
在已安裝 Pillow 時都交給 Pillow。現在**所有 PNG 檔都由 Rust 寫出**，不論是否安裝 Pillow：

- 每列以 libpng／Pillow 的準則選擇濾波（None／Sub／Up／Average／Paeth 中有號位元組絕對值總和最小者），
  以 32 位元組為一組的向量化迴圈計算；x86_64 執行時偵測 AVX2，否則使用 SSE2 版本，兩者輸出相同。
- 以 zlib-rs 壓縮，預設 **zlib 等級 6**，與 Pillow（也就是 Matplotlib）相同。
- 過濾後的資料切成約 256 KiB 的區段，各以前一段最後 32 KiB 作為字典、以 sync flush 結尾，
  在多個執行緒上同時過濾與壓縮，再接成單一 zlib 串流（與 pigz 相同的作法），每段一個 IDAT chunk。
  區段邊界只由影像大小決定，**檔案與執行緒數無關**（`MPLRS_THREADS=1`、3、8 逐位元組相同）。
- `savefig` 直接從 Rust renderer 的緩衝區編碼，不再複製 RGBA；壓縮期間釋放 GIL。
- chunk 與 Matplotlib 相同：IHDR、tEXt（`Software` 與 `metadata`，非 Latin-1 文字寫成 iTXt，
  與 Pillow 的 `PngInfo.add_text` 一致）、pHYs（`int(dpi / 0.0254 + 0.5)`）、IDAT、IEND。
- `pil_kwargs` 的 PNG 壓縮選項 `compress_level`、`optimize`（等級 9）、`compress_type`（zlib 策略）、
  `dpi`、`format` 由 Rust 處理，語意同 Pillow，**不需要 Pillow**；其他選項（如 `pnginfo`）仍交給 Pillow。

之後的版本完全不使用 Pillow：SVG／PDF 內嵌影像與其他影像格式也改由 Rust 處理，見 README〈影像檔〉。

## 測試方法

- [`tests/bench/png_save.py`](../../tests/bench/png_save.py)：先以 matplotlib-rs 繪製 9 張圖的 RGBA 畫布，
  每種編碼器在全新行程中預熱 1 次、量測 9 次取中位數；每個檔案都解碼比對，必須與畫布逐像素相同。
- `pillow`：Matplotlib 存 PNG 的方式（`PIL.Image.save`，帶 `PngInfo` 與 dpi，zlib 等級 6）。
- `0.0.4`：上一版 `savefig` 使用的 `_mplrs.encode_png(..., 1)`（commit `bdf710a` 建置）。
- `新 L6`／`新 L1`：本次的編碼器，等級 6（預設）與 1；「1 執行緒」為 `MPLRS_THREADS=1`。
- 環境：CPython 3.13.18、NumPy 2.3.5、Pillow 13.0.0、Linux x86_64、
  Intel Xeon Processor @ 2.30GHz、4 vCPU（支援 AVX2）。原始數據見 [`png-save.json`](png-save.json)。

## 結果：只量 PNG 編碼

每格為「毫秒／kB」。

| 畫布 | Pillow | 0.0.4 | 新 L6 | 新 L6 1 執行緒 | 新 L1 | 新 L1 1 執行緒 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| line，640 × 480 | 8.44／23.6 | 0.28／48.9 | 0.94／21.8 | 2.32／21.8 | 0.66／33.7 | 1.24／33.7 |
| grid（線、散佈、影像、直方圖），800 × 600 | 27.05／272.1 | 1.11／339.6 | 2.75／271.5 | 8.49／271.5 | 1.41／360.3 | 3.71／360.3 |
| grid，2400 × 1800 | 125.26／611.2 | 6.15／1012.3 | 10.35／599.1 | 39.74／599.1 | 5.93／842.9 | 18.58／842.9 |
| field（bilinear imshow），1600 × 1200 | 161.98／438.3 | 6.00／833.2 | 18.79／417.5 | 61.86／417.5 | 3.29／885.2 | 11.36／885.2 |
| contourf，1200 × 900 | 48.16／489.8 | 2.62／743.5 | 5.57／471.9 | 18.76／471.9 | 2.42／612.0 | 7.27／612.0 |
| hatch，1200 × 900 | 28.63／53.7 | 1.54／255.1 | 2.17／50.5 | 7.55／50.5 | 1.63／122.0 | 3.85／122.0 |
| 3 × 3 subplots，1200 × 960 | 44.75／279.8 | 2.37／625.0 | 4.15／256.9 | 15.54／256.9 | 2.02／339.2 | 6.01／339.2 |
| 3D surface，1200 × 900 | 36.99／219.0 | 1.97／269.6 | 3.38／201.8 | 12.85／201.8 | 1.60／307.1 | 5.35／307.1 |
| photo（bicubic 彩色影像），1200 × 900 | 243.48／762.2 | 3.78／2174.0 | 27.09／769.9 | 91.98／769.9 | 4.15／1428.7 | 12.39／1428.7 |
| **合計** | **724.7／3150** | **25.8／6301** | **75.2／3061** | **259.1／3061** | **23.1／4931** | **69.8／4931** |

- 預設（等級 6）比 Pillow 快 **9.6 倍**（單執行緒 2.8 倍），檔案合計小 2.8%；
  9 張中 8 張比 Pillow 小，photo 大 1.0%。
- 0.0.4 的 `savefig` 編碼器很快，但檔案是 Pillow 的 2.0 倍（hatch 4.8 倍、photo 2.9 倍）。
  等級 1 比它更快（23.1 對 25.8 毫秒）且小 22%。
- 等級 1 的檔案比 Pillow 等級 1 大（zlib-rs 沿用 zlib-ng 的 `deflate_quick`）。

## 結果：`savefig` 全程

`fig.savefig(BytesIO(), format="png")`，含重新繪製、壓縮與寫入；預熱 1 次後 7 次的中位數，「毫秒／kB」。

| 圖 | Matplotlib＋Pillow | 0.0.4 | 本次 |
| --- | ---: | ---: | ---: |
| grid，100 DPI | 88.0／271.6 | 13.2／338.9 | 14.7／270.9 |
| grid，300 DPI | 219.0／615.3 | 69.3／1020.2 | **47.2**／602.4 |
| field，200 DPI | 237.4／438.3 | 67.9／833.2 | **62.4**／417.5 |
| grid，100 DPI，`pil_kwargs={"compress_level": 1}` | 80.5／297.9 | 31.8／297.9 | **13.3**／359.8 |
| grid，300 DPI，`compress_level=1` | 195.5／715.4 | 134.8／715.4 | **36.2**／845.7 |
| field，200 DPI，`compress_level=1` | 133.3／613.3 | 122.0／613.3 | **53.5**／885.2 |

大圖不再複製緩衝區並平行壓縮，比 0.0.4 快且檔案約減半；800 × 600 的圖多 1.5 毫秒，換得小 20% 的檔案。
需要最快速度時可用 `pil_kwargs={"compress_level": 1}`，0.0.4 這種情況會交給 Pillow，現在快 2.4～3.7 倍。

## 驗證

- `cargo test`：各色彩型態（灰階、灰階＋alpha、RGB、RGBA）、多種大小、等級 0～9 與各 zlib 策略往返解碼相同，
  chunk CRC、zlib header 正確，且輸出不隨執行緒數改變。
- `tests/unit/test_native_ports.py`：Pillow 與內建讀取器都能讀回相同像素；
  非 IDAT chunk 與 Pillow 寫出的逐位元組相同（含 Latin-1、bytes 與非 Latin-1 的 metadata、不同的 x／y dpi）；
  Python 的 `zlib` 解壓並檢查合併後的 Adler-32；`pil_kwargs` 的選項與交回 Pillow 的情況。
- `tests/unit/test_numpy_only.py`：只有 NumPy 時 `compress_level`／`optimize` 可用，其他 `pil_kwargs` 仍提示需要 Pillow。
- `scripts/ci_test.sh`：59／59 個繪圖案例的 PNG 與 Matplotlib 逐像素相同，SVG／PDF／EPS 內容相同。

## 重跑

```bash
sh scripts/dev_build.sh --locked
python tests/bench/png_save.py --output tests/output/png-save
# 加上舊版：在另一個目錄建置舊版 extension 後
python tests/bench/png_save.py --output tests/output/png-save --base-source ../matplotlib-rs-0.0.4
```
