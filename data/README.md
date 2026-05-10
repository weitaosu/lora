# Data

Two datasets, one per project track.

## WebNLG (primary track)

`data/webnlg/raw/` is populated automatically on first run by [code/webnlg/webnlg_loader.py](../code/webnlg/webnlg_loader.py); we do not commit raw data.

The loader pulls the official corpus archive (~25 MB) from the WebNLG GitLab repo:

```
https://gitlab.com/shimorina/webnlg-dataset
```

It extracts only `release_v3.0/en/{train,dev,test}/**/*.xml` (or `release_v2.1/xml/...` for the paper-exact split) into `data/webnlg/raw/` and parses it into `{src, refs}` records.

After download you'll see:

```
data/webnlg/
└── raw/
    ├── webnlg-v3.zip
    └── v3.0_en/
        ├── train/  (~169 XML files across 1triples..7triples and category subdirs)
        ├── dev/
        └── test/
```

Expected counts: **train=13,211** input groupings, **dev=1,667**, **test=3,934**.

### Why a custom loader

Hugging Face `datasets >= 4.0` removed support for script-based loaders (which `web_nlg` uses), and even on `datasets < 4.0` the script's extracted paths exceed Windows' MAX_PATH (260 chars) on machines without long-path support enabled. The custom loader sidesteps both: it writes via the `\\?\` long-path prefix on Windows and is a single file with no HF dependency.

### Manual download (offline)

If your environment can't reach gitlab.com, download the archive manually and place it at `data/webnlg/raw/webnlg-v3.zip` — the loader will skip the download and go straight to extraction.

## Werner Bronkhorst paintings (extension track)

`data/diffusion_lora/train/` and `data/diffusion_lora/train_filtered/` hold the diffusion-LoRA training images. Both are gitignored (~150 MB combined). The filtered set (77 manually-curated images + hand-written .txt sidecar captions) is the one used for the v3 keeper LoRA. See [code/diffusion_lora/flux2/manual_review.json](../code/diffusion_lora/flux2/manual_review.json) for the per-image keep/drop manifest with reasons.
