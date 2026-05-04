# Data

WebNLG v3.0 English. Downloaded automatically on first run by [code/webnlg_loader.py](../code/webnlg_loader.py); we do not commit raw data.

## What gets downloaded

The loader pulls the official corpus archive from the WebNLG GitLab repo (~25 MB):

```
https://gitlab.com/shimorina/webnlg-dataset
```

It extracts only `release_v3.0/en/{train,dev,test}/**/*.xml` into `data/raw/v3.0_en/` and parses it into `{src, refs}` records.

After download you'll see:

```
data/raw/
├── webnlg-v3.zip
└── v3.0_en/
    ├── train/  (~169 XML files across 1triples..7triples and category subdirs)
    ├── dev/
    └── test/
```

Expected counts: **train=13,211** input groupings, **dev=1,667**, **test=3,934**.

## Why a custom loader

Hugging Face `datasets >= 4.0` removed support for script-based loaders (which `web_nlg` uses), and even on `datasets < 4.0` the script's extracted paths exceed Windows' MAX_PATH (260 chars) on machines without long-path support enabled. The custom loader sidesteps both: it writes via the `\\?\` long-path prefix on Windows and is a single file with no HF dependency.

## Manual download (offline)

If your environment can't reach gitlab.com, download the archive manually and place it at `data/raw/webnlg-v3.zip` — the loader will skip the download and go straight to extraction.
