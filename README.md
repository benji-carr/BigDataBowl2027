# NFL Big Data Bowl 2027

Collaborative analysis of NFL Scouting Combine tracking and player data for the [NFL Big Data Bowl 2027 Kaggle competition](https://www.kaggle.com/competitions/nfl-big-data-bowl-2027/).

## Before you start

1. Open the [competition page](https://www.kaggle.com/competitions/nfl-big-data-bowl-2027/) and **skim the Overview** to understand the competition goals and rules.
2. **Skim the Data section**, particularly the file descriptions, column definitions, and field-coordinate conventions.
3. Join the competition and accept its rules on Kaggle. Each collaborator must use their own Kaggle account and credentials to access the data.

> **Data is not stored in this Git repository.** Kaggle competition files are too large for ordinary Git tracking. Download them locally as described below, and follow the competition's terms regarding data use and sharing.

## Local setup (Windows / PowerShell)

### 1. Clone the repository

```powershell
git clone <REPOSITORY_URL>
cd BigDataBowl2027
```

Replace `<REPOSITORY_URL>` with this project's GitHub clone URL.

### 2. Create and activate a Python environment

**Python 3.11 is recommended** for compatibility with the project's data-science tools.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python --version
```

If activation is blocked by PowerShell's execution policy, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` and activate again.

Install dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install kaggle
```

If `requirements.txt` is not yet present, install the minimum required tooling with `python -m pip install kaggle pandas numpy pyarrow matplotlib ipykernel` instead.

### 3. Generate your Kaggle API token

1. Sign in to [Kaggle](https://www.kaggle.com/).
2. Open [Kaggle Settings → API](https://www.kaggle.com/settings/api).
3. Generate a new API token and copy its value (modern tokens commonly start with `KGAT_`).
4. In the **repository root**, create a file named `.env` containing:

```dotenv
KAGGLE_API_TOKEN=KGAT_replace_with_your_own_token
```

Do not add quotation marks or spaces around `=`. **Never commit `.env` or share your token.** The `.env` file does not automatically load into a PowerShell session, so use the next command before calling Kaggle.

### 4. Load the API token into your terminal

From the repository root, run:

```powershell
$tokenLine = Get-Content .env | Where-Object { $_ -match '^KAGGLE_API_TOKEN=' } | Select-Object -First 1
if (-not $tokenLine) { throw 'KAGGLE_API_TOKEN is missing from .env' }
$env:KAGGLE_API_TOKEN = $tokenLine.Substring('KAGGLE_API_TOKEN='.Length).Trim().Trim('"').Trim("'")
```

Repeat this step when opening a **new terminal session**. Avoid printing the token to the console.

### 5. Download the competition data

Make sure you've joined the competition and accepted its rules. Then run:

```powershell
New-Item -ItemType Directory -Force data/raw | Out-Null
kaggle competitions download -c nfl-big-data-bowl-2027 -p data/raw
```

This downloads the competition archive to `data/raw/`. To extract ZIP archives:

```powershell
Get-ChildItem data/raw -Filter *.zip | ForEach-Object {
    Expand-Archive -LiteralPath $_.FullName -DestinationPath data/raw -Force
}
```

Verify the extracted files:

```powershell
Get-ChildItem data/raw
```

**Note:** Some Kaggle CLI versions support `--unzip` on the download command, but the download-then-extract workflow above works without relying on that flag. If you receive a permission error, confirm you've joined the competition, accepted its rules, and loaded your token in the current terminal.

## Working with the data

Read files relative to the repository root, for example:

```python
from pathlib import Path
import pandas as pd

REPO_ROOT = Path.cwd().resolve()  # Run from repo root; adjust for notebooks/
combine_results = pd.read_csv(REPO_ROOT / 'data' / 'raw' / 'combine_results.csv')
```

For notebooks launched from `notebooks/`, use the repository-root discovery helper in your notebook instead of assuming the working directory is the root.

## Git and data hygiene

Keep these entries in `.gitignore`:

```gitignore
.venv/
.env
data/
__pycache__/
.ipynb_checkpoints/
.pytest_cache/
```

Do not commit Kaggle competition archives, extracted data, local credentials, or other large generated artifacts. Commit analysis code, notebooks, documentation, and appropriately sized outputs instead. Before pushing, check `git status` to ensure no raw data or secrets are staged.

## References

- [NFL Big Data Bowl 2027 — Overview and Data](https://www.kaggle.com/competitions/nfl-big-data-bowl-2027/)
- [Kaggle API settings](https://www.kaggle.com/settings/api)
- [Official Kaggle CLI documentation](https://github.com/Kaggle/kaggle-cli/blob/main/docs/README.md)
