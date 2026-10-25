import os
import sys
import zipfile

def load_dotenv(dotenv_path=".env"):
    """Simple parser to load environment variables from .env file."""
    if os.path.exists(dotenv_path):
        with open(dotenv_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    # Remove surrounding quotes if present
                    val = val.strip().strip("'\"")
                    os.environ[key.strip()] = val
        print("Loaded environment variables from .env")

def main():
    load_dotenv()

    # Verify Kaggle credentials
    kaggle_user = os.environ.get("KAGGLE_USERNAME")
    kaggle_key = os.environ.get("KAGGLE_KEY")
    user_home_kaggle = os.path.exists(os.path.expanduser("~/.kaggle/kaggle.json"))

    if not (kaggle_user and kaggle_key) and not user_home_kaggle:
        print("Error: Kaggle API credentials not found.", file=sys.stderr)
        print("Please configure one of the following:", file=sys.stderr)
        print("1. Create a '.env' file in the project root with:", file=sys.stderr)
        print("   KAGGLE_USERNAME=your_username", file=sys.stderr)
        print("   KAGGLE_KEY=your_api_key", file=sys.stderr)
        print("2. Set KAGGLE_USERNAME and KAGGLE_KEY environment variables.", file=sys.stderr)
        print("3. Place 'kaggle.json' in ~/.kaggle/ directory.", file=sys.stderr)
        sys.exit(1)

    # Import kaggle after setting env vars so it authenticates correctly
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except Exception as e:
        print(f"Error importing Kaggle API: {e}", file=sys.stderr)
        sys.exit(1)

    print("Authenticating with Kaggle API...")
    try:
        api = KaggleApi()
        api.authenticate()
    except Exception as e:
        print(f"Kaggle authentication failed: {e}", file=sys.stderr)
        sys.exit(1)

    dataset = os.environ.get("KAGGLE_DATASET", "wordsforthewise/lending-club")
    raw_dir = os.environ.get("RAW_DATA_DIR", "data/raw/lendingclub")
    os.makedirs(raw_dir, exist_ok=True)

    print(f"Downloading Kaggle dataset '{dataset}'...")
    try:
        api.dataset_download_files(dataset, path=raw_dir)
        print("Download completed successfully.")
    except Exception as e:
        print(f"Error downloading dataset: {e}", file=sys.stderr)
        sys.exit(1)

    zip_path = os.path.join(raw_dir, dataset.split("/")[-1] + ".zip")
    if os.path.exists(zip_path):
        print(f"Extracting {zip_path} to {raw_dir}...")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(raw_dir)
            print("Extraction completed successfully.")
            os.remove(zip_path)
            print(f"Removed ZIP archive: {zip_path}")
        except Exception as e:
            print(f"Error extracting ZIP archive: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        print("Warning: Expected ZIP file was not found, checking if files already extracted.")

    # List extracted files
    files = os.listdir(raw_dir)
    print(f"Files in {raw_dir}:")
    for f in files:
        print(f" - {f}")

if __name__ == "__main__":
    main()
