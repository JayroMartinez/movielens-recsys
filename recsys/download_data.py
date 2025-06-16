import pathlib
import urllib.request
import zipfile

URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
DATA_DIR = pathlib.Path(__file__).resolve().parents[1] / "data" / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)
ZIP_PATH = DATA_DIR / "ml-100k.zip"

def download():
    if ZIP_PATH.exists():
        print("Dataset already downloaded")
        return
    print("Downloading MovieLens-100K…")
    urllib.request.urlretrieve(URL, ZIP_PATH)
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        zf.extractall(DATA_DIR)
    print("Done")

if __name__ == "__main__":
    download()
