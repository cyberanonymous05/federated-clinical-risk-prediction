"""Download the PhysioNet/Computing in Cardiology Challenge 2019 training data (open access; 40,336 patient files from two hospital systems)
into data/sepsis/training_setA and data/sepsis/training_setB.  Usage: python -I download_sepsis.py   (about 160 MB, 24 parallel connections).
Licence: PhysioNet lists CC BY 4.0; the LICENSE.txt shipped with the files is ODbL 1.0. Both require attribution; see the paper's reference list."""
import os, re, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

BASE = "https://physionet.org/files/challenge-2019/1.0.0/training/"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "sepsis")
EXPECT = {"training_setA": 20336, "training_setB": 20000}


def get(url, dest, tries=4):
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=60) as r:
                b = r.read()
            if not b.startswith(b"HR|O2Sat"):
                raise ValueError("unexpected content")
            tmp = dest + ".part"; open(tmp, "wb").write(b); os.replace(tmp, dest); return True
        except Exception:
            time.sleep(1 + 2 * k)
    return False


def main():
    bad = 0
    for s, n in EXPECT.items():
        d = os.path.join(OUT, s); os.makedirs(d, exist_ok=True)
        idx = urllib.request.urlopen(urllib.request.Request(BASE + s + "/", headers={"User-Agent": "Mozilla/5.0"}), timeout=60).read().decode()
        files = sorted(set(re.findall(r'href="(p\d+\.psv)"', idx))); assert len(files) == n, (s, len(files))
        todo = [f for f in files if not os.path.exists(os.path.join(d, f))]
        with ThreadPoolExecutor(24) as ex:
            res = list(ex.map(lambda f: get(BASE + s + "/" + f, os.path.join(d, f)), todo))
        bad += res.count(False); print(s, "files", len(files), "downloaded now", res.count(True), "failed", res.count(False), flush=True)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
