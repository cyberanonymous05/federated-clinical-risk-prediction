"""Download the NHANES 2005-2018 public-use files used in task T4 into data/nhanes (7 cycles x 8 files)."""
import os, urllib.request
os.makedirs("data/nhanes", exist_ok=True)
for y, s in [(2005, "D"), (2007, "E"), (2009, "F"), (2011, "G"), (2013, "H"), (2015, "I"), (2017, "J")]:
    for t in ["DEMO", "BMX", "BPX", "TCHOL", "HDL", "GHB", "GLU", "DIQ"]:
        f = f"data/nhanes/{t}_{s}.xpt"
        if not os.path.exists(f):
            b = urllib.request.urlopen(urllib.request.Request(f"https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{y}/DataFiles/{t}_{s}.xpt", headers={"User-Agent": "Mozilla/5.0"}), timeout=60).read()
            assert b[:6] == b"HEADER", f
            open(f, "wb").write(b)
