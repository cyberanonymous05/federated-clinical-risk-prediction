"""Download the open COVID-19 registry of the Mexican Ministry of Health (Direccion General de Epidemiologia; 'Libre Uso MX' open data licence),
three yearly files (about 300 MB zipped) and the data dictionary, into data/mx.  Usage: python -I download_mx.py"""
import os, urllib.request
BASE = "https://datosabiertos.salud.gob.mx/gobmx/salud/datos_abiertos/"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "mx")
FILES = {f"COVID19MEXICO{y}.zip": f"{BASE}historicos/{y}/COVID19MEXICO{y}.zip" for y in (2020, 2021, 2022)}
FILES["diccionario_datos_abiertos.zip"] = BASE + "diccionario_datos_abiertos.zip"
os.makedirs(OUT, exist_ok=True)
for name, url in FILES.items():
    dest = os.path.join(OUT, name)
    if os.path.exists(dest) and os.path.getsize(dest) > 1000:
        print("have", name); continue
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as r, open(dest + ".part", "wb") as f:
        while True:
            b = r.read(1 << 20)
            if not b: break
            f.write(b)
    os.replace(dest + ".part", dest); print("downloaded", name, os.path.getsize(dest))
