"""Sondeo de la API de data912.com: qué endpoints existen y si alguno trae historia."""

import json
import urllib.request

BASE = "https://data912.com"
UA = {"User-Agent": "Mozilla/5.0 (compomercado sondeo)"}


def get(ruta: str):
    req = urllib.request.Request(BASE + ruta, headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.status, r.read()


def resumen(ruta: str) -> None:
    try:
        status, cuerpo = get(ruta)
        datos = json.loads(cuerpo)
    except Exception as e:  # noqa: BLE001
        print(f"{ruta}: ERROR {e}")
        return
    if isinstance(datos, list):
        print(f"{ruta}: {status}, lista de {len(datos)} elementos")
        if datos:
            print("   primero:", json.dumps(datos[0], ensure_ascii=False)[:300])
            print("   último: ", json.dumps(datos[-1], ensure_ascii=False)[:300])
    else:
        print(f"{ruta}: {status}, {json.dumps(datos, ensure_ascii=False)[:500]}")


try:
    _, cuerpo = get("/openapi.json")
    spec = json.loads(cuerpo)
    print("== Endpoints publicados ==")
    for ruta, metodos in spec.get("paths", {}).items():
        for metodo, info in metodos.items():
            print(f"{metodo.upper():6} {ruta}  — {info.get('summary', '')} {info.get('description', '')[:150]}")
    print("== Info ==", json.dumps(spec.get("info", {}), ensure_ascii=False)[:800])
except Exception as e:  # noqa: BLE001
    print("openapi.json: ERROR", e)

print("== Pruebas ==")
for ruta in ["/live/arg_stocks", "/live/arg_cedears", "/live/mep", "/live/ccl",
             "/historical/stocks/GGAL", "/historical/cedears/SPY", "/historical/bonds/AL30",
             "/historical/stocks/YPFD"]:
    resumen(ruta)
