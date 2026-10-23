"""Verifica el conjunto de fotos ANTES de subirlo a Drive / entrenar.

Uso:
    python3 scripts/verificar_fotos.py fotos
    python3 scripts/verificar_fotos.py fotos --por-clase-entrenamiento 60 --por-clase-prueba 10

Espera esta estructura (una subcarpeta por objeto; el nombre de la carpeta es la etiqueta):
    fotos/entrenamiento/<objeto>/*.jpg|png
    fotos/prueba/<objeto>/*.jpg|png

Revisa (solo necesita Pillow: `pip install pillow`):
  ERRORES  - imágenes ilegibles; clases distintas entre entrenamiento y prueba; fotos idénticas
             repetidas (sobre todo entre entrenamiento y prueba: eso infla el accuracy).
  AVISOS   - cantidades distintas de las pedidas; fotos muy pequeñas; las más borrosas de cada
             clase; clases con poca variación de luz; y un posible «atajo del fondo» (si el fondo
             de una clase es sistemáticamente distinto al de las otras, el modelo puede aprender
             el fondo en vez del objeto).
Además guarda hojas de contacto (miniaturas) en <carpeta>_revision/ para mirar la variedad a ojo.

Los umbrales de los avisos son heurísticos: son una pista para mirar, no un veredicto.
"""
import argparse
import hashlib
import math
import statistics
import sys
from pathlib import Path

try:
    from PIL import Image, ImageFilter, ImageStat, ImageOps
except ImportError:
    sys.exit("Falta Pillow. Instalar con: pip install pillow")

EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
LAPLACIANO = ImageFilter.Kernel((3, 3), [0, 1, 0, 1, -4, 1, 0, 1, 0], scale=1, offset=128)


def fotos_de(carpeta):
    return sorted(p for p in carpeta.rglob("*") if p.suffix.lower() in EXT and not p.name.startswith("."))


def clases_de(split):
    return sorted(d.name for d in split.iterdir() if d.is_dir() and not d.name.startswith("."))


def abrir(ruta):
    """Abre respetando la orientación EXIF del celular; devuelve None si está dañada."""
    try:
        with Image.open(ruta) as im:
            im.verify()
        im = Image.open(ruta)
        return ImageOps.exif_transpose(im).convert("RGB")
    except Exception:
        return None


def nitidez(img):
    """Varianza del laplaciano a tamaño fijo: más alta = más nítida. Solo sirve para COMPARAR fotos."""
    g = img.convert("L")
    g.thumbnail((256, 256))
    return ImageStat.Stat(g.filter(LAPLACIANO)).var[0]


def color_borde(img):
    """Color medio de una franja del 10 % en el borde de la foto (aproxima el fondo)."""
    w, h = img.size
    m = max(1, int(min(w, h) * 0.10))
    franjas = [img.crop((0, 0, w, m)), img.crop((0, h - m, w, h)),
               img.crop((0, 0, m, h)), img.crop((w - m, 0, w, h))]
    medias = [ImageStat.Stat(f).mean for f in franjas]
    return [sum(x[c] for x in medias) / 4 for c in range(3)]


def distancia(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def hoja_contacto(rutas, destino, columnas=10, lado=96):
    filas = math.ceil(len(rutas) / columnas)
    hoja = Image.new("RGB", (columnas * lado, filas * lado), "white")
    for i, r in enumerate(rutas):
        im = abrir(r)
        if im is None:
            continue
        im = ImageOps.fit(im, (lado, lado))
        hoja.paste(im, ((i % columnas) * lado, (i // columnas) * lado))
    hoja.save(destino, quality=85)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("carpeta", type=Path, help="carpeta con entrenamiento/ y prueba/")
    ap.add_argument("--por-clase-entrenamiento", type=int, default=60)
    ap.add_argument("--por-clase-prueba", type=int, default=10)
    args = ap.parse_args()

    errores, avisos = [], []
    base = args.carpeta
    esperado = {"entrenamiento": args.por_clase_entrenamiento, "prueba": args.por_clase_prueba}
    datos = {}   # split -> clase -> lista de (ruta, imagen_info)

    for split in esperado:
        carpeta = base / split
        if not carpeta.is_dir():
            errores.append(f"Falta la carpeta {carpeta}")
            continue
        datos[split] = {}
        for clase in clases_de(carpeta):
            filas = []
            for ruta in fotos_de(carpeta / clase):
                img = abrir(ruta)
                if img is None:
                    errores.append(f"Imagen ilegible o dañada: {ruta}")
                    continue
                filas.append(dict(ruta=ruta, lado=min(img.size), nitidez=nitidez(img),
                                  luz=ImageStat.Stat(img.convert("L")).mean[0], borde=color_borde(img)))
            datos[split][clase] = filas

    if len(datos) == 2:
        ce, cp = set(datos["entrenamiento"]), set(datos["prueba"])
        if ce != cp:
            errores.append(f"Las clases no coinciden: entrenamiento={sorted(ce)} prueba={sorted(cp)}")
    if "entrenamiento" in datos and len(datos["entrenamiento"]) != 3:
        avisos.append(f"El proyecto pide 3 objetos y hay {len(datos['entrenamiento'])} carpetas en entrenamiento.")

    # --- resumen y cantidades
    print("\n=== Cantidades ===")
    for split, clases in datos.items():
        for clase, filas in clases.items():
            n = len(filas)
            marca = "" if n == esperado[split] else f"   <-- se piden {esperado[split]}"
            print(f"  {split:14s} {clase:20s} {n:4d}{marca}")
            if n != esperado[split]:
                avisos.append(f"{split}/{clase}: hay {n} fotos y se piden {esperado[split]}.")

    # --- duplicados exactos (mismo contenido)
    vistos = {}
    for split, clases in datos.items():
        for clase, filas in clases.items():
            for f in filas:
                h = hashlib.md5(f["ruta"].read_bytes()).hexdigest()
                f["hash"] = h
                if h in vistos:
                    otro = vistos[h]
                    entre = otro["split"] != split
                    msg = f"Foto repetida: {f['ruta']}  ==  {otro['ruta']}"
                    (errores if entre else avisos).append(msg + ("  (¡entre entrenamiento y prueba!)" if entre else ""))
                else:
                    vistos[h] = dict(ruta=f["ruta"], split=split)

    # --- tamaño y nitidez
    print("\n=== Las 3 fotos menos nítidas de cada clase (nitidez relativa a la mediana de la clase) ===")
    for split, clases in datos.items():
        for clase, filas in clases.items():
            for f in filas:
                if f["lado"] < 128:
                    avisos.append(f"Foto muy pequeña (lado menor {f['lado']} px, la red usa 128): {f['ruta']}")
            if len(filas) >= 8:
                med = statistics.median(f["nitidez"] for f in filas)
                peores = sorted(filas, key=lambda f: f["nitidez"])[:3]
                print(f"  {split}/{clase}: " + ", ".join(f"{f['ruta'].name} ({f['nitidez'] / med:.2f})" for f in peores))
                for f in peores:
                    if f["nitidez"] < 0.55 * med:
                        avisos.append(f"Posible foto borrosa (nitidez {f['nitidez'] / med:.2f} de la mediana de su clase): {f['ruta']}")

    # --- variación de luz por clase
    if "entrenamiento" in datos:
        print("\n=== Variación de luz (entrenamiento) ===")
        for clase, filas in datos["entrenamiento"].items():
            if len(filas) >= 8:
                sd = statistics.pstdev(f["luz"] for f in filas)
                print(f"  {clase:20s} desviación del brillo medio: {sd:5.1f}")
                if sd < 12:
                    avisos.append(f"{clase}: las fotos tienen casi la misma luz (desviación {sd:.1f}). Variar la iluminación.")

        # --- atajo del fondo: ¿el borde de las fotos separa las clases?
        clases = {c: fl for c, fl in datos["entrenamiento"].items() if len(fl) >= 8}
        if len(clases) >= 2:
            centro = {c: [statistics.mean(f["borde"][k] for f in fl) for k in range(3)] for c, fl in clases.items()}
            dentro = {c: statistics.mean(distancia(f["borde"], centro[c]) for f in fl) for c, fl in clases.items()}
            print("\n=== Atajo del fondo (color de los bordes, entrenamiento) ===")
            nombres = list(clases)
            for i in range(len(nombres)):
                for j in range(i + 1, len(nombres)):
                    a, b = nombres[i], nombres[j]
                    entre = distancia(centro[a], centro[b])
                    ref = max(1.0, (dentro[a] + dentro[b]) / 2)
                    razon = entre / ref
                    print(f"  {a} vs {b}: distancia entre clases / dispersión interna = {razon:4.2f}")
                    if razon > 1.5:
                        avisos.append(f"{a} y {b} tienen fondos sistemáticamente distintos (razón {razon:.1f}): el modelo podría "
                                      f"aprender el fondo en lugar del objeto. Fotografiar cada objeto sobre los mismos fondos.")

    # --- hojas de contacto
    salida = base.parent / (base.name + "_revision")
    salida.mkdir(parents=True, exist_ok=True)
    for split, clases in datos.items():
        for clase, filas in clases.items():
            if filas:
                destino = salida / f"{split}_{clase}.jpg"
                hoja_contacto([f["ruta"] for f in filas], destino)
    print(f"\nHojas de contacto guardadas en: {salida}  (abrirlas y mirar si hay variedad de fondo, luz y ángulo)")

    print("\n=== Resultado ===")
    for e in errores:
        print("  ERROR :", e)
    for a in avisos:
        print("  AVISO :", a)
    if not errores and not avisos:
        print("  Sin errores ni avisos.")
    print(f"\n{len(errores)} error(es), {len(avisos)} aviso(s).")
    sys.exit(1 if errores else 0)


if __name__ == "__main__":
    main()
