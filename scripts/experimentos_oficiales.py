"""Los dos experimentos del proyecto, con 5 semillas cada uno, sobre las fotos reales.

  Experimento A: la MISMA resnet18 (224 px), con pesos preentrenados de ImageNet o con pesos aleatorios.
                 Solo cambia el punto de partida de los pesos.
  Experimento B: la red propia (CNN de 4 capas) con aumento de datos normal, fuerte o ninguno.
                 Solo cambia el aumento de datos.

Todas las corridas usan la misma partición entrenamiento/validación (semilla 42) y la misma base:
pad, 30 épocas (red propia) o 3 + 10 (resnet18), lote 16. Cada semilla (0 a 4) cambia el sorteo de los
pesos iniciales y del aumento. Las filas se van escribiendo en un CSV, así que si se interrumpe no se pierde lo hecho.

Tarda ~45 min en un Mac sin GPU (mucho menos con GPU). Requiere fastai.
Uso:  python experimentos_oficiales.py [carpeta_fotos] [archivo_salida.csv]
"""
import csv
import statistics
import sys
import time
import warnings

warnings.filterwarnings("ignore")
from fastai.vision.all import *
from torch import nn

RAIZ = Path(sys.argv[1] if len(sys.argv) > 1 else "fotos")
SALIDA = Path(sys.argv[2] if len(sys.argv) > 2 else "docs/resultados/experimentos.csv")
SEMILLAS = range(5)


def capa(entran, salen):
    return nn.Sequential(nn.Conv2d(entran, salen, 3, padding=1), nn.BatchNorm2d(salen), nn.ReLU(), nn.MaxPool2d(2))


def red_propia(n_clases=3):
    return nn.Sequential(capa(3, 16), capa(16, 32), capa(32, 64), capa(64, 128),
                         nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, n_clases))


def aumento(tipo, tam):
    if tipo == "ninguno":
        return []
    if tipo == "fuerte":
        return aug_transforms(size=tam, max_rotate=30., max_zoom=1.4, max_lighting=0.5, max_warp=0.3, flip_vert=True)
    return aug_transforms(size=tam)


def datos(tam, tipo_aumento, bs=16):
    lado_item = tam if tipo_aumento == "ninguno" else int(tam * 1.25)
    bloque = DataBlock(blocks=(ImageBlock, CategoryBlock), get_items=get_image_files, get_y=parent_label,
                       splitter=RandomSplitter(valid_pct=0.2, seed=42),
                       item_tfms=Resize(lado_item, method=ResizeMethod.Pad, pad_mode=PadMode.Zeros),
                       batch_tfms=aumento(tipo_aumento, tam))
    return bloque.dataloaders(RAIZ / "entrenamiento", bs=bs, num_workers=0)


def medir(learn):
    validacion = learn.recorder.values[-1][2]
    prueba = learn.dls.test_dl(get_image_files(RAIZ / "prueba"), with_labels=True, num_workers=0)
    _, acierto = learn.validate(dl=prueba)
    return round(float(validacion), 4), round(float(acierto), 4)


def corrida_propia(semilla, tipo_aumento):
    set_seed(semilla, reproducible=True)
    learn = Learner(datos(128, tipo_aumento), red_propia(), loss_func=CrossEntropyLossFlat(), metrics=accuracy)
    with learn.no_logging(), learn.no_bar():
        learn.fit_one_cycle(30, 3e-3)
    return medir(learn)


def corrida_resnet(semilla, preentrenada):
    set_seed(semilla, reproducible=True)
    learn = vision_learner(datos(224, "normal"), resnet18, pretrained=preentrenada, metrics=accuracy)
    with learn.no_logging(), learn.no_bar():
        learn.fine_tune(10, base_lr=1e-3, freeze_epochs=3)
    return medir(learn)


EXPERIMENTOS = [
    ("B", "Red propia, aumento normal (base)",   lambda s: corrida_propia(s, "normal")),
    ("B", "Red propia, aumento fuerte",          lambda s: corrida_propia(s, "fuerte")),
    ("B", "Red propia, sin aumento",             lambda s: corrida_propia(s, "ninguno")),
    ("A", "resnet18 SIN preentrenar (base)",     lambda s: corrida_resnet(s, False)),
    ("A", "resnet18 preentrenada",               lambda s: corrida_resnet(s, True)),
]


def main():
    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    with SALIDA.open("w", newline="", encoding="utf-8") as f:
        escritor = csv.writer(f)
        escritor.writerow(["experimento", "configuracion", "semilla", "validacion", "prueba", "segundos"])
        for exp, nombre, correr in EXPERIMENTOS:
            filas = []
            for s in SEMILLAS:
                t0 = time.time()
                val, pru = correr(s)
                filas.append((val, pru))
                escritor.writerow([exp, nombre, s, val, pru, round(time.time() - t0)])
                f.flush()
            v, p = [x[0] for x in filas], [x[1] for x in filas]
            print(f"[{exp}] {nombre:36s} validación {statistics.mean(v):.2f} ({min(v):.2f}-{max(v):.2f})   "
                  f"prueba {statistics.mean(p):.2f} ({min(p):.2f}-{max(p):.2f})", flush=True)


if __name__ == "__main__":
    main()
