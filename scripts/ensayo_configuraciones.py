"""Ensayo de configuraciones con datos SUSTITUTOS (manos sintéticas de la clase, 60 imágenes por clase).

NO son los datos del proyecto: sirve para saber, antes de tener las fotos propias, cómo se comporta
el notebook con un conjunto de este tamaño. Cada configuración se corre con 3 semillas.

Uso (desde una carpeta que tenga fotos/entrenamiento/<clase>/ y fotos/prueba/<clase>/):
    python ensayo_configuraciones.py
Requiere fastai.
"""
import sys, warnings, torch, random
warnings.filterwarnings("ignore")
from fastai.vision.all import *
from torch import nn

def capa(a, b):
    return nn.Sequential(nn.Conv2d(a, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU(), nn.MaxPool2d(2))
def red(n=3):
    return nn.Sequential(capa(3,16), capa(16,32), capa(32,64), capa(64,128),
                         nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, n))

def corrida(epocas, tasa, bs, aug=True, semilla=0):
    set_seed(semilla, reproducible=True)
    d = DataBlock(blocks=(ImageBlock, CategoryBlock), get_items=get_image_files, get_y=parent_label,
                  splitter=RandomSplitter(valid_pct=0.2, seed=42),
                  item_tfms=Resize(160) if aug else Resize(128),
                  batch_tfms=aug_transforms(size=128) if aug else [])
    dls = d.dataloaders(Path("fotos/entrenamiento"), bs=bs, num_workers=0)
    learn = Learner(dls, red(), loss_func=CrossEntropyLossFlat(), metrics=accuracy)
    with learn.no_logging(), learn.no_bar():
        learn.fit_one_cycle(epocas, tasa)
    v = learn.recorder.values[-1]
    prueba = learn.dls.test_dl(get_image_files(Path("fotos/prueba")), with_labels=True, num_workers=0)
    _, acc_test = learn.validate(dl=prueba)
    return v[2], float(acc_test), v[0], v[1]

configs = [
  ("base (12 ép, bs32, 3e-3)",       dict(epocas=12, tasa=3e-3, bs=32)),
  ("40 épocas, bs32",                dict(epocas=40, tasa=3e-3, bs=32)),
  ("12 ép, bs16",                    dict(epocas=12, tasa=3e-3, bs=16)),
  ("30 ép, bs16",                    dict(epocas=30, tasa=3e-3, bs=16)),
  ("30 ép, bs16, tasa 1e-2",         dict(epocas=30, tasa=1e-2, bs=16)),
  ("30 ép, bs16, sin aumento",       dict(epocas=30, tasa=3e-3, bs=16, aug=False)),
]
for nombre, kw in configs:
    res = [corrida(semilla=s, **kw) for s in range(3)]
    va = [r[0] for r in res]; te = [r[1] for r in res]
    print(f"{nombre:34s} valid {[round(x,2) for x in va]}  prueba {[round(x,2) for x in te]}", flush=True)
