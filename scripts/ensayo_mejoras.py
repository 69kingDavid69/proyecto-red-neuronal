"""Ensayo de mejoras con las FOTOS REALES, cambiando UNA cosa respecto a la base (pad, 30 épocas, lote 16, 128 px).

Cada configuración se entrena con 3 semillas; se imprime el accuracy de validación y de prueba y su media.
Tarda ~35-40 min en un Mac sin GPU. Requiere fastai. Uso:  python ensayo_mejoras.py [carpeta_fotos]
"""
import sys
import warnings; warnings.filterwarnings("ignore")
from fastai.vision.all import *
from torch import nn
RAIZ = Path(sys.argv[1] if len(sys.argv) > 1 else "fotos")

def capa(a, b):
    return nn.Sequential(nn.Conv2d(a, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU(), nn.MaxPool2d(2))

def red(anchos=(16, 32, 64, 128), dropout=0.0, n=3):
    capas, ent = [], 3
    for a in anchos:
        capas.append(capa(ent, a)); ent = a
    capas += [nn.AdaptiveAvgPool2d(1), nn.Flatten()]
    if dropout: capas.append(nn.Dropout(dropout))
    capas.append(nn.Linear(ent, n))
    return nn.Sequential(*capas)

def corrida(semilla=0, epocas=30, tasa=3e-3, bs=16, tam=128, aug="normal", anchos=(16,32,64,128), dropout=0.0, wd=None):
    set_seed(semilla, reproducible=True)
    if aug == "fuerte":
        bt = aug_transforms(size=tam, max_rotate=30., max_zoom=1.4, max_lighting=0.5, max_warp=0.3, flip_vert=True)
    else:
        bt = aug_transforms(size=tam)
    d = DataBlock(blocks=(ImageBlock, CategoryBlock), get_items=get_image_files, get_y=parent_label,
                  splitter=RandomSplitter(valid_pct=0.2, seed=42),
                  item_tfms=Resize(int(tam*1.25), method=ResizeMethod.Pad, pad_mode=PadMode.Zeros), batch_tfms=bt)
    dls = d.dataloaders(RAIZ/"entrenamiento", bs=bs, num_workers=0)
    learn = Learner(dls, red(anchos, dropout), loss_func=CrossEntropyLossFlat(), metrics=accuracy)
    with learn.no_logging(), learn.no_bar():
        learn.fit_one_cycle(epocas, tasa, wd=wd)
    v = learn.recorder.values[-1]
    prueba = learn.dls.test_dl(get_image_files(RAIZ/"prueba"), with_labels=True, num_workers=0)
    _, acc = learn.validate(dl=prueba)
    return round(v[2],2), round(float(acc),2)

configs = [
 ("BASE (pad, 30 ép, bs16, 128 px)",         dict()),
 ("resolución 192 px",                       dict(tam=192)),
 ("aumento FUERTE",                          dict(aug="fuerte")),
 ("red más ancha (32-64-128-256)",           dict(anchos=(32,64,128,256))),
 ("red con 5 capas (…-256)",                 dict(anchos=(16,32,64,128,256))),
 ("dropout 0,3",                             dict(dropout=0.3)),
 ("weight decay 0,1",                        dict(wd=0.1)),
]
for nombre, kw in configs:
    res = [corrida(semilla=s, **kw) for s in range(3)]
    mv = sum(r[0] for r in res)/3; mp = sum(r[1] for r in res)/3
    print(f"{nombre:36s} valid {[r[0] for r in res]} (media {mv:.2f})  prueba {[r[1] for r in res]} (media {mp:.2f})", flush=True)
