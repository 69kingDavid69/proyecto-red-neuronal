"""Ensayo de configuraciones con las FOTOS REALES del proyecto (fotos/entrenamiento y fotos/prueba).

Cada configuración se entrena con 3 semillas y se imprime el accuracy de validación y de prueba.
Tarda ~25-30 min en un Mac sin GPU. Requiere fastai. Uso:  python ensayo_fotos_reales.py [carpeta_fotos]
"""
import sys
import warnings; warnings.filterwarnings("ignore")
from fastai.vision.all import *
from torch import nn
RAIZ = Path(sys.argv[1] if len(sys.argv) > 1 else "fotos")

def capa(a, b):
    return nn.Sequential(nn.Conv2d(a, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU(), nn.MaxPool2d(2))
def red(n=3):
    return nn.Sequential(capa(3,16), capa(16,32), capa(32,64), capa(64,128),
                         nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, n))

def corrida(epocas, tasa, bs, metodo, aug=True, semilla=0):
    set_seed(semilla, reproducible=True)
    r = {"crop": ResizeMethod.Crop, "pad": ResizeMethod.Pad, "squish": ResizeMethod.Squish}[metodo]
    tam = 160 if aug else 128
    kw = dict(pad_mode=PadMode.Zeros) if metodo == "pad" else {}
    d = DataBlock(blocks=(ImageBlock, CategoryBlock), get_items=get_image_files, get_y=parent_label,
                  splitter=RandomSplitter(valid_pct=0.2, seed=42),
                  item_tfms=Resize(tam, method=r, **kw),
                  batch_tfms=aug_transforms(size=128) if aug else [])
    dls = d.dataloaders(RAIZ/"entrenamiento", bs=bs, num_workers=0)
    learn = Learner(dls, red(), loss_func=CrossEntropyLossFlat(), metrics=accuracy)
    with learn.no_logging(), learn.no_bar():
        learn.fit_one_cycle(epocas, tasa)
    v = learn.recorder.values[-1]
    prueba = learn.dls.test_dl(get_image_files(RAIZ/"prueba"), with_labels=True, num_workers=0, )
    # la prueba debe usar el MISMO método de resize que el entrenamiento: lo hereda del dls
    _, acc = learn.validate(dl=prueba)
    return round(v[2],2), round(float(acc),2)

configs = [
 ("clase: 12 ép, bs32, crop",         dict(epocas=12,  tasa=3e-3, bs=32, metodo="crop")),
 ("30 ép, bs16, crop",                dict(epocas=30,  tasa=3e-3, bs=16, metodo="crop")),
 ("30 ép, bs16, pad",                 dict(epocas=30,  tasa=3e-3, bs=16, metodo="pad")),
 ("30 ép, bs16, squish",              dict(epocas=30,  tasa=3e-3, bs=16, metodo="squish")),
 ("30 ép, bs16, crop, sin aumento",   dict(epocas=30,  tasa=3e-3, bs=16, metodo="crop", aug=False)),
 ("60 ép, bs16, pad",                 dict(epocas=60,  tasa=3e-3, bs=16, metodo="pad")),
 ("30 ép, bs16, pad, tasa 1e-2",      dict(epocas=30,  tasa=1e-2, bs=16, metodo="pad")),
 ("100 ép, bs16, pad, tasa 1e-2",     dict(epocas=100, tasa=1e-2, bs=16, metodo="pad")),
]
for nombre, kw in configs:
    res = [corrida(semilla=s, **kw) for s in range(3)]
    print(f"{nombre:36s} valid {[r[0] for r in res]}  prueba {[r[1] for r in res]}", flush=True)
