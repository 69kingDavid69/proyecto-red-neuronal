"""Ensayo con una red PREENTRENADA (resnet18 de ImageNet) sobre las fotos reales, y su control sin preentrenar.

Compara 3 configuraciones, cada una con 3 semillas (mismos datos, misma partición, mismo aumento):
  1. resnet18 preentrenada, 128 px     2. resnet18 preentrenada, 224 px     3. resnet18 SIN preentrenar, 224 px
La 3 sirve de control: misma arquitectura, solo cambia el punto de partida de los pesos.
La primera vez descarga los pesos (~45 MB, https://download.pytorch.org/models/resnet18-f37072fd.pth).
Tarda ~25 min en un Mac sin GPU. Requiere fastai. Uso:  python ensayo_preentrenada.py [carpeta_fotos]
"""
import sys, warnings; warnings.filterwarnings("ignore")
from fastai.vision.all import *
RAIZ = Path(sys.argv[1] if len(sys.argv) > 1 else "fotos")

def corrida(semilla, tam, libres=3, epocas=10, guardar=None, pre=True):
    set_seed(semilla, reproducible=True)
    d = DataBlock(blocks=(ImageBlock, CategoryBlock), get_items=get_image_files, get_y=parent_label,
                  splitter=RandomSplitter(valid_pct=0.2, seed=42),
                  item_tfms=Resize(int(tam*1.25), method=ResizeMethod.Pad, pad_mode=PadMode.Zeros),
                  batch_tfms=aug_transforms(size=tam))
    dls = d.dataloaders(RAIZ/"entrenamiento", bs=16, num_workers=0)
    learn = vision_learner(dls, resnet18, pretrained=pre, metrics=accuracy)
    with learn.no_logging(), learn.no_bar():
        learn.fine_tune(epocas, base_lr=1e-3, freeze_epochs=libres)
    v = learn.recorder.values[-1]
    prueba = learn.dls.test_dl(get_image_files(RAIZ/"prueba"), with_labels=True, num_workers=0)
    _, acc = learn.validate(dl=prueba)
    if guardar: learn.export(guardar)
    return round(v[2],2), round(float(acc),2)

if __name__ == "__main__":
    configs = (("resnet18 preentrenada, 128 px", 128, True),
               ("resnet18 preentrenada, 224 px", 224, True),
               ("resnet18 SIN preentrenar, 224 px", 224, False))
    for nombre, tam, pre in configs:
        res = [corrida(s, tam, pre=pre) for s in range(3)]
        mv = sum(r[0] for r in res)/3; mp = sum(r[1] for r in res)/3
        print(f"{nombre:34s} valid {[r[0] for r in res]} (media {mv:.2f})  prueba {[r[1] for r in res]} (media {mp:.2f})", flush=True)
