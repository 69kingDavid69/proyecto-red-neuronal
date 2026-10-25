# Fotos del proyecto

Conjunto de imágenes del clasificador de **lápiz · borrador · sacapuntas**, ya reducido (lado mayor de 256 px, `.jpg`,
sin metadatos EXIF ni ubicación). El nombre de cada carpeta es la **etiqueta** de las fotos que contiene.

```
fotos/
  entrenamiento/   60 fotos por objeto (180)    borrador/  lapiz/  sacapuntas/
  prueba/          10 fotos por objeto (30), tomadas en otro lugar y con otra luz
```

**Nombres de las carpetas.** En las carpetas originales la de entrenamiento se llamaba `Sacapuntas ` (con un espacio al
final) y la de prueba `Sacapuntas`; para el modelo eran dos etiquetas distintas. Aquí se normalizaron a minúsculas y sin
tildes (`borrador`, `lapiz`, `sacapuntas`). **No cambiar los nombres** ni mezclar fotos de entrenamiento con las de prueba.

Revisión: `python3 scripts/verificar_fotos.py fotos` (cantidades, fotos dañadas o repetidas, posible atajo del fondo).
Resultado de la revisión del 23/10: 0 errores; una foto posiblemente borrosa (`entrenamiento/lapiz/rsp_lap_17.jpg`) y
13 fotos muy alargadas (lápiz). Detalle en la sección 3.2 de `docs/documento-de-estudio.md`.

**Uso de las fotos:** son del grupo, para este trabajo. No subir ni compartir fuera del equipo.
