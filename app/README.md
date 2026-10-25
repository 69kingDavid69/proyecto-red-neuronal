# Programa para probar el clasificador

Una página local para probar los modelos entrenados con **lápiz · borrador · sacapuntas**: sube o toma una
foto y mira qué responde cada modelo. No sube nada a internet: el programa solo atiende en `127.0.0.1`.

## Cómo usarlo

Requiere los modelos guardados por el notebook (sección 7 → `modelo_objetos_base.pkl`; sección 8 →
`modelo_objetos_preentrenado.pkl`) dentro de la carpeta `modelos/`, y las fotos de prueba en `fotos/prueba/`.
Si se entrenó en Colab, descargar los `.pkl` desde Drive a `modelos/` **sin cambiarles el nombre**.

**Mac / Linux:**

```bash
python3 -m venv .venv                  # solo la primera vez
.venv/bin/pip install fastai           # solo la primera vez (descarga ~1 GB)
.venv/bin/python app/servidor.py
```

**Windows (PowerShell):**

```powershell
py -m venv .venv                       # solo la primera vez
.venv\Scripts\pip install fastai       # solo la primera vez (descarga ~1 GB)
.venv\Scripts\python app\servidor.py
```

> Los `.pkl` se deben cargar con una versión de fastai parecida a la de Colab. Si al cargarlos sale un error de
> versión, instalar la misma: en Colab `import fastai; fastai.__version__` y aquí `pip install fastai==<esa versión>`.

Abrir <http://127.0.0.1:8765>. Para cerrar el programa: `Ctrl + C` en la terminal.
Otras rutas: `--puerto 9000`, `--modelos otra_carpeta`, `--prueba otra_carpeta`.

## Qué hace

- **Una foto:** arrastra una foto (o pégala, o usa la cámara) y aparece el sello con la respuesta, la seguridad
  del modelo y la probabilidad de cada clase. Si ninguna clase pasa de 55 %, dice que el modelo duda.
- **Prueba en vivo:** 5 fotos de la carpeta de prueba al azar, clasificadas una tras otra, con acierto o error.
  Es la prueba que pide el enunciado para el video.
- **Evaluación completa:** pasa todas las fotos de prueba y muestra aciertos, la matriz de confusión y las
  fotos donde se equivocó.
- **Modelo:** *Red propia*, *Preentrenada* o *Comparar* (los dos a la vez).

## Seguridad

- Solo escucha en `127.0.0.1` y rechaza peticiones con otro `Host`.
- Solo sirve fotos de la carpeta de prueba (rutas validadas) y limita las imágenes subidas a 12 MB.
- Los `.pkl` usan *pickle* y pueden ejecutar código al cargarse: cargar solo modelos que hizo el propio grupo.
- Las fotos que subes se clasifican en memoria y no se guardan.

## Archivos

`servidor.py` (servidor y API, solo librería estándar + fastai) · `web/index.html` · `web/estilos.css` · `web/app.js`.
