# Proyecto: un clasificador de imágenes propio (IPCC – ML)

Proyecto del curso IPCC (Machine Learning). Recorremos el proceso completo de
clasificación de imágenes con una red neuronal convolucional, pero con un
conjunto de datos construido por el grupo: **tres objetos físicos fotografiados
con el celular**.

## Qué hay en este repositorio

| Archivo | Qué es |
|---|---|
| `proyecto_clasificador.pdf` / `.tex` | Enunciado del proyecto (fuente en LaTeX y PDF compilado). |
| `clase_clasificacion_imagenes.ipynb` | Notebook de la clase (piedra, papel o tijera con fastai/PyTorch). Es la base que vamos a reutilizar cambiando las imágenes. |
| `clasificador_propio.ipynb` | Notebook adaptado a nuestras fotos (carpetas como variables, parámetros de los experimentos arriba, prueba en vivo). |
| `app/` | **Programa para probar los modelos** desde el navegador: una foto, prueba en vivo con 5 fotos y evaluación completa (ver `app/README.md`). |
| `scripts/verificar_fotos.py` | Revisa el conjunto de fotos antes de entrenar: cantidades, fotos dañadas o repetidas, borrosas, y si el fondo delata la clase. |
| `docs/documento-de-estudio.md` | Documento vivo del equipo: reparto de tareas, protocolo de fotos, explicación del código celda por celda, experimentos, guion del video, calendario y bitácora. |

Lo que **no** se sube (ver `.gitignore`): los datos de ejemplo de la clase, la
lista del curso, el paper de referencia, el video final y los modelos entrenados (`modelos/`).

## Resumen del enunciado

- **Grupos de 4 personas.** Se entrega **solo un video de 5 minutos**
  (no el notebook ni las imágenes).
- **Plazo:** 26 de octubre, 10:00 am.
- **Envío:** a `mbastidaso@unal.edu.co` y `jfgarciava@unal.edu.co`, con el
  archivo nombrado con los cuatro apellidos en orden alfabético separados por
  guion bajo (ej. `Cardona_Gómez_Restrepo_Zapata.mp4`).

### Datos
- 3 objetos distinguibles entre sí, fotografiables con celular.
- **60 imágenes por objeto** para entrenar (180 en total, 45 por persona),
  variando fondo, luz, distancia y ángulo.
- **10 imágenes por objeto** para prueba, tomadas en condiciones distintas.
- Una carpeta por objeto: el nombre de la carpeta es la etiqueta.
- Con 3 clases el azar acierta el 33 %; la meta es superar el 90 %.

### Experimentos
Dos modificaciones distintas (una a la vez), reentrenando y registrando el
efecto sobre el accuracy.

### Qué debe aparecer en el video
1. El conjunto de imágenes.
2. El código, celda por celda (qué hace, por qué, qué pasaría si se quitara).
3. Accuracy y matriz de confusión.
4. Prueba en vivo con 5 imágenes del conjunto de prueba, sin cortes.
5. Análisis: qué par de objetos confunde, por qué, y cuándo usarían o no el clasificador.

### Rúbrica (100 pts)
Explicación del código 35 · Prueba en vivo y análisis del error 25 ·
Conjunto de imágenes 20 · Experimentos 10 · Realización del video 10.

El accuracy no da puntos por sí mismo: se evalúa el razonamiento.

## Cómo trabajar

### 1. Revisar las fotos antes de subirlas
Con las fotos en una carpeta `fotos/` (`entrenamiento/<objeto>/` y `prueba/<objeto>/`). **Los nombres de las carpetas
deben ser idénticos en entrenamiento y prueba**, en minúscula y sin tildes ni espacios (`borrador`, `lapiz`, `sacapuntas`):
la carpeta original `"Sacapuntas "` tenía un espacio al final y para el modelo eso es otra clase.

```bash
pip install pillow
python3 scripts/verificar_fotos.py fotos      # en Windows: py scripts\verificar_fotos.py fotos
```

Avisa de cantidades distintas de 60 y 10 por objeto, fotos dañadas, repetidas (sobre todo entre
entrenamiento y prueba), muy pequeñas o borrosas, poca variación de luz y un posible «atajo del
fondo». También guarda hojas de contacto en `fotos_revision/` para mirar la variedad a ojo. Los
avisos son heurísticos: son una pista para mirar, no un veredicto.

### 2. Abrir el notebook en Colab
1. Las fotos ya están en este repositorio (`fotos/`, ver `fotos/LEEME.md`). Para Colab, subir esa carpeta a Google Drive
   (descargar el repositorio como `.zip` desde GitHub y subir `fotos/`).
2. En Colab: *Archivo → Abrir cuaderno → GitHub* y pegar `69kingDavid69/proyecto-red-neuronal`
   (el repositorio es público: no hace falta autorizar nada).
   Alternativa: descargar `clasificador_propio.ipynb` y subirlo a Colab.
3. *Entorno de ejecución → Cambiar tipo de entorno de ejecución → T4 GPU*.
4. Ajustar las celdas marcadas **⚙ PARÁMETROS** (nombres de objetos y rutas) y ejecutar en orden.

### 3. Entrenar y registrar
Entrenar, evaluar y anotar los resultados y los dos experimentos en `docs/documento-de-estudio.md`
(secciones 5 y 6 y la bitácora).

## Equipo

- David (`69kingDavid69`)
- Juan Gómez (`juanj-g`)
- Emanuel Aristizábal (`emanuel02ss`)
- Nicolás Quintana (`NothingLogic`)
