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

Lo que **no** se sube (ver `.gitignore`): los datos de ejemplo de la clase, la
lista del curso, el paper de referencia y el video final.

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

1. Abrir el notebook en Google Colab con GPU activada.
2. Subir las fotos reducidas (el anexo del enunciado trae el script con PIL
   que las baja a 256 px) y apuntar `CARPETA` a la carpeta con las tres clases.
3. Entrenar, evaluar y registrar los experimentos.

## Equipo

- David (`69kingDavid69`)
- Juan Gómez (`juanj-g`)
- Emanuel Aristizábal (`emanuel02ss`)
- Nicolás Quintana (`NothingLogic`)
