# Documento de estudio y preparación — Clasificador de imágenes propio (IPCC – ML)

> **Para qué sirve este documento.** Es la memoria viva del proyecto: aquí se anota
> *qué se va construyendo*, *cómo funciona cada parte del código* y *qué se va a decir en el
> video*. Se estudia de aquí y se prepara la exposición desde aquí.
> Se actualiza después de cada avance (ver la sección 9, **Bitácora**).

**Entrega:** un video de 5 minutos, **lunes 26 de octubre de 2026, 10:00 am**.
**Envío:** `mbastidaso@unal.edu.co` y `jfgarciava@unal.edu.co`.
**Archivo:** los cuatro apellidos en orden alfabético con guion bajo, p. ej. `Apellido1_Apellido2_Apellido3_Apellido4.mp4`.
**Rúbrica (100 pts):** explicación del código 35 · prueba en vivo y análisis del error 25 ·
conjunto de imágenes 20 · experimentos 10 · realización del video 10.

---

## 1. Reparto de tareas

Dos personas se encargan de las fotos (**F1, F2**) y dos del código (**C1, C2**).
Completar los nombres:

| Rol | Persona | Responsabilidad principal |
|---|---|---|
| **F1** | _(por definir)_ | Fotos de entrenamiento, objetos 1–3 (la mitad) |
| **F2** | _(por definir)_ | Fotos de entrenamiento, objetos 1–3 (la otra mitad) |
| **C1** | David | Datos y *pipeline* (secciones 0–2 del notebook) + experimento A |
| **C2** | _(por definir)_ | Modelo, entrenamiento y evaluación (secciones 3–5) + experimento B |

> **Aunque el trabajo se reparta, el video lo hablan los cuatro y a todos se les puede
> preguntar por todo el código.** Por eso cada persona estudia la sección 4 completa; el reparto
> solo decide quién *construye* cada parte.

### 1.1 Fotógrafos (F1 y F2)

1. **Elegir los tres objetos entre los cuatro** (decisión conjunta, hoy). Criterios del enunciado:
   distinguibles entre sí y disponibles en cantidad o variedad (ver sección 2).
2. **Fotos de entrenamiento: 180 en total (60 por objeto).** Si solo F1 y F2 las toman, son
   **90 cada uno, 30 por objeto**.
3. Variar a propósito fondo, luz, distancia y ángulo (matriz de la sección 2.3).
4. Reducir las fotos con el script del anexo del enunciado, comparar original y reducida, y subir
   las carpetas a Drive (sección 2.5).
5. Contar y verificar: 60 por carpeta, sin duplicados, sin fotos borrosas.

### 1.2 Programadores (C1 y C2)

| | **C1 — datos y pipeline** | **C2 — modelo y evaluación** |
|---|---|---|
| Notebook | Secciones **0, 1 y 2**: entorno/GPU, cargar las carpetas propias, balance, partición, `DataBlock`, aumento de datos | Secciones **3, 4 y 5**: `crear_red`, entrenamiento, curvas, filtros, matriz de confusión, prueba |
| Adaptación | Cambiar `CARPETA` a las fotos propias y quitar la descarga de `rps.zip` | Revisar que las curvas y la matriz muestren las 3 clases nuevas |
| Experimento | **A** (modificación 1) | **B** (modificación 2) |
| Video | Explica el bloque de datos y *pipeline* | Explica el bloque de modelo y entrenamiento |

Tareas que son **de los dos**: hacer correr el notebook en Colab con GPU, anotar los resultados en
la sección 6, y escribir/estudiar la sección 4.

### 1.3 Propuesta para el conjunto de prueba

El enunciado pide que las **10 imágenes por objeto de prueba** se tomen en *condiciones distintas*
al entrenamiento (otro lugar, otro momento, otra luz). Propuesta: **que las tomen C1 y C2** (15 por
persona, 5 por objeto), con otro celular y en otro lugar distinto de donde se hizo el entrenamiento.
Así el conjunto de prueba es realmente independiente y los fotógrafos no cargan con 30 fotos más.

> **Decisión pendiente de confirmar con el grupo.** El enunciado sugiere que los cuatro tomen 45
> fotos cada uno. Si solo dos fotografían, hay menos celulares, menos manos y menos fondos, y eso
> puede bajar la variedad (parte de los 20 puntos del conjunto de imágenes). Si se puede, que
> C1 y C2 aporten también algunas fotos de entrenamiento con su propio celular.

---

## 2. Protocolo de fotos

### 2.1 Elegir los tres objetos

- Que se distingan entre sí a simple vista (si dos se parecen, el error se concentra en ese par).
- Que haya muchas unidades o muchas formas de verlos.
- Ejemplos del enunciado: lápiz / borrador / sacapuntas; tres frutas; llaves / audífonos /
  cargador; tres herramientas.

**Objetos elegidos:** **lápiz · borrador · sacapuntas** (carpetas `lapiz`, `borrador`, `sacapuntas`).

### 2.2 Estructura de carpetas
La etiqueta de cada imagen es **el nombre de su carpeta**:

```
fotos/
  entrenamiento/
    <objeto1>/   60 imágenes
    <objeto2>/   60 imágenes
    <objeto3>/   60 imágenes
  prueba/
    <objeto1>/   10 imágenes
    <objeto2>/   10 imágenes
    <objeto3>/   10 imágenes
```

Usar nombres de carpeta sin tildes ni espacios (`lapiz`, no `lápiz `).

### 2.3 Matriz de variación
Un conjunto de 60 imágenes casi idénticas aporta mucho menos que 60 variadas. Repartir las fotos
de cada objeto entre estas variaciones:

| Factor | Variaciones |
|---|---|
| Fondo | mesa de madera · pared blanca · piso · tela de color · escritorio desordenado |
| Luz | natural de ventana · luz artificial · sombra · contraluz suave |
| Distancia | cerca · media · lejos |
| Ángulo | de frente · de arriba · de lado · girado |
| Ejemplar | si es una categoría (p. ej. «fruta»), usar ejemplares distintos |

### 2.4 Qué evitar

- Que **un objeto siempre salga con el mismo fondo** y otro con otro: el modelo podría aprender
  el fondo en lugar del objeto.

- Fotos borrosas, mal enfocadas o donde el objeto casi no se ve.
- Mezclar fotos de entrenamiento con las de prueba.

### 2.5 Reducir y subir
Una foto de celular pesa ~3 MB; la red entrena con 128 píxeles. Reducir con el script del anexo
del enunciado (`img.thumbnail((256, 256))`, JPG calidad 85) y **abrir una original y su reducida
lado a lado** para comprobar que se ven equivalentes. Luego subir a Drive y apuntar `CARPETA` a la
carpeta de entrenamiento.

### 2.6 Verificar el conjunto con el script
Antes de subir a Drive, correr `python3 scripts/verificar_fotos.py fotos` (necesita `pip install pillow`).
Detecta fotos dañadas, **fotos repetidas entre entrenamiento y prueba** (eso infla el accuracy),
cantidades distintas de 60/10, fotos muy pequeñas o borrosas, poca variación de luz y un posible
**atajo del fondo** (si el borde de las fotos separa las clases, el modelo puede aprender el fondo
en vez del objeto). Guarda hojas de contacto en `fotos_revision/` para mirar la variedad a ojo.
Los avisos son heurísticos: son una pista, no un veredicto.

### 2.7 Lista de verificación

- [ ] 60 fotos por objeto en entrenamiento (180 en total)
- [ ] 10 fotos por objeto en prueba, tomadas en condiciones distintas (30 en total)
- [ ] Variedad en fondo, luz, distancia y ángulo
- [ ] Carpetas con el nombre del objeto
- [ ] Reducidas y subidas a Drive
- [ ] Original vs. reducida comparadas

---

## 3. Cómo se trabaja el notebook

- Se ejecuta en **Google Colab** con GPU: *Entorno de ejecución → Cambiar tipo de entorno de
  ejecución → T4 GPU*. Con GPU toma unos 3 minutos; sin ella, más de media hora.

- **Ya está listo el notebook adaptado: `clasificador_propio.ipynb`.** Parte del de clase
  (`clase_clasificacion_imagenes.ipynb`) con estos cambios: sin la descarga de `rps.zip`; las carpetas
  son variables (`CARPETA_ENTRENAMIENTO`, `CARPETA_PRUEBA`); una celda opcional para montar Drive y
  otra para reducir las fotos; los valores de los experimentos arriba (`AUMENTO`, `BS`, `EPOCAS`,
  `TASA`); matriz de confusión sobre las fotos de **prueba**; y una celda de **prueba en vivo** (5
  imágenes con su predicción y qué tan seguro estaba). Las celdas marcadas **⚙ PARÁMETROS** son las
  únicas que se tocan.
- **Qué falta para usarlo:** poner los nombres de los tres objetos en la primera celda y apuntar las
  carpetas a las fotos reales.

- Con tres clases el azar acierta el **33 %**: ese número es el punto de partida, no un resultado.
  La meta del enunciado es superar el **90 %**.

- Referencia: el notebook de clase, con las manos sintéticas, guarda una exactitud de
  **90,59 %** sobre las 372 imágenes de prueba.

### 3.1 Hallazgo importante: con la configuración de la clase el modelo falla con 180 imágenes

> _Esta sección es del ensayo con **datos sustitutos** (manos sintéticas). Con las **fotos reales** (lápiz · borrador · sacapuntas) el panorama es distinto: ver 3.2._

Se hizo un **ensayo con datos sustitutos** (60 imágenes por clase tomadas de las manos sintéticas de
la clase, 10 por clase de prueba; script en `scripts/ensayo_configuraciones.py`). **No son nuestros
datos ni nuestros resultados**: sirve para anticipar un problema, no para predecir el accuracy.

**Lo que se observó con los valores de la clase (12 épocas, lote 32):**
- Con ~144 imágenes de entrenamiento y lotes de 32 hay **4 ajustes por época, 48 en total**
  (en la clase eran ~63 por época).
- `train_loss` bajó (de 1,13 a 0,61) pero `valid_loss` **subió** (hasta ~6,8) y el accuracy de
  validación se quedó en **22 %**, **debajo del azar**. En las tres corridas el accuracy de prueba
  fue **33 %**; en la que se diagnosticó, el modelo contestó **«rock» en las 36 imágenes de validación**.
- **No era memorización (sobreajuste):** en modo evaluación el modelo falla **incluso sobre las
  mismas imágenes de entrenamiento** (accuracy 36,7 %, error 2,41), mientras que durante el
  entrenamiento el error era 0,61. Hay una diferencia entre cómo se comporta la red entrenando y
  evaluando; es **consistente con** que las estadísticas de `BatchNorm2d` (promedios móviles) no
  alcanzan a ajustarse en tan pocos pasos. _(Es una hipótesis razonable; no se probó aislando la
  capa.)_

**Resultados de las configuraciones (3 corridas cada una, datos sustitutos):**

| Configuración | Accuracy validación | Accuracy prueba |
|---|---|---|
| Clase: 12 épocas, lote 32, tasa 3e-3 | 0,22 · 0,22 · 0,22 | 0,33 · 0,33 · 0,33 |
| 40 épocas, lote 32 | 0,94 · 0,97 · 0,97 | 0,83 · 0,83 · 0,90 |
| 12 épocas, lote 16 | 0,94 · 0,94 · 0,92 | 0,87 · 0,80 · 0,63 |
| **30 épocas, lote 16 (base del notebook)** | 0,97 · 1,00 · 0,94 | 0,97 · 0,93 · 0,90 |
| 30 épocas, lote 16, tasa 1e-2 | 0,97 · 0,97 · 1,00 | 0,97 · 0,87 · 0,97 |
| 30 épocas, lote 16, **sin aumento de datos** | 0,97 · 1,00 · 0,97 | 0,87 · 0,80 · 0,90 |

**Qué se concluye:**
- Lo que importa es el **número total de ajustes** (imágenes ÷ lote × épocas), no la cantidad de
  épocas por sí sola. Con tan pocas imágenes hay que bajar el lote o subir las épocas.
- Por eso el notebook adaptado usa **`BS = 16` y `EPOCAS = 30`**. Es un cambio respecto a la clase
  que **hay que poder explicar en el video**.
- En este ensayo, quitar el aumento de datos **no cambió la validación pero sí bajó la prueba**
  (promedio ~0,86 contra ~0,93). Con 36 imágenes de validación esa diferencia no se ve; la de
  prueba sí. Es un buen candidato a experimento, **pero con datos sintéticos limpios**: con fotos
  reales el efecto puede ser distinto.
- Las corridas varían entre sí (el notebook no fija semilla para el entrenamiento): un mismo
  notebook dio 93 % y 97 % en prueba. **No sacar conclusiones de una sola corrida.**

### 3.2 Primera revisión y primera corrida con las fotos reales (23/10)

**Qué se recibió.** Dos carpetas: entrenamiento (`rsp/`) y prueba (`rps-test-set/`), con las clases
**Borrador, Lapiz y Sacapuntas**: **60 fotos por objeto en entrenamiento y 10 en prueba**, todas `.jpg`,
con el lado mayor en 256 px (ya reducidas con el script del anexo). _Detalle:_ la carpeta de
entrenamiento se llama `"Sacapuntas "` **con un espacio al final** y la de prueba `"Sacapuntas"`; para
el modelo son dos etiquetas distintas. Se trabajó con una copia en `fotos/` con los nombres
normalizados (`borrador`, `lapiz`, `sacapuntas`); **las carpetas originales no se tocaron.**

**Revisión con `scripts/verificar_fotos.py`: 0 errores, 4 avisos.**
- Cantidades exactas (60/60/60 y 10/10/10), sin fotos dañadas ni repetidas (ni entre entrenamiento y prueba).
- **Variedad de luz buena** (desviación del brillo medio 26–31) y **sin atajo del fondo**: el color de los
  bordes casi no separa las clases (razón 0,09–0,18; el aviso salta desde 1,5). Mirando las hojas de contacto del lápiz y del borrador
  (la del sacapuntas no se revisó a ojo), hay fondos, ejemplares y encuadres muy distintos.
- **Una foto posiblemente borrosa:** `entrenamiento/lapiz/rsp_lap_17.jpg` (índice 0,91). Mirarla; si no se ve bien, descartarla.
- **Fotos muy alargadas:** 10 del lápiz en entrenamiento (hasta 6:1), 2 del lápiz en prueba y 1 del borrador.
  **Problema real:** con el redimensionado por defecto (`Resize`, método *crop*) fastai **centra y recorta**, y en
  una foto así se pierde la mayor parte del lápiz. Por eso el notebook tiene el parámetro `REDIMENSION`
  (`"crop"`, `"pad"` o `"squish"`), con `"pad"` por defecto: conserva la foto entera con bordes negros.

**Corridas (3 semillas cada una; 36 fotos de validación, 30 de prueba):**

| Configuración | Accuracy validación | Accuracy prueba |
|---|---|---|
| Clase: 12 épocas, lote 32, *crop* | 0,53 · 0,44 · 0,53 | 0,57 · 0,50 · 0,43 |
| 30 épocas, lote 16, *crop* | 0,69 · 0,72 · 0,69 | 0,53 · 0,63 · 0,50 |
| 30 épocas, lote 16, ***pad*** (base del notebook) | 0,72 · 0,72 · 0,75 | 0,57 · 0,63 · 0,60 |
| 30 épocas, lote 16, *squish* | 0,72 · 0,69 · 0,69 | 0,47 · 0,50 · 0,57 |
| 30 épocas, lote 16, *crop*, **sin aumento** | 0,72 · 0,72 · 0,69 | 0,60 · 0,60 · 0,53 |
| 60 épocas, lote 16, *pad* | 0,75 · 0,75 · 0,69 | 0,53 · 0,57 · 0,53 |
| 30 épocas, lote 16, *pad*, tasa 1e-2 | 0,72 · 0,67 · 0,72 | 0,50 · 0,57 · 0,60 |
| 100 épocas, lote 16, *pad*, tasa 1e-2 | 0,75 · 0,75 · 0,75 | 0,57 · 0,60 · 0,60 |

Una corrida del notebook completo (base con `pad`) dio **56,67 % en prueba**. Los pares más confundidos
fueron **sacapuntas → borrador (5 veces)**, **borrador → sacapuntas (3)** y **lápiz → borrador (3)**.

**Qué se concluye (y qué no):**
- **La red actual llega a ~70 % en validación y ~55–60 % en prueba**, lejos del 90 % que sugiere el enunciado
  (el azar es 33 %). Sí aprende (con la configuración de la clase ni siquiera pasa de ~50 %).
- **Entrenar más no ayuda:** 60 o 100 épocas o una tasa mayor dan lo mismo. El límite parece estar en los datos y en la
  red (pocas fotos, objetos muy variados), no en el tiempo de entrenamiento.
- **Casi todas las diferencias entre configuraciones son menores que el ruido.** Con 30 fotos de prueba, una sola
  foto son 3,3 puntos, y entre corridas de la **misma** configuración hay más de 10 puntos de diferencia (p. ej. 0,50–0,63). **No se puede
  afirmar** que `pad` sea mejor que `crop`, ni que el aumento de datos ayude o no: aquí «sin aumento» no salió peor,
  a diferencia del ensayo sintético. Cada experimento debe **repetirse varias veces y compararse por el promedio**.
- La **caída de validación a prueba** (~0,72 → ~0,57) es coherente con que las fotos de prueba se tomaron en otras
  condiciones, pero con 30 fotos no se puede separar del azar.
- **Lo que no se probó** (ideas para el análisis o un experimento, sin evidencia todavía): una red preentrenada
  (transfer learning), más fotos por objeto, o cambiar el recorte. El notebook de la clase entrena desde cero.
- **Esto es un buen material para el video:** el enunciado dice que un 50 % bien explicado vale más que un 99 %
  sin entender. Explicar *por qué* la red se queda en ~70 % (y qué muestra la matriz de confusión) es justo lo que se califica.

### 3.3 Mejoras probadas, análisis de errores y modelo guardado (24/10)

**Mejoras: una sola cosa a la vez, partiendo de la base** (*pad*, 30 épocas, lote 16, 128 px; 3 semillas; medias):

| Cambio respecto a la base | Validación (media) | Prueba (media) | Prueba, 3 corridas |
|---|---|---|---|
| **Base** | **0,73** | **0,60** | 0,57 · 0,63 · 0,60 |
| Resolución 192 px | 0,71 | 0,56 | 0,47 · 0,63 · 0,57 |
| Aumento de datos **fuerte** (giros 30°, zoom 1,4, luz 0,5, volteo vertical) | 0,55 | 0,63 | 0,63 · 0,67 · 0,60 |
| Red más ancha (32-64-128-256) | 0,69 | 0,57 | 0,57 · 0,60 · 0,53 |
| Red con 5 capas (hasta 256) | 0,75 | 0,61 | 0,60 · 0,67 · 0,57 |
| *Dropout* 0,3 antes de la capa lineal | 0,68 | 0,65 | 0,63 · 0,70 · 0,63 |
| *Weight decay* 0,1 | 0,74 | 0,57 | 0,57 · 0,57 · 0,57 |

**Lectura honesta:** **ningún cambio mejora de forma clara.** Todas las medias de prueba están entre 0,56 y 0,65, y con
30 fotos de prueba una sola foto son 3,3 puntos. Lo que se ve, con cautela: el aumento fuerte **baja** la validación
(0,55: a la red le cuesta más ajustar el entrenamiento en 30 épocas) pero la prueba **no baja** (0,63); el *dropout* da la
mejor prueba (0,65) con menor validación (0,68). Son diferencias pequeñas, no una prueba de que ayuden. Es decir, la red
y los datos actuales topan en **~70 % de validación y ~55–65 % de prueba**.

**Errores del modelo guardado** (14 errores de 30 fotos de prueba = 53,3 %): acertó 7/10 lápices, 5/10 borradores y 4/10
sacapuntas.
- **Sacapuntas (6 errores, todos «borrador»):** tres de las seis (`rsp_test_saca_2`, `_3` y `_5`) tienen el sacapuntas **pequeño, lejos y sobre un
  piso gris**. En el entrenamiento casi no hay fotos así (la mayoría son primeros planos). _Hipótesis, no probada:_ la red
  no vio objetos tan pequeños ni ese fondo con ese objeto.
- **Lápiz (3 errores, todos «borrador»):** dos son las fotos **muy alargadas** de la prueba (proporción > 5:1), que con
  `pad` quedan como una franja fina dentro de un cuadro negro; la tercera es un lápiz rosado.
- **Borrador (5 errores):** 4 «sacapuntas» y 1 «lápiz». Tres de ellos son el mismo borrador amarillento sobre una tela
  oscura, en fotos casi idénticas.
- Es un **desplazamiento de condiciones** (distancia, fondo, ejemplares nuevos): justo lo que el enunciado pide al
  tomar la prueba «en otro lugar, otro momento, otra luz».

**Modelo guardado.** `learn.export` generó `modelos/modelo_objetos_base.pkl` (0,47 MB; la carpeta está en `.gitignore`).
Se volvió a cargar en un **proceso nuevo** y dio **53,33 % en prueba, igual que en el notebook**, así que el guardado es
fiel. El notebook tiene ahora la sección 7 (guardar y cargar), con la precaución de que el `.pkl` usa *pickle* y solo se
cargan modelos propios.

**Qué falta probar** (sin evidencia todavía): más fotos de las situaciones que fallan (sacapuntas lejos, lápices alargados), o juntar las fotos de prueba con las de
entrenamiento para validar mejor. _(La red preentrenada se probó después: ver 3.4.)_

### 3.4 La red preentrenada sí supera el 90 % (24/10)

Con permiso del grupo se descargaron los pesos de **resnet18 preentrenada en ImageNet** (`resnet18-f37072fd.pth`,
44,7 MB, de `download.pytorch.org`; el prefijo del nombre coincide con el hash SHA-256 del archivo) y se ajustó con las mismas
fotos, la misma partición y el mismo aumento (*fine-tuning*: 3 épocas con el resto congelado y 10 ajustando todo).

| Red | Validación (media) | Prueba (media) | Prueba, 3 corridas |
|---|---|---|---|
| CNN propia (base, 128 px) | 0,73 | 0,60 | 0,57 · 0,63 · 0,60 |
| resnet18 **preentrenada**, 128 px | 0,91 | 0,80 | 0,80 · 0,83 · 0,77 |
| resnet18 **preentrenada**, 224 px | **0,97** | **0,92** | 0,93 · 0,97 · 0,87 |
| resnet18 **sin preentrenar** (control), 224 px | 0,50 | 0,57 | 0,57 · 0,57 · 0,57 |

**Qué se concluye (con evidencia):**
- **El preentrenamiento es lo que produce la mejora.** La misma resnet18 con los mismos 224 px, pero con pesos aleatorios,
  se queda en 0,50 / 0,57, igual que la CNN propia; con los pesos de ImageNet pasa a 0,97 / 0,92. La diferencia (35 puntos de
  prueba) es mucho mayor que el ruido de las corridas (~10 puntos), así que aquí sí se puede afirmar.
- **La resolución también importa para la red preentrenada:** 128 px → 0,80 y 224 px → 0,92 (resnet18 se entrenó con 224 px).
- **Por qué funciona (explicación, no demostrada aquí):** con solo 144 fotos no alcanza para aprender a «ver» desde cero; la red
  preentrenada ya trae detectores de bordes, texturas y formas aprendidos con ~1,2 millones de fotos (la idea del paper de
  AlexNet), y solo aprende a combinarlos para nuestros tres objetos.
- **Cuidado al explicarlo:** cambiar a resnet18 preentrenada cambia **varias cosas a la vez** (arquitectura, pesos de partida y
  resolución). La comparación limpia «una sola cosa» es la del control: **misma resnet18, solo preentrenada o no.**

_(Con 5 semillas, sección 5: preentrenada 0,98 y 0,90; sin preentrenar 0,55 y 0,55. Las cifras de esta sección son de 3 semillas.)_

**En el notebook** (sección 8) una corrida dio **86,67 % en prueba** (26 de 30); los 4 errores fueron borrador ↔ sacapuntas
(2 y 2), y el lápiz no falló. La media de las 3 corridas del ensayo es 0,92: **una sola corrida puede dar 0,87 o 0,97.**
El modelo se guardó en `modelos/modelo_objetos_preentrenado.pkl` (47 MB, ignorado por Git) y se volvió a cargar en un proceso nuevo.

**Pendiente de decidir por el grupo:** cuál es «el modelo del proyecto» (la CNN propia que se explica celda por celda, o la
preentrenada como mejora) y cuáles son los dos experimentos. **Sugerencia:** explicar el código de la CNN propia (es lo que
pide la rúbrica, 35 pts) y presentar la preentrenada como un experimento con su control.

### 3.5 Programa para probar los modelos (`app/`)

Para probar los modelos sin abrir el notebook (y para la prueba en vivo del video) hay un programa local en `app/`:
un servidor de Python (`app/servidor.py`, solo librería estándar + fastai) y una página (`app/web/`). Se abre con
`.venv/bin/python app/servidor.py` y luego <http://127.0.0.1:8765>. Detalle de uso y seguridad en `app/README.md`.

- **Una foto:** arrastrar, pegar o tomar con la cámara; muestra el sello con la respuesta, qué tan seguro está el modelo y
  la probabilidad de cada clase. Si ninguna pasa de 55 %, dice que **el modelo duda** en vez de afirmar.
- **Prueba en vivo:** 5 fotos de prueba al azar, clasificadas una tras otra con acierto o error (con «Comparar», los dos modelos).
- **Evaluación completa:** las 30 fotos de prueba, con aciertos, matriz de confusión y las fotos donde se equivoca.
- **Diseño:** papel cuadriculado, colores de los objetos (amarillo lápiz, rosa borrador, azul acero sacapuntas) y el veredicto
  como un **sello de goma**; claro y oscuro, y se ve bien en pantalla pequeña.
- **Verificación:** se probó con las fotos reales y los dos modelos. La evaluación completa dio **26 de 30 (86,7 %)** para la
  preentrenada (igual que en el notebook); la API rechaza una petición con otro `Host`, una ruta fuera de la carpeta de
  prueba (`../../README.md`) y un archivo que no es imagen. **Limitaciones:** no se probó la cámara (el navegador del
  entorno de pruebas no tiene una), ni en Windows; el programa usa los `.pkl` entrenados con el notebook.
- Los modelos **no están en el repositorio** (`modelos/` está en `.gitignore`): cada integrante los genera con el notebook
  (secciones 7 y 8) o se los pasan por Drive. Los `.pkl` usan *pickle*: cargar solo modelos del propio grupo.

---

## 4. Explicación del código, celda por celda

> El enunciado pide que, de las celdas importantes, se diga **qué hace**, **por qué está** y **qué
> pasaría si se quitara**. Este es el material para eso. Las frases marcadas _(a verificar)_
> son hipótesis razonables que conviene comprobar con una corrida antes de afirmarlas en el video.

### Mapa del recorrido

```
imágenes en carpetas ──► DataBlock (etiqueta, partición, tamaño, aumento)
        ──► red CNN de 4 capas ──► entrenamiento (12 épocas) ──► evaluación
                                                                (accuracy, matriz de confusión, prueba)
```

### Numeración de celdas: clase → nuestro notebook

Las secciones de abajo usan la numeración del **notebook de clase**. En el video se muestra
`clasificador_propio.ipynb`, donde las celdas cambian de lugar (contando desde 0, como `nbformat`).
**Colab no muestra números de celda:** en el video guíense por los títulos de sección (`## 1. Los datos`, etc.).

| Qué es | Clase | Nuestro notebook |
|---|---|---|
| GPU · instalar · importar | 2 · 4 · 5 | 2 · 4 · 5 |
| Montar Drive · reducir fotos (nuevas) | — | 7 · 9 |
| Listar imágenes y etiquetas (+ **comprueba que las carpetas de prueba se llamen igual**) | 9 | 11 |
| Balance de clases (+ **galería de 8 fotos por objeto** y conteo de prueba) | 11 | 13 |
| Mirar una foto · píxeles | 12 · 14 | 14 · 16 |
| **`DataBlock`** (con `AUMENTO`, `BS`, `REDIMENSION`) | 16 | **19** |
| `show_batch` | 17 | 20 |
| **La red (`capa`, `crear_red`)** | 20 | **23** |
| **`Learner` y entrenamiento** (`EPOCAS`, `TASA`; imprime los ajustes por época) | 22 | **25** |
| Curvas · filtros | 23 · 25 | 26 · 28 |
| `show_results` · matriz (validación) · `plot_top_losses` | 27 · 29 · 30 | 30 · 32 · 33 |
| **Accuracy en prueba** (+ tabla `registro_exp` de experimentos) | 32 | **35** |
| **Matriz de confusión en prueba** (nueva) | — | **37** |
| **Prueba en vivo** (nueva) | — | **39** |
| Guardar · cargar el modelo (nuevas) | — | 41 · 43 |
| **resnet18, experimento A** (`PREENTRENADA`) · su evaluación y registro (nuevas) | — | **45** · 46 |

### Celda 2 — Verificar la GPU

- **Qué hace:** `torch.cuda.is_available()` revisa si Colab asignó una GPU y la nombra.
- **Por qué:** entrenar una red es hacer muchísimas operaciones iguales en paralelo, justo lo que
  hace una GPU.

- **Si se quita:** el código sigue funcionando, pero entrenaría en CPU y tardaría más de media hora
  en lugar de ~3 minutos.

### Celdas 4 y 5 — Instalar e importar fastai

- **Qué hace:** `pip install fastai` instala la biblioteca; `from fastai.vision.all import *` trae
  `DataBlock`, `Learner`, `Resize`, `aug_transforms`, etc.; `nn` de PyTorch trae las piezas para
  construir la red.

- **Por qué:** fastai permite entrenar un modelo de visión con poco código; por debajo usa PyTorch.
- **Si se quita:** nada de lo siguiente se encuentra (`NameError`).

### Celda 7 — Descargar los datos (**se elimina en nuestro proyecto**)

- **Qué hace:** descarga `rps.zip` y `rps-test-set.zip` y los descomprime en `datos/`.
- **En nuestro proyecto:** se reemplaza por nuestras carpetas de fotos. No se descarga nada.

### Celda 9 — Listar imágenes y sus etiquetas
```python
CARPETA = Path("datos/rps")
imagenes = get_image_files(CARPETA)
parent_label(imagenes[0])
```

- **Qué hace:** `get_image_files` recorre la carpeta y sus subcarpetas y devuelve las rutas de todas
  las imágenes; `parent_label` devuelve el nombre de la carpeta de una imagen, que es su **etiqueta**.

- **Por qué:** en clasificación de imágenes la respuesta correcta no está en un archivo aparte: está
  en el nombre de la carpeta.

- **Qué cambia:** `CARPETA` pasa a apuntar a nuestras fotos de entrenamiento.
- **Si se quita:** no hay lista de imágenes ni etiquetas para el resto del pipeline.

### Celda 11 — Verificar que las clases estén balanceadas

- **Qué hace:** cuenta cuántas imágenes hay por clase (`Counter`) y las grafica en barras.
- **Por qué:** si el 95 % fueran de una clase, responder siempre esa clase acertaría 95 % sin haber
  aprendido nada. Con clases balanceadas esa salida no existe.

- **En nuestro proyecto:** deben salir 60 / 60 / 60.
- **Si se quita:** el código funciona, pero se pierde la comprobación de que los datos son sanos.

### Celdas 12 y 14 — Mirar los datos y ver qué recibe el modelo

- **Qué hace:** abre una imagen, muestra su etiqueta y tamaño, y la convierte en un arreglo de
  números con `image2tensor`: forma `(3, alto, ancho)` = (canales R, G, B; alto; ancho), con valores
  de 0 a 255.

- **Por qué:** el modelo no ve una foto, ve una **tabla de números**. Una imagen de 300 × 300 en color
  son 270.000 números.

- **Regla general:** nunca se entrena sin haber mirado los datos antes.

### Celda 16 — `DataBlock`: la receta de los datos (**de las más importantes**)
```python
datos = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    get_items=get_image_files,
    get_y=parent_label,
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    item_tfms=Resize(160),
    batch_tfms=aug_transforms(size=128),
)
dls = datos.dataloaders(CARPETA, bs=BS)
```
_(En `clasificador_propio.ipynb` `CARPETA` se llama `CARPETA_ENTRENAMIENTO` y `bs` es la variable `BS`.)_
| Argumento | Qué hace | Si se quitara o cambiara |
|---|---|---|
| `blocks=(ImageBlock, CategoryBlock)` | Entra una imagen, sale una categoría | fastai no sabría qué tipo de problema es |
| `get_items=get_image_files` | De dónde sacar los elementos | no habría imágenes |
| `get_y=parent_label` | La etiqueta es el nombre de la carpeta | no habría respuesta correcta |
| `splitter=RandomSplitter(valid_pct=0.2, seed=42)` | 80 % entrenamiento / 20 % validación; `seed` fija el sorteo | sin `seed` la partición cambia en cada corrida y los resultados no son reproducibles |
| `item_tfms=Resize(160)` | Deja cada imagen del mismo tamaño, una por una | las imágenes de distinto tamaño no se pueden agrupar en un lote |
| `batch_tfms=aug_transforms(size=128)` | **Aumento de datos** sobre el lote, en la GPU, y deja todo en 128 × 128 | sin aumento, el modelo ve siempre las mismas fotos y es más fácil que memorice _(a verificar con el experimento)_ |
| `bs=BS` (`BS = 16`; en la clase, 32) | Imágenes procesadas juntas; **el modelo se ajusta una vez por lote** | más grande exige más memoria de la GPU y da **menos ajustes por época**; con ~144 imágenes y lote 32 solo hay 4 ajustes por época y el modelo no aprendió (sección 3.1) |

- **Con nuestros datos:** 180 imágenes → unas **144 de entrenamiento y 36 de validación**. Es poco:
  la validación es ruidosa (cada imagen mal clasificada mueve el accuracy ~2,8 puntos).

- **Por qué el aumento importa aquí:** con solo 144 imágenes de entrenamiento, el aumento de datos
  es una de las pocas formas de darle más variedad al modelo.

### Celda 17 — `show_batch`

- **Qué hace:** muestra 8 imágenes ya transformadas, con su etiqueta.
- **Por qué:** verifica dos cosas a la vez: que las etiquetas correspondan a lo que se ve (el pipeline
  es correcto) y que las transformaciones no destruyan la información.

- **Si se quita:** si hubiera un error en las etiquetas o el aumento fuera demasiado agresivo, no
  lo veríamos hasta el final.

### Celda 20 — La red neuronal convolucional (**la más importante**)
```python
def capa(entran, salen):
    return nn.Sequential(
        nn.Conv2d(entran, salen, kernel_size=3, padding=1),
        nn.BatchNorm2d(salen),
        nn.ReLU(),
        nn.MaxPool2d(2),
    )

def crear_red(n_clases=3):
    return nn.Sequential(
        capa(3,   16),    # 128 -> 64 px,  16 filtros
        capa(16,  32),    #  64 -> 32 px,  32 filtros
        capa(32,  64),    #  32 -> 16 px,  64 filtros
        capa(64, 128),    #  16 ->  8 px, 128 filtros
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(128, n_clases),
    )
```
**Las cuatro piezas de cada capa:**

| Pieza | Qué hace | Si se quitara |
|---|---|---|
| `Conv2d(…, kernel_size=3, padding=1)` | Aplica `salen` filtros de 3 × 3 que se deslizan por la imagen; `padding=1` conserva el tamaño | no habría detección de patrones: es el corazón de la capa |
| `BatchNorm2d` | Reescala los resultados a un rango comparable | el entrenamiento se vuelve más inestable y hay que afinar mucho más la tasa de aprendizaje |
| `ReLU` | Pone en cero los valores negativos | **sin ella apilar capas no sirve**: la composición de operaciones lineales es otra operación lineal, así que cuatro capas valdrían lo que una |
| `MaxPool2d(2)` | Reduce la resolución a la mitad quedándose con el mayor valor de cada región 2 × 2 | la siguiente capa vería una región mucho menor de la imagen y el cómputo crecería |

**Cierre de la red:**

- `AdaptiveAvgPool2d(1)`: resume cada mapa de activación en **un solo número** (el promedio).
- `Flatten`: deja un vector de 128 números.
- `Linear(128, n_clases)`: la decisión final, **una puntuación por clase**. `n_clases` pasa a 3.

**Cómo cambian las dimensiones** (lo imprime la propia celda, entrada de 4 imágenes de 128 × 128):

| Paso | Forma | Qué pasó |
|---|---|---|
| entrada | `(4, 3, 128, 128)` | 4 imágenes, 3 canales de color |
| capa 1 | `(4, 16, 64, 64)` | 16 filtros; resolución a la mitad |
| capa 2 | `(4, 32, 32, 32)` | 32 filtros |
| capa 3 | `(4, 64, 16, 16)` | 64 filtros |
| capa 4 | `(4, 128, 8, 8)` | 128 filtros |
| `AdaptiveAvgPool2d` | `(4, 128, 1, 1)` | cada mapa → 1 número |
| `Flatten` | `(4, 128)` | vector por imagen |
| `Linear` | `(4, 3)` | 3 puntuaciones por imagen |

**Idea clave:** al avanzar, la resolución baja y el número de filtros sube: se pierde precisión sobre
*dónde* está cada cosa y se gana información sobre *qué* es. Cada capa mira la salida de la anterior,
así que detecta patrones cada vez más complejos (bordes → esquinas → fragmentos → formas completas).

**Parámetros ajustables:** 98.307 (los que la red aprende). Con Conv2d de 3 × 3 el conteo de una
capa es `entran × salen × 9 + salen`, más `2 × salen` de BatchNorm; la capa lineal agrega
`128 × 3 + 3 = 387`.

### Celda 22 — `Learner` y entrenamiento
```python
learn = Learner(dls, crear_red(), loss_func=CrossEntropyLossFlat(), metrics=accuracy)
learn.fit_one_cycle(12, 3e-3)   # clase; en clasificador_propio.ipynb: fit_one_cycle(EPOCAS, TASA) con EPOCAS = 30
```

- **`Learner`** reúne los datos, el modelo y la función que mide el error.
- **`CrossEntropyLossFlat`** es la medida de error estándar en clasificación: penaliza según la
  probabilidad que el modelo le asignó a la clase **correcta**.

- **`metrics=accuracy`**: proporción de aciertos sobre validación.
- **`fit_one_cycle(12, 3e-3)`** (en nuestro notebook **30** épocas, por la razón de la sección 3.1): 12 **épocas** (una época = un recorrido completo del conjunto de
  entrenamiento) con tasa de aprendizaje máxima de `3e-3`. La tasa de aprendizaje es cuánto se mueven
  los parámetros en cada ajuste: demasiado alta y el entrenamiento se desestabiliza; demasiado baja
  y no avanza.

- **Lo que debe verse:** `train_loss` y `valid_loss` bajan, `accuracy` sube.
- **Señal de alarma (sobreajuste):** `train_loss` sigue bajando mientras `valid_loss` empieza a subir.
- **Si se quitara:** no hay entrenamiento: la red quedaría con sus valores aleatorios iniciales y
  acertaría más o menos al azar (≈ 33 %).

### Celda 23 — Curvas de entrenamiento

- **Qué hace:** grafica error de entrenamiento y de validación por época, y el accuracy con una línea
  punteada en 1/3 (el azar).

- **Por qué:** es la forma de **ver** si el modelo aprende, se estanca o sobreajusta.

### Celda 25 — Los filtros de la primera capa

- **Qué hace:** extrae `learn.model[0][0].weight` (forma `(16, 3, 3, 3)`: 16 filtros, 3 canales,
  3 × 3) y dibuja cada filtro como una pequeña imagen.

- **Por qué:** los filtros empezaron con valores aleatorios y nadie dijo qué debían buscar. Lo que
  se ve son gradientes, bordes y manchas de color: **emergieron del entrenamiento**.

- **Conexión con AlexNet:** la Figura 3 del paper muestra lo mismo para 96 filtros de 11 × 11 × 3.

### Celda 27 — `show_results`

- Muestra predicciones individuales (real / predicha); las incorrectas salen marcadas en rojo.

### Celda 29 — Matriz de confusión (**clave para el análisis**)

- **Cómo se lee:** cada **fila** es la clase real, cada **columna** la clase que el modelo predijo.
  La **diagonal** son aciertos; lo que queda fuera de la diagonal son errores y su posición dice
  **qué clase se confundió con cuál**.

- **Por qué:** dos modelos con el mismo accuracy pueden fallar de maneras distintas (repartidos entre
  todas las clases o concentrados en un par). Se corrigen de forma diferente.

- **Esto responde a la pregunta del video:** *«¿qué par de objetos confunde y por qué?»*

### Celda 30 — `plot_top_losses`

- Muestra los errores donde el modelo estaba **más seguro** de sí mismo. Son los casos que más
  conviene revisar: con frecuencia revelan etiquetas mal asignadas o fotos problemáticas.

### Celda 32 — Conjunto de prueba (**la medida limpia**)
```python
imagenes_prueba = get_image_files(Path("datos/rps-test-set"))
prueba = learn.dls.test_dl(imagenes_prueba, with_labels=True)
error, acierto = learn.validate(dl=prueba)
```

- **Por qué existe:** el conjunto de validación se usó para decidir épocas, tasa de aprendizaje y número
  de capas, así que su medida ya es algo optimista. El conjunto de **prueba** no se toca en ningún
  momento del proceso: es la única estimación limpia.

- **`test_dl`** aplica el mismo preprocesamiento a imágenes nuevas; `with_labels=True` toma también
  su etiqueta para poder comparar.

- **En nuestro proyecto:** apuntar a `fotos/prueba/` (30 imágenes) y **usar estas imágenes en vivo
  durante el video**.

### Celdas nuevas de nuestro notebook

**Celda 37 — Matriz de confusión sobre la prueba.** `ClassificationInterpretation.from_learner(learn, dl=prueba)`
hace la matriz con las **30 fotos de prueba** en vez de las 36 de validación; `most_confused()` lista los pares
(real, predicho, veces). *Por qué:* es la que responde «qué par confunde» con fotos tomadas en otras condiciones.
*Si se quitara:* solo tendríamos la matriz de validación, que son fotos del mismo lugar que el entrenamiento.

**Celda 39 — Prueba en vivo.** Elige 5 fotos de prueba al azar y, para cada una, `learn.predict(ruta)` devuelve la
clase predicha, su índice y la probabilidad de cada clase; se dibuja la foto con «real → predice (N % seguro)».
*Por qué:* es lo que el enunciado pide hacer sin cortes. *Ojo:* «95 % seguro» **no** significa que acierte el 95 %
de las veces; es la probabilidad que la red le asigna a su respuesta, y puede estar muy segura y equivocarse.

**Celda 45 — resnet18 (experimento A).** `vision_learner(dls_pre, resnet18, pretrained=PREENTRENADA)` arma una red de
18 capas y le cambia la última capa por una de 3 salidas. `fine_tune` entrena primero solo esa capa nueva
(`EPOCAS_LIBRES`, el resto congelado) y luego toda la red (`EPOCAS_AJUSTE`). El experimento es **cambiar solo
`PREENTRENADA`** (True/False). *Si se quitara el preentrenamiento:* la misma red cae al nivel de la nuestra (~55 % en
prueba), porque con 144 fotos no alcanza para aprender a ver desde cero.

### Cambios que hay que hacer para nuestros datos
> **Estos cambios ya están hechos en `clasificador_propio.ipynb`.** Solo falta poner los nombres de
> los tres objetos y apuntar las carpetas a las fotos reales.

1. Se quitó la celda de descarga (7); `CARPETA` pasa a `CARPETA_ENTRENAMIENTO` (`fotos/entrenamiento`).
2. La celda 32 usa `CARPETA_PRUEBA` (`fotos/prueba`).
3. `BS = 16` y `EPOCAS = 30` (sección 3.1).
4. La celda de píxeles calcula el centro de la imagen en vez de usar `147:153`, que solo servía para
   imágenes de 300 × 300.
5. Nuevas: montar Drive, reducir fotos, matriz de confusión sobre la prueba y **prueba en vivo**.

---

## 5. Experimentos (dos modificaciones, **una sola cosa a la vez**)

Regla del enunciado: modificar **una sola cosa**, reentrenar y registrar el efecto sobre el accuracy; **dos
modificaciones distintas, no al mismo tiempo.** Cambiar una pieza a la vez es lo que permite saber *cuál* causó el cambio.

**Cómo se hicieron.** Cada experimento parte de una base, cambia **una** variable, y se **repite con 5 semillas**
(0 a 4: cambian el sorteo de los pesos iniciales y del aumento; la partición entrenamiento/validación es la misma). Se
reporta la **media y el rango** (mínimo–máximo), no una corrida suelta. Todo lo ejecuta `scripts/experimentos_oficiales.py`
y las 25 corridas están en `docs/resultados/experimentos.csv`. En el notebook cada experimento es **una sola línea**:
`AUMENTO` (celda 19, el `DataBlock`) y `PREENTRENADA = True/False` (celda 45). Cada corrida se agrega a la tabla
`registro_exp` (celdas 35 y 46), que se puede mostrar en el video como «registro del efecto sobre el accuracy».
**Ojo:** para el experimento B hay que volver a ejecutar desde la celda 19 (no solo la 25), porque el aumento vive en el `DataBlock`.

> **Estos números son del ensayo local** (Mac, CPU, sus fotos). Hay que **repetirlos en Colab** y usar esas cifras en el
> video: pueden variar unos puntos. Lo que no debería cambiar es la **conclusión cualitativa**.

### Experimento A: la misma resnet18, con y sin pesos preentrenados

- **Qué cambia:** solo el punto de partida de los pesos (`pretrained=False` frente a `pretrained=True`). Todo lo demás
  igual: resnet18, 224 px, `pad`, mismo aumento, 3 épocas con el resto congelado y 10 ajustando todo, lote 16.
- **Hipótesis:** con 144 fotos no alcanza para aprender a ver desde cero, pero una red que ya vio ImageNet solo debe
  aprender a combinar lo que ya detecta.

| Configuración (5 corridas) | Validación: media (rango) | Prueba: media (rango) |
|---|---|---|
| resnet18 **sin** preentrenar (base) | 0,55 (0,42–0,67) | 0,55 (0,47–0,57) |
| resnet18 **preentrenada** | **0,98** (0,94–1,00) | **0,90** (0,87–0,97) |

- **Resultado:** el preentrenamiento sube la prueba **35 puntos**, y los rangos **no se tocan** (el mejor sin preentrenar,
  0,57, queda por debajo del peor preentrenado, 0,87). Con este margen sí se puede afirmar el efecto.
- **Observación sin explicar:** sin preentrenar, la prueba da **0,57 en 4 de 5 corridas**; no se analizó por qué.
- **Conclusión:** el preentrenamiento **es** lo que produce la mejora. Sin él, resnet18 queda al nivel de nuestra red
  propia (~0,55–0,60 en prueba) aunque tenga 11 millones de parámetros.
- **Lo que no se probó:** si la mejora se debe a las primeras capas (bordes, texturas) o a las últimas; ni qué pasa con
  otro tipo de objetos.

### Experimento B: aumento de datos en la red propia (ninguno, normal, fuerte)

- **Qué cambia:** solo el aumento de datos (`AUMENTO`). Base: **normal** (giros hasta 10°, zoom hasta 1,1, luz 0,2, deformación 0,2 y volteo
  horizontal). **Fuerte:** giros 30°, zoom 1,4, luz 0,5, deformación 0,3 y volteo vertical. **Ninguno:** sin transformaciones.
  La red propia, 30 épocas, lote 16, `pad`, 128 px, igual en los tres.
- **Hipótesis:** con 144 fotos, más variedad artificial debería ayudar a generalizar a las fotos de prueba.

| Configuración (5 corridas) | Validación: media (rango) | Prueba: media (rango) |
|---|---|---|
| Aumento **normal** (base) | 0,72 (0,67–0,75) | 0,60 (0,57–0,63) |
| Aumento **fuerte** | 0,56 (0,50–0,58) | 0,63 (0,60–0,67) |
| **Sin** aumento | 0,73 (0,72–0,78) | 0,59 (0,57–0,63) |

- **Sin aumento ≈ base** (0,73 y 0,59 contra 0,72 y 0,60): con estos datos y estos ajustes, **el aumento normal no aporta
  nada medible.** La hipótesis no se cumplió.
- **El aumento fuerte baja la validación 16 puntos** (los rangos no se tocan: 0,50–0,58 contra 0,67–0,75). Eso sí es un
  efecto real. _Explicación posible, sin comprobar:_ la red de 4 capas no alcanza a ajustar tanta variación en 30 épocas
  (se podría confirmar mirando el error de entrenamiento).
- **En la prueba, el aumento fuerte sube 3 puntos** (0,63 contra 0,60), pero los rangos se solapan (0,60–0,67 y
  0,57–0,63) y comparando semilla por semilla 3 de 5 suben, 1 queda igual y 1 baja: **no es concluyente.** Con 30 fotos de
  prueba, una foto son 3,3 puntos.
- **Conclusión honesta:** el aumento de datos **no resolvió** el problema de fondo (la red propia topa en ~0,60 en prueba).
  Que un experimento no mejore **también es un resultado**, y es lo que el enunciado pide explicar.

### Otras ideas probadas antes (una sola corrida de 3 semillas, sección 3.3 y 3.2)

Resolución 192 px, red más ancha, red de 5 capas, *dropout* 0,3, *weight decay* 0,1, más épocas, otra tasa de aprendizaje y
otro redimensionado: ninguna mejoró de forma clara la red propia. No se eligieron como experimentos por eso.

**Qué decir en el video (por experimento):** qué se cambió y en qué línea; las cifras **del Colab del grupo** (media de varias
corridas, no una sola); y una explicación **propia** de por qué pasó. Si el resultado no es el esperado, **mostrarlo igual.**

---

## 6. Resultados

**Del ensayo local (5 corridas por configuración; reemplazar con las cifras del Colab del grupo):**

| Configuración | Qué se cambió | Validación: media (rango) | Prueba: media (rango) | Observaciones |
|---|---|---|---|---|
| Base: red propia | nada (aumento normal, 30 épocas, lote 16, `pad`, 128 px) | 0,72 (0,67–0,75) | 0,60 (0,57–0,63) | punto de partida |
| Experimento B | aumento **fuerte** | 0,56 (0,50–0,58) | 0,63 (0,60–0,67) | validación baja; prueba igual |
| Experimento B | **sin** aumento | 0,73 (0,72–0,78) | 0,59 (0,57–0,63) | igual que la base |
| Base de A: resnet18 | pesos aleatorios, 224 px | 0,55 (0,42–0,67) | 0,55 (0,47–0,57) | como la red propia |
| Experimento A | **preentrenada** | 0,98 (0,94–1,00) | 0,90 (0,87–0,97) | la mejora clara |

**Matriz de confusión de una corrida** (modelo guardado `modelo_objetos_base.pkl`, red propia, 16 de 30 aciertos; filas =
clase real, columnas = respuesta):

| Real ↓ · Respondió → | Lápiz | Borrador | Sacapuntas |
|---|---|---|---|
| Lápiz | **7** | 3 | 0 |
| Borrador | 1 | **5** | 4 |
| Sacapuntas | 0 | 6 | **4** |

Con la resnet18 preentrenada (una corrida, 26 de 30): lápiz 10/10, borrador 8/10 y sacapuntas 8/10; los 4 errores son
**borrador ↔ sacapuntas**.

**Par de objetos más confundido:** **sacapuntas ↔ borrador**, en los dos modelos.
**Por qué creemos que ocurre** _(hipótesis; confirmar con `plot_top_losses` en el Colab del grupo)_: varias fotos de prueba
muestran un sacapuntas **pequeño, lejos y sobre un piso gris** (`rsp_test_saca_2`, `_3` y `_5`), algo casi ausente en el
entrenamiento, que tiene sobre todo primeros planos; además, tres errores del borrador son la misma pieza amarillenta
sobre una tela oscura, en fotos casi iguales. Es un cambio de condiciones entre entrenamiento y prueba, que el enunciado pide.
**En qué situación usaríamos / no usaríamos este clasificador** _(borrador de respuesta, que el grupo debe reescribir)_: lo
usaríamos para ordenar fotos de útiles de escritorio vistos de cerca y con fondos parecidos a los del entrenamiento, donde un
error no cuesta. **No** lo usaríamos para una decisión donde el error importe, ni con objetos lejanos, de otras marcas o
con otra luz, porque solo vio 60 fotos por objeto y en nuestras condiciones.

---

## 7. El video de 5 minutos

### 7.1 Reparto sugerido de tiempo
Hablan los cuatro. El código es lo que más pesa (35 pts): darle la mayor parte del tiempo.

| Tiempo | Bloque | Quién | Qué mostrar / decir |
|---|---|---|---|
| 0:00–0:45 | Introducción + **conjunto de imágenes** (20 pts) | F1, F2 | Qué problema es y qué objetos; abrir las carpetas y recorrer varias imágenes de cada objeto, decir cuántas hay y qué variaciones |
| 0:45–1:45 | **Código I: datos** (35 pts) | C1 | Celdas 11, 13, **19** de nuestro notebook (`DataBlock`, partición, `pad`, aumento, lote 16) |
| 1:45–2:45 | **Código II: modelo** (35 pts) | C2 | Celda **23** (capa, dimensiones), celda **25** (`Learner`, épocas, tasa de aprendizaje) |
| 2:45–3:30 | **Resultados y experimentos** (10 pts) | C2 / C1 | Curvas, accuracy, matriz de confusión; los dos experimentos y su efecto (p. ej. resnet18 con y sin preentrenar) |
| 3:30–4:20 | **Prueba en vivo** (25 pts) | F2 | 5 imágenes de prueba, **sin cortes**, comentando cada respuesta |
| 4:20–5:00 | **Análisis** | C2 / F1 | Qué par confunde, por qué, cuándo usarlo y cuándo no |

> **Ensayar antes de grabar.** 5 minutos es muy poco; con un tiempo por bloque, el ensayo detecta
> dónde se pasan.

### 7.2 Reglas que dicta el enunciado

- **Jerarquizar las celdas.** No alcanza el tiempo para todas por igual. De las que importan:
  **qué hace, por qué está, qué pasaría si se quitara.**

- **El código «celda por celda»** es el componente de mayor peso.
- **Prueba en vivo: 5 imágenes de prueba, sin cortes.** **Si el modelo falla, mostrarlo y explicarlo:**
  no cortar ni repetir hasta que salga bien. Un error explicado vale más que una ejecución perfecta
  sin análisis.

- **El accuracy no da puntos por sí mismo.** Un 50 % bien explicado califica más que un 99 % donde el
  código se lee sin entender. Lo que se evalúa es el razonamiento.

### 7.3 Cómo grabar

- **macOS:** `Cmd + Shift + 5` → grabar pantalla. **Windows:** `Windows + G`.
- Alternativa: una reunión de Meet con pantalla compartida, grabada.
- Nombrar el archivo con los **cuatro apellidos en orden alfabético**, separados por guion bajo, `.mp4`.
- Enviar a `mbastidaso@unal.edu.co` y `jfgarciava@unal.edu.co` **antes del lunes 26/10 a las 10:00**.

### 7.4 Preguntas para ensayar (todos deben poder contestarlas)
1. ¿Por qué se separan datos de entrenamiento y validación?
2. ¿Qué es una época? ¿Qué es la tasa de aprendizaje?
3. ¿Qué hace un filtro de 3 × 3? ¿Quién decide qué detecta?
4. ¿Por qué hay una ReLU entre capas? ¿Qué pasaría sin ella?
5. ¿Para qué sirve el aumento de datos y por qué importa con 60 imágenes por clase?
6. ¿Por qué hay un conjunto de prueba distinto del de validación?
7. ¿Cómo se lee la matriz de confusión? ¿Qué par se confunde y por qué?
8. ¿Qué cambió en cada experimento y qué efecto tuvo?
9. ¿Por qué un 33 % no es un resultado en tres clases?
10. ¿En qué situación no usarían este clasificador?

### 7.5 Guion de partida (borrador para leer y ajustar)

> Es un **borrador**: lo que va entre `[corchetes]` se llena con **las cifras de la corrida del propio grupo en
> Colab** (los números del ensayo local varían de una corrida a otra: no copiarlos). Ritmo de referencia: unas
> **145 palabras por minuto**. Cada quien debe **reescribirlo con sus palabras**: leído de corrido suena a lectura, y a
> los cuatro se les puede preguntar por todo el código. Lo que **sí** está respaldado por el ensayo local: los objetos,
> las cantidades de fotos, `pad`, lote 16, 30 épocas y que la preentrenada supera a la red propia y al control sin preentrenar.

**Bloque 1 · 0:00–0:45 · F1 y F2 — introducción y conjunto de imágenes** _(~90 palabras)_

**F1:** «Somos [nombres]. Hicimos el proceso completo de la clase —datos, modelo y evaluación— con fotos propias: nuestro clasificador distingue **lápiz, borrador y sacapuntas**.»

**F2** _(mostrando las carpetas y abriendo varias imágenes de cada una; de apoyo, la galería de la celda 13)_: «Tenemos **60 fotos por objeto** para entrenar, 180 en total, y **10 por objeto de prueba**, tomadas en otro lugar y con otra luz. Variamos el fondo, la luz, la distancia y el ángulo, y usamos ejemplares distintos: lápices de varios colores, borradores de varias formas [mostrar ejemplos]. La etiqueta de cada foto es el nombre de su carpeta, y las reducimos a 256 píxeles.»

**Bloque 2 · 0:45–1:45 · C1 — datos y `DataBlock`** _(~124 palabras)_

«El `DataBlock` es la receta de los datos: entra una imagen, sale una categoría, y **la etiqueta es el nombre de la carpeta**. Separamos 80 % para entrenar y 20 % para validar; la validación **nunca se usa para aprender**, porque medir con las mismas fotos sería medir memorización.

Cada foto se deja cuadrada **agregando bordes** (`pad`) y no recortando el centro: así un lápiz muy alargado no pierde casi todo el objeto. Además aplicamos **aumento de datos** —giros, zoom, cambios de luz— para darle variedad al modelo con solo 144 fotos, y miramos un lote con `show_batch` para comprobar las etiquetas.

Un cambio frente a la clase: lote de **16**, no 32. Con 144 imágenes, 32 daba solo 4 ajustes por época y [decir lo que salió en su corrida con la configuración de la clase].» _(**Verificar antes de decirlo:** en el ensayo, con EPOCAS = 12 y BS = 32 la validación quedó entre 0,44 y 0,53.)_

**Bloque 3 · 1:45–2:45 · C2 — la red y el entrenamiento** _(~106 palabras)_

«La red es una CNN de cuatro capas iguales. `Conv2d` aplica filtros de 3×3 que buscan patrones; nadie les dice qué buscar, empiezan aleatorios. `BatchNorm2d` reescala para estabilizar el entrenamiento. `ReLU` anula los negativos; sin ella, apilar capas valdría lo mismo que una sola. `MaxPool2d` reduce la imagen a la mitad. La resolución baja de 128 a 8 y los filtros suben de 3 a 128; al final, una capa lineal da **una puntuación por cada objeto**.

Entrenamos con `Learner`, entropía cruzada y **30 épocas** —recorridos completos de las fotos— con tasa de aprendizaje de 3e-3, que es cuánto se mueven los parámetros en cada ajuste.»

**Bloque 4 · 2:45–3:30 · C2 y C1 — resultados y experimento** _(~115 palabras)_

**C2:** «Nuestra red llegó a **[X %]** en validación y **[Y %]** en prueba; el azar da 33 %. En la matriz de confusión las filas son la clase real y las columnas la predicha.»

**C1:** «Hicimos dos experimentos, cambiando **una sola cosa** cada uno. **A:** la misma resnet18, con y sin preentrenar: **[W %]** contra **[Z %]** en prueba; con 144 fotos no alcanza para aprender a ver desde cero. **B:** aumento de datos fuerte en nuestra red: la validación bajó a **[V %]** y la prueba quedó en **[U %]**, casi igual; [conclusión propia].» _(Referencia del ensayo local, 5 corridas: A 0,55 → 0,90 en prueba; B validación 0,72 → 0,56 y prueba 0,60 → 0,63. **Usar las cifras del Colab del grupo.**)_

**Bloque 5 · 3:30–4:20 · F2 — prueba en vivo** _(~40 palabras)_

«Ahora la prueba en vivo: **cinco fotos de prueba**, que el modelo nunca vio, sin cortes.» _(Se puede hacer con la celda del notebook o con la pestaña **Prueba en vivo** del programa de `app/`: elige 5 fotos al azar y las clasifica una tras otra. Decidirlo antes de grabar y no mezclar.)_ _(Ejecutar la celda y comentar cada una, sin repetir si falla):_ «Real: **[objeto]**; el modelo responde **[objeto]** con **[N %]** de seguridad: [acertó / se equivocó].» _(Si hay un error:)_ «Aquí confundió **[A]** con **[B]**; lo dejamos porque sirve para el análisis.»

**Bloque 6 · 4:20–5:00 · C2 y F1 — análisis** _(~57 palabras)_

**C2:** «El par más confundido es **[sacapuntas con borrador — confirmar con la matriz de su corrida]**, posiblemente porque varias fotos de prueba muestran un sacapuntas pequeño y lejos sobre un piso gris, algo casi ausente en el entrenamiento. Lo vimos en los errores de mayor confianza (`plot_top_losses`).»

**F1:** «Lo usaríamos para [situación de bajo riesgo]; **no** para [situación grave], porque solo vio 60 fotos por objeto, en nuestras condiciones. Gracias.»

_Total ~532 palabras. La prueba en vivo y mostrar gráficos o carpetas llevan tiempo sin hablar, por eso el guion no llena los 5 minutos completos._

---

## 8. Calendario

Hoy es **jueves 22/10**. Quedan **cuatro días** hasta el lunes 26/10 a las 10:00.

| Día | Meta |
|---|---|
| **Jue 22/10** | Elegir los tres objetos. Repartir roles. Empezar las fotos. C1/C2: subir `clasificador_propio.ipynb` a Colab con GPU y comprobar que corre (ya probado localmente con datos sustitutos). |
| **Vie 23/10** | ✔ Fotos de entrenamiento (180) y de prueba (30) **recibidas y revisadas** (sección 3.2). Pendiente: decidir si se descarta la foto borrosa y qué hacer con las muy alargadas. C1/C2: adaptar el notebook a las carpetas propias. |
| **Sáb 24/10** | Fotos de prueba (30, en otras condiciones). Corrida base + **dos experimentos**. Llenar la sección 6. |
| **Dom 25/10** | Cerrar el análisis del error. Repartir el guion, **ensayar y grabar**. Revisar el video completo. |
| **Lun 26/10** | Enviar el video **antes de las 10:00 am**. Dejar margen: no dejar la grabación para la mañana de la entrega. |

### 8.1 Lista final (dom 25/10 → lun 26/10, 10:00)

- [ ] **Asignar F1, F2 y C2** en la tabla de la sección 1 (en el video hablan los cuatro).
- [ ] Subir a Drive la copia con carpetas **en minúscula y sin espacios** (`borrador`, `lapiz`, `sacapuntas`): la carpeta
      original `"Sacapuntas "` tiene un espacio al final. La celda 11 ahora se detiene si no coinciden.
- [ ] **Correr el notebook completo en Colab con GPU** (nunca se ha probado allí) y guardar los `.pkl` en Drive.
- [ ] Repetir en Colab cada experimento 2–3 veces y **anotar las cifras propias** en la sección 6 (A: `PREENTRENADA`
      True/False en la celda 45; B: `AUMENTO` en la celda 19).
- [ ] Decidir **con qué modelo se hace la prueba en vivo** (red propia ~60 % o resnet18 ~90 %) y si se hace en el
      notebook (celda 39) o en `app/`. Con la red propia es casi seguro que falle alguna: preparar cómo explicarlo.
- [ ] Reescribir con palabras propias el «cuándo sí / cuándo no» de la sección 6 y el guion de la 7.5.
- [ ] Ensayo cronometrado (≤ 5:00), grabar, revisar audio y que se lean las celdas.
- [ ] Nombrar el archivo con los 4 apellidos en orden alfabético y **enviarlo antes de las 10:00 del lunes** a los dos correos.

---

## 9. Bitácora de construcción

> Una entrada por avance: **qué se hizo · qué se decidió y por qué · cómo se verificó.**
> No se editan las entradas viejas; se agregan nuevas debajo.

### 2026-10-19 — Repositorio y enunciado

- Se creó el repositorio `69kingDavid69/proyecto-red-neuronal` (privado) y se invitó al resto del equipo.
- Se subió el enunciado (`.tex`/`.pdf`), el notebook de clase y un README; los datos de ejemplo, la
  lista del curso y el paper de referencia quedaron fuera del repo con `.gitignore`.

- **Verificación:** se comprobó en GitHub que están los cinco archivos esperados.

### 2026-10-22 — Plan, reparto y documento de estudio

- Se leyó el notebook completo (33 celdas) y se escribió la sección 4 con la explicación celda por
  celda, usando las salidas guardadas del notebook (98.307 parámetros; dimensiones por capa).

- Se propuso el reparto F1/F2 (fotos) y C1/C2 (código), y que el conjunto de prueba lo tomen C1 y C2.
- **Pendiente:** elegir los tres objetos, nombres de los roles, los dos experimentos.
- **Verificación:** el conteo de parámetros (98.307) se recalculó a mano:
  `448+32 + 4.640+64 + 18.496+128 + 73.856+256 + 387 = 98.307`. Coincide con la salida del notebook.

### 2026-10-22 (tarde) — Notebook adaptado y ensayo con datos sustitutos
- Se creó `clasificador_propio.ipynb` a partir del de clase (ver la sección 3) y `scripts/ensayo_configuraciones.py`.
- **Hallazgo:** con los valores de la clase (12 épocas, lote 32) y 180 imágenes el modelo no aprende
  (22 % en validación, siempre la misma clase). Con lote 16 y 30 épocas llega a ~93–97 % en prueba
  (datos sintéticos). Detalle en la sección 3.1. **Decisión:** el notebook usa `BS = 16`, `EPOCAS = 30`.
- **Se corrigió un error propio** en la celda de prueba en vivo (la imagen se dibujaba en una figura
  y el título en otra); ahora usa un solo eje.
- **Verificación:** el notebook se ejecutó de punta a punta sin errores (fastai 2.8.12) con 60
  imágenes por clase de entrenamiento y 10 de prueba; la celda de prueba en vivo muestra las 5
  imágenes con su predicción. Probado en CPU/Mac, **no en Colab con GPU**.
- **Límite:** los datos sustitutos son renders limpios; con fotos reales (fondos, luz) el accuracy
  será menor y el efecto de cada experimento puede cambiar.

### 2026-10-23 — Verificador de fotos
- Se escribió `scripts/verificar_fotos.py` (solo requiere Pillow) y se documentó en el README y en 2.6.
- **Verificación:** con el conjunto limpio de ensayo (60/10 por clase de manos sintéticas) no da avisos.
  Con una copia a la que se le inyectaron fallos conocidos detectó: una imagen dañada, una foto de
  entrenamiento colada en prueba, una foto repetida dentro de entrenamiento, cantidades distintas de
  las pedidas, una foto de 60 px, una foto borrosa y el fondo distinto de una clase (7 de 7).
- **Ajuste:** el umbral de «borrosa» empezó en 0,35 de la mediana y no detectó la foto borrosa
  (estaba en 0,52); se subió a 0,55 calibrando con el conjunto limpio (mínimos de 0,62 o más).
- **Límite:** esa calibración es con renders limpios. Con fotos de celular puede dar falsos avisos o
  dejar pasar fotos mediocres: por eso siempre imprime las 3 menos nítidas de cada clase para mirarlas.

### 2026-10-23 (noche) — Llegaron las fotos reales
- Objetos: **lápiz · borrador · sacapuntas**. Se recibieron las carpetas de entrenamiento (60 por objeto) y prueba
  (10 por objeto) desde Descargas; se trabajó con una copia normalizada en `fotos/` (la carpeta `"Sacapuntas "` tenía un
  espacio al final y no coincidía con la de prueba). Los originales no se modificaron.
- `verificar_fotos.py`: 0 errores. **Se corrigió el propio verificador** tras verlo con fotos reales: (1) el aviso de
  «foto pequeña» miraba el lado menor y saltaba en fotos delgadas que sí están a 256 px; ahora mira el lado mayor y avisa
  aparte de las **muy alargadas**; (2) el detector de borrosa daba falsos avisos con fondos lisos (medía el detalle
  absoluto); se cambió por un índice de borrosidad (cuánto detalle conserva la foto al difuminarla). Se validó en los 3
  conjuntos: fotos reales (1 aviso de borrosa), sintético limpio (0) y sintético con una foto borrosa inyectada (detectada).
- **Hallazgo:** con las fotos reales la red llega a ~70 % (validación) y ~55–60 % (prueba), y entrenar más no ayuda (3.2).
- **Decisión:** el notebook gana el parámetro `REDIMENSION` (por defecto `"pad"`, porque `crop` recorta los lápices muy
  alargados). No se justifica con los números (la diferencia está dentro del ruido) sino con la geometría de las fotos.
- **Verificación:** el notebook se ejecutó de punta a punta con las fotos reales (56,67 % en prueba) y sin errores;
  las 8 configuraciones, con 3 semillas cada una (`scripts/ensayo_fotos_reales.py`). Probado en CPU, **no en Colab**.

### 2026-10-24 — Modelo guardado, mejoras probadas y análisis de errores
- Se agregó al notebook la **sección 7 (guardar y cargar el modelo)** y se generó `modelos/modelo_objetos_base.pkl`
  (ignorado por Git). Se cargó en un proceso nuevo: 53,33 % en prueba, igual que en el notebook.
- Se probaron **7 configuraciones** (una sola cosa a la vez, 3 semillas, `scripts/ensayo_mejoras.py`): ninguna mejora
  claramente (sección 3.3). La red actual topa en ~70 % de validación y ~55–65 % de prueba.
- **Análisis de errores:** 14 de 30; los 6 errores de sacapuntas son «borrador» y 3 de ellos son objetos pequeños y lejanos
  sobre piso gris, algo casi ausente en el entrenamiento. Hipótesis, sin comprobar.
- **Decisión pendiente:** si se prueba una red preentrenada (requiere descargar pesos de internet).
- **Verificación:** el notebook con las celdas nuevas se ejecutó de punta a punta con las fotos reales sin errores
  (CPU, no Colab). Las cifras de 3.3 vienen de las corridas y de los listados impresos de errores.

### 2026-10-24 (tarde) — Red preentrenada
- Con permiso del usuario se descargaron los pesos de resnet18 (44,7 MB, `download.pytorch.org`; integridad: el prefijo
  `f37072fd` del nombre coincide con el SHA-256). Se agregó al notebook la **sección 8** (transfer learning).
- **Resultado:** resnet18 preentrenada a 224 px: **0,97 validación y 0,92 prueba** (3 corridas; una sola dio 0,87–0,97),
  frente a 0,73 / 0,60 de la CNN propia. El **control sin preentrenar** (misma arquitectura) dio 0,50 / 0,57: la mejora viene
  del preentrenamiento. Detalle en 3.4.
- Se guardó `modelos/modelo_objetos_preentrenado.pkl` (ignorado por Git). `scripts/ensayo_preentrenada.py` reproduce el ensayo.
- **Verificación:** el notebook de 47 celdas se ejecutó de punta a punta con las fotos reales (CNN propia 56,67 %,
  preentrenada 86,67 % en esa corrida) y el modelo se recargó en un proceso nuevo. Probado en CPU, **no en Colab**.

### 2026-10-24 (noche) — Programa para probar los modelos
- Se creó `app/` (servidor local + página, UI/UX con tema de papelería y sello de goma); ver 3.5 y `app/README.md`.
  Se creó un `.venv` en el proyecto (ignorado por Git) con fastai para ejecutarlo.
- **Errores propios corregidos al probarlo:** los botones ocultos seguían visibles (`display:flex` anulaba `hidden`), el
  balance de la prueba en vivo imprimía «null», los nombres de las barras salían cortados y los sellos pequeños se salían
  de la tarjeta en modo «Comparar». Todos se vieron en capturas y se corrigieron.
- **Verificación:** una foto, prueba en vivo con comparación, evaluación completa (26 de 30), tema oscuro y ancho de celular
  (sin desplazamiento horizontal). **No se probó la cámara.**

### 2026-10-24 (madrugada del 25) — Los dos experimentos, con 5 semillas
- Se escribió `scripts/experimentos_oficiales.py` y se corrieron **25 corridas** (resultados en `docs/resultados/experimentos.csv`).
  **A:** resnet18 con y sin preentrenar: prueba 0,55 → 0,90 (rangos que no se tocan). **B:** aumento de datos ninguno / normal / fuerte
  en la red propia: sin aumento ≈ base; fuerte baja la validación 16 puntos y la prueba sube 3, no concluyente. Detalle en la sección 5.
- **Cambio en el notebook:** el parámetro `USAR_AUMENTO` (sí/no) pasó a `AUMENTO = "ninguno" | "normal" | "fuerte"`, para poder hacer el
  experimento B con una sola línea.
- **Corrección de una cifra:** con 3 semillas la preentrenada daba 0,92 en prueba; con 5 da 0,90. Se dejó nota en 3.4.
- **Pendiente:** repetir los dos experimentos en Colab y usar esas cifras en el video; este ensayo es en CPU.

### 2026-10-25 — Revisión contra el enunciado y limpieza del notebook

- Se revisó el repo contra el enunciado (`proyecto_clasificador.tex`) y el notebook de clase.
- **Notebook:** quedaban textos del ejemplo de clase que se habrían leído en el video: «una mano», «piedra», dedos en la
  tabla de capas, «2.016 / 504 imágenes», «300 × 300» y «372 imágenes de prueba». Ahora hablan de nuestros datos
  (144 / 36, 256 px, 30 de prueba, lápiz · borrador · sacapuntas). Se agregó la nota de ruido (3,3 puntos por foto).
- **Experimento A en una línea:** la celda 45 tiene el parámetro `PREENTRENADA` (antes `pretrained=True` fijo, así
  que el control no se podía correr desde el notebook). El control se guarda con otro nombre para no pisar el modelo bueno.
- **Documento:** tabla de numeración clase → nuestro notebook (la sección 4 y el guion citaban las celdas de la clase),
  explicación de las celdas nuevas (37, 39, 45) y lista final 8.1.
- **Verificación:** el `.ipynb` se volvió a leer desde GitHub y es JSON válido con 47 celdas. **No se ejecutó** (no hay
  Python/fastai en el equipo donde se hizo): correrlo en Colab es el primer punto de la lista 8.1.

### 2026-10-25 (tarde) — Segunda revisión: robustez para Colab, galería y registro de experimentos

- **Celda 11:** compara los nombres de las carpetas de entrenamiento y prueba y se detiene con un mensaje claro si no
  coinciden. Motivo: las carpetas originales tenían `"Sacapuntas "` (con espacio) en entrenamiento y `"Sacapuntas"` en
  prueba. Si se suben así a Drive, la prueba falla o mide mal.
- **Celda 13:** conteo de las fotos de prueba y una **galería de 8 fotos al azar por objeto**, para el bloque del conjunto
  de imágenes (20 pts): se ve la variedad sin abrir foto por foto.
- **Celda 25:** imprime los ajustes por época y en total (9 y 270 con 144 fotos, lote 16 y 30 épocas). Sirve para explicar
  por qué se cambió el lote de 32 a 16.
- **Celdas 35 y 46:** la tabla `registro_exp` acumula cada corrida (configuración, validación y prueba). El enunciado pide
  «registrar el efecto sobre el accuracy».
- **Celda 41:** el modelo se guarda como `modelo_objetos_base.pkl`, que es el nombre que busca `app/` (antes era
  `modelo_objetos.pkl` y la app no lo encontraba).
- **READMEs:** comandos para Windows además de Mac.
- **Verificación:** el `.ipynb` es JSON válido (47 celdas) y el código de las celdas se revisó a mano. **No se ejecutó.**

---

## Referencias

- Enunciado: `proyecto_clasificador.pdf` (IPCC – ML).
- Notebook de clase: `clase_clasificacion_imagenes.ipynb`.
- Krizhevsky, Sutskever y Hinton (2012). *ImageNet Classification with Deep Convolutional Neural
  Networks* (AlexNet). NIPS 2012.
