# Guía para correr, grabar y entregar

Entrega: **lunes 26 de octubre, antes de las 10:00 am**. Se entrega **solo el video** (5 minutos, hablan los cuatro).

Lo que hay que tener a mano:
- `fotos.zip` (1,75 MB, **no está en el repo** porque las fotos quedan fuera de Git; lo comparte Nicolás por Drive o
  WhatsApp): las 210 fotos con las carpetas ya normalizadas:
  `fotos/entrenamiento/{borrador,lapiz,sacapuntas}` (60 c/u) y `fotos/prueba/{…}` (10 c/u).
  Se corrigió la carpeta original `"Sacapuntas "`, que tenía un espacio al final.
- `clasificador_propio.ipynb`: en GitHub, abrir el archivo → botón **Download raw file**.
- `docs/documento-de-estudio.md`: la explicación celda por celda (sección 4), la tabla de numeración de celdas
  y el guion (sección 7.5).

---

## Paso 1 · Correr el notebook en Colab (unos 30 min, una sola persona)

1. Entrar a <https://colab.research.google.com> → **Archivo → Subir cuaderno** → elegir `clasificador_propio.ipynb`.
2. **Entorno de ejecución → Cambiar tipo de entorno de ejecución → T4 GPU → Guardar.**
3. Panel izquierdo, ícono de carpeta (**Archivos**) → botón de subir → elegir `fotos.zip`. Esperar a que termine de subir.
4. **Entorno de ejecución → Ejecutar todo** (Ctrl + F9). Si Colab pide reiniciar después de instalar fastai, aceptar
   y volver a hacer «Ejecutar todo».
5. Revisar:
   - **Celda 11:** debe decir `Total de imagenes de entrenamiento: 180` y mostrar las mismas carpetas en entrenamiento y prueba.
   - **Celda 13:** gráfico de barras 60 / 60 / 60, `Fotos de prueba por objeto` 10 / 10 / 10 y la galería.
   - **Celda 35:** el accuracy sobre la prueba (se espera alrededor de 55–65 % con la red propia).
   - **Celda 46:** el accuracy de la resnet18 preentrenada (se espera alrededor de 85–95 %).

### Los dos experimentos (cambiar **una sola cosa** a la vez)

Cada corrida se suma sola a la tabla `registro_exp`, que se imprime al final de las celdas 35 y 46.

- **Experimento B, aumento de datos.** En la celda 19 cambiar `AUMENTO = "fuerte"` y ejecutar **de la celda 19 a la 35**.
  Después `AUMENTO = "ninguno"` y lo mismo. Al final dejar `AUMENTO = "normal"` y volver a ejecutar de la 19 a la 39.
  Así la matriz y la prueba en vivo quedan con el modelo base.
- **Experimento A, preentrenamiento.** En la celda 45 cambiar `PREENTRENADA = False` y ejecutar las celdas 45 y 46.
  Después dejar `PREENTRENADA = True` y volver a ejecutarlas.
- Si da tiempo, repetir cada configuración 2 veces: con 30 fotos de prueba, una sola foto mueve el accuracy 3,3 puntos.
- **Anotar las cifras** de la tabla `registro_exp`: son las que se dicen en el video (los `[X %]` del guion, sección 7.5).

> Si la sesión de Colab se reinicia, la tabla se borra y hay que volver a correr. Por eso conviene correr todo
> **el mismo día de grabar** y no cerrar la pestaña.

---

## Paso 2 · Ensayar (unos 30 min, los cuatro)

Reparto sugerido. Los nombres se pueden cambiar, pero **cada uno habla en al menos un bloque**:

| Tiempo | Bloque | Quién | Qué se muestra en pantalla |
|---|---|---|---|
| 0:00–0:45 | Intro y **conjunto de imágenes** (20 pts) | Persona 1 | Panel Archivos de Colab con `fotos/` desplegado (3 carpetas) + **galería de la celda 13** |
| 0:45–1:45 | **Código I: datos** (35 pts) | Persona 2 (David, C1) | Celdas 11, 19 (`DataBlock`: partición, `pad`, aumento, lote 16) y 20 (`show_batch`) |
| 1:45–2:45 | **Código II: red y entrenamiento** | Persona 3 | Celda 23 (`capa`, `crear_red`, dimensiones), celda 25 (ajustes por época, `fit_one_cycle`), curvas (26) |
| 2:45–3:30 | **Resultados y experimentos** (10 pts) | Persona 3 + 2 | Celda 35 (accuracy + `registro_exp`), matriz de prueba (37), los dos experimentos |
| 3:30–4:20 | **Prueba en vivo** (25 pts) | Persona 4 | Ejecutar **celda 39** una vez, sin cortar, comentando cada una de las 5 fotos |
| 4:20–5:00 | **Análisis** | Persona 4 + 1 | Par más confundido (sacapuntas ↔ borrador), por qué, cuándo sí y cuándo no usar el clasificador |

Para cada celda importante decir **qué hace**, **por qué está** y **qué pasaría si se quitara** (sección 4 del
documento de estudio). Cronometrar el ensayo: si pasa de 5:00, recortar en la intro, nunca en el código.

**Si en la prueba en vivo el modelo falla, se deja y se explica.** El enunciado dice que un error explicado vale más que
una ejecución perfecta. Con la red propia es casi seguro que falle 1 o 2 de 5: preparar la explicación de antemano
(fotos de prueba lejanas o sobre piso gris, lápices muy alargados que con `pad` quedan como una franja delgada).

---

## Paso 3 · Grabar

**Hagan todo dentro del navegador** (Colab y su panel Archivos). Así la grabación captura todo sin cambiar de ventana.

- **Opción recomendada, Google Meet:** una reunión con los cuatro. Quien tiene el Colab corriendo comparte la
  **pestaña** del navegador y se graba la reunión (Actividades → Grabación). La grabación llega como `.mp4` al Drive
  del organizador. Si su cuenta no permite grabar, usar la opción siguiente.
- **Opción Windows:** **Windows + G** → Capturar → botón grabar, con el navegador en primer plano y los demás en la
  llamada. Ojo: Windows + G graba **una sola ventana**, así que no abran el Explorador de archivos durante la grabación.
  El video queda en `Videos\Capturas`.
- **macOS:** Cmd + Shift + 5 → grabar pantalla completa.

Antes de grabar: celdas ya ejecutadas con salidas visibles, zoom del navegador al 110–125 % para que se lea el código,
notificaciones silenciadas y micrófonos probados.

Durante la grabación **solo se ejecuta en vivo la celda 39** (prueba en vivo). Lo demás ya tiene sus salidas y se va
recorriendo con scroll mientras cada uno explica.

---

## Paso 4 · Revisar y entregar

1. Ver el video completo: dura **5:00 o menos**, se oye a los cuatro, se lee el código y la prueba en vivo no tiene cortes.
2. Nombrar el archivo con los **cuatro apellidos en orden alfabético**, separados por guion bajo, `.mp4`.
   Con tres apellidos ya conocidos, el nombre queda `Aristizábal_Gómez_Quintana.mp4` más el apellido de David en su
   lugar alfabético. Confirmar cómo escribe cada uno su apellido (con o sin tilde).
3. Enviar **un solo correo** a `mbastidaso@unal.edu.co` y `jfgarciava@unal.edu.co`:
   - Asunto sugerido: `IPCC-ML · Proyecto clasificador de imágenes · <Apellidos>`
   - Cuerpo: nombres completos de los cuatro y los tres objetos (lápiz, borrador, sacapuntas).
   - Si el video pesa más de 25 MB, Gmail lo adjunta como enlace de Drive. En ese caso dar acceso a los dos correos
     (o a «cualquier persona con el enlace») y **abrir el enlace en una ventana de incógnito** para comprobar que se ve.
4. Enviarlo **con margen** (domingo en la noche o lunes temprano) y avisar al grupo cuando esté enviado.

---

## Checklist rápido

- [ ] Roles asignados (Persona 1–4)
- [ ] Notebook corrido en Colab con GPU y `fotos.zip`
- [ ] Experimentos A y B corridos y cifras anotadas
- [ ] Guion con las cifras propias, reescrito con palabras de cada uno
- [ ] Ensayo cronometrado ≤ 5:00
- [ ] Video grabado y revisado
- [ ] Archivo con el nombre correcto
- [ ] Correo enviado a los dos profesores antes de las 10:00 del lunes
