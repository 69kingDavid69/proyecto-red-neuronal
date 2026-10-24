"use strict";

/* Clasificador de objetos: lógica de la interfaz. Todo el texto de la página se inserta con textContent. */

const $ = (id) => document.getElementById(id);
const ESPERA_ENTRE_FOTOS_MS = 900;

const estado = {
  modelos: [],          // [{id, nombre, detalle, disponible}]
  clases: [],           // [{id, nombre}]
  fotos5: [],           // fotos elegidas para la prueba en vivo
  ocupado: false,
  flujoCamara: null,
};

const porcentaje = (p, decimales = 0) =>
  (p * 100).toLocaleString("es-CO", { minimumFractionDigits: decimales, maximumFractionDigits: decimales }) + " %";
const nombreClase = (id) => (estado.clases.find((c) => c.id === id) || { nombre: id }).nombre;
const nombreModelo = (id) => (estado.modelos.find((m) => m.id === id) || { nombre: id }).nombre;
const pausa = (ms) => new Promise((r) => setTimeout(r, ms));

function el(etiqueta, opciones = {}, hijos = []) {
  const nodo = document.createElement(etiqueta);
  for (const [k, v] of Object.entries(opciones)) {
    if (k === "clase") nodo.className = v;
    else if (k === "texto") nodo.textContent = v;
    else if (k in nodo) nodo[k] = v;
    else nodo.setAttribute(k, v);
  }
  for (const h of [].concat(hijos)) if (h) nodo.append(h);
  return nodo;
}

function icono(clase) {
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("aria-hidden", "true");
  const uso = document.createElementNS("http://www.w3.org/2000/svg", "use");
  uso.setAttribute("href", `#i-${clase}`);
  svg.append(uso);
  return svg;
}

function mostrarError(mensaje) {
  const caja = $("error");
  caja.textContent = mensaje || "";
  caja.hidden = !mensaje;
}

async function pedir(url, opciones) {
  let respuesta;
  try {
    respuesta = await fetch(url, opciones);
  } catch {
    throw new Error("No se pudo hablar con el programa. ¿Sigue abierto en la terminal?");
  }
  const datos = await respuesta.json().catch(() => ({}));
  if (!respuesta.ok) throw new Error(datos.error || "Algo salió mal al clasificar.");
  return datos;
}

/* ---------- modelos activos según el selector ---------- */
function modelosActivos() {
  const modo = document.querySelector('input[name="modelo"]:checked').value;
  const disponibles = estado.modelos.filter((m) => m.disponible).map((m) => m.id);
  return modo === "comparar" ? disponibles : [modo];
}

function actualizarAviso() {
  const faltan = estado.modelos.filter((m) => !m.disponible);
  const aviso = $("aviso-modelo");
  if (!faltan.length) { aviso.hidden = true; return; }
  const archivos = { propia: "sección 7", preentrenada: "sección 8" };
  aviso.textContent = faltan
    .map((m) => `Falta el modelo «${m.nombre}». Se genera con el notebook (${archivos[m.id]}) y se guarda en la carpeta modelos.`)
    .join(" ");
  aviso.hidden = false;
}

/* ---------- veredicto: sello + barras ---------- */
function textoConfianza(resp) {
  const p = resp.confianza;
  if (p >= 0.8) return [el("strong", { texto: `Está seguro: ${porcentaje(p)}.` }), " La respuesta es " + resp.nombre.toLowerCase() + "."];
  if (p >= 0.55) return [el("strong", { texto: `Probable: ${porcentaje(p)}.` }), " Hay otra clase con peso; revisa las barras."];
  return [el("strong", { clase: "duda", texto: "El modelo duda." }), ` Ninguna clase pasa de 55 %; la más alta es ${resp.nombre.toLowerCase()} con ${porcentaje(p)}.`];
}

function construirVeredicto(idModelo, resp) {
  const barras = el("ul", { clase: "barras" });
  for (const c of estado.clases) {
    const p = resp.probabilidades[c.id] ?? 0;
    const relleno = el("span", { clase: "relleno" });
    barras.append(el("li", { clase: `barra ${c.id}${c.id === resp.clase ? " gana" : ""}` }, [
      el("span", { clase: "nombre" }, [icono(c.id), c.nombre]),
      el("span", { clase: "pista" }, [relleno]),
      el("span", { clase: "pct", texto: porcentaje(p) }),
    ]));
    requestAnimationFrame(() => requestAnimationFrame(() => { relleno.style.width = (p * 100).toFixed(1) + "%"; }));
  }
  return el("div", { clase: "veredicto" }, [
    el("h2", { texto: nombreModelo(idModelo) }),
    el("div", { clase: `sello ${resp.clase}`, role: "img", "aria-label": `Respuesta: ${resp.nombre}` }, [icono(resp.clase), el("span", { texto: resp.nombre })]),
    el("p", { clase: "confianza" }, textoConfianza(resp)),
    barras,
    el("p", { clase: "pie-resultado", texto: `Tardó ${resp.ms} ms en responder.` }),
  ]);
}

async function mostrarResultados(obtener) {
  const activos = modelosActivos();
  $("vacio").hidden = true;
  const contenedor = el("div", { clase: `modelos-resultado${activos.length > 1 ? " doble" : ""}` });
  $("resultados-modelos").replaceChildren(contenedor);
  contenedor.append(el("p", { clase: "cargando", texto: "Clasificando…" }));
  try {
    const resultados = [];
    for (const id of activos) resultados.push([id, await obtener(id)]);
    contenedor.replaceChildren(...resultados.map(([id, r]) => construirVeredicto(id, r)));
    mostrarError("");
  } catch (e) {
    contenedor.replaceChildren();
    $("vacio").hidden = false;
    mostrarError(e.message);
  }
}

/* ---------- una foto ---------- */
async function clasificarArchivo(archivo) {
  if (!archivo) return;
  if (!archivo.type.startsWith("image/")) { mostrarError("Ese archivo no es una imagen. Prueba con una foto .jpg o .png."); return; }
  detenerCamara();
  const vista = $("vista-previa");
  if (vista.src.startsWith("blob:")) URL.revokeObjectURL(vista.src);
  vista.src = URL.createObjectURL(archivo);
  vista.alt = "La foto que se está clasificando";
  vista.hidden = false;
  $("zona-vacia").hidden = true;
  await mostrarResultados((id) => pedir(`/api/predecir?modelo=${encodeURIComponent(id)}`, { method: "POST", headers: { "Content-Type": archivo.type }, body: archivo }));
}

function configurarZona() {
  const zona = $("zona"), entrada = $("archivo");
  const elegir = () => entrada.click();
  zona.addEventListener("click", () => { if (!estado.flujoCamara) elegir(); });
  zona.addEventListener("keydown", (e) => { if ((e.key === "Enter" || e.key === " ") && !estado.flujoCamara) { e.preventDefault(); elegir(); } });
  $("b-elegir").addEventListener("click", elegir);
  entrada.addEventListener("change", () => { clasificarArchivo(entrada.files[0]); entrada.value = ""; });
  for (const nombre of ["dragenter", "dragover"]) zona.addEventListener(nombre, (e) => { e.preventDefault(); zona.classList.add("arrastrando"); });
  for (const nombre of ["dragleave", "drop"]) zona.addEventListener(nombre, (e) => { e.preventDefault(); zona.classList.remove("arrastrando"); });
  zona.addEventListener("drop", (e) => clasificarArchivo(e.dataTransfer.files[0]));
  window.addEventListener("paste", (e) => { const f = [...(e.clipboardData?.files || [])].find((x) => x.type.startsWith("image/")); if (f) clasificarArchivo(f); });
}

/* ---------- cámara ---------- */
async function abrirCamara() {
  if (!navigator.mediaDevices?.getUserMedia) { mostrarError("Este navegador no permite usar la cámara aquí."); return; }
  try {
    estado.flujoCamara = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment" } });
  } catch {
    mostrarError("No se pudo abrir la cámara. Revisa el permiso del navegador o elige una foto del computador.");
    return;
  }
  mostrarError("");
  const video = $("video");
  video.srcObject = estado.flujoCamara;
  video.hidden = false;
  video.play();
  $("vista-previa").hidden = true;
  $("zona-vacia").hidden = true;
  $("acciones-foto").hidden = true;
  $("acciones-camara").hidden = false;
}

function detenerCamara() {
  estado.flujoCamara?.getTracks().forEach((t) => t.stop());
  estado.flujoCamara = null;
  const video = $("video");
  video.srcObject = null;
  video.hidden = true;
  $("acciones-foto").hidden = false;
  $("acciones-camara").hidden = true;
  if ($("vista-previa").hidden) $("zona-vacia").hidden = false;
}

function tomarFoto() {
  const video = $("video");
  if (!video.videoWidth) return;
  const lienzo = el("canvas", { width: video.videoWidth, height: video.videoHeight });
  lienzo.getContext("2d").drawImage(video, 0, 0);
  lienzo.toBlob((blob) => clasificarArchivo(new File([blob], "camara.jpg", { type: "image/jpeg" })), "image/jpeg", 0.92);
}

/* ---------- prueba en vivo ---------- */
function dibujarFotos5() {
  const lista = $("fotos5");
  lista.replaceChildren(...estado.fotos5.map((f, i) => el("li", { clase: "instantanea", id: `foto5-${i}` }, [
    el("div", { clase: "foto" }, [el("img", { src: `/api/foto?archivo=${encodeURIComponent(f.archivo)}`, alt: `Foto de prueba ${i + 1}` })]),
    el("div", { clase: "pie" }, [el("span", { clase: "esperando", texto: "Sin clasificar" })]),
  ])));
  $("balance").replaceChildren();
}

async function elegirAzar() {
  if (estado.ocupado) return;
  try {
    estado.fotos5 = (await pedir("/api/azar?n=5")).fotos;
    dibujarFotos5();
    mostrarError("");
  } catch (e) { mostrarError(e.message); }
}

async function clasificarLas5() {
  if (estado.ocupado) return;
  if (!estado.fotos5.length) await elegirAzar();
  if (!estado.fotos5.length) return;
  estado.ocupado = true;
  $("b-clasificar5").disabled = $("b-azar").disabled = true;
  dibujarFotos5();
  const activos = modelosActivos();
  const aciertos = Object.fromEntries(activos.map((id) => [id, 0]));
  try {
    for (let i = 0; i < estado.fotos5.length; i++) {
      const f = estado.fotos5[i], tarjeta = $(`foto5-${i}`), pie = tarjeta.querySelector(".pie");
      tarjeta.classList.add("activa");
      pie.replaceChildren(el("span", { clase: "esperando", texto: "Clasificando…" }));
      const filas = [];
      for (const id of activos) {
        const r = await pedir(`/api/prueba_foto?modelo=${encodeURIComponent(id)}&archivo=${encodeURIComponent(f.archivo)}`);
        if (r.acierto) aciertos[id]++;
        filas.push(el("div", { clase: "fila-modelo" }, [
          activos.length > 1 ? el("span", { clase: "esperando", texto: nombreModelo(id) }) : null,
          el("div", { clase: `sello mini ${r.clase}` }, [icono(r.clase), el("span", { texto: r.nombre })]),
          el("span", { clase: `marca ${r.acierto ? "ok" : "mal"}`, texto: r.acierto ? "Acertó" : `Error: era ${nombreClase(r.real).toLowerCase()}` }),
          el("span", { clase: "esperando", texto: `${porcentaje(r.confianza)} seguro` }),
        ]));
      }
      pie.replaceChildren(...filas);
      tarjeta.classList.remove("activa");
      await pausa(ESPERA_ENTRE_FOTOS_MS);
    }
    const n = estado.fotos5.length;
    $("balance").replaceChildren(...activos.flatMap((id, k) => [
      k ? el("br") : null,
      activos.length > 1 ? `${nombreModelo(id)}: ` : "",
      el("strong", { texto: `${aciertos[id]} de ${n}` }),
      aciertos[id] === n ? " correctas." : aciertos[id] === n - 1 ? " correctas, con un error para analizar." : " correctas; los errores sirven para el análisis.",
    ]).filter(Boolean));
    mostrarError("");
  } catch (e) {
    mostrarError(e.message);
  } finally {
    estado.ocupado = false;
    $("b-clasificar5").disabled = $("b-azar").disabled = false;
    document.querySelectorAll(".instantanea.activa").forEach((n) => n.classList.remove("activa"));
  }
}

/* ---------- evaluación completa ---------- */
function construirMatriz(datos) {
  const cabeza = el("tr", {}, [el("th", { scope: "col", texto: "Real ↓ · Predicho →" }), ...datos.clases.map((c) => el("th", { scope: "col", texto: c.nombre }))]);
  const cuerpo = datos.clases.map((fila) => {
    const total = Object.values(datos.matriz[fila.id]).reduce((a, b) => a + b, 0) || 1;
    return el("tr", {}, [
      el("th", { scope: "row", texto: fila.nombre }),
      ...datos.clases.map((col) => {
        const n = datos.matriz[fila.id][col.id];
        const celda = el("td", { clase: `${fila.id}${n === 0 ? " cero" : ""}${fila.id === col.id ? " diagonal" : ""}`, texto: String(n) });
        celda.style.setProperty("--peso", String(n / total));
        celda.setAttribute("aria-label", `${fila.nombre} real, predicho ${col.nombre}: ${n}`);
        return celda;
      }),
    ]);
  });
  return el("table", { clase: "matriz" }, [
    el("caption", { texto: "Matriz de confusión: cada fila es la clase real; cada columna, lo que respondió el modelo." }),
    el("thead", {}, [cabeza]), el("tbody", {}, cuerpo),
  ]);
}

function construirEvaluacion(datos) {
  const errores = el("div", { clase: "errores" }, [el("h3", { texto: datos.errores.length ? `Fotos donde se equivocó (${datos.errores.length})` : "No se equivocó en ninguna foto." })]);
  if (datos.errores.length) {
    errores.append(el("ul", {}, datos.errores.map((e) => el("li", {}, [
      el("img", { src: `/api/foto?archivo=${encodeURIComponent(e.archivo)}`, alt: `Foto de ${nombreClase(e.real).toLowerCase()} que el modelo confundió` }),
      `Era ${nombreClase(e.real).toLowerCase()}; dijo ${nombreClase(e.prediccion).toLowerCase()} (${porcentaje(e.confianza)}).`,
    ]))));
  }
  return el("section", { clase: "eval-modelo" }, [
    el("h2", { texto: nombreModelo(datos.modelo) }),
    el("p", { clase: "eval-frase" }, ["Acertó ", el("strong", { texto: `${datos.aciertos} de ${datos.total}` }), ` fotos (${porcentaje(datos.accuracy, 1)}). Con tres clases, responder al azar acierta 33 %.`]),
    el("div", { clase: "eval-cuerpo" }, [construirMatriz(datos), errores]),
  ]);
}

async function evaluarTodas() {
  if (estado.ocupado) return;
  estado.ocupado = true;
  $("b-evaluar").disabled = true;
  const zona = $("evaluacion");
  zona.replaceChildren(el("p", { clase: "cargando", texto: "Pasando todas las fotos por el modelo…" }));
  try {
    const bloques = [];
    for (const id of modelosActivos()) bloques.push(construirEvaluacion(await pedir(`/api/evaluar?modelo=${encodeURIComponent(id)}`)));
    zona.replaceChildren(el("div", { clase: "eval" }, bloques));
    mostrarError("");
  } catch (e) {
    zona.replaceChildren();
    mostrarError(e.message);
  } finally {
    estado.ocupado = false;
    $("b-evaluar").disabled = false;
  }
}

/* ---------- pestañas ---------- */
function configurarPestanas() {
  const pestanas = [...document.querySelectorAll(".pestana")];
  const activar = (pestana, enfocar) => {
    for (const p of pestanas) {
      const activa = p === pestana;
      p.setAttribute("aria-selected", String(activa));
      p.tabIndex = activa ? 0 : -1;
      $(p.getAttribute("aria-controls")).hidden = !activa;
    }
    if (enfocar) pestana.focus();
    if (pestana.id === "p-vivo" && !estado.fotos5.length) elegirAzar();
    mostrarError("");
  };
  pestanas.forEach((p, i) => {
    p.addEventListener("click", () => activar(p, false));
    p.addEventListener("keydown", (e) => {
      if (e.key === "ArrowRight") activar(pestanas[(i + 1) % pestanas.length], true);
      if (e.key === "ArrowLeft") activar(pestanas[(i - 1 + pestanas.length) % pestanas.length], true);
    });
  });
}

/* ---------- arranque ---------- */
async function iniciar() {
  configurarZona();
  configurarPestanas();
  $("b-camara").addEventListener("click", abrirCamara);
  $("b-tomar").addEventListener("click", tomarFoto);
  $("b-cancelar-camara").addEventListener("click", detenerCamara);
  $("b-azar").addEventListener("click", elegirAzar);
  $("b-clasificar5").addEventListener("click", clasificarLas5);
  $("b-evaluar").addEventListener("click", evaluarTodas);

  try {
    const datos = await pedir("/api/estado");
    estado.modelos = datos.modelos;
    estado.clases = datos.clases;
  } catch (e) { mostrarError(e.message); return; }

  const entradas = [...document.querySelectorAll('input[name="modelo"]')];
  const disponibles = estado.modelos.filter((m) => m.disponible);
  for (const entrada of entradas) {
    if (entrada.value === "comparar") entrada.disabled = disponibles.length < 2;
    else entrada.disabled = !estado.modelos.find((m) => m.id === entrada.value)?.disponible;
  }
  const primero = entradas.find((e) => !e.disabled);
  if (primero) primero.checked = true;
  actualizarAviso();
  if (!disponibles.length) mostrarError("No hay ningún modelo guardado. Entrena el notebook y guarda el modelo (secciones 7 y 8) en la carpeta modelos.");
}

iniciar();
