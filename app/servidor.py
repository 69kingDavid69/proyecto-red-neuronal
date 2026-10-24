"""Programa para probar los modelos entrenados (lápiz · borrador · sacapuntas) desde el navegador.

Uso (desde la carpeta del proyecto, con fastai instalado):
    .venv/bin/python app/servidor.py
    .venv/bin/python app/servidor.py --puerto 8765 --modelos modelos --prueba fotos/prueba

Abre http://127.0.0.1:8765 en el navegador. Solo atiende a este computador (127.0.0.1): no se expone a la red.

Modelos: archivos .pkl que genera el notebook (`learn.export`). Se buscan en la carpeta --modelos:
    modelo_objetos_base.pkl           red propia (CNN de 4 capas)
    modelo_objetos_preentrenado.pkl   resnet18 preentrenada
Los .pkl usan pickle (pueden ejecutar código al cargarse): usar solo modelos que hizo el propio grupo.
"""
import argparse
import io
import json
import mimetypes
import random
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

AQUI = Path(__file__).resolve().parent
LIMITE_BYTES = 12 * 1024 * 1024
EXTENSIONES = {".jpg", ".jpeg", ".png", ".webp"}

NOMBRES = {"lapiz": "Lápiz", "borrador": "Borrador", "sacapuntas": "Sacapuntas"}

MODELOS = [
    {"id": "propia", "archivo": "modelo_objetos_base.pkl", "nombre": "Red propia",
     "detalle": "CNN de 4 capas entrenada desde cero con nuestras 180 fotos."},
    {"id": "preentrenada", "archivo": "modelo_objetos_preentrenado.pkl", "nombre": "Preentrenada",
     "detalle": "resnet18 que ya aprendió a ver con ImageNet, ajustada a nuestros tres objetos."},
]


class Estado:
    """Carga los modelos una sola vez y los comparte entre peticiones."""

    def __init__(self, carpeta_modelos, carpeta_prueba):
        self.carpeta_modelos = carpeta_modelos
        self.carpeta_prueba = carpeta_prueba.resolve()
        self._aprendices = {}
        self._cerrojo = threading.Lock()
        self.fotos_prueba = self._listar_prueba()

    def _listar_prueba(self):
        if not self.carpeta_prueba.is_dir():
            return []
        fotos = []
        for p in sorted(self.carpeta_prueba.rglob("*")):
            if p.suffix.lower() in EXTENSIONES and not p.name.startswith("."):
                fotos.append({"archivo": p.relative_to(self.carpeta_prueba).as_posix(), "real": p.parent.name})
        return fotos

    def modelos(self):
        salida = []
        for m in MODELOS:
            ruta = self.carpeta_modelos / m["archivo"]
            salida.append({k: m[k] for k in ("id", "nombre", "detalle")} | {"disponible": ruta.is_file()})
        return salida

    def aprendiz(self, id_modelo):
        with self._cerrojo:
            if id_modelo not in self._aprendices:
                meta = next((m for m in MODELOS if m["id"] == id_modelo), None)
                if meta is None:
                    raise KeyError("modelo desconocido")
                ruta = self.carpeta_modelos / meta["archivo"]
                if not ruta.is_file():
                    raise FileNotFoundError(f"Falta {ruta.name} en {self.carpeta_modelos}. Genérelo con el notebook (sección 7 u 8).")
                from fastai.vision.all import load_learner  # import tardío: arranca más rápido y falla con mensaje claro
                self._aprendices[id_modelo] = load_learner(ruta, cpu=True)
            return self._aprendices[id_modelo]

    def clasificar(self, id_modelo, imagen):
        """imagen: PIL.Image. Devuelve (clase, {clase: probabilidad}, milisegundos)."""
        from fastai.vision.all import PILImage
        aprendiz = self.aprendiz(id_modelo)
        t0 = time.perf_counter()
        clase, indice, probs = aprendiz.predict(PILImage.create(_a_bytes(imagen)))
        ms = (time.perf_counter() - t0) * 1000
        return str(clase), {str(c): float(p) for c, p in zip(aprendiz.dls.vocab, probs)}, ms

    def ruta_prueba(self, archivo):
        """Convierte un nombre relativo en ruta segura dentro de la carpeta de prueba (o None)."""
        try:
            ruta = (self.carpeta_prueba / archivo).resolve()
            ruta.relative_to(self.carpeta_prueba)
        except (ValueError, OSError):
            return None
        return ruta if ruta.is_file() and ruta.suffix.lower() in EXTENSIONES else None


def _a_bytes(imagen):
    buffer = io.BytesIO()
    imagen.convert("RGB").save(buffer, format="PNG")
    buffer.seek(0)
    return buffer


def abrir_imagen(datos):
    from PIL import Image, ImageOps, UnidentifiedImageError
    try:
        imagen = Image.open(io.BytesIO(datos))
        imagen.load()
    except (UnidentifiedImageError, OSError):
        raise ValueError("El archivo no es una imagen que se pueda leer. Probar con una foto .jpg o .png.")
    return ImageOps.exif_transpose(imagen).convert("RGB")   # respeta la orientación del celular


class Manejador(BaseHTTPRequestHandler):
    estado = None
    puerto = 0
    protocol_version = "HTTP/1.1"

    # ---- utilidades ---------------------------------------------------------------------
    def log_message(self, *_):
        pass   # sin ruido en la terminal

    def _host_valido(self):
        host = (self.headers.get("Host") or "").lower()
        return host in {f"127.0.0.1:{self.puerto}", f"localhost:{self.puerto}"}

    def _json(self, datos, codigo=200):
        cuerpo = json.dumps(datos, ensure_ascii=False).encode("utf-8")
        self.send_response(codigo)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(cuerpo)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(cuerpo)

    def _error(self, codigo, mensaje):
        self._json({"error": mensaje}, codigo)

    def _archivo(self, ruta, cache=False):
        datos = ruta.read_bytes()
        tipo = mimetypes.guess_type(ruta.name)[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", tipo + ("; charset=utf-8" if tipo.startswith("text/") or tipo.endswith("javascript") else ""))
        self.send_header("Content-Length", str(len(datos)))
        self.send_header("Cache-Control", "max-age=300" if cache else "no-store")
        self.end_headers()
        self.wfile.write(datos)

    def _respuesta_clasificacion(self, id_modelo, imagen, real=None):
        clase, probs, ms = self.estado.clasificar(id_modelo, imagen)
        salida = {"clase": clase, "nombre": NOMBRES.get(clase, clase), "probabilidades": probs,
                  "confianza": probs[clase], "ms": round(ms)}
        if real is not None:
            salida |= {"real": real, "acierto": clase == real}
        return salida

    # ---- rutas --------------------------------------------------------------------------
    def do_GET(self):
        if not self._host_valido():
            return self._error(403, "Host no permitido.")
        url = urlparse(self.path)
        consulta = parse_qs(url.query)
        try:
            if url.path in ("/", "/index.html"):
                return self._archivo(AQUI / "web" / "index.html")
            if url.path in ("/estilos.css", "/app.js"):
                return self._archivo(AQUI / "web" / url.path.lstrip("/"))
            if url.path == "/api/estado":
                return self._json({"modelos": self.estado.modelos(), "clases": [{"id": k, "nombre": v} for k, v in NOMBRES.items()],
                                   "prueba": len(self.estado.fotos_prueba)})
            if url.path == "/api/prueba":
                return self._json({"fotos": self.estado.fotos_prueba})
            if url.path == "/api/azar":
                n = max(1, min(int(consulta.get("n", ["5"])[0]), 30))
                return self._json({"fotos": random.sample(self.estado.fotos_prueba, min(n, len(self.estado.fotos_prueba)))})
            if url.path == "/api/foto":
                ruta = self.estado.ruta_prueba(consulta.get("archivo", [""])[0])
                return self._archivo(ruta, cache=True) if ruta else self._error(404, "Esa foto no está en la carpeta de prueba.")
            if url.path == "/api/prueba_foto":
                ruta = self.estado.ruta_prueba(consulta.get("archivo", [""])[0])
                if not ruta:
                    return self._error(404, "Esa foto no está en la carpeta de prueba.")
                return self._json(self._respuesta_clasificacion(consulta.get("modelo", ["propia"])[0], abrir_imagen(ruta.read_bytes()), ruta.parent.name))
            if url.path == "/api/evaluar":
                return self._json(self._evaluar(consulta.get("modelo", ["propia"])[0]))
            return self._error(404, "No existe esa página.")
        except FileNotFoundError as e:
            self._error(409, str(e))
        except (KeyError, ValueError) as e:
            self._error(400, str(e))
        except Exception as e:   # noqa: BLE001 - se informa al navegador en vez de cerrar la conexión
            self._error(500, f"Error inesperado: {type(e).__name__}: {e}")

    def do_POST(self):
        if not self._host_valido():
            return self._error(403, "Host no permitido.")
        url = urlparse(self.path)
        if url.path != "/api/predecir":
            return self._error(404, "No existe esa página.")
        try:
            largo = int(self.headers.get("Content-Length") or 0)
            if largo <= 0:
                return self._error(400, "No llegó ninguna imagen.")
            if largo > LIMITE_BYTES:
                return self._error(413, "La imagen pesa más de 12 MB. Probar con una más pequeña.")
            datos = self.rfile.read(largo)
            id_modelo = parse_qs(url.query).get("modelo", ["propia"])[0]
            self._json(self._respuesta_clasificacion(id_modelo, abrir_imagen(datos)))
        except FileNotFoundError as e:
            self._error(409, str(e))
        except (KeyError, ValueError) as e:
            self._error(400, str(e))
        except Exception as e:   # noqa: BLE001
            self._error(500, f"Error inesperado: {type(e).__name__}: {e}")

    # ---- evaluación completa ------------------------------------------------------------
    def _evaluar(self, id_modelo):
        fotos = self.estado.fotos_prueba
        if not fotos:
            raise ValueError("No hay fotos en la carpeta de prueba.")
        clases = list(NOMBRES)
        matriz = {r: {c: 0 for c in clases} for r in clases}
        errores, aciertos = [], 0
        for f in fotos:
            ruta = self.estado.ruta_prueba(f["archivo"])
            clase, probs, _ = self.estado.clasificar(id_modelo, abrir_imagen(ruta.read_bytes()))
            matriz[f["real"]][clase] += 1
            if clase == f["real"]:
                aciertos += 1
            else:
                errores.append({"archivo": f["archivo"], "real": f["real"], "prediccion": clase, "confianza": round(probs[clase], 2)})
        return {"modelo": id_modelo, "total": len(fotos), "aciertos": aciertos, "accuracy": aciertos / len(fotos),
                "clases": [{"id": c, "nombre": NOMBRES[c]} for c in clases], "matriz": matriz, "errores": errores}


def main():
    raiz = AQUI.parent
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--puerto", type=int, default=8765)
    ap.add_argument("--modelos", type=Path, default=raiz / "modelos")
    ap.add_argument("--prueba", type=Path, default=raiz / "fotos" / "prueba")
    args = ap.parse_args()

    Manejador.estado = Estado(args.modelos, args.prueba)
    Manejador.puerto = args.puerto
    servidor = ThreadingHTTPServer(("127.0.0.1", args.puerto), Manejador)   # solo este computador
    disp = [m["nombre"] for m in Manejador.estado.modelos() if m["disponible"]]
    print(f"Modelos disponibles: {', '.join(disp) or 'ninguno (falta exportar con el notebook)'}")
    print(f"Fotos de prueba: {len(Manejador.estado.fotos_prueba)}")
    print(f"Abrir en el navegador: http://127.0.0.1:{args.puerto}   (Ctrl+C para cerrar)")
    try:
        servidor.serve_forever()
    except KeyboardInterrupt:
        print("\nCerrado.")


if __name__ == "__main__":
    main()
