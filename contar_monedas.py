"""
Conteo de monedas en imagen con:
- CLAHE (mejora de contraste local)
- Filtro pasa-altas en Fourier (realce de bordes)
- Suavizado (GaussianBlur)
- HoughCircles (detección de círculos) con barrido adaptativo
- Deduplicación (NMS)
- Clasificación física (1, 2, 5, 10 pesos) con cortes sesgados.

Cada etapa del procesamiento guarda una imagen intermedia en la carpeta "salidas/".

Uso:
  python contar_monedas.py --img dineros.webp --save resultado.png --target 25

Requisitos: Python 3, OpenCV (cv2), NumPy.
"""

import argparse
import os
from math import sqrt
import cv2
import numpy as np

# ======= CONFIGURACIÓN AJUSTABLE =======
# Sesgos de corte entre denominaciones. Mayores valores favorecen clasificar hacia la moneda de menor valor.
BIAS_UMBRAL_5_VS_10 = 1.90
BIAS_UMBRAL_2_VS_5 = 2.49757
BIAS_UMBRAL_1_VS_2 = 1.77
# =======================================


# ---------------------- FUNCIONES AUXILIARES ----------------------


def fourier_highpass(img_gray: np.ndarray, cutoff: int) -> np.ndarray:
    """
    Aplica un filtro pasa-altas ideal en el dominio de Fourier.
    Elimina bajas frecuencias (fondo) y realza bordes de las monedas.
    """
    h, w = img_gray.shape
    # Transformada de Fourier 2D
    f = np.fft.fft2(img_gray)
    fshift = np.fft.fftshift(f)

    # Crea máscara centrada
    Y, X = np.ogrid[:h, :w]
    cy, cx = h // 2, w // 2
    R = np.sqrt((Y - cy) ** 2 + (X - cx) ** 2)

    # Máscara pasa-altas
    mask = np.ones((h, w), dtype=np.float32)
    mask[R <= cutoff] = 0.0  # elimina bajas frecuencias (fondo liso)

    # Aplica máscara y reconstruye la imagen
    f_hp = fshift * mask
    ishift = np.fft.ifftshift(f_hp)
    img_hp = np.fft.ifft2(ishift).real

    # Normaliza intensidades
    img_hp -= img_hp.min()
    if img_hp.max() > 0:
        img_hp = img_hp / img_hp.max()
    return (img_hp * 255).astype(np.uint8)


def nms_circles(circles, center_thr_rel=0.80, radius_thr_rel=0.45):
    """
    Non-Maximum Suppression (NMS) para eliminar círculos redundantes.
    Conserva el círculo más grande si hay solapamientos significativos.
    """
    circles = sorted(circles, key=lambda t: t[2], reverse=True)
    keep = []
    for x, y, r in circles:
        ok = True
        for X, Y, R in keep:
            d = sqrt((x - X) ** 2 + (y - Y) ** 2)
            if d < center_thr_rel * min(r, R) and abs(r - R) < radius_thr_rel * max(
                r, R
            ):
                ok = False
                break
        if ok:
            keep.append((x, y, r))
    return keep


def hough_sweep(gray, H, W, target=None):
    """
    Barrido adaptativo de parámetros de HoughCircles.
    Prueba diferentes configuraciones para acercarse al número objetivo de monedas.
    """
    rmin = max(8, int(0.045 * H))  # radio mínimo relativo
    rmax = int(0.20 * H)  # radio máximo relativo

    deteccion_mejor = []
    mejor_gap = 1e9

    for minDist in [30, 28, 26, 24, 22, 20, 18]:
        for p2 in range(64, 16, -2):
            circles = cv2.HoughCircles(
                gray,
                cv2.HOUGH_GRADIENT,
                dp=1.2,
                minDist=minDist,
                param1=120,
                param2=p2,
                minRadius=rmin,
                maxRadius=rmax,
            )
            cur = []
            if circles is not None:
                for x, y, r in np.round(circles[0, :]).astype(int):
                    if rmin <= r <= rmax:
                        cur.append((x, y, r))
            cur = nms_circles(cur, 0.80, 0.45)

            # Selecciona la detección más cercana al número objetivo
            if target is not None:
                gap = abs(len(cur) - target)
                if gap < mejor_gap:
                    mejor_gap = gap
                    deteccion_mejor = cur
                if gap <= 1:
                    return cur

            # Si no hay target, se queda con la detección más poblada
            if target is None and len(cur) > len(deteccion_mejor):
                deteccion_mejor = cur

    return deteccion_mejor


def clasificar_por_radiokmeans(radii: np.ndarray, k: int = 4):
    """
    Clasifica los radios detectados usando K-Means para encontrar grupos.
    Devuelve etiquetas y centros ordenados de menor a mayor.
    """
    if len(radii) == 0:
        return None, None, 0
    Z = radii.reshape(-1, 1).astype(np.float32)
    k = int(min(k, len(Z)))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 120, 0.05)
    _compact, labels, centers = cv2.kmeans(
        Z, k, None, criteria, 12, cv2.KMEANS_PP_CENTERS
    )
    centers = centers.flatten()
    order = np.argsort(centers)
    remap = {int(order[i]): i for i in range(k)}
    labels_ranked = np.array([remap[int(c)] for c in labels.flatten()], dtype=int)
    centers_sorted = np.sort(centers)
    return labels_ranked, centers_sorted, k


def asignar_denominaciones_por_fisica_umbral_sesgado(
    radii, centers_sorted, bias_5_10=0.65
):
    """
    Asigna denominaciones a partir del tamaño físico de las monedas y umbrales sesgados.
    Usa proporciones aproximadas (en mm) para monedas mexicanas.
    """
    diam_mm = np.array([21.0, 23.0, 25.5, 28.0], dtype=np.float32)
    r_mm = diam_mm / 2.0

    if len(radii) == 0:
        return None, None

    # Estima el factor de escala (pixeles por milímetro)
    k_eff = 0 if centers_sorted is None else min(len(centers_sorted), 4)
    if k_eff >= 2:
        obs = centers_sorted[:k_eff].astype(np.float32)
        exp = r_mm[:k_eff]
        denom = float(np.dot(exp, exp))
        alpha = float(np.dot(exp, obs) / denom) if denom > 1e-9 else None
    elif k_eff == 1:
        alpha = float(centers_sorted[0]) / float(r_mm[0])
    else:
        alpha = float(np.max(radii) / r_mm[-1])

    if alpha is None or not np.isfinite(alpha) or alpha <= 0:
        alpha = float(np.median(radii) / np.median(r_mm))

    # Radios esperados en pixeles según la escala estimada
    r1, r2, r5, r10 = (alpha * r_mm).tolist()

    # Cortes sesgados entre denominaciones
    mid12 = r1 + float(BIAS_UMBRAL_1_VS_2) * (r2 - r1)
    mid25 = r2 + float(BIAS_UMBRAL_2_VS_5) * (r5 - r2)
    mid510 = r5 + float(bias_5_10) * (r10 - r5)

    denoms = ["1 Peso", "2 Pesos", "5 Pesos", "10 Pesos"]
    values = [1, 2, 5, 10]

    asignaciones = []
    for r in radii:
        if r < mid12:
            asignaciones.append(0)
        elif r < mid25:
            asignaciones.append(1)
        elif r < mid510:
            asignaciones.append(2)
        else:
            asignaciones.append(3)
    return np.array(asignaciones, dtype=int), (denoms, values)


# ---------------------- PROCESAMIENTO PRINCIPAL ----------------------


def main():
    # --- Argumentos CLI ---
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", required=True, help="Ruta de la imagen de entrada")
    ap.add_argument("--save", default=None, help="Ruta de salida anotada (opcional)")
    ap.add_argument(
        "--target", type=int, default=None, help="Número esperado de monedas (opcional)"
    )
    args = ap.parse_args()

    # Crea carpeta de salida si no existe
    os.makedirs("salidas", exist_ok=True)

    # 1) Lectura de imagen
    img = cv2.imread(args.img)
    if img is None:
        raise FileNotFoundError(f"No se pudo abrir: {args.img}")
    H, W = img.shape[:2]

    # 2) Mejora de contraste local (CLAHE)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    L2 = clahe.apply(L)
    eq = cv2.cvtColor(cv2.merge([L2, A, B]), cv2.COLOR_LAB2BGR)
    cv2.imwrite("salidas/1_contraste_CLAHE.png", eq)

    # 3) Filtro pasa-altas en Fourier
    gray0 = cv2.cvtColor(eq, cv2.COLOR_BGR2GRAY)
    cutoff = max(10, min(H, W) // 40)
    hp = fourier_highpass(gray0, cutoff=cutoff)
    cv2.imwrite("salidas/2_pasa_altas.png", hp)

    # 4) Suavizado para eliminar ruido fino antes de Hough
    gray = cv2.GaussianBlur(hp, (7, 7), 1.2)
    cv2.imwrite("salidas/3_suavizado.png", gray)

    # 5) Detección de círculos (Hough)
    circles = hough_sweep(gray, H, W, target=args.target)
    out_hough = img.copy()
    for x, y, r in circles:
        cv2.circle(out_hough, (x, y), r, (0, 255, 0), 2)
    cv2.imwrite("salidas/4_circulos_detectados.png", out_hough)

    # 6) Clasificación física por tamaño
    radii = np.array([r for (_, _, r) in circles], dtype=np.float32)
    _labels_ranked, centers_sorted, _k_eff = clasificar_por_radiokmeans(radii, k=4)
    labels_fisica, dv = asignar_denominaciones_por_fisica_umbral_sesgado(
        radii, centers_sorted, bias_5_10=BIAS_UMBRAL_5_VS_10
    )
    denoms, values = (
        (["1 Peso", "2 Pesos", "5 Pesos", "10 Pesos"], [1, 2, 5, 10])
        if dv is None
        else dv
    )

    # 7) Cálculo de conteo y valor total
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (0, 255, 255)]
    conteo = {d: 0 for d in denoms}
    valor_total = 0

    if labels_fisica is not None:
        for i, (x, y, r) in enumerate(circles):
            idx = int(labels_fisica[i])
            conteo[denoms[idx]] += 1
            valor_total += values[idx]

    # --- Reporte en consola ---
    print("\n--- RESUMEN ---")
    for d in denoms:
        print(f"{d}: {conteo[d]}")
    print(f"Total detectado: {len(circles)} monedas")
    print(f"Valor total estimado: ${valor_total} MXN")

    # 8) Imagen final anotada
    if args.save:
        out = img.copy()
        for i, (x, y, r) in enumerate(circles):
            idx = 0 if labels_fisica is None else int(labels_fisica[i])
            color = colors[idx]
            cv2.circle(out, (x, y), r, color, 2)
            cv2.putText(
                out,
                denoms[idx],
                (x - r, y - r - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )
        cv2.putText(
            out,
            f"Total: {len(circles)} monedas",
            (10, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            out,
            f"Valor: ${valor_total} MXN",
            (10, 44),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )
        cv2.imwrite(args.save, out)
        print(f"Imagen anotada guardada en: {args.save}")


# --- EJECUCIÓN DIRECTA ---
if __name__ == "__main__":
    main()


# ============================ NOTA DE USO ============================
# Para ejecutar el script desde la terminal:
#
#   python contar_monedas.py --img dineros.webp --save salidas/resultado.png --target 25
#
# Donde:
#   --img    -> ruta a la imagen de monedas
#   --save   -> (opcional) ruta de salida con nombre de archivo
#   --target -> (opcional) número esperado de monedas, ayuda a ajustar la búsqueda de Hough
#
# Las imágenes intermedias se guardan automáticamente en la carpeta "salidas/".
# =====================================================================
