#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Conteo de monedas en imagen con:
- CLAHE (mejora de contraste local)
- Pasa-altas en Fourier (realce de bordes)
- HoughCircles (detección de círculos) con barrido adaptativo
- Deduplicación (NMS sencillo)
- Clasificación anclada a tamaños físicos con cortes sesgados 1–2, 2–5 y 5–10.

Uso:
  python contar_monedas.py --img dineros.webp --save resultado.png --target 25

Requisitos: Python 3, OpenCV (cv2), NumPy.
"""

import argparse
from math import sqrt
import cv2
import numpy as np

# ======= CONFIGURACIÓN AJUSTABLE =======
# Empuja el corte entre 5 y 10 hacia 10 para no sobre-clasificar 10.
# 0.50 = punto medio; valores mayores favorecen la clase menor (5$).
BIAS_UMBRAL_5_VS_10 = 1.90

# Empuja el corte entre 2 y 5 hacia 5 para no sobre-clasificar 5.
# 0.50 = punto medio; valores mayores favorecen la clase menor (2$).
BIAS_UMBRAL_2_VS_5 = 2.4975

# Empuja el corte entre 1 y 2 hacia 2 para no sobre-clasificar 2.
# 0.50 = punto medio; valores mayores favorecen la clase menor (1$).
BIAS_UMBRAL_1_VS_2 = 1.70
# =======================================


# ---------------------- utilidades ----------------------


def fourier_highpass(img_gray: np.ndarray, cutoff: int) -> np.ndarray:
    """Filtro pasa-altas ideal en Fourier para realzar bordes."""
    h, w = img_gray.shape
    f = np.fft.fft2(img_gray)
    fshift = np.fft.fftshift(f)

    Y, X = np.ogrid[:h, :w]
    cy, cx = h // 2, w // 2
    R = np.sqrt((Y - cy) ** 2 + (X - cx) ** 2)

    mask = np.ones((h, w), dtype=np.float32)
    mask[R <= cutoff] = 0.0  # suprime bajas frecuencias

    f_hp = fshift * mask
    ishift = np.fft.ifftshift(f_hp)
    img_hp = np.fft.ifft2(ishift).real

    img_hp -= img_hp.min()
    if img_hp.max() > 0:
        img_hp = img_hp / img_hp.max()
    return (img_hp * 255).astype(np.uint8)


def nms_circles(circles, center_thr_rel=0.80, radius_thr_rel=0.45):
    """
    Non-Maximum Suppression para círculos: preserva el de mayor radio
    cuando hay detecciones muy cercanas con radios parecidos.
    circles: lista de (x,y,r) enteros.
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
    Devuelve lista deduplicada de (x,y,r).
    """
    rmin = max(8, int(0.045 * H))
    rmax = int(0.20 * H)

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

            cur = nms_circles(cur, center_thr_rel=0.80, radius_thr_rel=0.45)

            if target is not None:
                gap = abs(len(cur) - target)
                if gap < mejor_gap:
                    mejor_gap = gap
                    deteccion_mejor = cur
                if gap <= 1:
                    return cur

            if target is None and len(cur) > len(deteccion_mejor):
                deteccion_mejor = cur

    return deteccion_mejor


def clasificar_por_radiokmeans(radii: np.ndarray, k: int = 4):
    """
    K-Means sobre radios (px) para estimar centros y usar su escala.
    Retorna labels_ranked, centers_sorted, k_eff.
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
    Asigna denominaciones usando diámetros físicos y cortes sesgados:
    - mid12 sesgado por BIAS_UMBRAL_1_VS_2
    - mid25 sesgado por BIAS_UMBRAL_2_VS_5
    - mid510 sesgado por bias_5_10 (BIAS_UMBRAL_5_VS_10)
    """
    # Diámetros aprox. (mm); radios físicos (mm)
    diam_mm = np.array([21.0, 23.0, 25.5, 28.0], dtype=np.float32)
    r_mm = diam_mm / 2.0

    if len(radii) == 0:
        return None, None

    # Estimar alpha (px/mm) con centros si existen; fallback robusto
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

    # Radios esperados en pixeles
    r1, r2, r5, r10 = (alpha * r_mm).tolist()

    # Cortes sesgados
    mid12 = r1 + float(BIAS_UMBRAL_1_VS_2) * (r2 - r1)
    mid25 = r2 + float(BIAS_UMBRAL_2_VS_5) * (r5 - r2)
    mid510 = r5 + float(bias_5_10) * (r10 - r5)

    denoms = ["1 Peso", "2 Pesos", "5 Pesos", "10 Pesos"]
    values = [1, 2, 5, 10]

    asignaciones = []
    for r in radii:
        if r < mid12:
            asignaciones.append(0)  # $1
        elif r < mid25:
            asignaciones.append(1)  # $2
        elif r < mid510:
            asignaciones.append(2)  # $5
        else:
            asignaciones.append(3)  # $10
    return np.array(asignaciones, dtype=int), (denoms, values)


# ---------------------- script principal ----------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--img", required=True, help="Ruta de la imagen (e.g., dineros.webp)"
    )
    ap.add_argument("--save", default=None, help="Ruta de salida anotada (opcional)")
    ap.add_argument(
        "--target", type=int, default=None, help="Número objetivo de objetos (opcional)"
    )
    args = ap.parse_args()

    img = cv2.imread(args.img)
    if img is None:
        raise FileNotFoundError(f"No se pudo abrir: {args.img}")

    H, W = img.shape[:2]

    # 1) CLAHE en canal L* (LAB)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    L2 = clahe.apply(L)
    eq = cv2.cvtColor(cv2.merge([L2, A, B]), cv2.COLOR_LAB2BGR)

    # 2) Pasa-altas en Fourier
    gray0 = cv2.cvtColor(eq, cv2.COLOR_BGR2GRAY)
    cutoff = max(10, min(H, W) // 40)
    hp = fourier_highpass(gray0, cutoff=cutoff)

    # 3) Suavizado antes de Hough
    # Si hay brillos especulares, prueba median en lugar de Gaussian:
    # gray = cv2.medianBlur(hp, 5)
    gray = cv2.GaussianBlur(hp, (7, 7), 1.2)

    # 4) Hough con barrido
    circles = hough_sweep(gray, H, W, target=args.target)

    # 5) Clasificación con cortes sesgados
    radii = np.array([r for (_, _, r) in circles], dtype=np.float32)
    _labels_ranked, centers_sorted, _k_eff = clasificar_por_radiokmeans(radii, k=4)

    labels_fisica, dv = asignar_denominaciones_por_fisica_umbral_sesgado(
        radii, centers_sorted, bias_5_10=BIAS_UMBRAL_5_VS_10
    )
    if dv is not None:
        denoms, values = dv
    else:
        denoms = ["1 Peso", "2 Pesos", "5 Pesos", "10 Pesos"]
        values = [1, 2, 5, 10]

    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (0, 255, 255)]

    conteo = {d: 0 for d in denoms}
    valor_total = 0
    if labels_fisica is not None:
        for i, (x, y, r) in enumerate(circles):
            idx = int(labels_fisica[i])
            conteo[denoms[idx]] += 1
            valor_total += values[idx]

    total_objetos = len(circles)

    # ---- SALIDA EN CONSOLA ----
    print("\n--- RESUMEN ---")
    for d in denoms:
        print(f"{d}: {conteo[d]}")
    print(f"Total de objetos detectados: {total_objetos}")
    print(f"Valor total aproximado: ${valor_total} MXN")

    # ---- OPCIONAL: imagen anotada ----
    if args.save:
        out = img.copy()
        for i, (x, y, r) in enumerate(circles):
            idx = 0 if labels_fisica is None else int(labels_fisica[i])
            color = colors[idx]
            cv2.circle(out, (int(x), int(y)), int(r), color, 2)
            cv2.putText(
                out,
                denoms[idx],
                (int(x - r), int(y - r - 2)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )
        cv2.putText(
            out,
            f"Objetos: {total_objetos}",
            (10, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            out,
            f"Valor: ${valor_total} MXN",
            (10, 44),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imwrite(args.save, out)
        print(f"Imagen anotada guardada en: {args.save}")


if __name__ == "__main__":
    main()
