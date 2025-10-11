#!/usr.bin/env python3
# -*- coding: utf-8 -*-
"""
Conteo de monedas en imagen con:
- CLAHE y Ecualización de Histograma (Pre-procesamiento mejorado)
- Pasa-altas en Fourier (realce de bordes)
- HoughCircles con barrido adaptativo mejorado
- Clasificación estricta por tamaño de clúster K-Means (k=4) basada en el tamaño físico: $1 < $2 < $10 < $5.
"""

import argparse
from math import sqrt
import cv2
import numpy as np
from pathlib import Path


# ---------------------- utilidades ----------------------


def fourier_highpass(img_gray: np.ndarray, cutoff: int) -> np.ndarray:
    """Filtro pasa-altas ideal en el dominio de Fourier para realzar bordes."""
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


def nms_circles(circles, center_thr_rel=0.60, radius_thr_rel=0.35):
    """Non-Maximum Suppression para círculos."""
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
    """Barrido adaptativo de parámetros de HoughCircles."""
    rmin = max(8, int(0.045 * H))
    rmax = int(0.20 * H)

    deteccion_mejor = []
    mejor_gap = 1e9

    for minDist in range(16, 30, 2):
        for p2 in range(60, 14, -2):
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

    return deteccion_mejor


def clasificar_por_radiokmeans(radii: np.ndarray, k: int = 4):
    """
    K-Means con OpenCV sobre radios.
    Devuelve labels_ranked (0=más chico) y centers_sorted.
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

    # 1) CLAHE y Ecualización Global (Pre-procesamiento mejorado)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    L2 = clahe.apply(L)
    L2_eq = cv2.equalizeHist(L2)
    eq_bgr = cv2.cvtColor(cv2.merge([L2_eq, A, B]), cv2.COLOR_LAB2BGR)

    # 2) Pasa-altas en Fourier
    gray0 = cv2.cvtColor(eq_bgr, cv2.COLOR_BGR2GRAY)
    cutoff = max(10, min(H, W) // 40)
    hp = fourier_highpass(gray0, cutoff=cutoff)

    # 3) Suavizado ligero
    gray = cv2.GaussianBlur(hp, (7, 7), 1.2)

    # 4) Hough con barrido
    circles = hough_sweep(gray, H, W, target=args.target)

    # 5) Clasificación por radios (k=4)
    radii = np.array([r for (_, _, r) in circles], dtype=np.float32)
    labels_ranked, centers_sorted, k_eff = clasificar_por_radiokmeans(radii, k=4)

    # *** Mapeo Estricto Basado en Tamaño Físico Real ***
    # K-Means devuelve los clusters ordenados por tamaño (rank 0, 1, 2, 3),
    # donde 0 es el más pequeño.
    # Mapeo físico: R_0=$1 < R_1=$2 < R_2=$10 < R_3=$5

    # Definición de la asignación basada en el rank (0=chico, 3=grande)
    # Si k_eff < 4, el último elemento de la lista será el que se use como fallback.

    # Mapeo de Rango de K-Means (0..3) a la Denominación Real:
    DENOM_MAP = {0: "1 Peso", 1: "2 Pesos", 2: "10 Pesos", 3: "5 Pesos"}
    VALUE_MAP = {0: 1, 1: 2, 2: 10, 3: 5}
    # Colores: Verde (1P), Azul (2P), Amarillo (10P), Rojo (5P)
    COLOR_MAP = {0: (0, 255, 0), 1: (255, 0, 0), 2: (0, 255, 255), 3: (0, 0, 255)}

    # Lista de todas las denominaciones posibles para el resumen final
    all_denoms = ["1 Peso", "2 Pesos", "10 Pesos", "5 Pesos"]

    conteo = {d: 0 for d in all_denoms}
    valor_total = 0

    if labels_ranked is not None:
        for i, (x, y, r) in enumerate(circles):
            rank = int(labels_ranked[i])

            # Asegurar que el rank no exceda el mapeo, incluso si k_eff < 4
            effective_rank = min(rank, k_eff - 1)

            # Asignación usando el mapa corregido
            current_denom = DENOM_MAP.get(effective_rank, "Desconocido")
            current_value = VALUE_MAP.get(effective_rank, 0)

            conteo[current_denom] += 1
            valor_total += current_value

    total_objetos = len(circles)

    # ---- SALIDA EN CONSOLA ----
    print("\n--- RESUMEN ---")
    # Imprimir en el orden tradicional
    for d in all_denoms:
        if d in conteo and conteo[d] > 0:
            print(f"{d}: {conteo[d]}")
    print(f"Total de objetos detectados: {total_objetos}")
    print(f"Valor total aproximado: ${valor_total} MXN")

    # ---- OPCIONAL: imagen anotada ----
    if args.save:
        out = img.copy()
        for i, (x, y, r) in enumerate(circles):
            color = (255, 255, 255)
            denom_text = "N/A"
            if labels_ranked is not None:
                rank = int(labels_ranked[i])
                effective_rank = min(rank, k_eff - 1)

                # Asignación para el dibujo
                color = COLOR_MAP.get(effective_rank, (255, 255, 255))
                denom_text = DENOM_MAP.get(effective_rank, "Desconocido")

            cv2.circle(out, (int(x), int(y)), int(r), color, 2)
            cv2.putText(
                out,
                denom_text,
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
