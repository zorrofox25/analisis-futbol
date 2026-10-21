"""Modelo de goles, resultados y mercados derivados mediante Poisson."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy.stats import poisson


def calcular_partido_poisson(
    equipos: pd.DataFrame,
    liga: Mapping[str, Any],
    equipo_local: str,
    equipo_visitante: str,
    max_goles: int = 10,
) -> dict[str, Any]:
    """Calcula fuerzas, lambdas, marcadores y probabilidades de varios mercados.

    La matriz exacta se presenta entre 0 y ``max_goles``. Las probabilidades
    1X2 se calculan sobre un soporte ampliado hasta cubrir al menos el 99.9999999999%
    de la masa de cada Poisson. Los totales usan que la suma de dos Poisson
    independientes también es Poisson; ambos marcan se deriva de las masas en 0.
    """
    if max_goles < 1:
        raise ValueError("max_goles debe ser al menos 1.")

    ausentes = [equipo for equipo in (equipo_local, equipo_visitante) if equipo not in equipos.index]
    if ausentes:
        raise ValueError(f"No se encontraron estos equipos: {', '.join(ausentes)}")

    columnas = ("GF_local", "GC_local", "GF_visitante", "GC_visitante")
    faltantes = [columna for columna in columnas if columna not in equipos.columns]
    if faltantes:
        raise ValueError(f"Faltan métricas de la Fase 1: {', '.join(faltantes)}")

    try:
        media_local = float(liga["goles_local_por_partido"])
        media_visitante = float(liga["goles_visitante_por_partido"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("El diccionario liga no contiene medias válidas de goles.") from exc
    if not np.isfinite(media_local) or not np.isfinite(media_visitante):
        raise ValueError("Las medias de liga deben ser finitas.")
    if media_local <= 0 or media_visitante <= 0:
        raise ValueError("Las medias de goles de liga deben ser mayores que cero.")

    local = equipos.loc[equipo_local]
    visitante = equipos.loc[equipo_visitante]
    necesarios = [local[col] for col in columnas] + [visitante[col] for col in columnas]
    if any(pd.isna(valor) for valor in necesarios):
        raise ValueError(
            "Faltan promedios en condición local o visitante para uno de los equipos."
        )

    ataque_local = float(local["GF_local"]) / media_local
    defensa_local = float(local["GC_local"]) / media_visitante
    ataque_visitante = float(visitante["GF_visitante"]) / media_visitante
    defensa_visitante = float(visitante["GC_visitante"]) / media_local

    lambda_local = media_local * ataque_local * defensa_visitante
    lambda_visitante = media_visitante * ataque_visitante * defensa_local
    if not np.isfinite(lambda_local) or not np.isfinite(lambda_visitante):
        raise ValueError("El modelo produjo lambdas no finitas.")
    if lambda_local < 0 or lambda_visitante < 0:
        raise ValueError("Las lambdas no pueden ser negativas.")

    # Matriz de marcadores para presentación (límite configurable por el llamador).
    goles_mostrados = np.arange(max_goles + 1)
    pmf_local_mostrada = poisson.pmf(goles_mostrados, lambda_local)
    pmf_visitante_mostrada = poisson.pmf(goles_mostrados, lambda_visitante)
    matriz = pd.DataFrame(
        pmf_local_mostrada[:, None] * pmf_visitante_mostrada[None, :],
        index=pd.Index(goles_mostrados, name="Goles_local"),
        columns=pd.Index(goles_mostrados, name="Goles_visitante"),
    )
    masa_matriz = float(matriz.to_numpy().sum())

    # Ampliamos el soporte para calcular 1X2 sin truncar la matriz que se muestra.
    cuantiles = [
        float(poisson.ppf(1.0 - 1e-12, lam)) if lam > 0 else 0.0
        for lam in (lambda_local, lambda_visitante)
    ]
    max_soporte = max(max_goles, int(max(cuantiles)) + 1)
    max_soporte = min(max_soporte, 1000)
    goles_soporte = np.arange(max_soporte + 1)
    pmf_local = poisson.pmf(goles_soporte, lambda_local)
    pmf_visitante = poisson.pmf(goles_soporte, lambda_visitante)

    # P(local gana) = sum P(H=i) P(A<i), y análogo para visitante.
    prob_1 = float(np.sum(pmf_local * poisson.cdf(goles_soporte - 1, lambda_visitante)))
    prob_x = float(np.sum(pmf_local * pmf_visitante))
    prob_2 = float(np.sum(pmf_visitante * poisson.cdf(goles_soporte - 1, lambda_local)))
    masa_1x2 = prob_1 + prob_x + prob_2
    if masa_1x2 <= 0:
        raise ValueError("No se pudo calcular la distribución de resultados 1X2.")

    # Normalización por el residuo numérico del soporte finito.
    probabilidades = {
        "1": prob_1 / masa_1x2,
        "X": prob_x / masa_1x2,
        "2": prob_2 / masa_1x2,
    }
    cuotas_justas = {
        mercado: 1.0 / probabilidad if probabilidad > 0 else float("inf")
        for mercado, probabilidad in probabilidades.items()
    }

    lambda_total = lambda_local + lambda_visitante
    lineas_totales: dict[str, dict[str, float]] = {}
    for linea in (0.5, 1.5, 2.5, 3.5, 4.5):
        # En una línea x.5, over significa un total >= ceil(x + 0.5).
        minimo_over = int(linea + 0.5)
        prob_over = float(poisson.sf(minimo_over - 1, lambda_total))
        lineas_totales[f"{linea:.1f}"] = {
            "over": prob_over,
            "under": 1.0 - prob_over,
        }

    prob_local_marca = float(1.0 - poisson.pmf(0, lambda_local))
    prob_visitante_marca = float(1.0 - poisson.pmf(0, lambda_visitante))
    ambos_marcan_si = prob_local_marca * prob_visitante_marca

    matriz_np = matriz.to_numpy()
    cantidad_top = min(5, matriz_np.size)
    indices_top = np.argsort(matriz_np, axis=None)[-cantidad_top:][::-1]
    marcadores_top = []
    for indice in indices_top:
        goles_local, goles_visitante = np.unravel_index(indice, matriz_np.shape)
        marcadores_top.append(
            {
                "marcador": f"{goles_local}-{goles_visitante}",
                "probabilidad": float(matriz_np[goles_local, goles_visitante]),
            }
        )

    return {
        "equipo_local": equipo_local,
        "equipo_visitante": equipo_visitante,
        "fuerzas": {
            "ataque_local": ataque_local,
            "defensa_local": defensa_local,
            "ataque_visitante": ataque_visitante,
            "defensa_visitante": defensa_visitante,
        },
        "desglose_lambda": {
            "local": {
                "media_liga_local": media_local,
                "fuerza_ataque_local": ataque_local,
                "fuerza_defensa_visitante": defensa_visitante,
                "formula": "media_liga_local × ataque_local × defensa_visitante",
            },
            "visitante": {
                "media_liga_visitante": media_visitante,
                "fuerza_ataque_visitante": ataque_visitante,
                "fuerza_defensa_local": defensa_local,
                "formula": "media_liga_visitante × ataque_visitante × defensa_local",
            },
        },
        "lambda_local": lambda_local,
        "lambda_visitante": lambda_visitante,
        "lambda_total": lambda_total,
        "matriz_marcadores": matriz,
        "marcadores_mas_probables": marcadores_top,
        "probabilidades_1x2": probabilidades,
        "cuotas_justas": cuotas_justas,
        "soporte_1x2_max_goles": max_soporte,
        "masa_omitida_1x2": max(0.0, 1.0 - masa_1x2),
        "masa_omitida_matriz": max(0.0, 1.0 - masa_matriz),
        "totales_goles": lineas_totales,
        "ambos_marcan": {
            "si": ambos_marcan_si,
            "no": 1.0 - ambos_marcan_si,
        },
    }
