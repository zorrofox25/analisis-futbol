"""Fase 2: modelo básico de goles y resultados usando Poisson.

Uso desde otro script:
    from fase1_estadisticas import cargar_partidos, calcular_estadisticas
    from fase2_poisson import calcular_partido_poisson

    partidos = cargar_partidos("partidos.csv")
    equipos, liga = calcular_estadisticas(partidos)
    prediccion = calcular_partido_poisson(
        equipos, liga, "Real Madrid", "Barcelona"
    )
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import poisson


def calcular_partido_poisson(
    equipos: pd.DataFrame,
    liga: dict[str, float],
    equipo_local: str,
    equipo_visitante: str,
    max_goles: int = 10,
) -> dict[str, Any]:
    """Calcula fuerzas, goles esperados, matriz exacta, 1/X/2 y cuotas justas.

    Las fuerzas comparan los promedios de cada equipo con el promedio de liga
    correspondiente a esa perspectiva. La probabilidad de cada marcador es el
    producto de dos Poisson independientes.

    Las probabilidades 1/X/2 se normalizan sobre la matriz calculada para que
    sumen 1. Con max_goles=10, la masa omitida suele ser pequeña; se devuelve
    también como ``masa_omitida`` para que el truncamiento sea visible.
    """
    if max_goles < 1:
        raise ValueError("max_goles debe ser al menos 1 para calcular 1/X/2.")

    nombres = (equipo_local, equipo_visitante)
    ausentes = [nombre for nombre in nombres if nombre not in equipos.index]
    if ausentes:
        raise ValueError(f"No se encontraron estos equipos: {', '.join(ausentes)}")

    columnas = ("GF_local", "GC_local", "GF_visitante", "GC_visitante")
    faltantes = [col for col in columnas if col not in equipos.columns]
    if faltantes:
        raise ValueError(f"Faltan métricas de la Fase 1: {', '.join(faltantes)}")

    try:
        media_local = float(liga["goles_local_por_partido"])
        media_visitante = float(liga["goles_visitante_por_partido"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("El diccionario liga no tiene medias válidas de local y visitante.") from exc
    if media_local <= 0 or media_visitante <= 0:
        raise ValueError("Las medias de goles de liga deben ser mayores que cero.")

    local = equipos.loc[equipo_local]
    visitante = equipos.loc[equipo_visitante]
    valores = [
        local[col] for col in ("GF_local", "GC_local", "GF_visitante", "GC_visitante")
    ] + [
        visitante[col]
        for col in ("GF_local", "GC_local", "GF_visitante", "GC_visitante")
    ]
    if any(pd.isna(valor) for valor in valores):
        raise ValueError(
            "Faltan promedios de local o visitante para uno de los equipos; "
            "se necesitan partidos en ambas condiciones."
        )

    # Fuerzas ofensivas: goles marcados frente al promedio de esa condición.
    ataque_local = float(local["GF_local"]) / media_local
    ataque_visitante = float(visitante["GF_visitante"]) / media_visitante

    # Fuerzas defensivas: goles recibidos frente al promedio que anota el rival.
    defensa_local = float(local["GC_local"]) / media_visitante
    defensa_visitante = float(visitante["GC_visitante"]) / media_local

    # Modelo clásico: promedio de liga × ataque propio × debilidad defensiva rival.
    lambda_local = media_local * ataque_local * defensa_visitante
    lambda_visitante = media_visitante * ataque_visitante * defensa_local

    goles = list(range(max_goles + 1))
    prob_local = poisson.pmf(goles, lambda_local)
    prob_visitante = poisson.pmf(goles, lambda_visitante)
    matriz = pd.DataFrame(
        prob_local[:, None] * prob_visitante[None, :],
        index=pd.Index(goles, name="Goles_local"),
        columns=pd.Index(goles, name="Goles_visitante"),
    )

    masa_matriz = float(matriz.to_numpy().sum())
    if masa_matriz <= 0:
        raise ValueError("No se pudo obtener masa de probabilidad en la matriz.")
    valores_matriz = matriz.to_numpy()
    goles_local = np.array(goles)[:, None]
    goles_visitante = np.array(goles)[None, :]
    prob_1 = float(valores_matriz[goles_local > goles_visitante].sum())
    prob_x = float(sum(matriz.loc[g, g] for g in goles))
    prob_2 = float(valores_matriz[goles_local < goles_visitante].sum())

    # Corrige el pequeño truncamiento por marcadores superiores a max_goles.
    probabilidades = {
        "1": prob_1 / masa_matriz,
        "X": prob_x / masa_matriz,
        "2": prob_2 / masa_matriz,
    }
    cuotas_justas = {
        signo: (1.0 / probabilidad if probabilidad > 0 else float("inf"))
        for signo, probabilidad in probabilidades.items()
    }

    return {
        "equipo_local": equipo_local,
        "equipo_visitante": equipo_visitante,
        "fuerzas": {
            "ataque_local": ataque_local,
            "defensa_local": defensa_local,
            "ataque_visitante": ataque_visitante,
            "defensa_visitante": defensa_visitante,
        },
        "lambda_local": lambda_local,
        "lambda_visitante": lambda_visitante,
        "matriz_marcadores": matriz,
        "probabilidades_1x2": probabilidades,
        "cuotas_justas": cuotas_justas,
        "masa_omitida": 1.0 - masa_matriz,
    }
