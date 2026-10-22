"""Poisson con regularización de ataques/defensas y ajuste Dixon–Coles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson


@dataclass(frozen=True)
class ModeloAjustado:
    """Parámetros ajustados sobre los resultados históricos de la liga."""

    equipos: tuple[str, ...]
    intercepto: float
    ventaja_local: float
    ataques: np.ndarray
    defensas: np.ndarray
    rho: float


def _validar_partidos(partidos: pd.DataFrame) -> pd.DataFrame:
    requeridas = {"HomeTeam", "AwayTeam", "FTHG", "FTAG"}
    faltantes = requeridas - set(partidos.columns)
    if faltantes:
        raise ValueError(f"Faltan columnas requeridas: {sorted(faltantes)}")

    datos = partidos[["HomeTeam", "AwayTeam", "FTHG", "FTAG"]].copy()
    datos["FTHG"] = pd.to_numeric(datos["FTHG"], errors="coerce")
    datos["FTAG"] = pd.to_numeric(datos["FTAG"], errors="coerce")
    datos = datos.dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG"])
    if datos.empty:
        raise ValueError("No hay resultados válidos para ajustar el modelo.")
    goles = datos[["FTHG", "FTAG"]].to_numpy(dtype=float)
    if (goles < 0).any() or not np.allclose(goles, np.round(goles)):
        raise ValueError("Los goles deben ser enteros no negativos.")
    datos["FTHG"] = datos["FTHG"].astype(int)
    datos["FTAG"] = datos["FTAG"].astype(int)
    return datos.reset_index(drop=True)


def _tau_dc(
    goles_local: np.ndarray,
    goles_visitante: np.ndarray,
    lambda_local: np.ndarray,
    lambda_visitante: np.ndarray,
    rho: float,
) -> np.ndarray:
    """Factor Dixon–Coles, distinto de 1 solo en los cuatro marcadores bajos."""
    tau = np.ones_like(lambda_local, dtype=float)
    m00 = (goles_local == 0) & (goles_visitante == 0)
    m01 = (goles_local == 0) & (goles_visitante == 1)
    m10 = (goles_local == 1) & (goles_visitante == 0)
    m11 = (goles_local == 1) & (goles_visitante == 1)
    tau[m00] = 1 - lambda_local[m00] * lambda_visitante[m00] * rho
    tau[m01] = 1 + lambda_local[m01] * rho
    tau[m10] = 1 + lambda_visitante[m10] * rho
    tau[m11] = 1 - rho
    return tau


def ajustar_modelo(
    partidos: pd.DataFrame,
    *,
    fuerza_prior: float = 8.0,
    limite_rho: float = 0.10,
) -> ModeloAjustado:
    """Ajusta fuerzas con un prior de liga y corrige marcadores bajos.

    ``fuerza_prior`` es el peso del prior centrado en la media de liga. Al ser
    una penalización fija frente a la verosimilitud, los equipos con pocas
    observaciones quedan más cerca del promedio que los equipos con más datos.
    """
    if fuerza_prior < 0:
        raise ValueError("fuerza_prior no puede ser negativo.")
    if not 0 < limite_rho <= 0.15:
        raise ValueError("limite_rho debe estar entre 0 y 0.15.")

    datos = _validar_partidos(partidos)
    equipos = tuple(sorted(set(datos["HomeTeam"]) | set(datos["AwayTeam"])))
    n = len(equipos)
    if n < 2:
        raise ValueError("Se necesitan al menos dos equipos.")

    indice = {equipo: i for i, equipo in enumerate(equipos)}
    idx_h = datos["HomeTeam"].map(indice).to_numpy(dtype=int)
    idx_a = datos["AwayTeam"].map(indice).to_numpy(dtype=int)
    goles_h = datos["FTHG"].to_numpy(dtype=float)
    goles_a = datos["FTAG"].to_numpy(dtype=float)

    def expandir(parametros: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ataque_libre = parametros[: n - 1]
        defensa_libre = parametros[n - 1 : 2 * (n - 1)]
        ataque = np.append(ataque_libre, -ataque_libre.sum())
        defensa = np.append(defensa_libre, -defensa_libre.sum())
        return ataque, defensa

    def objetivo(parametros: np.ndarray) -> float:
        intercepto, ventaja = parametros[:2]
        ataque, defensa = expandir(parametros[2:-1])
        rho = float(parametros[-1])
        try:
            lam_h = np.exp(intercepto + ventaja + ataque[idx_h] - defensa[idx_a])
            lam_a = np.exp(intercepto + ataque[idx_a] - defensa[idx_h])
        except FloatingPointError:
            return 1e100
        if not np.isfinite(lam_h).all() or not np.isfinite(lam_a).all():
            return 1e100
        tau = _tau_dc(goles_h, goles_a, lam_h, lam_a, rho)
        if (tau <= 0).any():
            return 1e100

        log_likelihood = (
            poisson.logpmf(goles_h, lam_h)
            + poisson.logpmf(goles_a, lam_a)
            + np.log(tau)
        ).sum()
        penalizacion = 0.5 * fuerza_prior * (
            np.square(ataque).sum() + np.square(defensa).sum()
        )
        return float(-log_likelihood + penalizacion)

    media_goles = max(float(np.mean(np.r_[goles_h, goles_a])), 0.05)
    inicial = np.zeros(2 + 2 * (n - 1) + 1, dtype=float)
    inicial[0] = np.log(media_goles)
    limites = [(None, None)] * (len(inicial) - 1) + [(-limite_rho, limite_rho)]
    ajuste = minimize(objetivo, inicial, method="L-BFGS-B", bounds=limites)
    if not np.isfinite(ajuste.fun) or not np.isfinite(ajuste.x).all():
        raise RuntimeError(f"No se pudo ajustar el modelo: {ajuste.message}")
    if not ajuste.success:
        raise RuntimeError(
            "El optimizador no convergió; no se generará un pronóstico poco fiable. "
            f"Detalle: {ajuste.message}"
        )

    ataque, defensa = expandir(ajuste.x[2:-1])
    return ModeloAjustado(
        equipos=equipos,
        intercepto=float(ajuste.x[0]),
        ventaja_local=float(ajuste.x[1]),
        ataques=ataque,
        defensas=defensa,
        rho=float(ajuste.x[-1]),
    )


def _matriz_dc_sin_normalizar(
    lambda_local: float, lambda_visitante: float, rho: float, max_goles: int
) -> np.ndarray:
    """Matriz de probabilidades sin renormalizar, para medir truncamiento."""
    goles = np.arange(max_goles + 1)
    matriz = np.outer(poisson.pmf(goles, lambda_local), poisson.pmf(goles, lambda_visitante))
    matriz[0, 0] *= 1 - lambda_local * lambda_visitante * rho
    if max_goles >= 1:
        matriz[0, 1] *= 1 + lambda_local * rho
        matriz[1, 0] *= 1 + lambda_visitante * rho
        matriz[1, 1] *= 1 - rho
    if not np.isfinite(matriz).all() or (matriz < 0).any() or matriz.sum() <= 0:
        raise ValueError("La corrección Dixon–Coles produjo probabilidades inválidas.")
    return matriz


def _matriz_dc(lambda_local: float, lambda_visitante: float, rho: float, max_goles: int) -> np.ndarray:
    """Matriz Dixon–Coles normalizada dentro del soporte solicitado."""
    matriz = _matriz_dc_sin_normalizar(lambda_local, lambda_visitante, rho, max_goles)
    return matriz / matriz.sum()


def calcular_partido_poisson(
    partidos: pd.DataFrame,
    equipo_local: str,
    equipo_visitante: str,
    *,
    max_goles: int = 10,
    fuerza_prior: float = 8.0,
    limite_rho: float = 0.10,
) -> dict[str, Any]:
    """Ajusta el modelo histórico y calcula probabilidades para el encuentro."""
    if equipo_local == equipo_visitante:
        raise ValueError("El equipo local y el visitante deben ser distintos.")
    if max_goles < 5:
        raise ValueError("max_goles debe ser al menos 5.")

    modelo = ajustar_modelo(partidos, fuerza_prior=fuerza_prior, limite_rho=limite_rho)
    posiciones = {equipo: i for i, equipo in enumerate(modelo.equipos)}
    ausentes = [e for e in (equipo_local, equipo_visitante) if e not in posiciones]
    if ausentes:
        raise ValueError(f"Equipos no encontrados en el histórico: {', '.join(ausentes)}")

    i_h, i_a = posiciones[equipo_local], posiciones[equipo_visitante]
    lambda_local = float(np.exp(
        modelo.intercepto + modelo.ventaja_local + modelo.ataques[i_h] - modelo.defensas[i_a]
    ))
    lambda_visitante = float(np.exp(
        modelo.intercepto + modelo.ataques[i_a] - modelo.defensas[i_h]
    ))
    rho = modelo.rho

    # La matriz visible usa el límite indicado; los mercados se calculan sobre
    # un soporte mayor para que el truncamiento de marcadores altos sea mínimo.
    matriz = _matriz_dc(lambda_local, lambda_visitante, rho, max_goles)
    soporte = max(
        max_goles,
        int(poisson.ppf(1 - 1e-12, lambda_local)),
        int(poisson.ppf(1 - 1e-12, lambda_visitante)),
    )
    soporte = min(soporte, 100)
    matriz_completa = _matriz_dc(lambda_local, lambda_visitante, rho, soporte)
    goles = np.arange(soporte + 1)
    hh, aa = np.meshgrid(goles, goles, indexing="ij")

    probabilidades = {
        "1": float(matriz_completa[hh > aa].sum()),
        "X": float(matriz_completa[hh == aa].sum()),
        "2": float(matriz_completa[hh < aa].sum()),
    }
    suma_1x2 = sum(probabilidades.values())
    probabilidades = {k: v / suma_1x2 for k, v in probabilidades.items()}
    cuotas_justas = {k: 1.0 / v if v > 0 else float("inf") for k, v in probabilidades.items()}

    total = hh + aa
    btts_si = float(matriz_completa[(hh > 0) & (aa > 0)].sum())
    top_indices = np.argsort(matriz.ravel())[-5:][::-1]
    marcadores_top = []
    for flat_idx in top_indices:
        gh, ga = np.unravel_index(flat_idx, matriz.shape)
        marcadores_top.append({
            "marcador": f"{gh}-{ga}",
            "probabilidad": float(matriz[gh, ga]),
        })

    return {
        "equipo_local": equipo_local,
        "equipo_visitante": equipo_visitante,
        "fuerzas": {
            "ataque_local": float(np.exp(modelo.ataques[i_h])),
            "defensa_local": float(np.exp(modelo.defensas[i_h])),
            "ataque_visitante": float(np.exp(modelo.ataques[i_a])),
            "defensa_visitante": float(np.exp(modelo.defensas[i_a])),
        },
        "lambda_local": lambda_local,
        "lambda_visitante": lambda_visitante,
        "lambda_total": lambda_local + lambda_visitante,
        "rho_dixon_coles": rho,
        "matriz_marcadores": pd.DataFrame(
            matriz, index=pd.Index(range(max_goles + 1), name="Goles_local"),
            columns=pd.Index(range(max_goles + 1), name="Goles_visitante"),
        ),
        "marcadores_mas_probables": marcadores_top,
        "probabilidades_1x2": probabilidades,
        "cuotas_justas": cuotas_justas,
        "totales_goles": {
            f"{linea:.1f}": {
                "over": float(matriz_completa[total >= int(linea + 0.5)].sum()),
                "under": float(matriz_completa[total < int(linea + 0.5)].sum()),
            }
            for linea in (0.5, 1.5, 2.5, 3.5, 4.5)
        },
        "ambos_marcan": {"si": btts_si, "no": 1.0 - btts_si},
        # La matriz visible se mide sin normalizar; así el truncamiento no queda
        # oculto por la normalización de _matriz_dc.
        "masa_omitida_matriz": float(
            max(0.0, 1.0 - _matriz_dc_sin_normalizar(
                lambda_local, lambda_visitante, rho, max_goles
            ).sum())
        ),
    }
