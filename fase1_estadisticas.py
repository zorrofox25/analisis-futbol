"""Fase 1: estadísticas básicas de goles a partir de resultados históricos.

Requiere pandas. Ejemplo de uso:
    python fase1_estadisticas.py partidos.csv "Real Madrid" "Barcelona"
"""

from __future__ import annotations

import argparse
import re
from typing import Any

import pandas as pd
import requests


COLUMNAS_REQUERIDAS = {"HomeTeam", "AwayTeam", "FTHG", "FTAG"}
URL_BASE = "https://api.football-data.org/v4/competitions"


def cargar_partidos(ruta_csv: str) -> pd.DataFrame:
    """Lee y valida un CSV de resultados históricos.

    Las filas con nombres de equipo vacíos o goles no numéricos se descartan.
    """
    partidos = pd.read_csv(ruta_csv)
    faltantes = COLUMNAS_REQUERIDAS.difference(partidos.columns)
    if faltantes:
        nombres = ", ".join(sorted(faltantes))
        raise ValueError(f"Faltan columnas obligatorias en el CSV: {nombres}")

    partidos = partidos[list(COLUMNAS_REQUERIDAS)].copy()
    partidos["FTHG"] = pd.to_numeric(partidos["FTHG"], errors="coerce")
    partidos["FTAG"] = pd.to_numeric(partidos["FTAG"], errors="coerce")
    partidos = partidos.dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG"])
    partidos["HomeTeam"] = partidos["HomeTeam"].astype(str).str.strip()
    partidos["AwayTeam"] = partidos["AwayTeam"].astype(str).str.strip()
    partidos = partidos[
        (partidos["HomeTeam"] != "") & (partidos["AwayTeam"] != "")
    ]

    if partidos.empty:
        raise ValueError("El CSV no contiene partidos válidos para analizar.")
    return partidos


def cargar_partidos_api(codigo_liga: str, api_token: str) -> pd.DataFrame:
    """Descarga partidos finalizados de football-data.org y los normaliza.

    Devuelve las mismas cuatro columnas que usa el resto del proyecto:
    ``HomeTeam``, ``AwayTeam``, ``FTHG`` y ``FTAG``. Solo conserva partidos
    con goles de tiempo completo disponibles.
    """
    codigo_liga = codigo_liga.strip().upper()
    if not re.fullmatch(r"[A-Z0-9]{2,5}", codigo_liga):
        raise ValueError("Código de liga inválido; usa un código como PD o PL.")
    if not api_token or not api_token.strip():
        raise ValueError("Falta el token de football-data.org.")

    url = f"{URL_BASE}/{codigo_liga}/matches"
    try:
        respuesta = requests.get(
            url,
            headers={"X-Auth-Token": api_token.strip()},
            params={"status": "FINISHED"},
            timeout=30,
        )
    except requests.exceptions.Timeout as exc:
        raise RuntimeError("Tiempo de espera agotado al consultar football-data.org.") from exc
    except requests.exceptions.ConnectionError as exc:
        raise RuntimeError("No se pudo conectar con football-data.org.") from exc
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(f"Error de conexión con football-data.org: {exc}") from exc

    if respuesta.status_code == 401:
        raise RuntimeError("football-data.org rechazó el token (401). Verifica FOOTBALL_DATA_API_TOKEN.")
    if respuesta.status_code == 403:
        raise RuntimeError(
            "football-data.org denegó el acceso (403); revisa que tu plan permita esa liga."
        )
    if respuesta.status_code == 404:
        raise RuntimeError(f"No se encontró la liga/código '{codigo_liga}' (404).")
    if respuesta.status_code == 429:
        raise RuntimeError("Límite de solicitudes alcanzado en football-data.org (429).")
    try:
        respuesta.raise_for_status()
    except requests.exceptions.HTTPError as exc:
        raise RuntimeError(
            f"football-data.org respondió con HTTP {respuesta.status_code}."
        ) from exc

    try:
        contenido = respuesta.json()
    except ValueError as exc:
        raise RuntimeError("football-data.org devolvió una respuesta que no es JSON válido.") from exc

    if not isinstance(contenido, dict) or not isinstance(contenido.get("matches"), list):
        raise RuntimeError("La respuesta de football-data.org no contiene una lista 'matches'.")

    filas: list[dict[str, Any]] = []
    for indice, partido in enumerate(contenido["matches"]):
        try:
            equipo_local = partido["homeTeam"]["name"]
            equipo_visitante = partido["awayTeam"]["name"]
            goles_local = partido["score"]["fullTime"]["home"]
            goles_visitante = partido["score"]["fullTime"]["away"]
        except (KeyError, TypeError) as exc:
            raise RuntimeError(
                f"La respuesta contiene datos incompletos en el partido #{indice + 1}."
            ) from exc

        # Algunos partidos FINISHED pueden no tener el marcador completo informado.
        if goles_local is None or goles_visitante is None:
            continue
        if not equipo_local or not equipo_visitante:
            raise RuntimeError(f"Falta el nombre de un equipo en el partido #{indice + 1}.")
        try:
            fthg = int(goles_local)
            ftag = int(goles_visitante)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"Marcador inválido en el partido #{indice + 1}.") from exc
        if fthg < 0 or ftag < 0:
            raise RuntimeError(f"Marcador negativo en el partido #{indice + 1}.")

        filas.append(
            {
                "HomeTeam": str(equipo_local).strip(),
                "AwayTeam": str(equipo_visitante).strip(),
                "FTHG": fthg,
                "FTAG": ftag,
            }
        )

    partidos = pd.DataFrame(filas, columns=["HomeTeam", "AwayTeam", "FTHG", "FTAG"])
    if partidos.empty:
        raise ValueError(
            f"La liga {codigo_liga} no devolvió partidos finalizados con marcador completo."
        )
    return partidos


def calcular_estadisticas(partidos: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    """Calcula promedios de goles por equipo y promedios generales de liga.

    Para cada equipo se devuelven goles a favor y en contra por partido,
    separando sus partidos como local y como visitante.
    """
    requeridas = COLUMNAS_REQUERIDAS
    faltantes = requeridas.difference(partidos.columns)
    if faltantes:
        raise ValueError(f"Faltan columnas obligatorias: {', '.join(sorted(faltantes))}")
    if partidos.empty:
        raise ValueError("No hay partidos para calcular estadísticas.")

    local = partidos.groupby("HomeTeam").agg(
        GF_local=("FTHG", "mean"),
        GC_local=("FTAG", "mean"),
    )
    visitante = partidos.groupby("AwayTeam").agg(
        GF_visitante=("FTAG", "mean"),
        GC_visitante=("FTHG", "mean"),
    )

    # La unión conserva equipos aunque solo tengan datos como local o visitante.
    equipos = local.join(visitante, how="outer").sort_index()
    liga = {
        "goles_local_por_partido": float(partidos["FTHG"].mean()),
        "goles_visitante_por_partido": float(partidos["FTAG"].mean()),
    }
    return equipos, liga


def obtener_metricas_dos_equipos(
    equipos: pd.DataFrame,
    liga: dict[str, float],
    equipo_local: str,
    equipo_visitante: str,
) -> dict[str, Any]:
    """Devuelve métricas del equipo local, visitante y la media de liga."""
    ausentes = [
        nombre for nombre in (equipo_local, equipo_visitante)
        if nombre not in equipos.index
    ]
    if ausentes:
        raise ValueError(f"No se encontraron estos equipos: {', '.join(ausentes)}")

    return {
        "equipo_local": equipo_local,
        "metricas_local": equipos.loc[equipo_local].to_dict(),
        "equipo_visitante": equipo_visitante,
        "metricas_visitante": equipos.loc[equipo_visitante].to_dict(),
        "media_liga": liga.copy(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calcula promedios básicos de goles por equipo desde un CSV."
    )
    parser.add_argument("csv", help="Ruta al archivo CSV de partidos")
    parser.add_argument("equipo_local", help="Nombre del equipo que juega en casa")
    parser.add_argument("equipo_visitante", help="Nombre del equipo que juega fuera")
    args = parser.parse_args()

    partidos = cargar_partidos(args.csv)
    equipos, liga = calcular_estadisticas(partidos)
    resultado = obtener_metricas_dos_equipos(
        equipos, liga, args.equipo_local, args.equipo_visitante
    )

    print(pd.Series(resultado["metricas_local"], name=args.equipo_local).to_string())
    print()
    print(pd.Series(resultado["metricas_visitante"], name=args.equipo_visitante).to_string())
    print("\nMedia de liga (goles por partido):")
    print(pd.Series(liga).to_string())


if __name__ == "__main__":
    main()
