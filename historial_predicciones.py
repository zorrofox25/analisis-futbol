"""Persistencia local SQLite para las predicciones calculadas por la app."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import pandas as pd


RUTA_DB_POR_DEFECTO = Path(__file__).with_name("predicciones.db")


@dataclass(frozen=True)
class PrediccionGuardada:
    """Campos que se guardan para dejar constancia de una predicción."""

    codigo_liga: str
    nombre_liga: str
    equipo_local: str
    equipo_visitante: str
    cuota_1: float
    cuota_x: float
    cuota_2: float
    probabilidad_1: float
    probabilidad_x: float
    probabilidad_2: float
    ev_1: float
    ev_x: float
    ev_2: float
    ev_1_porcentaje: float
    ev_x_porcentaje: float
    ev_2_porcentaje: float
    lambda_local: float
    lambda_visitante: float
    pronostico_1x2: str
    mercado_recomendado: str | None
    cuota_recomendada: float | None


@contextmanager
def _conectar(ruta_db: str | Path) -> Iterator[sqlite3.Connection]:
    """Crea la carpeta destino y cierra la conexión al salir del bloque."""
    ruta = Path(ruta_db)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    conexion = sqlite3.connect(ruta, timeout=20)
    try:
        yield conexion
        conexion.commit()
    except Exception:
        conexion.rollback()
        raise
    finally:
        conexion.close()


def inicializar_historial(ruta_db: str | Path = RUTA_DB_POR_DEFECTO) -> None:
    """Crea la tabla e índice si todavía no existen."""
    with _conectar(ruta_db) as conexion:
        conexion.execute(
            """
            CREATE TABLE IF NOT EXISTS predicciones (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fecha_utc TEXT NOT NULL,
                codigo_liga TEXT NOT NULL,
                nombre_liga TEXT NOT NULL,
                equipo_local TEXT NOT NULL,
                equipo_visitante TEXT NOT NULL,
                cuota_1 REAL NOT NULL,
                cuota_x REAL NOT NULL,
                cuota_2 REAL NOT NULL,
                probabilidad_1 REAL NOT NULL,
                probabilidad_x REAL NOT NULL,
                probabilidad_2 REAL NOT NULL,
                ev_1 REAL NOT NULL,
                ev_x REAL NOT NULL,
                ev_2 REAL NOT NULL,
                ev_1_porcentaje REAL NOT NULL,
                ev_x_porcentaje REAL NOT NULL,
                ev_2_porcentaje REAL NOT NULL,
                lambda_local REAL NOT NULL,
                lambda_visitante REAL NOT NULL,
                pronostico_1x2 TEXT,
                mercado_recomendado TEXT,
                cuota_recomendada REAL,
                estado_resultado TEXT NOT NULL DEFAULT 'PENDIENTE',
                resultado_real TEXT,
                acierto INTEGER,
                roi_unidades REAL,
                fecha_resultado_utc TEXT
            )
            """
        )
        # Migración sencilla para bases creadas por una versión anterior de la app.
        columnas = {
            fila[1] for fila in conexion.execute("PRAGMA table_info(predicciones)").fetchall()
        }
        columnas_nuevas = {
            "ev_1": "REAL NOT NULL DEFAULT 0.0",
            "ev_x": "REAL NOT NULL DEFAULT 0.0",
            "ev_2": "REAL NOT NULL DEFAULT 0.0",
            "pronostico_1x2": "TEXT",
            "mercado_recomendado": "TEXT",
            "cuota_recomendada": "REAL",
            "estado_resultado": "TEXT NOT NULL DEFAULT 'PENDIENTE'",
            "resultado_real": "TEXT",
            "acierto": "INTEGER",
            "roi_unidades": "REAL",
            "fecha_resultado_utc": "TEXT",
        }
        for nombre, tipo in columnas_nuevas.items():
            if nombre not in columnas:
                conexion.execute(f"ALTER TABLE predicciones ADD COLUMN {nombre} {tipo}")
                if nombre in {"ev_1", "ev_x", "ev_2"}:
                    sufijo = nombre[-1].lower()
                    conexion.execute(
                        f"UPDATE predicciones SET {nombre} = ev_{sufijo}_porcentaje / 100.0"
                    )
        conexion.execute(
            "CREATE INDEX IF NOT EXISTS idx_predicciones_fecha "
            "ON predicciones (fecha_utc DESC)"
        )


def guardar_prediccion(
    prediccion: PrediccionGuardada,
    ruta_db: str | Path = RUTA_DB_POR_DEFECTO,
) -> int:
    """Inserta una predicción y devuelve el identificador asignado."""
    inicializar_historial(ruta_db)
    fecha_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
    valores = (
        fecha_utc,
        prediccion.codigo_liga,
        prediccion.nombre_liga,
        prediccion.equipo_local,
        prediccion.equipo_visitante,
        prediccion.cuota_1,
        prediccion.cuota_x,
        prediccion.cuota_2,
        prediccion.probabilidad_1,
        prediccion.probabilidad_x,
        prediccion.probabilidad_2,
        prediccion.ev_1,
        prediccion.ev_x,
        prediccion.ev_2,
        prediccion.ev_1_porcentaje,
        prediccion.ev_x_porcentaje,
        prediccion.ev_2_porcentaje,
        prediccion.lambda_local,
        prediccion.lambda_visitante,
        prediccion.pronostico_1x2,
        prediccion.mercado_recomendado,
        prediccion.cuota_recomendada,
    )
    with _conectar(ruta_db) as conexion:
        cursor = conexion.execute(
            """
            INSERT INTO predicciones (
                fecha_utc, codigo_liga, nombre_liga, equipo_local, equipo_visitante,
                cuota_1, cuota_x, cuota_2,
                probabilidad_1, probabilidad_x, probabilidad_2,
                ev_1, ev_x, ev_2,
                ev_1_porcentaje, ev_x_porcentaje, ev_2_porcentaje,
                lambda_local, lambda_visitante,
                pronostico_1x2, mercado_recomendado, cuota_recomendada
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            valores,
        )
        return int(cursor.lastrowid)


def leer_historial(
    limite: int = 500,
    ruta_db: str | Path = RUTA_DB_POR_DEFECTO,
) -> pd.DataFrame:
    """Lee las predicciones más recientes, de la más nueva a la más antigua."""
    if limite < 1:
        raise ValueError("El límite del historial debe ser mayor que cero.")
    inicializar_historial(ruta_db)
    with _conectar(ruta_db) as conexion:
        return pd.read_sql_query(
            "SELECT * FROM predicciones ORDER BY fecha_utc DESC, id DESC LIMIT ?",
            conexion,
            params=(limite,),
        )


def liquidar_predicciones(
    partidos: pd.DataFrame,
    codigo_liga: str,
    ruta_db: str | Path = RUTA_DB_POR_DEFECTO,
) -> int:
    """Liquida pronósticos pendientes al aparecer un resultado posterior coincidente.

    Se requiere el mismo orden local/visitante y una fecha de partido posterior
    a la creación del pronóstico. El ROI usa una unidad apostada al mercado +EV
    recomendado: cuota - 1 si gana, -1 si pierde. Sin recomendación, no hay ROI.
    """
    requeridas = {"HomeTeam", "AwayTeam", "FTHG", "FTAG", "MatchDate"}
    if not requeridas.issubset(partidos.columns):
        return 0
    resultados = partidos.copy()
    resultados["MatchDate"] = pd.to_datetime(resultados["MatchDate"], utc=True, errors="coerce")
    resultados = resultados.dropna(subset=["MatchDate", "FTHG", "FTAG"])
    if resultados.empty:
        return 0

    inicializar_historial(ruta_db)
    liquidadas = 0
    with _conectar(ruta_db) as conexion:
        pendientes = conexion.execute(
            """
            SELECT id, fecha_utc, equipo_local, equipo_visitante,
                   probabilidad_1, probabilidad_x, probabilidad_2,
                   pronostico_1x2, mercado_recomendado, cuota_recomendada
            FROM predicciones
            WHERE codigo_liga = ? AND estado_resultado = 'PENDIENTE'
            """,
            (codigo_liga,),
        ).fetchall()

        for fila in pendientes:
            (
                registro_id, fecha_pronostico, equipo_local, equipo_visitante,
                prob_1, prob_x, prob_2, pronostico, recomendado, cuota_recomendada,
            ) = fila
            fecha_pronostico_utc = pd.to_datetime(fecha_pronostico, utc=True, errors="coerce")
            if pd.isna(fecha_pronostico_utc):
                continue

            candidatos = resultados[
                (resultados["HomeTeam"] == equipo_local)
                & (resultados["AwayTeam"] == equipo_visitante)
                & (resultados["MatchDate"] > fecha_pronostico_utc)
            ].sort_values("MatchDate")
            if candidatos.empty:
                continue

            partido = candidatos.iloc[0]
            goles_local, goles_visitante = int(partido["FTHG"]), int(partido["FTAG"])
            resultado_real = "1" if goles_local > goles_visitante else "2" if goles_local < goles_visitante else "X"
            if pronostico not in {"1", "X", "2"}:
                pronostico = max(
                    (("1", prob_1), ("X", prob_x), ("2", prob_2)),
                    key=lambda par: par[1],
                )[0]
            acierto = int(pronostico == resultado_real)
            apuesta_ganada: bool | None = None
            if recomendado in {"1", "X", "2", "1X", "12", "X2", "O2.5", "U2.5", "BTTS_SI", "BTTS_NO"}:
                ambos_marcan = goles_local > 0 and goles_visitante > 0
                selecciones_ganadoras = {
                    "1": resultado_real == "1",
                    "X": resultado_real == "X",
                    "2": resultado_real == "2",
                    "1X": resultado_real in {"1", "X"},
                    "12": resultado_real in {"1", "2"},
                    "X2": resultado_real in {"X", "2"},
                    "O2.5": goles_local + goles_visitante >= 3,
                    "U2.5": goles_local + goles_visitante <= 2,
                    "BTTS_SI": ambos_marcan,
                    "BTTS_NO": not ambos_marcan,
                }
                apuesta_ganada = selecciones_ganadoras[recomendado]
            if apuesta_ganada is not None and cuota_recomendada is not None:
                roi = float(cuota_recomendada) - 1.0 if apuesta_ganada else -1.0
            else:
                roi = None

            fecha_resultado = pd.Timestamp(partido["MatchDate"]).isoformat()
            conexion.execute(
                """
                UPDATE predicciones
                SET estado_resultado = 'LIQUIDADA', resultado_real = ?, acierto = ?,
                    roi_unidades = ?, fecha_resultado_utc = ?
                WHERE id = ?
                """,
                (resultado_real, acierto, roi, fecha_resultado, registro_id),
            )
            liquidadas += 1
    return liquidadas


def resumen_rendimiento(ruta_db: str | Path = RUTA_DB_POR_DEFECTO) -> dict[str, float | int | None]:
    """Calcula acierto del pronóstico y ROI real de apuestas recomendadas liquidadas."""
    inicializar_historial(ruta_db)
    with _conectar(ruta_db) as conexion:
        total = int(conexion.execute("SELECT COUNT(*) FROM predicciones").fetchone()[0])
        liquidadas = int(
            conexion.execute(
                "SELECT COUNT(*) FROM predicciones WHERE estado_resultado = 'LIQUIDADA'"
            ).fetchone()[0]
        )
        correctas = int(
            conexion.execute(
                "SELECT COALESCE(SUM(acierto), 0) FROM predicciones WHERE estado_resultado = 'LIQUIDADA'"
            ).fetchone()[0]
        )
        apuestas_liquidadas, apuestas_ganadas, roi_total = conexion.execute(
            """
            SELECT COUNT(*),
                   COALESCE(SUM(CASE WHEN roi_unidades > 0 THEN 1 ELSE 0 END), 0),
                   COALESCE(SUM(roi_unidades), 0)
            FROM predicciones
            WHERE estado_resultado = 'LIQUIDADA'
              AND mercado_recomendado IS NOT NULL
              AND roi_unidades IS NOT NULL
            """
        ).fetchone()

    apuestas_liquidadas = int(apuestas_liquidadas)
    return {
        "predicciones_guardadas": total,
        "partidos_liquidados": liquidadas,
        "aciertos_pronostico": correctas,
        "porcentaje_acierto": correctas / liquidadas * 100 if liquidadas else None,
        "apuestas_liquidadas": apuestas_liquidadas,
        "apuestas_ganadas": int(apuestas_ganadas),
        "roi_porcentaje": roi_total / apuestas_liquidadas * 100 if apuestas_liquidadas else None,
    }
