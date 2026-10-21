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
                lambda_visitante REAL NOT NULL
            )
            """
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
                lambda_local, lambda_visitante
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
