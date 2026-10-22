"""Dashboard Streamlit para analizar partidos y seguir el rendimiento histórico."""

from __future__ import annotations

import os
from typing import Any

import pandas as pd
import psycopg
import streamlit as st

from fase1_estadisticas import calcular_estadisticas, cargar_partidos_api
from fase2_poisson import calcular_partido_poisson
from fase3_valor_esperado import analizar_valor_esperado
from historial_predicciones import (
    PrediccionGuardada,
    leer_historial,
    liquidar_predicciones,
    guardar_prediccion,
    resumen_rendimiento,
)


LIGAS = {
    "La Liga (España)": "PD",
    "Premier League (Inglaterra)": "PL",
    "Bundesliga (Alemania)": "BL1",
    "Serie A (Italia)": "SA",
    "Ligue 1 (Francia)": "FL1",
    "Champions League": "CL",
}
NOMBRES_1X2 = {"1": "Victoria local", "X": "Empate", "2": "Victoria visitante"}


@st.cache_data(ttl=1800, show_spinner=False)
def cargar_datos_liga(
    codigo_liga: str,
    api_token: str,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    """Carga y cachea partidos finalizados, estadísticas y metadatos."""
    partidos = cargar_partidos_api(codigo_liga, api_token)
    equipos, liga = calcular_estadisticas(partidos)
    return equipos, liga, partidos


def obtener_token_configurado() -> str:
    """Lee el token desde Streamlit Secrets o una variable de entorno."""
    try:
        secreto = st.secrets.get("FOOTBALL_DATA_API_TOKEN", "")
    except FileNotFoundError:
        secreto = ""
    return str(secreto or os.environ.get("FOOTBALL_DATA_API_TOKEN", "")).strip()


def fecha_bogota(valor: Any, incluir_hora: bool = True) -> str:
    """Formatea una fecha ISO en la zona horaria de Bogotá."""
    if valor is None or valor == "":
        return "N/D"
    fecha = pd.to_datetime(valor, utc=True, errors="coerce")
    if pd.isna(fecha):
        return str(valor)
    formato = "%d %b %Y · %H:%M" if incluir_hora else "%d %b %Y"
    return fecha.tz_convert("America/Bogota").strftime(formato)


def construir_ultimos_resultados(
    partidos: pd.DataFrame,
    equipo: str,
    cantidad: int = 5,
) -> pd.DataFrame:
    """Devuelve los últimos resultados desde la perspectiva del equipo."""
    if partidos.empty:
        return pd.DataFrame()
    seleccion = partidos[
        (partidos["HomeTeam"] == equipo) | (partidos["AwayTeam"] == equipo)
    ].copy()
    seleccion = seleccion.sort_values("MatchDate", ascending=False, na_position="last")

    filas: list[dict[str, Any]] = []
    for _, partido in seleccion.head(cantidad).iterrows():
        es_local = partido["HomeTeam"] == equipo
        gf = int(partido["FTHG"] if es_local else partido["FTAG"])
        gc = int(partido["FTAG"] if es_local else partido["FTHG"])
        resultado = "G" if gf > gc else "E" if gf == gc else "P"
        filas.append(
            {
                "Fecha": fecha_bogota(partido.get("MatchDate"), incluir_hora=False),
                "Sede": "Local" if es_local else "Visitante",
                "Rival": partido["AwayTeam"] if es_local else partido["HomeTeam"],
                "Marcador": f"{gf}–{gc}",
                "Forma": resultado,
            }
        )
    return pd.DataFrame(filas)


def construir_tabla_rendimiento(equipos: pd.DataFrame, equipo: str) -> pd.DataFrame:
    """Construye la tabla local/visitante de un equipo."""
    metricas = equipos.loc[equipo]
    filas = []
    for sede, sufijo in (("Local", "local"), ("Visitante", "visitante")):
        filas.append(
            {
                "Sede": sede,
                "PJ": int(metricas[f"PJ_{sufijo}"]),
                "G–E–P": (
                    f"{int(metricas[f'PG_{sufijo}'])}–"
                    f"{int(metricas[f'PE_{sufijo}'])}–"
                    f"{int(metricas[f'PP_{sufijo}'])}"
                ),
                "Pts": int(metricas[f"Puntos_{sufijo}"]),
                "GF": int(metricas[f"GF_total_{sufijo}"]),
                "GC": int(metricas[f"GC_total_{sufijo}"]),
                "GF / PJ": float(metricas[f"GF_{sufijo}"]),
                "GC / PJ": float(metricas[f"GC_{sufijo}"]),
            }
        )
    return pd.DataFrame(filas)


def construir_mercados(
    prediccion: dict[str, Any],
    resumen_ev: dict[str, Any],
    cuotas_extra: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Reúne probabilidades y EV para mercados 1X2 y secundarios."""
    cuotas = {**(cuotas_extra or {})}
    probabilidades = prediccion["probabilidades_1x2"]
    cuotas.update({key: resumen_ev["mercados"][key]["cuota_casa"] for key in ("1", "X", "2")})

    mercados: list[tuple[str, str, float, str]] = [
        ("1", "Victoria local", probabilidades["1"], "1"),
        ("X", "Empate", probabilidades["X"], "X"),
        ("2", "Victoria visitante", probabilidades["2"], "2"),
        ("1X", "Doble oportunidad · 1X", probabilidades["1"] + probabilidades["X"], "1X"),
        ("12", "Doble oportunidad · 12", probabilidades["1"] + probabilidades["2"], "12"),
        ("X2", "Doble oportunidad · X2", probabilidades["X"] + probabilidades["2"], "X2"),
    ]
    for linea in (0.5, 1.5, 2.5, 3.5, 4.5):
        totales = prediccion["totales_goles"][f"{linea:.1f}"]
        over_key, under_key = f"O{linea:.1f}", f"U{linea:.1f}"
        mercados.append((over_key, f"Over {linea:.1f}", totales["over"], over_key))
        mercados.append((under_key, f"Under {linea:.1f}", totales["under"], under_key))
    mercados.extend(
        [
            ("BTTS_SI", "Ambos marcan · Sí", prediccion["ambos_marcan"]["si"], "BTTS_SI"),
            ("BTTS_NO", "Ambos marcan · No", prediccion["ambos_marcan"]["no"], "BTTS_NO"),
        ]
    )

    filas = []
    for codigo, etiqueta, probabilidad, cuota_key in mercados:
        cuota = cuotas.get(cuota_key)
        ev_pct = (probabilidad * cuota - 1.0) * 100.0 if cuota is not None else None
        filas.append(
            {
                "Código": codigo,
                "Mercado": etiqueta,
                "Prob. modelo": probabilidad,
                "Cuota justa": 1.0 / probabilidad if probabilidad > 0 else float("inf"),
                "Cuota casa": cuota,
                "EV %": ev_pct,
            }
        )
    return pd.DataFrame(filas)


def estilizar_mercados(tabla: pd.DataFrame) -> Any:
    """Resalta en verde filas con EV positivo y en rojo las negativas."""
    def color_fila(fila: pd.Series) -> list[str]:
        ev = fila["EV %"]
        if pd.isna(ev):
            color = ""
        elif ev > 0:
            color = "background-color: #dcfce7; color: #166534; font-weight: 600;"
        else:
            color = "background-color: #fff1f2; color: #9f1239;"
        return [color] * len(fila)

    return (
        tabla.style.apply(color_fila, axis=1)
        .format(
            {
                "Prob. modelo": "{:.2%}",
                "Cuota justa": "{:.2f}",
                "Cuota casa": lambda valor: "—" if pd.isna(valor) else f"{valor:.2f}",
                "EV %": lambda valor: "—" if pd.isna(valor) else f"{valor:+.2f}%",
            }
        )
        .hide(axis="columns", subset=["Código"])
    )


def mostrar_pestana_resumen(analisis: dict[str, Any] | None) -> None:
    """Presenta salidas del modelo ajustado, cuotas y goles esperados."""
    if not analisis:
        st.info("Selecciona equipos y pulsa **Calcular análisis** para generar el pronóstico.")
        return
    pred = analisis["prediccion"]
    ev = analisis["resumen_ev"]
    with st.container(border=True):
        st.markdown(f"#### {pred['equipo_local']} vs {pred['equipo_visitante']}")
        cols = st.columns(3)
        for col, mercado in zip(cols, ("1", "X", "2")):
            col.metric(
                NOMBRES_1X2[mercado],
                f"{pred['probabilidades_1x2'][mercado]:.1%}",
                f"Cuota justa {pred['cuotas_justas'][mercado]:.2f}",
            )
    cols = st.columns(2)
    with cols[0], st.container(border=True):
        st.markdown("**Goles esperados (xG del modelo)**")
        xg = st.columns(2)
        xg[0].metric(pred["equipo_local"], f"{pred['lambda_local']:.2f}")
        xg[1].metric(pred["equipo_visitante"], f"{pred['lambda_visitante']:.2f}")
        st.caption(
            "Las lambdas se estiman conjuntamente con los resultados históricos; "
            "las fuerzas de ataque y defensa incluyen ajuste por rival y shrinkage."
        )
        fuerzas = pred["fuerzas"]
        st.markdown("**Fuerzas relativas a la media de liga**")
        fuerza_cols = st.columns(2)
        fuerza_cols[0].metric("Ataque local", f"{fuerzas['ataque_local']:.2f}×")
        fuerza_cols[1].metric("Defensa local", f"{fuerzas['defensa_local']:.2f}×")
        fuerza_cols[0].metric("Ataque visitante", f"{fuerzas['ataque_visitante']:.2f}×")
        fuerza_cols[1].metric("Defensa visitante", f"{fuerzas['defensa_visitante']:.2f}×")
        st.metric("ρ · Dixon–Coles", f"{pred['rho_dixon_coles']:+.4f}")
    with cols[1], st.container(border=True):
        st.markdown("**Mercado 1X2 frente a la casa**")
        filas = []
        for mercado in ("1", "X", "2"):
            datos = ev["mercados"][mercado]
            filas.append(
                {
                    "Selección": mercado,
                    "Prob. modelo": pred["probabilidades_1x2"][mercado],
                    "Cuota justa": pred["cuotas_justas"][mercado],
                    "Cuota casa": datos["cuota_casa"],
                    "EV %": datos["ev_porcentaje"],
                }
            )
        tabla = pd.DataFrame(filas)
        st.dataframe(
            tabla.style.format(
                {
                    "Prob. modelo": "{:.1%}",
                    "Cuota justa": "{:.2f}",
                    "Cuota casa": "{:.2f}",
                    "EV %": "{:+.2f}%",
                }
            ),
            hide_index=True,
            use_container_width=True,
        )
    probs = pred["probabilidades_1x2"]
    pick = max(probs, key=probs.get)
    recomendacion = analisis.get("etiqueta_recomendada")
    if recomendacion:
        st.success(
            f"Mayor valor positivo: **{recomendacion}** · EV "
            f"**+{analisis['ev_recomendado_pct']:.2f}%**. "
            f"Pronóstico modal: **{pick}**."
        )
    else:
        st.info(f"Pronóstico modal 1X2: **{pick}**. No hay apuesta +EV en los mercados con cuota disponible.")


def mostrar_pestana_mercados(analisis: dict[str, Any] | None) -> None:
    """Muestra la tabla completa de mercados y resalta el valor positivo."""
    if not analisis:
        st.info("Calcula un pronóstico para ver probabilidades y cuotas justas.")
        return
    st.caption(
        "Las probabilidades usan Poisson con shrinkage y corrección Dixon–Coles. "
        "Las filas verdes tienen EV positivo y requieren una cuota de casa."
    )
    tabla = construir_mercados(
        analisis["prediccion"], analisis["resumen_ev"], analisis["cuotas_extra"]
    )
    st.dataframe(estilizar_mercados(tabla), hide_index=True, use_container_width=True)
    st.caption("Las cuotas no ingresadas se muestran como —; se presenta la probabilidad del modelo y la cuota justa.")


def mostrar_pestana_estadisticas(
    equipos: pd.DataFrame,
    partidos: pd.DataFrame,
    equipo_local: str,
    equipo_visitante: str,
) -> None:
    """Organiza rendimiento, forma reciente y enfrentamientos directos."""
    st.markdown("#### Rendimiento local / visitante")
    cols = st.columns(2)
    for col, equipo in zip(cols, (equipo_local, equipo_visitante)):
        with col, st.container(border=True):
            st.markdown(f"**{equipo}**")
            st.dataframe(
                construir_tabla_rendimiento(equipos, equipo).style.format(
                    {"GF / PJ": "{:.2f}", "GC / PJ": "{:.2f}"}
                ),
                hide_index=True,
                use_container_width=True,
            )

    st.markdown("#### Últimos 5 resultados")
    cols = st.columns(2)
    for col, equipo in zip(cols, (equipo_local, equipo_visitante)):
        with col, st.container(border=True):
            st.markdown(f"**{equipo}**")
            forma = construir_ultimos_resultados(partidos, equipo)
            if forma.empty:
                st.caption("Sin partidos finalizados disponibles.")
            else:
                st.dataframe(forma, hide_index=True, use_container_width=True)

    st.markdown("#### Enfrentamientos directos (H2H)")
    h2h = partidos[
        ((partidos["HomeTeam"] == equipo_local) & (partidos["AwayTeam"] == equipo_visitante))
        | ((partidos["HomeTeam"] == equipo_visitante) & (partidos["AwayTeam"] == equipo_local))
    ].copy()
    if h2h.empty:
        st.caption("No hay enfrentamientos directos disponibles en los partidos cargados.")
    else:
        h2h = h2h.sort_values("MatchDate", ascending=False).head(5)
        h2h_vista = pd.DataFrame(
            {
                "Fecha": h2h["MatchDate"].map(lambda fecha: fecha_bogota(fecha, False)),
                "Local": h2h["HomeTeam"],
                "Marcador": h2h.apply(lambda fila: f"{int(fila['FTHG'])}–{int(fila['FTAG'])}", axis=1),
                "Visitante": h2h["AwayTeam"],
            }
        )
        st.dataframe(h2h_vista, hide_index=True, use_container_width=True)


def mostrar_pestana_historial(estadisticas: dict[str, Any]) -> None:
    """Muestra rendimiento liquidado, tabla histórica y descarga CSV."""
    st.markdown("#### Rendimiento histórico")
    cols = st.columns(4)
    pct_acierto = estadisticas["porcentaje_acierto"]
    roi = estadisticas["roi_porcentaje"]
    cols[0].metric("Predicciones guardadas", f"{estadisticas['predicciones_guardadas']}")
    cols[1].metric("Acierto 1X2", "N/D" if pct_acierto is None else f"{pct_acierto:.1f}%")
    cols[2].metric("ROI apuestas +EV", "N/D" if roi is None else f"{roi:+.1f}%")
    cols[3].metric(
        "Partidos liquidados",
        f"{estadisticas['partidos_liquidados']}",
        f"{estadisticas['apuestas_liquidadas']} apuestas evaluadas",
    )
    st.caption(
        "El acierto usa el resultado 1X2 más probable. El ROI supone una unidad por apuesta +EV recomendada; "
        "solo se calcula después de liquidar el partido. Como no se selecciona una fecha de fixture, "
        "se liquida con el primer resultado posterior de esa misma localía que publique la API."
    )

    try:
        historial = leer_historial(limite=1000)
    except (OSError, psycopg.Error, RuntimeError, ValueError) as exc:
        st.error(f"No se pudo leer el historial: {exc}")
        return
    if historial.empty:
        st.info("Todavía no hay pronósticos guardados.")
        return

    fechas = pd.to_datetime(historial["fecha_utc"], utc=True, errors="coerce")
    vista = pd.DataFrame(
        {
            "Fecha": fechas.dt.tz_convert("America/Bogota").dt.strftime("%Y-%m-%d %H:%M"),
            "Liga": historial["nombre_liga"],
            "Partido": historial["equipo_local"] + " vs " + historial["equipo_visitante"],
            "Prob. 1/X/2": historial.apply(
                lambda f: f"{f['probabilidad_1']:.0%} / {f['probabilidad_x']:.0%} / {f['probabilidad_2']:.0%}",
                axis=1,
            ),
            "Cuotas 1/X/2": historial.apply(
                lambda f: f"{f['cuota_1']:.2f} / {f['cuota_x']:.2f} / {f['cuota_2']:.2f}",
                axis=1,
            ),
            "Pick modal": historial["pronostico_1x2"].fillna("N/D"),
            "Pick +EV": historial["mercado_recomendado"].fillna("—"),
            "Estado": historial["estado_resultado"].replace({"PENDIENTE": "Pendiente", "LIQUIDADA": "Liquidada"}),
            "Real": historial["resultado_real"].fillna("—"),
            "Acierto": historial["acierto"].map({1: "✅", 0: "❌"}).fillna("—"),
            "ROI (u)": historial["roi_unidades"].map(lambda valor: "—" if pd.isna(valor) else f"{valor:+.2f}"),
        }
    )
    st.dataframe(vista, hide_index=True, use_container_width=True)
    st.download_button(
        "Descargar historial CSV",
        data=historial.to_csv(index=False).encode("utf-8-sig"),
        file_name="historial_predicciones.csv",
        mime="text/csv",
    )


def main() -> None:
    """Construye el dashboard completo."""
    st.set_page_config(page_title="SportIQ AI · Football Analytics", page_icon="⚽", layout="wide")
    st.markdown(
        """
        <style>
        .block-container {padding-top: 1.4rem; padding-bottom: 2.5rem;}
        [data-testid="stMetric"] {background: rgba(120, 120, 128, 0.08); padding: 0.8rem; border-radius: 0.7rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("⚽ SportIQ AI")
    st.caption("Football analytics · Poisson probabilities · Value tracking")

    with st.sidebar:
        st.header("Configuración")
        token = obtener_token_configurado()
        if token:
            st.caption("Token cargado desde Streamlit Secrets.")
        else:
            token = st.text_input("API Token", type="password").strip()
        nombre_liga = st.selectbox("Competición", list(LIGAS.keys()))
        codigo_liga = LIGAS[nombre_liga]

    if not token:
        st.info("Configura `FOOTBALL_DATA_API_TOKEN` en Secrets para cargar los datos.")
        st.stop()
    try:
        with st.spinner("Actualizando competición y estadísticas…"):
            equipos, liga, partidos = cargar_datos_liga(codigo_liga, token)
            liquidar_predicciones(partidos, codigo_liga)
            rendimiento = resumen_rendimiento()
    except (RuntimeError, ValueError, OSError, psycopg.Error) as exc:
        st.error(f"No se pudieron cargar los datos: {exc}")
        st.stop()

    with st.container(border=True):
        kpis = st.columns(4)
        kpis[0].metric("Partidos analizados", f"{liga['partidos_analizados']:,}")
        roi_hist = rendimiento["roi_porcentaje"]
        kpis[1].metric("ROI historial +EV", "N/D" if roi_hist is None else f"{roi_hist:+.1f}%")
        acierto = rendimiento["porcentaje_acierto"]
        kpis[2].metric("Acierto 1X2", "N/D" if acierto is None else f"{acierto:.1f}%")
        kpis[3].metric("Actualizado", fecha_bogota(liga.get("consultado_en_utc")))
        st.caption(
            f"Último partido: {fecha_bogota(liga.get('ultimo_partido_utc'))} · "
            f"Última actualización reportada por la API: "
            f"{fecha_bogota(liga.get('ultima_actualizacion_api_utc'))} · "
            f"{nombre_liga} · {len(partidos)} resultados finalizados"
        )

    disponibles = sorted(
        equipos.dropna(subset=["GF_local", "GC_local", "GF_visitante", "GC_visitante"])
        .index.astype(str)
        .tolist()
    )
    if len(disponibles) < 2:
        st.error("No hay suficientes equipos con estadísticas locales y visitantes completas.")
        st.stop()

    with st.container(border=True):
        st.markdown("#### Configura el partido")
        with st.form("formulario_partido"):
            col_local, col_visitante = st.columns(2)
            with col_local:
                equipo_local = st.selectbox("Equipo local", disponibles, key="home_team")
            with col_visitante:
                away_options = [equipo for equipo in disponibles if equipo != equipo_local]
                equipo_visitante = st.selectbox("Equipo visitante", away_options, key="away_team")
            st.markdown("**Cuotas 1X2**")
            odds_cols = st.columns(3)
            with odds_cols[0]:
                cuota_1 = st.number_input("Local (1)", min_value=1.01, value=2.10, step=0.01)
            with odds_cols[1]:
                cuota_x = st.number_input("Empate (X)", min_value=1.01, value=3.40, step=0.01)
            with odds_cols[2]:
                cuota_2 = st.number_input("Visitante (2)", min_value=1.01, value=3.60, step=0.01)

            cuotas_extra: dict[str, float] = {}
            with st.expander("Cuotas opcionales · Doble oportunidad, Over/Under 2.5, ambos marcan"):
                agregar_extra = st.checkbox("Tengo cuotas para mercados adicionales")
                if agregar_extra:
                    extra_cols = st.columns(4)
                    campos_extra = [
                        ("1X", "Doble oportunidad 1X", 1.30),
                        ("12", "Doble oportunidad 12", 1.30),
                        ("X2", "Doble oportunidad X2", 1.30),
                        ("O2.5", "Over 2.5", 1.90),
                        ("U2.5", "Under 2.5", 1.90),
                        ("BTTS_SI", "Ambos marcan · Sí", 1.90),
                        ("BTTS_NO", "Ambos marcan · No", 1.90),
                    ]
                    for indice, (codigo, etiqueta, valor_inicial) in enumerate(campos_extra):
                        with extra_cols[indice % len(extra_cols)]:
                            cuotas_extra[codigo] = st.number_input(
                                etiqueta,
                                min_value=1.01,
                                value=valor_inicial,
                                step=0.01,
                                key=f"odds_{codigo}",
                            )
            calcular = st.form_submit_button("Calcular y guardar pronóstico", type="primary", use_container_width=True)

    if calcular:
        try:
            # El motor avanzado ajusta fuerzas sobre resultados históricos, con
            # regularización por muestra y corrección Dixon–Coles para marcadores bajos.
            prediccion = calcular_partido_poisson(
                partidos,
                equipo_local,
                equipo_visitante,
            )
            resumen_ev = analizar_valor_esperado(
                prediccion["probabilidades_1x2"],
                {"1": cuota_1, "X": cuota_x, "2": cuota_2},
            )
            tabla_mercados = construir_mercados(prediccion, resumen_ev, cuotas_extra)
        except (ValueError, KeyError, RuntimeError) as exc:
            st.error(f"No se pudo calcular el análisis: {exc}")
        else:
            candidatos = tabla_mercados.dropna(subset=["Cuota casa", "EV %"])
            candidatos = candidatos[candidatos["EV %"] > 0]
            if candidatos.empty:
                mercado_recomendado, cuota_recomendada, ev_recomendado, etiqueta_recomendada = (
                    None, None, None, None
                )
            else:
                mejor = candidatos.sort_values("EV %", ascending=False).iloc[0]
                mercado_recomendado = str(mejor["Código"])
                cuota_recomendada = float(mejor["Cuota casa"])
                ev_recomendado = float(mejor["EV %"])
                etiqueta_recomendada = str(mejor["Mercado"])

            pronostico_modal = max(
                prediccion["probabilidades_1x2"],
                key=prediccion["probabilidades_1x2"].get,
            )
            registro = PrediccionGuardada(
                codigo_liga=codigo_liga,
                nombre_liga=nombre_liga,
                equipo_local=equipo_local,
                equipo_visitante=equipo_visitante,
                cuota_1=cuota_1,
                cuota_x=cuota_x,
                cuota_2=cuota_2,
                probabilidad_1=prediccion["probabilidades_1x2"]["1"],
                probabilidad_x=prediccion["probabilidades_1x2"]["X"],
                probabilidad_2=prediccion["probabilidades_1x2"]["2"],
                ev_1=resumen_ev["mercados"]["1"]["ev"],
                ev_x=resumen_ev["mercados"]["X"]["ev"],
                ev_2=resumen_ev["mercados"]["2"]["ev"],
                ev_1_porcentaje=resumen_ev["mercados"]["1"]["ev_porcentaje"],
                ev_x_porcentaje=resumen_ev["mercados"]["X"]["ev_porcentaje"],
                ev_2_porcentaje=resumen_ev["mercados"]["2"]["ev_porcentaje"],
                lambda_local=prediccion["lambda_local"],
                lambda_visitante=prediccion["lambda_visitante"],
                pronostico_1x2=pronostico_modal,
                mercado_recomendado=mercado_recomendado,
                cuota_recomendada=cuota_recomendada,
                metricas_avanzadas={
                    "modelo": prediccion,
                    "valor_esperado": resumen_ev,
                    "cuotas_extra": cuotas_extra,
                    "mercados": tabla_mercados.astype(object)
                    .where(pd.notna(tabla_mercados), None)
                    .to_dict(orient="records"),
                },
            )
            try:
                guardar_prediccion(registro)
            except (OSError, psycopg.Error, RuntimeError) as exc:
                st.warning(f"Análisis calculado; no se pudo guardar el historial: {exc}")

            st.session_state["analisis_actual"] = {
                "prediccion": prediccion,
                "resumen_ev": resumen_ev,
                "cuotas_extra": cuotas_extra,
                "mercado_recomendado": mercado_recomendado,
                "etiqueta_recomendada": etiqueta_recomendada,
                "cuota_recomendada": cuota_recomendada,
                "ev_recomendado_pct": ev_recomendado,
            }
            st.session_state["partido_actual"] = (equipo_local, equipo_visitante)

    analisis = st.session_state.get("analisis_actual")
    equipo_stats = st.session_state.get("partido_actual", (disponibles[0], disponibles[1]))
    tab_resumen, tab_mercados, tab_stats, tab_historial = st.tabs(
        ["Resumen & Probabilidades", "Mercados & +EV", "Estadísticas & H2H", "Historial & Rendimiento"]
    )
    with tab_resumen:
        mostrar_pestana_resumen(analisis)
    with tab_mercados:
        mostrar_pestana_mercados(analisis)
    with tab_stats:
        mostrar_pestana_estadisticas(equipos, partidos, equipo_stats[0], equipo_stats[1])
    with tab_historial:
        mostrar_pestana_historial(rendimiento)


if __name__ == "__main__":
    main()
