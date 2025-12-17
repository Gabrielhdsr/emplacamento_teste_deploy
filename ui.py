import streamlit as st
import re

# ============================================================
# 1. ESTILOS CSS
# ============================================================
def apply_style():
    # Retornando ao estilo original mais leve e clean
    css = """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
        
        html, body, [class*="css"] { font-family: 'Inter', sans-serif; color: #1e293b; }
        .stApp { background-color: #fffff; }
        header[data-testid="stHeader"] { display: none; }
        
        .block-container { padding-top: 0.5rem !important; padding-bottom: 2rem !important; max-width: 1600px; } 
        .main-header { margin-bottom: 1.5rem; }
        .main-title { font-size: 2.2rem; font-weight: 800; color: #0f172a; margin-bottom: 0.2rem; letter-spacing: -1px; }
        .sub-title { font-size: 1rem; color: #64748b; }
        
        /* CARDS KPI */
        .metric-card { background: #fff; border-radius: 16px; padding: 30px; border: 1px solid #e2e8f0; box-shadow: 0 4px 8px rgba(0,0,0,0.03); height: 100%; display: flex; flex-direction: column; justify-content: space-between; }
        .metric-card:hover { border-color: #cbd5e1; transform: translateY(-3px); }
        .metric-label { font-size: 0.9rem; font-weight: 700; text-transform: uppercase; color: #64748b; margin-bottom: 10px; }
        .metric-value { font-size: 3.5rem; font-weight: 900; color: #0f172a; line-height: 1; letter-spacing: -2px; }
        .metric-footer { margin-top: 20px; padding-top: 15px; border-top: 1px solid #f1f5f9; display: flex; justify-content: space-between; align-items: flex-end; }
        
        /* TRIMESTRES */
        .quarter-card { background: white; border-radius: 12px; padding: 22px; border: 1px solid #e2e8f0; position: relative; }
        .quarter-card.future { background: #f8fafc; border: 1px dashed #e2e8f0; opacity: 0.7; }
        .q-header { display: flex; justify-content: space-between; margin-bottom: 10px; font-size: 0.9rem; font-weight: 700; color: #64748b; }
        .q-value { font-size: 2.2rem; font-weight: 900; color: #0f172a; margin-bottom: 4px; }
        
        /* MESES */
        .month-tile { background: #fff; border-radius: 12px; padding: 18px; text-align: center; border: 1px solid #e2e8f0; height: 100%; margin-bottom: 1rem; }
        .month-tile:hover { border-color: #cbd5e1; }
        .month-name { font-size: 0.8rem; font-weight: 700; color: #94a3b8; text-transform: uppercase; margin-bottom: 6px; }
        .month-val { font-size: 1.7rem; font-weight: 800; color: #0f172a; }
        .month-delta { font-size: 0.8rem; font-weight: 600; margin-top: 4px; }

        /* BADGES */
        .delta-badge { font-size: 0.8rem; font-weight: 700; padding: 5px 12px; border-radius: 99px; display: inline-flex; align-items: center; gap: 4px; }
        .delta-pos { background: #dcfce7; color: #15803d; }
        .delta-neg { background: #fee2e2; color: #b91c1c; }
        .delta-neu { background: #f1f5f9; color: #475569; }

        /* UTILITÁRIOS */
        .section-divider { margin: 2rem 0 1rem 0; display: flex; align-items: center; gap: 12px; }
        .section-title { font-size: 1.1rem; font-weight: 800; color: #334155; text-transform: uppercase; letter-spacing: 1.2px; }
        .divider-line { height: 2px; background: #e2e8f0; width: 100%; }
        
        /* BORDAS COLORIDAS */
        .border-left-dark { border-left: 6px solid #1e293b !important; }
        .border-left-blue { border-left: 6px solid #3b82f6 !important; }
        .border-left-red { border-left: 6px solid #ef4444 !important; }
    </style>
    """
    st.markdown(_clean_html(css), unsafe_allow_html=True)

def _clean_html(html):
    return re.sub(r'\s+', ' ', html).strip()

def header(titulo, subtitulo):
    html = f"<div class='main-header'><div class='main-title'>{titulo}</div><div class='sub-title'>{subtitulo}</div></div>"
    st.markdown(_clean_html(html), unsafe_allow_html=True)

def section(titulo):
    html = f"<div class='section-divider'><div class='section-title'>{titulo}</div><div class='divider-line'></div></div>"
    st.markdown(_clean_html(html), unsafe_allow_html=True)

def kpi_card(
    label,
    value,
    delta_val=None,
    delta_fmt="",
    context_html="",
    is_percent=False,
    color=None,
    show_delta=True
):
    """
    KPI Card padrão.

    - show_delta=True  -> comportamento antigo (mostra badge de delta)
    - show_delta=False -> card limpo, sem delta
    """

    # Classe do delta
    if delta_val is None:
        cls = "delta-neu"
        icon = "—"
    else:
        cls = "delta-pos" if delta_val > 0 else "delta-neg" if delta_val < 0 else "delta-neu"
        icon = "▲" if delta_val > 0 else "▼" if delta_val < 0 else "—"

    border = f"border-left-{color}" if color else ""

    delta_html = (
        f"<div class='delta-badge {cls}'>{icon} {delta_fmt}</div>"
        if show_delta
        else ""
    )

    html = f"""
    <div class="metric-card {border}">
        <div>
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>

        <div class="metric-footer">
            <div style="font-size:0.85rem; color:#94a3b8; line-height:1.4">
                {context_html}
            </div>
            {delta_html}
        </div>
    </div>
    """

    return _clean_html(html)

def quarter_card(periodo, value_str, delta_val, delta_str, html_share, val_ant_str, is_future=False, color=None):
    c_delta = "#15803d" if delta_val > 0 else "#b91c1c"
    cls = "quarter-card" + (" future" if is_future else "")
    border = f"border-left-{color}" if color else ""
    
    # Mantendo a estrutura nova (LY em baixo), mas com o visual antigo (mais clean)
    html = f"""
    <div class='{cls} {border}'>
        <div class='q-header'>
            <span>{periodo}</span>
            <span style='color:{c_delta}; background:#f1f5f9; padding:2px 6px; border-radius:4px'>{delta_str}</span>
        </div>
        <div class='q-value'>{value_str}</div>
        
        <div style='font-size:0.8rem; color:#94a3b8; margin-bottom:12px; font-weight:500'>
            vs {val_ant_str} (LY)
        </div>
        
        <div style='margin: 8px 0; height:1px; background:#f1f5f9'></div>
        
        <div style='font-size:0.85rem; color:#64748b; display:flex; justify-content:space-between; align-items:center'>
            <span>Share FACCHINI: <b style='color:#475569'>{html_share}</b></span>
        </div>
    </div>"""
    return _clean_html(html)

def month_card(nome_mes, value_str, delta_val, delta_str, html_share, has_data=True, color=None):
    if not has_data:
        return _clean_html(f"<div class='month-tile' style='opacity:0.4; border:1px dashed #e2e8f0; background:#f9fafb'><div class='month-name'>{nome_mes}</div><div class='month-val' style='color:#cbd5e1'>—</div></div>")
    
    c_delta = "#16a34a" if delta_val >= 0 else "#dc2626"
    border = f"border-left-{color}" if color else ""
    
    html = f"""
    <div class='month-tile {border}'>
        <div class='month-name'>{nome_mes}</div>
        <div class='month-val'>{value_str}</div>
        <div class='month-delta' style='color:{c_delta}'>{delta_str}</div>
        
        <div style='font-size:0.75rem; color:#64748b; margin-top:8px; padding-top:6px; border-top:1px solid #f1f5f9'>
            Share FACCHINI: <b style='color:#475569'>{html_share}</b>
        </div>
    </div>"""
    return _clean_html(html)