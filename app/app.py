"""Cyber stress test: pick a company and an attack, see which ratios it moves, how the IG/SG model reacts, and where the line is.

Run with app/run.sh (it sets the library path LightGBM needs on macOS).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from engine import ANCHORS, ATTACKS, CALL, LABELS, TAX, Attack, Engine  # noqa: E402

BLUE, ORANGE, SURFACE, INK, INK2, MUTED, GRID = "#2a78d6", "#eb6834", "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
SHARE = ANCHORS.set_index("incident").share_of_revenue

st.set_page_config(page_title="Cyber stress test", page_icon="🛡️", layout="wide")


@st.cache_resource(show_spinner="Fitting the five cross-fitted models (about 5 seconds)…")
def engine():
    return Engine()


eng = engine()
co = eng.companies
G = eng.grade_names


def layout(fig, height, xtitle, ytitle):
    fig.update_layout(height=height, margin=dict(l=10, r=60, t=10, b=10), plot_bgcolor=SURFACE, paper_bgcolor=SURFACE,
                      font=dict(color=INK2, size=12), showlegend=False, hoverlabel=dict(bgcolor="white", font_color=INK))
    fig.update_xaxes(title=xtitle, gridcolor=GRID, zeroline=False, linecolor=MUTED, ticks="outside", tickcolor=MUTED)
    fig.update_yaxes(title=ytitle, gridcolor=GRID, zeroline=False, linecolor=MUTED)
    return fig


# ---------------- sidebar ----------------
with st.sidebar:
    st.header("1 · Company")
    sector = st.selectbox("Sector", ["All sectors"] + sorted(co.Sector.unique()))
    only_ig = st.checkbox("Only rated IG and below the review screen", value=True,
                          help="The 169 companies the headroom result is about: rated investment grade, not yet on the review list.")
    pool = co if sector == "All sectors" else co[co.Sector == sector]
    if only_ig:
        pool = pool[(pool.SG == 0) & (pool.p_sg < eng.screen)]
    if pool.empty:
        st.warning("No company matches these filters."); st.stop()
    pool = pool.sort_values("p_sg", ascending=False)
    labels = {n: f"{n} · {r.Sector} · rated {r.Rating} · {G[r.grade]} · P(SG) {r.p_sg:.2f}" for n, r in pool.iterrows()}
    default = (pool.headroom_screen - 0.015).abs().idxmin()   # a company an M&S-scale incident tips: a good first look
    name = st.selectbox("Company (closest to the screen first)", list(labels), index=list(labels).index(default), format_func=labels.get)
    gm = eng.gross_margin_used(name)

    st.header("2 · Attack")
    kind = st.radio("Type of attack", list(ATTACKS))
    st.caption(ATTACKS[kind])

    def outage_picker(options, default_days):
        pick = st.selectbox("Size", options + ["Custom"])
        if pick == "Custom":
            return st.slider("Days of full shutdown", 1, 120, default_days) / 365
        incident = pick.split(" scale")[0]
        lost = SHARE[incident] / gm                    # days whose lost profit equals the reported cost, for this company
        st.caption(f"Lost profit of {SHARE[incident]:.2%} of revenue, as {incident} reported = **{lost * 365:.0f} days** of full shutdown "
                   f"for this company (gross margin {gm:.0%}).")
        return lost

    if kind == "Operations shutdown":
        attack = Attack(lost=outage_picker(["Marks & Spencer scale", "MGM Resorts scale", "Norsk Hydro scale"], 21))
    elif kind == "Data breach":
        cost = st.slider("Response, legal and regulatory costs (% of revenue)", 0.0, 10.0, 1.0, 0.1) / 100
        accrued = st.toggle("Provisioned now, paid in a later year", value=False,
                            help="Profit falls now, but no cash leaves this year: a new short-term liability instead.")
        attack = Attack(cost=cost, accrued=accrued)
        st.caption("None of the six reported incidents is a pure breach; for scale, M&S reported 2.17% of revenue.")
    elif kind == "Billing / payments disruption":
        pick = st.selectbox("Direct costs", ["UnitedHealth scale (0.77% of revenue)", "Custom"])
        cost = SHARE["UnitedHealth"] if pick != "Custom" else st.slider("Direct costs (% of revenue)", 0.0, 5.0, 0.5, 0.05) / 100
        delay = st.slider("Collections still late at year-end (extra days of sales)", 0, 60, 10,
                          help="Receivables still unpaid at the year-end close, on top of normal: that cash is missing from this year's "
                               "operating cash flow. A backlog cleared before year-end would not count.")
        attack = Attack(cost=cost, delay=delay)
    else:
        lost = outage_picker(["Maersk scale", "Merck scale"], 14)
        capex = st.slider("IT rebuild, capitalised (% of revenue)", 0.0, 3.0, 0.25, 0.05,
                          help="Illustrative: Maersk and Merck did not report the rebuild separately.") / 100
        attack = Attack(lost=lost, capex=capex)

    st.header("3 · How it lands")
    cash_funded = st.radio("Paid for with", ["New debt", "Cash on hand"], horizontal=True) == "Cash on hand"
    recovery = st.slider("Insurance recovery (% of the profit hit)", 0, 80, 0, 5,
                         help="Reported incident costs are mostly gross of insurance.") / 100
    tax = st.slider("Tax rate", 15, 35, int(TAX * 100), 5) / 100
    feed_all = st.toggle("Also feed liquidity and working-capital ratios to the model", value=False,
                         help="Off (recommended): ratios the model is not monotone in (cash, quick and current ratios, days sales outstanding, "
                              "asset turnover) are shown but held at today's value. The model reads more cash as riskier, so feeding them "
                              "can make an attack look like an improvement.")
    attack = Attack(**{**attack.__dict__, "cash_funded": cash_funded, "recovery": recovery, "tax": tax})

    st.header("4 · Money (optional)")
    revenue = st.number_input("Annual revenue, millions (0 = show % only)", min_value=0.0, value=0.0, step=100.0)
    currency = st.selectbox("Currency", ["€", "$", "£"], label_visibility="collapsed")


def money(share):
    if revenue <= 0 or not np.isfinite(share):
        return ""
    amount = share * revenue
    return f" ≈ {currency}{amount:,.1f}m" if amount < 10 else f" ≈ {currency}{amount:,.0f}m"


# the size people think in, per attack type: days of shutdown, days of delay, a cost, or a rebuild
if attack.lost > 0:
    unit, base_value, fmt, axis = "days", attack.lost * 365, (lambda v: f"{v:.0f} days of shutdown"), "days of full shutdown"
elif attack.delay > 0:
    unit, base_value, fmt, axis = "delay", attack.delay, (lambda v: f"{v:.0f} days of delayed collections"), "extra days to collect"
elif attack.cost > 0:
    unit, base_value, fmt, axis = "cost", attack.cost * 100, (lambda v: f"costs of {v:.2f}% of revenue"), "costs (% of revenue)"
else:
    unit, base_value, fmt, axis = "capex", attack.capex * 100, (lambda v: f"a rebuild of {v:.2f}% of revenue"), "IT rebuild (% of revenue)"
others = {"delay": "collections delay", "cost": "direct costs", "capex": "IT rebuild"}
scaled_with = [others[k] for k in ("delay", "cost", "capex") if k != unit and getattr(attack, k) > 0]

r = co.loc[name]
st.title("Cyber stress test")
st.caption("Pick a company and an attack. The attack moves the ratios it would hit in the accounts; the IG/SG screening model then "
           "re-scores the company. A triage screen for analysts, not a rating opinion and not a probability of default.")
st.subheader(f"{name} · {r.Sector}")
if base_value == 0:
    st.info("Set a size for the attack in the sidebar."); st.stop()

s = eng.attack_scenario(name, attack, feed_all)
b, a_ = s["before"], s["after"]
m_screen = eng.attack_line(name, attack, eng.screen, feed_all)
m_call = eng.attack_line(name, attack, CALL, feed_all)

# ---------------- verdict ----------------
st.caption(f"Latest statement {r.Date:%d %b %Y} · rated **{r.Rating}** by {r['Rating Agency Name']} "
           f"({'speculative' if r.SG else 'investment'} grade) · scored by fold model {s['fold'] + 1} of 5, which never saw this company")

imp = s["impacts"]
moved = a_["grade"] - b["grade"]
call_before, call_after = b["p_sg"] >= CALL, a_["p_sg"] >= CALL
tipped = (call_after and not call_before) or a_["p_sg"] >= eng.screen > b["p_sg"]
verdict = (f"**{kind}**, {fmt(base_value)}: a profit hit of **{imp['pnl']:.2%} of revenue**{money(imp['pnl'])}"
           + (f" and cash going out of {imp['cash_out']:.2%}{money(imp['cash_out'])}" if abs(imp["cash_out"] - imp["pnl"]) > 1e-4 else "")
           + f". P(SG) goes from **{b['p_sg']:.2f} to {a_['p_sg']:.2f}**"
           + (f", down {moved} grade{'s' if moved > 1 else ''} (**{G[b['grade']]} → {G[a_['grade']]}**)" if moved else f", grade stays **{G[b['grade']]}**")
           + (". **The model now calls it SG.**" if call_after and not call_before
              else ". It **lands on the review list**." if a_["p_sg"] >= eng.screen > b["p_sg"] else "."))
(st.error if tipped else st.info)(verdict)

c1, c2, c3, c4 = st.columns(4)
c1.metric("P(SG), calibrated", f"{a_['p_sg']:.2f}", f"{a_['p_sg'] - b['p_sg']:+.2f} from {b['p_sg']:.2f}", delta_color="inverse")
on_after, on_before = a_["p_sg"] >= eng.screen, b["p_sg"] >= eng.screen
c2.metric("Internal grade", G[a_["grade"]], f"from {G[b['grade']]}" if moved else None, delta_color="inverse")
c3.metric("Model call", "SG" if call_after else "IG", "from IG" if call_after and not call_before else None, delta_color="inverse",
          help="SG when calibrated P(SG) ≥ 0.5; not an agency decision.")
c4.metric("Review list", "on it" if on_after else "off it", "from off it" if on_after and not on_before else None, delta_color="inverse",
          help=f"The review screen: calibrated P(SG) ≥ {eng.screen:.2f}.")
if a_["p_sg"] < b["p_sg"] - 1e-9:
    st.warning("The model reads this attack as an improvement. That comes from the liquidity and working-capital ratios now fed to it: "
               "the model learned that more cash goes with more risk, so spending cash lowers P(SG). This is why those ratios are off by default.")
if r.negative_equity == 1:
    st.warning("Negative equity: the saved model's rule puts this company at SG-1 or worse whatever its P(SG).")

# ---------------- the line ----------------
st.markdown("#### The line: how big this attack can get before the credit view changes")


def line_text(m, what):
    if m == 0:
        return f"Already {what} before any attack."
    if not np.isfinite(m):
        return f"Not reached within the sizes searched (up to {attack.max_scale():.0f}× this scenario)."
    i = eng.attack_scenario(name, attack.scaled(m), feed_all)["impacts"]
    extra = f", with the {' and '.join(scaled_with)} scaled in proportion" if scaled_with else ""
    return (f"**{fmt(base_value * m)}**{extra}: a profit hit of **{i['pnl']:.2%} of revenue**{money(i['pnl'])}. "
            f"That is {m:.1f}× the scenario above.")


l1, l2 = st.columns(2)
with l1:
    st.markdown(f"**Review list** (P(SG) ≥ {eng.screen:.2f})")
    st.markdown(line_text(m_screen, "on the review list"))
with l2:
    st.markdown("**SG call** (P(SG) ≥ 0.50)")
    st.markdown(line_text(m_call, "called SG"))
st.caption("\"Affordable\" here means before the credit screen reacts, not before the company is in distress. "
           "It assumes the whole loss lands in one year's accounts.")

# ---------------- chart 1: P(SG) as the attack grows ----------------
finite = [m for m in (m_screen, m_call) if np.isfinite(m) and m > 0]
top = min(attack.max_scale(), max(2.0, 1.25 * max(finite, default=0)))
scales = np.linspace(0, top, 241)
curve = eng.attack_curve(name, attack, scales, feed_all)
x = base_value * scales
grades_on_curve = np.asarray(G)[eng.grade(co.loc[[name] * len(scales)], curve)]

fig = go.Figure()
for lo, hi, g in zip([0] + eng.cuts, eng.cuts + [1], G):
    fig.add_annotation(x=1, xref="paper", xanchor="left", y=(lo + hi) / 2, text=g, showarrow=False, font=dict(color=MUTED, size=11))
for y, label, dash in ((eng.screen, f"review screen {eng.screen:.2f}", "dash"), (CALL, "SG call 0.50", "dot")):
    fig.add_hline(y=y, line=dict(color=INK2, width=1.2, dash=dash))
    fig.add_annotation(x=0.005, xref="paper", xanchor="left", y=y, yanchor="bottom", text=label, showarrow=False, font=dict(color=INK2, size=11))
if unit in ("days", "cost"):   # reported incidents on the same axis (Maersk, MGM and UnitedHealth sit together)
    for incident, label in (("Norsk Hydro", "Norsk Hydro"), ("UnitedHealth", "Maersk, MGM, UnitedHealth"), ("Merck", "Merck"), ("Marks & Spencer", "M&S")):
        v = SHARE[incident] / gm * 365 if unit == "days" else SHARE[incident] * 100
        if v <= x[-1]:
            fig.add_vline(x=v, line=dict(color=GRID, width=1))
            fig.add_annotation(x=v, y=1, yanchor="top", xanchor="left", text=label, textangle=-90, showarrow=False, font=dict(color=MUTED, size=10))
fig.add_trace(go.Scatter(x=x, y=curve, mode="lines", line=dict(color=BLUE, width=2), customdata=grades_on_curve,
                         hovertemplate=f"%{{x:.2f}} {axis}<br>P(SG) %{{y:.3f}} · %{{customdata}}<extra></extra>"))
fig.add_trace(go.Scatter(x=[0, base_value], y=[b["p_sg"], a_["p_sg"]], mode="markers+text", text=["", "this scenario"],
                         textposition="top left", textfont=dict(color=INK2), customdata=["today", "this scenario"],
                         marker=dict(size=[9, 11], color=[BLUE, ORANGE], line=dict(color=SURFACE, width=2)),
                         hovertemplate="%{customdata}: P(SG) %{y:.3f}<extra></extra>"))
layout(fig, 380, axis + (f" ({', '.join(scaled_with)} scaled in proportion)" if scaled_with else ""), "calibrated P(SG)")
fig.update_yaxes(range=[0, 1], tickvals=[0] + eng.cuts + [1], gridcolor=GRID)
fig.update_xaxes(range=[0, x[-1]], showgrid=False)
st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
st.caption("Where the blue line crosses the dashed and dotted lines is the line above. Horizontal lines are the grade cut-offs; "
           "vertical lines are reported incidents at this company's scale. P(SG) never falls as the attack grows; the steps come from the decision trees.")

# ---------------- chart 2: SHAP before and after ----------------
st.markdown("#### What moved: each input's push towards SG, before and after")
sv_b, sv_a = b["shap"], a_["shap"]
size = pd.concat([sv_b.abs(), sv_a.abs()], axis=1).max(axis=1)
changed = (sv_a - sv_b).abs() > 1e-6
show = size[changed | size.isin(size.nlargest(12))].nlargest(14).index[::-1]
lab = lambda f: LABELS.get(f, f.replace("Sector=", "Sector: "))


def value_text(f):
    if f not in eng.features and f not in eng.flags:
        return "sector (not changed by the attack)"
    vb, va = b["given"][f], a_["given"][f]
    text = f"value {vb:.3f} → {va:.3f}" if pd.notna(vb) else "missing (training median used)"
    if f in eng.unconstrained and not feed_all and not np.isclose(vb, va, equal_nan=True):
        text += "<br>not fed to the model (held at today's value)"
    elif f in eng.features and pd.notna(va) and (va < s["clip_lo"][f] or va > s["clip_hi"][f]):
        text += f"<br>model sees {a_['seen'][f]:.3f}: clipped to the training 1st–99th percentile"
    return text


fig2 = go.Figure()
for f in show:
    fig2.add_trace(go.Scatter(x=[sv_b[f], sv_a[f]], y=[lab(f)] * 2, mode="lines", line=dict(color=GRID, width=3), hoverinfo="skip", showlegend=False))
for sv, color, nm in ((sv_b, BLUE, "before"), (sv_a, ORANGE, "after")):
    fig2.add_trace(go.Scatter(x=sv[show], y=[lab(f) for f in show], mode="markers", name=nm,
                              marker=dict(size=10, color=color, line=dict(color=SURFACE, width=2)),
                              customdata=[[value_text(f), sv_b[f], sv_a[f]] for f in show],
                              hovertemplate="<b>%{y}</b><br>%{customdata[0]}<br>SHAP %{customdata[1]:+.3f} → %{customdata[2]:+.3f}<extra></extra>"))
fig2.add_vline(x=0, line=dict(color=MUTED, width=1))
layout(fig2, 34 * len(show) + 90, "SHAP value, raw log-odds of SG (right = pushes towards SG)", "")
fig2.update_layout(showlegend=True, legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0, font=dict(color=INK2)))
st.plotly_chart(fig2, width="stretch", config={"displayModeBar": False})
logit = lambda p: np.log(p / (1 - p))
st.caption(f"Blue = today, orange = after the attack; hover for the values. Inputs the attack does not touch can still move a little: "
           f"in a tree model, an input's push depends on the values of the others. SHAP values add up to the model's raw log-odds: "
           f"{s['base']:+.2f} for the average company plus {sv_b.sum():+.2f} from this company's inputs gives {logit(b['raw']):+.2f} today; "
           f"the attack adds {sv_a.sum() - sv_b.sum():+.2f}, to {logit(a_['raw']):+.2f}. Calibration then maps raw log-odds to P(SG).")

# ---------------- table: the ratios the attack changed ----------------
with st.expander("The ratios this attack changed", expanded=True):
    rows = []
    for f in eng.features + eng.flags:
        vb, va = b["given"][f], a_["given"][f]
        if np.isclose(vb, va, equal_nan=True):
            continue
        held = f in eng.unconstrained and not feed_all
        clipped = f in eng.features and not held and pd.notna(va) and (va < s["clip_lo"][f] or va > s["clip_hi"][f])
        rows.append({"input": lab(f), "before": vb, "after": va, "fed to the model": "no: shown only" if held else "yes",
                     "SHAP before": sv_b[f], "SHAP after": sv_a[f], "SHAP change": sv_a[f] - sv_b[f],
                     "note": ("model not monotone in this ratio" if held else
                              "clipped at the training range: further damage no longer moves it" if clipped else "")})
    table = pd.DataFrame(rows).sort_values(["fed to the model", "SHAP change"], ascending=[False, False])
    st.dataframe(table, hide_index=True, width="stretch",
                 column_config={c: st.column_config.NumberColumn(format="%.3f") for c in table.columns if c not in ("input", "fed to the model", "note")})

# ---------------- portfolio view ----------------
with st.expander("Where this company sits among the 169 rated-IG companies (the deck's headroom curve)"):
    ig = co[(co.SG == 0) & (co.p_sg < eng.screen)]
    grid = np.linspace(0, 0.08, 321)
    share = [(ig.headroom_screen <= v).mean() * 100 for v in grid]
    fig3 = go.Figure(go.Scatter(x=grid * 100, y=share, mode="lines", line=dict(color=BLUE, width=2),
                                hovertemplate="%{x:.2f}% of revenue: %{y:.0f}% of companies on the review list<extra></extra>"))
    h = r.headroom_screen
    if name in ig.index and np.isfinite(h):
        fig3.add_vline(x=h * 100, line=dict(color=ORANGE, width=2))
        fig3.add_annotation(x=h * 100, y=95, xanchor="left", text=f" {name}: {h:.2%}", showarrow=False, font=dict(color=INK2))
    fig3.add_vline(x=SHARE["Marks & Spencer"] * 100, line=dict(color=GRID, width=1, dash="dash"))
    layout(fig3, 300, "incident cost charged to one year (% of revenue)", "rated-IG companies on the review list (%)")
    fig3.update_yaxes(range=[0, 100]); fig3.update_xaxes(range=[0, 8])
    st.plotly_chart(fig3, width="stretch", config={"displayModeBar": False})
    st.caption(f"The notebook's single-cost shock (paid, debt-funded, {TAX:.0%} tax) for every rated-IG company below the screen; "
               f"median {np.median(ig.headroom_screen):.2%}. The dashed line is M&S scale. "
               + ("This company is not in that group." if name not in ig.index else ""))

with st.expander("How each attack is charged, and what this does not show"):
    st.markdown(f"""
| Attack | What it does in the accounts | Ratios moved |
|---|---|---|
| Operations shutdown | Sales lost for the days chosen; the profit lost is lost sales × gross margin ({gm:.0%} here) | margins, returns, operating and free cash flow, debt ratio; asset turnover |
| Data breach | One-off costs through the P&L; if provisioned, no cash leaves this year and a short-term liability appears | margins, returns, debt ratio; cash flow only if paid now; current, quick and cash ratios if provisioned |
| Billing / payments disruption | Direct costs, plus receivables still unpaid at year-end, so this year's cash flow falls | margins, returns, operating and free cash flow, debt ratio; days sales outstanding, cash ratio |
| Destructive malware / IT rebuild | A shutdown, plus replacing IT as capital spending | as a shutdown, plus free cash flow and debt ratio; fixed-asset turnover |

- **The mapping from attack to ratios is ours, not the model's**: the model only sees ratios. Margins and cash flow are measured per unit of a
  normal year's revenue, as in the notebook, so lost profit is charged like a cost. Tax ({tax:.0%}) shields the hit only up to existing profit.
- **Paid for with new debt** raises the debt ratio by the cash going out; **cash on hand** spends cash first (current liabilities are
  estimated from receivables ÷ (quick − cash ratio)), then borrows the rest.
- **Ratios the model is not monotone in** (cash, quick and current ratios, days sales outstanding, asset and fixed-asset turnover) are
  shown but held at today's value unless the toggle is on. The model reads more liquidity as riskier, likely because companies with
  market access hold less cash; feeding a cash drain could make an attack look like an improvement.
- **A paid, debt-funded data breach is exactly the notebook's shock**, so the deck's numbers hold for it; the other attack types are
  explorations built on the same model and are not in the deck. Each company is scored by the fold model that never saw it.
- **One year, one company**: no recovery over time, no contagion, no management response. Not a rating; an SG call is the model's classification.
""")
