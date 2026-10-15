"""Cyber stress test: pick a company and an attack, see which ratios it moves, how the IG/SG model reacts, and where the line is.

Run with app/run.sh (it sets the library path LightGBM needs on macOS).
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from engine import ANCHORS, ATTACKS, CALL, TAX, Attack, Engine, label as input_label  # noqa: E402

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
                          help="The companies the headroom result is about: rated investment grade, not yet on the review list.")
    pool = co if sector == "All sectors" else co[co.Sector == sector]
    if only_ig:
        pool = pool[(pool.SG == 0) & (pool.p_sg < eng.screen)]
    if pool.empty:
        st.warning("No company matches these filters."); st.stop()
    pool = pool.sort_values("p_sg", ascending=False)
    labels = {n: f"{n} · {r.Sector} · rated {r.Rating} · {G[r.grade]} · P(SG) {r.p_sg:.2f}" for n, r in pool.iterrows()}
    default = (pool.headroom_screen - 0.015).abs().idxmin()   # a company an M&S-scale incident tips: a good first look
    name = st.selectbox("Company (closest to the screen first)" if only_ig else "Company (highest P(SG) first)", list(labels),
                        index=list(labels).index(default), format_func=labels.get)
    gm, gm_source = eng.gross_margin_used(name)

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
                   f"for this company (profit lost per unit of sales lost: gross margin {gm:.0%}, {gm_source}).")
        return lost

    if kind == "Operations shutdown":
        attack = Attack(lost=outage_picker(["Marks & Spencer scale", "MGM Resorts scale", "Norsk Hydro scale"], 21))
    elif kind == "Data breach":
        cost = st.slider("Response, legal and regulatory costs (% of revenue)", 0.0, 10.0, 1.0, 0.1) / 100
        accrued = st.toggle("Provisioned now, paid in a later year", value=False,
                            help="Profit falls now, but no cash leaves this year: a short-term liability for the full cost instead, with a "
                                 "deferred-tax asset for its tax shield. The debt ratio therefore rises a little more than when paying with new "
                                 "debt (which borrows only the after-tax cash), while cash flow is untouched.")
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
        capex = st.slider("IT rebuild, capitalised, on top (% of revenue)", 0.0, 3.0, 0.25, 0.05,
                          help="An assumption added on top of the outage. Maersk and Merck did not report a rebuild figure separately; "
                               "part of it may already sit in their reported cost, so with a reported-incident size this scenario is "
                               "somewhat larger than the real incident. Set it to 0 to match the reported cost alone.") / 100
        attack = Attack(lost=lost, capex=capex)

    st.header("3 · How it lands")
    cash_funded = st.radio("Paid for with", ["New debt", "Cash on hand"], horizontal=True) == "Cash on hand"
    recovery = st.slider("Insurance recovery (% of the profit hit)", 0, 80, 0, 5,
                         help="Reported incident costs are mostly gross of insurance.") / 100
    tax = st.slider("Tax rate", 15, 35, int(TAX * 100), 5) / 100
    feed_all = st.toggle("Also feed liquidity and working-capital ratios to the model", value=False,
                         help="Off (recommended): ratios the model is not monotone in (cash, quick and current ratios, days sales outstanding, "
                              "asset and fixed-asset turnover) are shown but held at today's value. The model reads more liquidity as riskier, "
                              "so feeding them can make an attack look like an improvement.")
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
           f"({'speculative' if r.SG else 'investment'} grade) · "
           + ("holdout company: scored by the saved model, as on the slides" if r.holdout
              else f"scored by cross-fitted fold model {s['fold'] + 1} of 5, which never saw this company"))

imp = s["impacts"]
moved = a_["grade"] - b["grade"]
call_before, call_after = b["p_sg"] >= CALL, a_["p_sg"] >= CALL
tipped = (call_after and not call_before) or a_["p_sg"] >= eng.screen > b["p_sg"]
verdict = (f"**{kind}**, {fmt(base_value)}: a profit hit of **{imp['pnl']:.2%} of revenue**{money(imp['pnl'])}"
           + (" and no cash out this year" if imp["cash_out"] < 1e-9
              else f" and {imp['cash_out']:.2%} less cash this year{money(imp['cash_out'])}" if abs(imp["cash_out"] - imp["pnl"]) > 1e-4 else "")
           + f". P(SG) goes from **{b['p_sg']:.2f} to {a_['p_sg']:.2f}**"
           + (f", {'down' if moved > 0 else 'up'} {abs(moved)} grade{'s' if abs(moved) > 1 else ''} (**{G[b['grade']]} → {G[a_['grade']]}**)"
              if moved else f", grade stays **{G[b['grade']]}**")
           + (". **The model now calls it SG.**" if call_after and not call_before
              else ". It **lands on the review list**." if a_["p_sg"] >= eng.screen > b["p_sg"] else "."))
(st.error if tipped else st.info)(verdict)

c1, c2, c3, c4 = st.columns(4)
c1.metric("P(SG), calibrated", f"{a_['p_sg']:.2f}", f"{a_['p_sg'] - b['p_sg']:+.2f} from {b['p_sg']:.2f}", delta_color="inverse")
on_after, on_before = a_["p_sg"] >= eng.screen, b["p_sg"] >= eng.screen
c2.metric("Grade from P(SG)", G[a_["grade"]], f"{moved:+d} (was {G[b['grade']]})" if moved else None, delta_color="inverse",
          help=f"Internal grade from calibrated P(SG) alone (cut-offs {' / '.join(f'{c:g}' for c in eng.cuts)}), as in the deck's migration table. "
               "The saved model also applies a rule (negative equity: SG-1 or worse), flagged below when it applies.")
change = int(call_after) - int(call_before)
c3.metric("Model call", "SG" if call_after else "IG", f"{change:+d} (was {'SG' if call_before else 'IG'})" if change else None,
          delta_color="inverse", help="SG when calibrated P(SG) ≥ 0.5; not an agency decision.")
change = int(on_after) - int(on_before)
c4.metric("Review list", "on it" if on_after else "off it", f"{change:+d} (was {'on it' if on_before else 'off it'})" if change else None,
          delta_color="inverse", help=f"The review screen: calibrated P(SG) ≥ {eng.screen:.2f}.")
if a_["p_sg"] < b["p_sg"] - 1e-9:
    st.warning("The model reads this attack as an improvement. That comes from the liquidity and working-capital ratios now fed to it "
               "(for example a higher current ratio from receivables still unpaid, or a lower cash ratio after paying from cash): the model "
               "learned that more liquidity goes with more risk, so these ratios can move P(SG) the wrong way. This is why they are off by default.")
if r.negative_equity == 1:
    st.warning("Negative equity: the saved model's rule puts this company at SG-1 or worse whatever its P(SG), so its internal grade is at "
               "least SG-1. Leverage effects of an attack are approximate here: the attack raises the debt ratio but never changes equity's sign.")
if not 0 < r.assetTurnover <= 20:
    st.warning(f"This statement reports an implausible asset turnover ({r.assetTurnover:.3g}), so returns on assets, capital and equity "
               "after an attack are not meaningful for this company.")

# ---------------- the line ----------------
st.markdown("#### The line: how big this attack can get before the credit view changes")


def line_text(m, what):
    if m == 0:
        return f"Already {what} before any attack."
    if not np.isfinite(m):
        return f"Not reached within the sizes searched (up to {attack.max_scale():.1f}× this scenario)."
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
fig.add_trace(go.Scatter(x=[0, base_value], y=[b["p_sg"], a_["p_sg"]], mode="markers", customdata=["today", "this scenario"],
                         marker=dict(size=[9, 11], color=[BLUE, ORANGE], line=dict(color=SURFACE, width=2)),
                         hovertemplate="%{customdata}: P(SG) %{y:.3f}<extra></extra>"))
layout(fig, 380, axis + (f" ({', '.join(scaled_with)} scaled in proportion)" if scaled_with else ""), "calibrated P(SG)")
fig.update_yaxes(range=[0, 1], tickvals=[0] + eng.cuts + [1], gridcolor=GRID)
fig.update_xaxes(range=[0, x[-1]], showgrid=False)
st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
st.caption("Blue dot: today; orange dot: this scenario. Where the blue line crosses the dashed and dotted lines is the line above. Horizontal lines are the grade cut-offs; "
           "vertical lines are reported incidents at this company's scale (lost profit only, without any IT rebuild). "
           + ("P(SG) never falls as the attack grows; the steps come from the decision trees." if not feed_all else
              "With liquidity and working-capital ratios fed to the model, P(SG) can fall as the attack grows; the line above is where it first crosses."))

# ---------------- SHAP: this company (local), what the attack changed, all companies (global) ----------------
st.markdown("#### Why: the SHAP explanation")
AXIS = "raw log-odds of SG (uncalibrated model)"
sv_b, sv_a = b["shap"], a_["shap"]
logit = lambda p: np.log(p / (1 - p))
lab = input_label


def clip_note(f, vb, va):
    """What the training range (1st-99th percentile clip) does to this change, or "" when it does nothing."""
    if f not in eng.features or pd.isna(va):
        return ""
    lo_, hi_ = s["clip_lo"][f], s["clip_hi"][f]
    out_b, out_a = pd.notna(vb) and not lo_ <= vb <= hi_, not lo_ <= va <= hi_
    if out_b and out_a:
        return "outside the training range before and after: the model sees the same clipped value, so this change does not reach it"
    if out_a:
        return "reaches the edge of the training range: damage beyond it no longer moves the model"
    if out_b:
        return "comes back inside the training range: the model sees only the part of the change inside it"
    return ""


def marks(side, after):
    """Input name with the slide's marks: [filled] = missing, entered as the training median; [clipped] = cut to the training range;
    [held] = moved by the attack but not fed to the model (not monotone in it)."""
    out = []
    for c in eng.columns:
        name_ = lab(c)
        if c in eng.features:
            v, given = side["fed"][c], side["given"][c]
            if after and c in eng.unconstrained and not feed_all and not np.isclose(v, given, equal_nan=True):
                name_ += " [held]"
            elif pd.isna(v):
                name_ += " [filled]"
            elif v < s["clip_lo"][c] or v > s["clip_hi"][c]:
                name_ += " [clipped]"
        out.append(name_)
    return out


def waterfall(side, after):
    """shap's waterfall for one statement, as on the slides: preprocessed values, readable names, 12 rows."""
    one = shap.Explanation(values=side["shap"].to_numpy(), base_values=s["base"], data=np.round(side["seen"].to_numpy(), 3),
                           feature_names=marks(side, after))
    plt.close("all")
    shap.plots.waterfall(one, max_display=12, show=False)
    fig = plt.gcf()
    fig.axes[0].set_xlabel(AXIS, labelpad=22)
    st.pyplot(fig, width="content")
    plt.close(fig)


t1, t2, t3, t4 = st.tabs(["This company today", "This company after the attack", "What the attack changed", "All companies (global)"])
with t1:
    waterfall(b, after=False)
    st.caption(f"Start at E[f(X)] = {s['base']:+.2f}, the average company. Each bar is one input's push: red towards SG, blue away. "
               f"They add up to f(x) = {logit(b['raw']):+.2f} raw log-odds, which calibration turns into P(SG) {b['p_sg']:.2f}. "
               "Values are as the model sees them (missing values filled with the training median, extremes clipped).")
with t2:
    waterfall(a_, after=True)
    st.caption(f"The same company after the attack: f(x) = {logit(a_['raw']):+.2f} raw log-odds, P(SG) {a_['p_sg']:.2f}. "
               "[held] marks ratios the attack moved but the model is not monotone in, kept at today's value unless the toggle is on.")
with t3:
    size = pd.concat([sv_b.abs(), sv_a.abs()], axis=1).max(axis=1)
    changed = (sv_a - sv_b).abs() > 1e-6
    show = size[changed | size.isin(size.nlargest(12))].nlargest(14).index[::-1]

    def value_text(f):
        if f not in eng.features and f not in eng.flags:
            return "sector (not changed by the attack)"
        vb, va = b["given"][f], a_["given"][f]
        text = f"value {vb:.3f} → {va:.3f}" if pd.notna(vb) else "missing (training median used)"
        if f in eng.unconstrained and not feed_all and not np.isclose(vb, va, equal_nan=True):
            text += "<br>not fed to the model (held at today's value)"
        elif clip_note(f, vb, va):
            text += f"<br>model sees {b['seen'][f]:.3f} → {a_['seen'][f]:.3f}: {clip_note(f, vb, va)}"
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
    layout(fig2, 34 * len(show) + 90, f"SHAP value, {AXIS} (right = pushes towards SG)", "")
    fig2.update_layout(showlegend=True, legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0, font=dict(color=INK2)))
    st.plotly_chart(fig2, width="stretch", config={"displayModeBar": False})
    st.caption(f"Blue = today, orange = after the attack; hover for the values. Inputs the attack does not touch can still move a little: "
               f"in a tree model, an input's push depends on the values of the others. The attack adds {sv_a.sum() - sv_b.sum():+.2f} "
               f"raw log-odds in total, from {logit(b['raw']):+.2f} to {logit(a_['raw']):+.2f}.")
with t4:
    expl = eng.global_explanation()
    plt.close("all")
    np.random.seed(0)   # shap shuffles dot order with NumPy's global generator (as in 04)
    shap.plots.beeswarm(expl, max_display=10, group_remaining_features=False, show=False)
    ax = plt.gca()
    rows = {t.get_text(): y for t, y in zip(ax.get_yticklabels(), ax.get_yticks())}
    for side, marker, size_, face, nm in ((a_, "D", 40, "black", "after the attack"), (b, "o", 110, "none", "today")):   # ring on top
        pts = [(side["shap"][c], rows[lab(c)]) for c in eng.columns if lab(c) in rows]
        ax.scatter([p[0] for p in pts], [p[1] for p in pts], marker=marker, s=size_, facecolor=face, edgecolor="black", linewidth=1.4,
                   zorder=5, label=f"{name}, {nm}")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, fontsize=8, frameon=False)
    st.pyplot(plt.gcf(), width="content")
    plt.close("all")
    size = np.abs(expl.values).mean(0)
    rest = size[np.argsort(-size)[10:]].sum() / size.sum()
    st.caption(f"Every holdout firm-year ({len(expl.values)}), scored by the saved model, as on the slides: one dot per firm-year, "
               f"coloured by the input's value (red high, blue low). The other {len(size) - 10} inputs carry {rest:.0%} of the total push. "
               "Black markers: this company today (ring) and after the attack (diamond), from the model that scores it"
               + (" (the saved model, like the dots)." if r.holdout else
                  f" (cross-fitted fold model {s['fold'] + 1}, not the saved model behind the dots, so its pushes are comparable but not identical).")
               + " Inputs the attack does not touch can still shift a little, because in a tree model each push depends on the other inputs.")

# ---------------- table: the ratios the attack changed ----------------
with st.expander("The ratios this attack changed", expanded=True):
    rows = []
    for f in eng.features + eng.flags:
        vb, va = b["given"][f], a_["given"][f]
        if np.isclose(vb, va, equal_nan=True):
            continue
        held = f in eng.unconstrained and not feed_all
        rows.append({"input": lab(f), "before": vb, "after": va, "fed to the model": "no: shown only" if held else "yes",
                     "SHAP before": sv_b[f], "SHAP after": sv_a[f], "SHAP change": sv_a[f] - sv_b[f],
                     "note": "model not monotone in this ratio" if held else clip_note(f, vb, va)})
    if rows:
        table = pd.DataFrame(rows).sort_values(["fed to the model", "SHAP change"], ascending=[False, False])
        st.dataframe(table, hide_index=True, width="stretch",
                     column_config={c: st.column_config.NumberColumn(format="%.3f") for c in table.columns if c not in ("input", "fed to the model", "note")})
    else:
        st.write("This attack changes no ratio.")
    st.caption("Debt / equity and the equity multiplier are not moved by any attack (as in the notebook), and the negative-equity flag never "
               "switches on, so the leverage effect is understated.")

# ---------------- portfolio view ----------------
with st.expander("Where this company sits among the 169 rated-IG companies (the deck's headroom curve)"):
    ig = co[(co.SG == 0) & (co.p_sg_cf < eng.screen)]   # 04's every-company view (cross-fitted), as in the deck
    grid = np.linspace(0, 0.08, 321)
    share = [(ig.headroom_screen_cf <= v).mean() * 100 for v in grid]
    fig3 = go.Figure(go.Scatter(x=grid * 100, y=share, mode="lines", line=dict(color=BLUE, width=2),
                                hovertemplate="%{x:.2f}% of revenue: %{y:.0f}% of companies on the review list<extra></extra>"))
    h = r.headroom_screen_cf
    in_deck = name in ig.index
    if in_deck:
        fig3.add_vline(x=h * 100, line=dict(color=ORANGE, width=2))
        fig3.add_annotation(x=h * 100, y=95, xanchor="left", text=f" {name}: {h:.2%}", showarrow=False, font=dict(color=INK2))
    fig3.add_vline(x=SHARE["Marks & Spencer"] * 100, line=dict(color=GRID, width=1, dash="dash"))
    layout(fig3, 300, "incident cost charged to one year (% of revenue)", "rated-IG companies on the review list (%)")
    fig3.update_yaxes(range=[0, 100]); fig3.update_xaxes(range=[0, 8])
    st.plotly_chart(fig3, width="stretch", config={"displayModeBar": False})
    st.caption(f"The deck's view: the notebook's single-cost shock (paid, debt-funded, {TAX:.0%} tax, no insurance) for the {len(ig)} rated-IG "
               f"companies below the screen, each scored by a cross-fitted fold model; median {np.median(ig.headroom_screen_cf):.2%}. "
               "The dashed line is M&S scale; the orange line is this company in that same view. "
               + ("" if in_deck else "This company is not in that group in the deck's view. ")
               + ("For holdout companies the rest of this page uses the saved model, as the slides do, so the line above can differ "
                  "a little from this deck figure." if r.holdout else ""))

with st.expander("How each attack is charged, and what this does not show"):
    st.markdown(f"""
| Attack | What it does in the accounts | Ratios the model uses | Ratios shown only (toggle off) |
|---|---|---|---|
| Operations shutdown | Sales lost for the days chosen; profit lost = lost sales × gross margin ({gm:.0%} here, {gm_source}) | margins, returns, operating and free cash flow, debt ratio, loss flags | asset and fixed-asset turnover (sales fall) |
| Data breach, paid | One-off costs through the P&L, paid this year | margins, returns, operating and free cash flow, debt ratio, loss flags | none |
| Data breach, provisioned | Profit falls now; no cash leaves this year; a short-term liability for the cost and a deferred-tax asset for its tax shield | margins, returns, debt ratio, loss flag | current, quick and cash ratios (short-term liabilities up) |
| Billing / payments disruption | Direct costs, plus receivables still unpaid at year-end, so this year's cash flow falls | margins, returns, operating and free cash flow, debt ratio, loss flags | days sales outstanding, quick and current ratios (receivables up), asset turnover |
| Destructive malware / IT rebuild | A shutdown, plus replacing IT as capital spending (an assumption on top of the reported cost) | as a shutdown | as a shutdown |

**How it is paid** adds to every attack: with **new debt**, the cash going out (and any capital spending) is borrowed, raising the debt ratio;
with **cash on hand**, cash is spent first, lowering the cash, quick and current ratios and total assets (so asset turnover rises), and only the rest is borrowed.

- **The mapping from attack to ratios is ours, not the model's**: the model only sees ratios. Margins and cash flow are measured per unit of a
  normal year's revenue, as in the notebook, so lost profit is charged like a cost. Tax ({tax:.0%}) shields the hit only up to existing profit.
- **Cash on hand** needs current liabilities, which the ratios do not give directly: they are estimated from receivables ÷ (quick − cash ratio),
  or the sector median where that is unstable.
- **Not moved by any attack**: the implied interest rate and the operating-margin volatility, the two credit features of 03_features.
  The first is the price lenders charge today and the second is the company's history, so both stay at today's value: the attack
  adds no interest on the new debt and no spread widening, which understates its effect. The negative-equity flag never switches on.
- **Ratios the model is not monotone in** (cash, quick and current ratios, days sales outstanding, asset and fixed-asset turnover) are
  shown but held at today's value unless the toggle is on. The model reads more liquidity as riskier, likely because companies with
  market access hold less cash; feeding a cash drain could make an attack look like an improvement.
- **A paid, debt-funded data breach at 25% tax and no insurance is exactly the notebook's shock**, so the deck's numbers hold for it; the other attack types are
  explorations built on the same model and are not in the deck. Each company is scored by a model that never saw it: the saved model
  for the 123 holdout companies (as on the slides), a cross-fitted fold model for the others (as in the deck's every-company headroom).
- **One year, one company**: no recovery over time, no contagion, no management response. Not a rating; an SG call is the model's classification.
""")
