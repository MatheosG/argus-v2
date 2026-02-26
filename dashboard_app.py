# dashboard_app.py — ARGUS (multi-class rigs)
# Run: python -m streamlit run dashboard_app.py
import numpy as np, pandas as pd, streamlit as st, plotly.graph_objects as go
from pathlib import Path; from copy import deepcopy
from config import load_config, resolve_stochastic, resolve_deterministic
from staffing import load_staffing; from running_costs import load_running_costs
from model import run_single_trial

LOGO = Path(__file__).parent / "logo.png"

# ═══ CSS ═══
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');
html,body,[class*="css"]{font-family:'IBM Plex Sans',sans-serif!important}
h1,h2,h3{font-family:'IBM Plex Sans',sans-serif!important;font-weight:700!important}
[data-testid="stMetricValue"]{font-family:'IBM Plex Mono',monospace!important;font-size:22px!important;font-weight:600!important}
[data-testid="stMetricLabel"]{font-size:12px!important;font-weight:600!important;text-transform:uppercase!important;letter-spacing:.5px!important}
.pnl-box{overflow-x:auto;max-height:720px;overflow-y:auto;border-radius:8px;border:1px solid #64748b}
.pnl-box table{border-collapse:collapse;white-space:nowrap;width:max-content;min-width:100%;font-family:'IBM Plex Mono',monospace;font-size:11px}
.pnl-box thead th{background:#1e293b;color:#e2e8f0;font-weight:600;text-align:center;padding:8px 12px;border:1px solid #334155;position:sticky;top:0;z-index:10}
.pnl-box thead th.lbl{text-align:left;position:sticky;left:0;z-index:20;min-width:220px;background:#1e293b}
.pnl-box thead th.yr{background:#1d4ed8;color:#fff}.pnl-box thead th.mvp{background:#92400e;color:#fef3c7}
.pnl-box td{text-align:right;padding:5px 12px;border:1px solid #cbd5e1;background:#fff;color:#1e293b}
.pnl-box td.lbl{text-align:left;position:sticky;left:0;z-index:5;background:#f8fafc;color:#0f172a;border-right:2px solid #94a3b8;min-width:220px}
.pnl-box tr.hdr td{background:#e2e8f0;font-weight:700;color:#0f172a;border-top:2px solid #64748b}
.pnl-box tr.hdr td.lbl{background:#e2e8f0;font-weight:700}
.pnl-box tr.sub td{font-weight:700;border-top:2px solid #94a3b8;background:#f1f5f9}
.pnl-box tr.sub td.lbl{background:#f1f5f9;font-weight:700}
.pnl-box tr.gt td{font-weight:700;border-top:3px double #0f172a;background:#e2e8f0}
.pnl-box tr.gt td.lbl{background:#e2e8f0;font-weight:700}
.pnl-box td.yr{background:#eff6ff}.pnl-box tr.hdr td.yr{background:#dbeafe}
.pnl-box tr.sub td.yr{background:#dbeafe}.pnl-box tr.gt td.yr{background:#bfdbfe}
.pnl-box .neg{color:#dc2626}.pnl-box .pos{color:#059669}.pnl-box .zr{color:#94a3b8}
</style>
"""

# ═══ Helpers ═══
def money(x):
    if x is None or (isinstance(x,float) and np.isnan(x)):return "—"
    x=float(x);s="-" if x<0 else "";a=abs(x)
    if a>=1e6:return f"{s}${a/1e6:,.2f}M"
    if a>=1e3:return f"{s}${a/1e3:,.1f}K"
    return f"{s}${a:,.0f}"

def mcell(x):
    if x is None or(isinstance(x,float)and np.isnan(x)):return"—","zr"
    x=float(x)
    if abs(x)<.5:return"—","zr"
    s="-"if x<0 else"";a=abs(x)
    if a>=1e6:t=f"{s}${a/1e6:,.2f}M"
    elif a>=1e3:t=f"{s}${a/1e3:,.1f}K"
    else:t=f"{s}${a:,.0f}"
    return t,("neg"if x<0 else"pos")

def pct(x):
    if x is None or(isinstance(x,float)and np.isnan(x)):return"—"
    return f"{100*float(x):.1f}%"

# ═══ Data ═══
def to_df(results):
    rows=[]
    for r in results:
        d={k:(float(v)if isinstance(v,(int,float,np.integer,np.floating))else v)for k,v in r.items()}
        d["month"]=int(r["month"]);rows.append(d)
    df=pd.DataFrame(rows);df["year"]=((df["month"]-1)//12)+1;return df

def annual(df):
    return df.groupby("year",as_index=False).agg(
        rigs=("rig_count","last"),hc=("total_headcount","last"),
        svc=("service_revenue","sum"),inst=("installation_revenue","sum"),
        rev=("total_revenue","sum"),cogs=("total_cogs","sum"),gp=("gross_profit","sum"),
        comp=("total_compensation","sum"),dep=("total_depreciation","sum"),
        ga=("total_ga","sum"),it=("total_it","sum"),
        ebitda=("ebitda","sum"),ebit=("ebit","sum"),profit=("profit","sum"))

def be_month(df):
    h=df[df["cumulative_profit"]>0]
    return float(h["month"].iloc[0])if not h.empty else None

def get_class_names(config):
    return list(config.get("rig_classes",{}).keys())

# ═══ P&L Table ═══
def pnl_html(df, class_names):
    mvp_end=int(df[df["phase"]=="mvp"]["month"].max())if"mvp"in df["phase"].values else 0

    # Build line items dynamically based on rig classes
    items=[("OPERATING METRICS",None,"hdr")]
    items.append(("  Total Rigs","rig_count",""))
    for cn in class_names:
        items.append((f"    {cn.title()} Rigs",lambda r,c=cn:r["by_class"].get(c,{}).get("rigs",0),""))
    items.append(("  New Rigs (Total)","new_rigs",""))
    items.append(("  Rigs Lost (Total)","lost_rigs",""))
    items.append(("  Headcount","total_headcount",""))

    items.append(("REVENUE",None,"hdr"))
    for cn in class_names:
        items.append((f"    {cn.title()} Service Rev",lambda r,c=cn:r["by_class"].get(c,{}).get("service_revenue",0),""))
        items.append((f"    {cn.title()} Install Rev",lambda r,c=cn:r["by_class"].get(c,{}).get("installation_revenue",0),""))
    items.append(("  Total Revenue","total_revenue","sub"))

    items.append(("COST OF GOODS SOLD",None,"hdr"))
    items.append(("  Installation COGS","total_cogs",""))
    items.append(("  Gross Profit","gross_profit","sub"))

    items.append(("OPERATING EXPENSES",None,"hdr"))
    items.append(("  Compensation","total_compensation",""))
    items.append(("  Depreciation","total_depreciation",""))
    items.append(("  G&A (Internet + Cloud)","total_ga",""))
    items.append(("  IT Services","total_it",""))
    items.append(("  Total OpEx",lambda r:r["total_compensation"]+r["total_depreciation"]+r["total_ga"]+r["total_it"],"sub"))

    items.append(("PROFITABILITY",None,"hdr"))
    items.append(("  EBITDA","ebitda",""))
    items.append(("  EBIT","ebit",""))
    items.append(("  Net Income","profit","sub"))
    items.append(("  Cumulative Profit","cumulative_profit","gt"))

    cnts={"rig_count","new_rigs","lost_rigs","total_headcount"}

    cols=[{"t":"m","l":f"M{int(r['month'])}","d":r,"mo":int(r["month"])}for _,r in df.iterrows()]
    ann=annual(df)
    for _,a in ann.iterrows():
        yr=int(a["year"])
        if yr<=0:continue
        ydf=df[df["year"]==yr]
        # Build annual class data
        acls={}
        for cn in class_names:
            acls[cn]={
                "rigs":ydf.apply(lambda r:r["by_class"].get(cn,{}).get("rigs",0),axis=1).iloc[-1],
                "service_revenue":ydf.apply(lambda r:r["by_class"].get(cn,{}).get("service_revenue",0),axis=1).sum(),
                "installation_revenue":ydf.apply(lambda r:r["by_class"].get(cn,{}).get("installation_revenue",0),axis=1).sum(),
            }
        d=pd.Series({"rig_count":a["rigs"],"new_rigs":ydf["new_rigs"].sum(),"lost_rigs":ydf["lost_rigs"].sum(),
            "total_headcount":a["hc"],"service_revenue":a["svc"],"installation_revenue":a["inst"],
            "total_revenue":a["rev"],"total_cogs":a["cogs"],"gross_profit":a["gp"],
            "total_compensation":a["comp"],"total_depreciation":a["dep"],"total_ga":a["ga"],
            "total_it":a["it"],"ebitda":a["ebitda"],"ebit":a["ebit"],"profit":a["profit"],
            "cumulative_profit":ydf["cumulative_profit"].iloc[-1],"by_class":acls})
        cols.append({"t":"a","l":f"Year {yr}","d":d,"mo":None})

    h='<div class="pnl-box"><table><thead><tr><th class="lbl">P&L & TEA Line Item</th>'
    for c in cols:
        if c["t"]=="a":h+=f'<th class="yr">{c["l"]}</th>'
        elif c["mo"]<=mvp_end:h+=f'<th class="mvp">{c["l"]}</th>'
        else:h+=f'<th>{c["l"]}</th>'
    h+='</tr></thead><tbody>'

    for name,key,rt in items:
        h+=f'<tr class="{rt}"><td class="lbl">{name}</td>'
        if key is None:
            for c in cols:h+=f'<td class="{"yr"if c["t"]=="a"else""}"></td>'
        else:
            for c in cols:
                d=c["d"]
                v=key(d)if callable(key)else(d.get(key,0)if isinstance(d,dict)else(d[key]if key in d.index else 0))
                yc=" yr"if c["t"]=="a"else""
                is_cnt=(isinstance(key,str)and key in cnts)or(callable(key)and"rigs"in name.lower())
                if is_cnt:
                    iv=int(float(v));txt=str(iv)if iv else"—";vc="zr"if not iv else""
                else:txt,vc=mcell(v)
                h+=f'<td class="{vc}{yc}">{txt}</td>'
        h+='</tr>'
    h+='</tbody></table></div>';return h

# ═══ Charts ═══
def _layout(title,yt):
    return dict(title=dict(text=title,font=dict(size=14,color="#1e293b")),
        font=dict(family="IBM Plex Sans",size=12,color="#334155"),
        plot_bgcolor="#fff",paper_bgcolor="#fff",margin=dict(l=60,r=20,t=50,b=50),
        xaxis=dict(showgrid=False,linecolor="#1e293b",linewidth=1.5,tickfont=dict(color="#475569",size=11)),
        yaxis=dict(title=yt,gridcolor="#e2e8f0",gridwidth=.5,linecolor="#1e293b",linewidth=1.5,
            zeroline=True,zerolinecolor="#94a3b8",zerolinewidth=1,tickfont=dict(color="#475569",size=11)),
        legend=dict(orientation="h",yanchor="bottom",y=1.04,xanchor="left",x=0,
            font=dict(size=11,color="#334155"),bgcolor="rgba(255,255,255,.95)",bordercolor="#cbd5e1",borderwidth=1))

def _xt(mx):
    v=[0]+list(range(12,mx+1,12));t=["M0"]+[f"Y{m//12}"for m in range(12,mx+1,12)]
    return dict(tickvals=v,ticktext=t)

def _fan(bd,lo,mid,hi,title,yt):
    fig=go.Figure();m=bd["month"]
    fig.add_trace(go.Scatter(x=m,y=bd[hi],line=dict(width=0),showlegend=False,hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=m,y=bd[lo],fill="tonexty",line=dict(width=0),fillcolor="rgba(59,130,246,.18)",name=f"{lo.upper()}–{hi.upper()}",hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=m,y=bd[mid],line=dict(color="#2563eb",width=3),name=f"{mid.upper()} (median)"))
    fig.update_layout(**_layout(title,yt));fig.update_xaxes(**_xt(int(m.max())));return fig

def _pband(dt,col,ps=(5,25,50,75,95)):
    o=pd.DataFrame({"month":sorted(dt["month"].unique())})
    for p in ps:o[f"p{p}"]=dt.groupby("month")[col].quantile(p/100).values
    return o

CLASS_COLORS={"onshore":"#2563EB","offshore":"#059669"}

# ═══ MC ═══
@st.cache_data(show_spinner=False)
def run_mc(cp,nt,nm):
    cfg=load_config(cp);cfg["simulation"]["months"]=nm
    ro=load_staffing(cfg["files"]["staffing"]);ci=load_running_costs(cfg["files"]["running_costs"])
    sd=int(cfg["simulation"]["seed"]);rows=[];bel=[];cum=np.zeros(nt)
    cnames=get_class_names(cfg)
    disc_rates=[]  # per-trial sampled annual discount rates
    for t in range(nt):
        rng=np.random.default_rng(sd+t)
        resolved=resolve_stochastic(cfg,rng)
        # Sample discount rate directly (in case config.py doesn't resolve economics)
        econ_dr = cfg.get("economics",{}).get("discount_rate", 0.10)
        if isinstance(econ_dr, dict) and "distribution" in econ_dr:
            from config import sample_param
            dr = sample_param(econ_dr, rng)
        elif isinstance(econ_dr, dict):
            dr = econ_dr.get("params",{}).get("mode", 0.10)
        else:
            dr = float(econ_dr)
        disc_rates.append(dr)
        res=run_single_trial(resolved,ro,ci,rng);df=to_df(res)
        for _,r in df.iterrows():
            row={"trial":t,"month":int(r["month"]),"rigs":float(r["rig_count"]),
                "revenue":float(r["total_revenue"]),"ebitda":float(r["ebitda"]),
                "profit":float(r["profit"]),"cumulative_profit":float(r["cumulative_profit"]),
                "total_cogs":float(r["total_cogs"]),"total_compensation":float(r["total_compensation"]),
                "total_depreciation":float(r["total_depreciation"]),"total_ga":float(r["total_ga"]),
                "total_it":float(r["total_it"])}
            for cn in cnames:
                bc=r["by_class"].get(cn,{})if isinstance(r.get("by_class"),dict)else{}
                row[f"rigs_{cn}"]=float(bc.get("rigs",0))
                row[f"rev_{cn}"]=float(bc.get("total_revenue",0))
            rows.append(row)
        b=be_month(df)
        if b:bel.append(b)
        cum[t]=float(df["profit"].sum())
    dfm=pd.DataFrame(rows);vb=np.array(bel)
    return{"dfm":dfm,"bel":bel,"cnames":cnames,"disc_rates":np.array(disc_rates),
        "s":{"n":nt,"be%":100*len(vb)/nt,"bep50":float(np.median(vb))if len(vb)else None,
             "cp5":float(np.percentile(cum,5)),"cp50":float(np.percentile(cum,50)),"cp95":float(np.percentile(cum,95))}}


# ═══════════════════════════════════════════════
# TEA — Techno-Economic Analysis
# ═══════════════════════════════════════════════
def compute_tea(df, r_monthly):
    """Compute TEA metrics from monthly DataFrame. r_monthly = monthly discount rate."""
    months = df["month"].values.astype(float)
    cf = df["profit"].values.astype(float)
    rev = df["total_revenue"].values.astype(float)
    cogs = df["total_cogs"].values.astype(float)
    comp = df["total_compensation"].values.astype(float)
    depr = df["total_depreciation"].values.astype(float)
    ga = df["total_ga"].values.astype(float)
    it_ = df["total_it"].values.astype(float)

    disc = np.array([(1 + r_monthly) ** (-m) for m in months])
    dcf = cf * disc
    cum_dcf = np.cumsum(dcf)
    npv = float(np.sum(dcf))
    mfe = float(np.min(cum_dcf))

    # Payback
    pot = None; neg = False
    for i, v in enumerate(cum_dcf):
        if v < 0: neg = True
        if neg and v > 0: pot = float(months[i]); break

    # IRR
    irr = None
    try:
        from scipy.optimize import brentq
        if np.any(cf > 0) and np.any(cf < 0):
            def f(r):
                if r <= -1: return 1e10
                return float(np.sum(cf * np.array([(1+r)**(-m) for m in months])))
            irr_m = brentq(f, -0.05, 1.0, maxiter=200)
            irr = (1 + irr_m) ** 12 - 1
    except: pass

    return {"npv": npv, "irr": irr, "pot": pot, "pot_yr": pot/12 if pot else None,
            "mfe": mfe, "months": months, "dcf": dcf, "cum_dcf": cum_dcf,
            "cum_rev": np.cumsum(rev*disc), "cum_cogs": np.cumsum(cogs*disc),
            "cum_comp": np.cumsum(comp*disc), "cum_depr": np.cumsum(depr*disc),
            "cum_ga": np.cumsum(ga*disc), "cum_it": np.cumsum(it_*disc),
            "disc_ann": (1+r_monthly)**12-1}


def tea_html(df, r_monthly):
    """Build an HTML table for the TEA / DCF statement, same style as P&L."""
    months = df["month"].values.astype(float)
    disc = np.array([(1 + r_monthly) ** (-m) for m in months])
    ann_rate = (1 + r_monthly) ** 12 - 1

    # Build discounted monthly series
    tdf = pd.DataFrame({
        "month": df["month"].values,
        "disc_factor": disc,
        "disc_revenue": df["total_revenue"].values * disc,
        "disc_cogs": df["total_cogs"].values * disc,
        "disc_gp": (df["total_revenue"].values - df["total_cogs"].values) * disc,
        "disc_comp": df["total_compensation"].values * disc,
        "disc_depr": df["total_depreciation"].values * disc,
        "disc_ga": df["total_ga"].values * disc,
        "disc_it": df["total_it"].values * disc,
        "disc_opex": (df["total_compensation"].values + df["total_depreciation"].values + df["total_ga"].values + df["total_it"].values) * disc,
        "dcf": df["profit"].values * disc,
    })
    tdf["cum_dcf"] = tdf["dcf"].cumsum()
    tdf["year"] = ((tdf["month"] - 1) // 12) + 1

    mvp_end = int(df[df["phase"] == "mvp"]["month"].max()) if "mvp" in df["phase"].values else 0

    # Line items
    items = [
        ("DISCOUNTED CASHFLOW", None, "hdr"),
        ("  Discount Factor", "disc_factor", "df"),
        ("DISCOUNTED REVENUE", None, "hdr"),
        ("  Revenue (PV)", "disc_revenue", ""),
        ("  COGS (PV)", "disc_cogs", ""),
        ("  Gross Profit (PV)", "disc_gp", "sub"),
        ("DISCOUNTED OPEX", None, "hdr"),
        ("  Compensation (PV)", "disc_comp", ""),
        ("  Depreciation (PV)", "disc_depr", ""),
        ("  G&A (PV)", "disc_ga", ""),
        ("  IT Services (PV)", "disc_it", ""),
        ("  Total OpEx (PV)", "disc_opex", "sub"),
        ("NET CASHFLOW", None, "hdr"),
        ("  Period DCF", "dcf", "sub"),
        ("  Cumulative DCF (NPV)", "cum_dcf", "gt"),
    ]

    # Monthly + annual columns
    cols = [{"t": "m", "l": f"M{int(r['month'])}", "d": r, "mo": int(r["month"])} for _, r in tdf.iterrows()]

    # Annual summaries
    for yr in sorted(tdf[tdf["year"] > 0]["year"].unique()):
        ydf = tdf[tdf["year"] == yr]
        ad = pd.Series({
            "disc_factor": ydf["disc_factor"].mean(),
            "disc_revenue": ydf["disc_revenue"].sum(),
            "disc_cogs": ydf["disc_cogs"].sum(),
            "disc_gp": ydf["disc_gp"].sum(),
            "disc_comp": ydf["disc_comp"].sum(),
            "disc_depr": ydf["disc_depr"].sum(),
            "disc_ga": ydf["disc_ga"].sum(),
            "disc_it": ydf["disc_it"].sum(),
            "disc_opex": ydf["disc_opex"].sum(),
            "dcf": ydf["dcf"].sum(),
            "cum_dcf": ydf["cum_dcf"].iloc[-1],
        })
        cols.append({"t": "a", "l": f"Year {int(yr)}", "d": ad, "mo": None})

    h = '<div class="pnl-box"><table><thead><tr><th class="lbl">TEA / DCF Statement</th>'
    for c in cols:
        if c["t"] == "a":
            h += f'<th class="yr">{c["l"]}</th>'
        elif c["mo"] <= mvp_end:
            h += f'<th class="mvp">{c["l"]}</th>'
        else:
            h += f'<th>{c["l"]}</th>'
    h += '</tr></thead><tbody>'

    df_keys = {"disc_factor"}  # format as decimal not money

    for name, key, rt in items:
        h += f'<tr class="{rt}"><td class="lbl">{name}</td>'
        if key is None:
            for c in cols:
                h += f'<td class="{"yr" if c["t"] == "a" else ""}"></td>'
        else:
            for c in cols:
                d = c["d"]
                v = d[key] if key in d.index else 0
                yc = " yr" if c["t"] == "a" else ""
                if key in df_keys:
                    txt = f"{float(v):.3f}" if float(v) > 0 else "—"
                    vc = ""
                else:
                    txt, vc = mcell(v)
                h += f'<td class="{vc}{yc}">{txt}</td>'
        h += '</tr>'
    h += '</tbody></table></div>'
    return h


def render_tea_det(df, r_monthly, nm):
    """TEA section for deterministic mode."""
    tea = compute_tea(df, r_monthly)
    st.subheader("💹 Techno-Economic Analysis")
    st.caption(f"Discount rate: {tea['disc_ann']:.1%}/yr")

    k1,k2,k3,k4 = st.columns(4)
    k1.metric("NPV", money(tea["npv"]))
    k2.metric("IRR", f"{tea['irr']:.1%}" if tea["irr"] else "N/A")
    k3.metric("Payback", f"{tea['pot_yr']:.1f} yr" if tea["pot_yr"] else "Never")
    k4.metric("Max Funding Exposure", money(tea["mfe"]))

    # TEA Statement Table
    st.divider()
    st.subheader("📋 TEA / DCF Statement")
    st.caption(f"All values discounted at {tea['disc_ann']:.1%}/yr · Scroll right → | Blue = annual | Yellow = MVP")
    st.markdown(tea_html(df, r_monthly), unsafe_allow_html=True)

    st.divider()
    st.subheader("📈 TEA Charts")

    la, lb = st.columns(2)
    with la:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=tea["months"],y=tea["cum_dcf"],fill="tozeroy",
            fillcolor="rgba(37,99,235,.12)",line=dict(color="#1e293b",width=2.5),name="Cum DCF"))
        mi = int(np.argmin(tea["cum_dcf"]))
        fig.add_trace(go.Scatter(x=[tea["months"][mi]],y=[tea["mfe"]],mode="markers+text",
            marker=dict(size=12,color="#DC2626",symbol="circle"),
            text=[f"MFE: {money(tea['mfe'])}"],textposition="bottom center",
            textfont=dict(size=11,color="#DC2626"),showlegend=False))
        if tea["pot"]:
            fig.add_vline(x=tea["pot"],line_dash="dash",line_color="#059669",line_width=2,
                annotation_text=f"Payback: {tea['pot_yr']:.1f}yr",annotation_position="top right",
                annotation_font_color="#059669")
        fig.add_hline(y=0,line_color="#94a3b8",line_width=1,line_dash="dot")
        fig.update_layout(**_layout("Discounted Cumulative Net Cashflow","USD"))
        fig.update_xaxes(**_xt(nm))
        st.plotly_chart(fig,use_container_width=True,key="tea_dcf_d")

    with lb:
        fig = go.Figure()
        m = tea["months"]
        fig.add_trace(go.Scatter(x=m,y=tea["cum_rev"],fill="tozeroy",
            fillcolor="rgba(37,99,235,.3)",line=dict(color="#2563EB",width=1.5),name="Gross Revenue"))
        net_after_cogs = tea["cum_rev"]-tea["cum_cogs"]
        fig.add_trace(go.Scatter(x=m,y=net_after_cogs,fill="tonexty",
            fillcolor="rgba(220,38,38,.15)",line=dict(color="#DC2626",width=1),name="After COGS"))
        fig.add_trace(go.Scatter(x=m,y=tea["cum_dcf"],fill="tonexty",
            fillcolor="rgba(249,115,22,.15)",line=dict(color="#0f172a",width=2),name="Net (after OpEx)"))
        fig.add_hline(y=0,line_color="#94a3b8",line_width=1,line_dash="dot")
        fig.update_layout(**_layout("Cashflow Breakdown (Discounted Cumulative)","USD"))
        fig.update_xaxes(**_xt(nm))
        st.plotly_chart(fig,use_container_width=True,key="tea_bk_d")

    # Annual waterfall
    st.markdown("#### Annual Discounted Cashflow Waterfall")
    disc = np.array([(1+r_monthly)**(-m) for m in tea["months"]])
    ad = pd.DataFrame({"month":tea["months"],"dcf":tea["dcf"],
        "rev":df["total_revenue"].values*disc,"cogs":df["total_cogs"].values*disc,
        "comp":df["total_compensation"].values*disc,"depr":df["total_depreciation"].values*disc,
        "ga":df["total_ga"].values*disc,"it":df["total_it"].values*disc})
    ad["year"]=((ad["month"]-1)//12)+1
    ya=ad[ad["year"]>0].groupby("year").sum(numeric_only=True)
    labs=[f"Y{int(y)}"for y in ya.index]
    fig=go.Figure()
    fig.add_trace(go.Bar(x=labs,y=ya["rev"],name="Revenue",marker_color="#2563EB"))
    fig.add_trace(go.Bar(x=labs,y=-ya["cogs"],name="COGS",marker_color="#F97316"))
    fig.add_trace(go.Bar(x=labs,y=-ya["comp"],name="Compensation",marker_color="#DC2626"))
    fig.add_trace(go.Bar(x=labs,y=-(ya["depr"]+ya["ga"]+ya["it"]),name="G&A+IT+Depr",marker_color="#8B5CF6"))
    fig.add_trace(go.Scatter(x=labs,y=ya["dcf"],mode="lines+markers",
        line=dict(color="#0f172a",width=2.5),marker=dict(size=6),name="Net DCF"))
    fig.update_layout(**_layout("Annual Discounted Cashflow","USD"),barmode="relative")
    st.plotly_chart(fig,use_container_width=True,key="tea_wf_d")


def render_tea_mc(dfm, mc_data, r_monthly_fallback, nm, band):
    """TEA section for Monte Carlo mode. Uses per-trial sampled discount rates."""
    bmap={"P5–P95":(5,95),"P10–P90":(10,90),"P25–P75":(25,75)};blo,bhi=bmap[band]
    disc_rates=mc_data.get("disc_rates",None)
    st.subheader("💹 Techno-Economic Analysis (Monte Carlo)")
    if disc_rates is not None and len(disc_rates):
        st.caption(f"Discount rate: sampled per trial (P50={np.median(disc_rates):.1%}/yr) · {mc_data['s']['n']:,} trials")
    else:
        st.caption(f"Discount rate: {((1+r_monthly_fallback)**12-1):.1%}/yr · {mc_data['s']['n']:,} trials")

    trials=sorted(dfm["trial"].unique());months_list=sorted(dfm["month"].unique())
    npvs=[];irrs=[];pots=[];mfes=[];cdcf_all=[]

    for idx,t in enumerate(trials):
        tdf=dfm[dfm["trial"]==t].sort_values("month")
        cf=tdf["profit"].values.astype(float);ms=tdf["month"].values.astype(float)
        # Use per-trial discount rate if available
        if disc_rates is not None and idx < len(disc_rates):
            r_ann=disc_rates[idx]
            r_mo=(1+r_ann)**(1/12)-1
        else:
            r_mo=r_monthly_fallback
        d=np.array([(1+r_mo)**(-m)for m in ms]);dcf=cf*d;cdcf=np.cumsum(dcf)
        cdcf_all.append(cdcf);npvs.append(float(np.sum(dcf)));mfes.append(float(np.min(cdcf)))
        pot=None;neg=False
        for i,v in enumerate(cdcf):
            if v<0:neg=True
            if neg and v>0:pot=float(ms[i]);break
        pots.append(pot)
        try:
            from scipy.optimize import brentq
            if np.any(cf>0)and np.any(cf<0):
                def f(r,_cf=cf,_ms=ms):
                    if r<=-1:return 1e10
                    return float(np.sum(_cf*np.array([(1+r)**(-m)for m in _ms])))
                irr_m=brentq(f,-0.05,1.0,maxiter=100);irrs.append((1+irr_m)**12-1)
            else:irrs.append(np.nan)
        except:irrs.append(np.nan)

    npvs=np.array(npvs);mfes=np.array(mfes)
    vp=np.array([p for p in pots if p is not None])
    vi=np.array([i for i in irrs if not np.isnan(i)])

    k1,k2,k3,k4=st.columns(4)
    k1.metric("NPV P50",money(np.median(npvs)))
    k2.metric("IRR P50",f"{np.median(vi):.1%}"if len(vi)else"N/A")
    k3.metric("Payback P50",f"{np.median(vp)/12:.1f} yr"if len(vp)else"Never")
    k4.metric("MFE P50",money(np.median(mfes)))

    # Fan chart
    mat=np.array(cdcf_all);pcts=sorted(set([5,blo,25,50,75,bhi,95]))
    fan=pd.DataFrame({"month":months_list})
    for p in pcts:fan[f"p{p}"]=np.percentile(mat,p,axis=0)
    la,lb=st.columns(2)
    with la:
        st.plotly_chart(_fan(fan,f"p{blo}","p50",f"p{bhi}","Discounted Cumulative Cashflow","USD"),
            use_container_width=True,key="tea_fan_mc")
    with lb:
        fig=go.Figure()
        fig.add_trace(go.Box(y=npvs,name="NPV",marker_color="#059669",boxpoints="outliers"))
        fig.add_trace(go.Box(y=mfes,name="MFE",marker_color="#DC2626",boxpoints="outliers"))
        fig.update_layout(**_layout("NPV & MFE Distribution","USD"))
        fig.add_hline(y=0,line_dash="dot",line_color="#94a3b8")
        st.plotly_chart(fig,use_container_width=True,key="tea_box_mc")

    la2,lb2=st.columns(2)
    with la2:
        if len(vp):
            fig=go.Figure(go.Box(y=vp/12,name="Payback (yr)",marker_color="#DC2626",boxpoints="outliers"))
            fig.update_layout(**_layout("Payback Distribution","Years"))
            st.plotly_chart(fig,use_container_width=True,key="tea_pot_mc")
        else:st.info("No trials reached payback.")
    with lb2:
        if len(vi):
            fig=go.Figure(go.Box(y=vi*100,name="IRR (%)",marker_color="#059669",boxpoints="outliers"))
            fig.update_layout(**_layout("IRR Distribution","%"))
            st.plotly_chart(fig,use_container_width=True,key="tea_irr_mc")
        else:st.info("IRR not computable.")

    st.markdown("#### TEA Percentiles")
    rows=[]
    for p in[5,25,50,75,95]:
        r={"Pctl":f"P{p}","NPV":money(np.percentile(npvs,p)),"MFE":money(np.percentile(mfes,p))}
        if len(vp):r["Payback (yr)"]=f"{np.percentile(vp,p)/12:.1f}"
        if len(vi):r["IRR"]=f"{np.percentile(vi,p):.1%}"
        rows.append(r)
    st.dataframe(pd.DataFrame(rows),hide_index=True,use_container_width=True)


# ═══════════════════════════════════════════════
# APP
# ═══════════════════════════════════════════════
st.set_page_config(page_title="ARGUS",layout="wide")
st.markdown(CSS,unsafe_allow_html=True)
c1,c2=st.columns([1,8])
with c1:
    if LOGO.exists():st.image(str(LOGO),width=70)
with c2:st.title("ARGUS — Venture P&L & TEA Dashboard")

cp="config.yaml"
try:_c=load_config(cp);_dm=int(_c["simulation"]["months"]);_dt=int(_c["simulation"]["n_trials"]);_ds=int(_c["simulation"]["seed"])
except:_dm,_dt,_ds=120,5000,42

with st.sidebar:
    if LOGO.exists():st.image(str(LOGO),width=160)
    st.header("⚙️ Config")
    cp=st.text_input("Config",value="config.yaml")
    nm=st.number_input("Months",12,240,_dm,12)
    mode=st.radio("Mode",["📊 Deterministic","🎲 Monte Carlo","📐 Sensitivity","📖 Assumptions"])
    st.divider()
    # Discount rate for TEA (shown for Determ and MC)
    if "Determ" in mode or "Monte" in mode or "Sensit" in mode:
        st.header("💹 TEA Settings")
        # Read from config, show mode as default
        _econ = _c.get("economics", {}) if '_c' in dir() else {}
        _dr_spec = _econ.get("discount_rate", 0.10)
        if isinstance(_dr_spec, dict) and "params" in _dr_spec:
            _dr_default = _dr_spec["params"].get("mode", 0.10)
        else:
            _dr_default = float(_dr_spec) if not isinstance(_dr_spec, dict) else 0.10
        disc_rate_annual = st.number_input("Discount Rate (%/yr)", 0.0, 50.0, _dr_default * 100, 1.0) / 100
        disc_rate_monthly = (1 + disc_rate_annual) ** (1/12) - 1
        st.caption(f"From config: Tri({_dr_spec['params']['low']:.0%}, {_dr_spec['params']['mode']:.0%}, {_dr_spec['params']['high']:.0%})" if isinstance(_dr_spec, dict) else f"Fixed: {_dr_default:.0%}")
    if"Sensit"in mode:
        st.header("Sensitivity")
        sa_steps=st.number_input("Steps",5,30,11,2)
        sa_output=st.selectbox("Metric",["Cumulative Profit","Breakeven Month","Total Revenue","NPV"])
    elif"Monte"in mode:
        mc_n=st.number_input("Trials",100,50000,_dt,500)
        mc_b=st.selectbox("Band",["P5–P95","P10–P90","P25–P75"])

try:
    base=load_config(cp);base["simulation"]["months"]=nm
    ro=load_staffing(base["files"]["staffing"]);ci=load_running_costs(base["files"]["running_costs"])
    cnames=get_class_names(base)
except Exception as e:st.error(f"Load error: {e}");st.stop()

# ─── DETERMINISTIC ───
if"Determ"in mode:
    rv=resolve_deterministic(base)
    df=to_df(run_single_trial(rv,ro,ci,np.random.default_rng(0),deterministic=True))
    be=be_month(df);tr=df["total_revenue"].sum();gm=df["gross_profit"].sum()/tr if tr>0 else np.nan

    # Mode values caption
    parts=[]
    for cn in cnames:
        m=rv["rig_classes"][cn]["market"]
        parts.append(f"**{cn.title()}:** rate=${m['daily_rate']:.0f}, util={m['utilization_rate']:.0%}, rigs={m['total_rigs_added']:.0f}")
    st.caption(" · ".join(parts))

    k1,k2,k3,k4,k5,k6=st.columns(6)
    k1.metric("Rigs (End)",f"{int(df.iloc[-1]['rig_count'])}")
    k2.metric(f"Revenue ({nm//12}yr)",money(tr))
    k3.metric("EBITDA",money(df["ebitda"].sum()))
    k4.metric("Net Income",money(df["profit"].sum()))
    k5.metric("Gross Margin",pct(gm))
    k6.metric("Breakeven","—"if be is None else f"Month {int(be)}")

    st.divider()
    st.subheader("📋 P&L & TEA Statement")
    st.caption("Scroll right → | Blue = annual | Yellow = MVP")
    st.markdown(pnl_html(df,cnames),unsafe_allow_html=True)

    st.divider();st.subheader("📈 Charts")
    xt=_xt(nm);la,lb=st.columns(2)

    with la:
        fig=go.Figure()
        for cn in cnames:
            rigs=[r["by_class"].get(cn,{}).get("rigs",0)if isinstance(r.get("by_class"),dict)else 0 for _,r in df.iterrows()]
            fig.add_trace(go.Scatter(x=df["month"],y=rigs,stackgroup="one",name=cn.title(),
                line=dict(width=0),fillcolor=CLASS_COLORS.get(cn,"#6366f1")))
        fig.update_layout(**_layout("Rigs by Class","Rigs"));fig.update_xaxes(**xt)
        st.plotly_chart(fig,use_container_width=True)

        fig=go.Figure()
        for cn in cnames:
            rev=[r["by_class"].get(cn,{}).get("service_revenue",0)if isinstance(r.get("by_class"),dict)else 0 for _,r in df.iterrows()]
            fig.add_trace(go.Bar(x=df["month"],y=rev,name=f"{cn.title()} Svc",marker_color=CLASS_COLORS.get(cn,"#6366f1")))
        fig.update_layout(**_layout("Service Revenue by Class","USD"),barmode="stack");fig.update_xaxes(**xt)
        st.plotly_chart(fig,use_container_width=True)

    with lb:
        fig=go.Figure()
        fig.add_trace(go.Bar(x=df["month"],y=df["total_compensation"],name="Comp",marker_color="#DC2626"))
        fig.add_trace(go.Bar(x=df["month"],y=df["total_cogs"],name="COGS",marker_color="#F97316"))
        fig.add_trace(go.Bar(x=df["month"],y=df["total_ga"]+df["total_it"]+df["total_depreciation"],name="G&A+IT+Dep",marker_color="#8B5CF6"))
        fig.update_layout(**_layout("Costs","USD"),barmode="stack");fig.update_xaxes(**xt)
        st.plotly_chart(fig,use_container_width=True)

        fig=go.Figure()
        fig.add_trace(go.Scatter(x=df["month"],y=df["cumulative_profit"],fill="tozeroy",
            fillcolor="rgba(37,99,235,.12)",line=dict(color="#1e293b",width=2.5),name="Cumulative"))
        fig.update_layout(**_layout("Cumulative Profit","USD"));fig.update_xaxes(**xt)
        st.plotly_chart(fig,use_container_width=True)

    st.subheader("🏗️ Waterfall")
    sel=st.slider("Month",0,nm,min(12,nm))
    row=df[df["month"]==sel].iloc[0]
    labs=["Revenue","COGS","Comp","Depr","G&A","IT","Net Income"]
    vals=[float(row["total_revenue"]),-float(row["total_cogs"]),-float(row["total_compensation"]),
          -float(row["total_depreciation"]),-float(row["total_ga"]),-float(row["total_it"]),0]
    fig=go.Figure(go.Waterfall(orientation="v",measure=["absolute"]+["relative"]*5+["total"],
        x=labs,y=vals,connector=dict(line=dict(color="#94a3b8",width=1,dash="dot")),
        increasing=dict(marker=dict(color="#2563EB")),decreasing=dict(marker=dict(color="#DC2626")),
        totals=dict(marker=dict(color="#0f172a")),textposition="outside",text=[money(abs(v))for v in vals]))
    fig.update_layout(**_layout(f"Waterfall — Month {sel}","USD"),showlegend=False)
    st.plotly_chart(fig,use_container_width=True)

    with st.expander("🔍 Drilldown"):
        d1,d2=st.columns(2)
        with d1:
            st.markdown("**Comp by Dept**")
            st.dataframe(pd.Series(row["comp_by_department"]).sort_values(ascending=False).apply(money).to_frame("USD"),use_container_width=True)
        with d2:
            st.markdown("**Running Costs**")
            st.dataframe(pd.Series(row["running_by_class"]).sort_values(ascending=False).apply(money).to_frame("USD"),use_container_width=True)

    st.divider()
    render_tea_det(df, disc_rate_monthly, nm)

# ─── MONTE CARLO ───
elif"Monte"in mode:
    bm={"P5–P95":(5,95),"P10–P90":(10,90),"P25–P75":(25,75)};blo,bhi=bm[mc_b]
    with st.spinner(f"Running {mc_n:,} trials..."):mc=run_mc(cp,mc_n,nm)
    s=mc["s"];dfm=mc["dfm"]

    k1,k2,k3,k4,k5,k6=st.columns(6)
    k1.metric("Trials",f"{s['n']:,}");k2.metric("Breakeven %",f"{s['be%']:.0f}%")
    k3.metric("Breakeven P50","—"if s["bep50"]is None else f"Mo {int(s['bep50'])}")
    k4.metric("Cum Profit P50",money(s["cp50"]));k5.metric("Cum Profit P5",money(s["cp5"]));k6.metric("Cum Profit P95",money(s["cp95"]))

    st.divider();st.subheader(f"📊 Fan Charts ({mc_b})")
    ps=sorted(set([5,blo,25,50,75,bhi,95]))
    ca,cb=st.columns(2)
    with ca:
        st.plotly_chart(_fan(_pband(dfm,"cumulative_profit",ps),f"p{blo}","p50",f"p{bhi}","Cumulative Profit","USD"),use_container_width=True)
        st.plotly_chart(_fan(_pband(dfm,"revenue",ps),f"p{blo}","p50",f"p{bhi}","Monthly Revenue","USD"),use_container_width=True)
    with cb:
        st.plotly_chart(_fan(_pband(dfm,"rigs",ps),f"p{blo}","p50",f"p{bhi}","Total Rigs","Rigs"),use_container_width=True)
        # Per-class rig fans
        for cn in mc["cnames"]:
            col=f"rigs_{cn}"
            if col in dfm.columns:
                st.plotly_chart(_fan(_pband(dfm,col,ps),f"p{blo}","p50",f"p{bhi}",f"{cn.title()} Rigs","Rigs"),use_container_width=True)

    st.divider();st.subheader("📋 Annual Percentiles")
    dfm["year"]=((dfm["month"]-1)//12)+1;years=[y for y in sorted(dfm["year"].unique())if y>0]
    def ptbl(met):
        rows=[]
        for p in[5,25,50,75,95]:
            r={"Pctl":f"P{p}"}
            for y in years:v=dfm[dfm["year"]==y].groupby("trial")[met].sum().values;r[f"Y{y}"]=np.percentile(v,p)if len(v)else np.nan
            rows.append(r)
        return pd.DataFrame(rows)
    t1,t2,t3=st.tabs(["Revenue","Profit","Rigs"])
    with t1:t=ptbl("revenue");[t.__setitem__(c,t[c].map(money))for c in t.columns[1:]];st.dataframe(t,hide_index=True,use_container_width=True)
    with t2:t=ptbl("profit");[t.__setitem__(c,t[c].map(money))for c in t.columns[1:]];st.dataframe(t,hide_index=True,use_container_width=True)
    with t3:
        rows=[]
        for p in[5,25,50,75,95]:
            r={"Pctl":f"P{p}"}
            for y in years:v=dfm[(dfm["year"]==y)&(dfm["month"]==y*12)].groupby("trial")["rigs"].last().values;r[f"Y{y}"]=int(np.percentile(v,p))if len(v)else"—"
            rows.append(r)
        st.dataframe(pd.DataFrame(rows),hide_index=True,use_container_width=True)

    if mc["bel"]:
        st.divider();st.subheader("⏱️ Breakeven Distribution")
        fig=go.Figure(go.Histogram(x=mc["bel"],nbinsx=min(nm,len(set(mc["bel"]))),
            marker_color="#2563EB",marker_line_color="#0f172a",marker_line_width=.5))
        fig.update_layout(**_layout("Breakeven Month","Count"));fig.update_xaxes(**_xt(nm))
        st.plotly_chart(fig,use_container_width=True)

    st.divider()
    render_tea_mc(mc["dfm"], mc, disc_rate_monthly, nm, mc_b)

# ─── SENSITIVITY ───
elif"Sensit"in mode:
    det_cfg=resolve_deterministic(base)

    # Gather all params across classes + economics
    all_params={}
    for cn in cnames:
        mkt=base["rig_classes"][cn]["market"]
        for key,val in mkt.items():
            if isinstance(val,dict)and val.get("distribution")=="triangular":
                p=val["params"];uid=f"{cn}.{key}"
                all_params[uid]={"class":cn,"key":key,"name":f"{cn.title()} {key.replace('_',' ').title()}",
                    "low":p["low"],"mode":p["mode"],"high":p["high"]}
    # Economics params
    for key,val in base.get("economics",{}).items():
        if isinstance(val,dict)and val.get("distribution")=="triangular":
            p=val["params"];uid=f"economics.{key}"
            all_params[uid]={"class":"economics","key":key,"name":f"Econ {key.replace('_',' ').title()}",
                "low":p["low"],"mode":p["mode"],"high":p["high"]}

    def run_scenario(overrides,metric):
        cfg=deepcopy(det_cfg)
        # Track if discount rate is overridden
        dr_override = None
        for uid,v in overrides.items():
            cn,key=uid.split(".",1)
            if cn=="economics":
                cfg["economics"][key]=v
                if key=="discount_rate": dr_override=v
            else:
                cfg["rig_classes"][cn]["market"][key]=v
        rng=np.random.default_rng(0)
        res=run_single_trial(cfg,ro,ci,rng,deterministic=True);df=to_df(res)
        # Use overridden or default discount rate for TEA metrics
        r_ann = dr_override if dr_override is not None else disc_rate_annual
        r_mo = (1+r_ann)**(1/12)-1
        if metric=="Cumulative Profit":return df["cumulative_profit"].iloc[-1]
        elif metric=="Breakeven Month":b=be_month(df);return b if b else nm+1
        elif metric=="Total Revenue":return df["total_revenue"].sum()
        elif metric=="NPV":
            tea=compute_tea(df,r_mo)
            return tea["npv"]
        return 0.0

    baseline=run_scenario({},sa_output)
    is_month="Month"in sa_output
    st.caption(f"**Baseline:** {f'Month {int(baseline)}'if is_month and baseline<=nm else('Never'if is_month else money(baseline))}")

    # Tornado
    st.subheader("🌪️ Tornado")
    tdata=[]
    for uid,pd_ in all_params.items():
        vl=run_scenario({uid:pd_["low"]},sa_output);vh=run_scenario({uid:pd_["high"]},sa_output)
        tdata.append({"uid":uid,"name":pd_["name"],"low":pd_["low"],"high":pd_["high"],"rl":vl,"rh":vh,"swing":abs(vh-vl)})
    tdata.sort(key=lambda x:x["swing"])

    fig=go.Figure()
    for td in tdata:
        lo=td["rl"]-baseline;hi=td["rh"]-baseline
        if lo>hi:lo,hi=hi,lo
        fig.add_trace(go.Bar(y=[td["name"]],x=[lo],orientation="h",base=[baseline],marker_color="#DC2626",
            showlegend=False,text=[money(td["rl"])],textposition="outside",
            hovertemplate=f'{td["name"]} @ {td["low"]:.2f}: {money(td["rl"])}<extra></extra>'))
        fig.add_trace(go.Bar(y=[td["name"]],x=[hi],orientation="h",base=[baseline],marker_color="#2563EB",
            showlegend=False,text=[money(td["rh"])],textposition="outside",
            hovertemplate=f'{td["name"]} @ {td["high"]:.2f}: {money(td["rh"])}<extra></extra>'))
    fig.add_vline(x=baseline,line_dash="dash",line_color="#0f172a",line_width=2,
        annotation_text=f"Base: {money(baseline)}",annotation_position="top")
    fig.update_layout(**_layout(f"Tornado — {sa_output}",sa_output),barmode="overlay",height=max(300,len(tdata)*65+100))
    st.plotly_chart(fig,use_container_width=True)

    # Swing table
    srows=[{"Param":td["name"],"Low":f'{td["low"]:.2f}',"High":f'{td["high"]:.2f}',
            f"{sa_output} @ Low":money(td["rl"]),f"{sa_output} @ High":money(td["rh"]),"Swing":money(td["swing"])}
           for td in sorted(tdata,key=lambda x:-x["swing"])]
    st.dataframe(pd.DataFrame(srows),hide_index=True,use_container_width=True)

    # Heatmap
    st.divider();st.subheader("🔥 Two-Way Heatmap")
    uids=list(all_params.keys())
    sa_x=st.selectbox("X-axis",uids,index=0)
    sa_y=st.selectbox("Y-axis",uids,index=min(1,len(uids)-1))
    if sa_x==sa_y:st.warning("Pick two different parameters.")
    else:
        px_=all_params[sa_x];py_=all_params[sa_y]
        xv=np.linspace(px_["low"],px_["high"],sa_steps);yv=np.linspace(py_["low"],py_["high"],sa_steps)
        with st.spinner(f"Running {sa_steps**2} scenarios..."):
            z=np.zeros((len(yv),len(xv)))
            for i,y_ in enumerate(yv):
                for j,x_ in enumerate(xv):
                    z[i,j]=run_scenario({sa_x:float(x_),sa_y:float(y_)},sa_output)

        xl=[f"{v:.2f}"for v in xv];yl=[f"{v:.2f}"for v in yv]
        zmin=z.min();zmax=z.max()
        if zmin<0 and zmax>0:
            am=max(abs(zmin),abs(zmax));cs=[[0,"#dc2626"],[.5,"#ffffff"],[1,"#059669"]];zmin,zmax=-am,am
        elif zmax<=0:cs=[[0,"#dc2626"],[1,"#fecaca"]]
        else:cs=[[0,"#d1fae5"],[1,"#059669"]]
        fig=go.Figure(go.Heatmap(z=z,x=xl,y=yl,colorscale=cs,zmin=zmin,zmax=zmax,
            text=[[money(z[i,j])for j in range(len(xv))]for i in range(len(yv))],
            texttemplate="%{text}",textfont=dict(size=10)))
        fig.update_layout(**_layout(f"{sa_output}",py_["name"]),height=max(500,sa_steps*45+100))
        fig.update_xaxes(title=px_["name"])
        st.plotly_chart(fig,use_container_width=True)
        if"Profit"in sa_output:
            bc=np.sum(z>0);st.caption(f"**{bc}/{z.size}** scenarios ({100*bc/z.size:.0f}%) profitable")

# ─── ASSUMPTIONS ───
elif"Assum"in mode:
    from scipy import stats as sp_stats
    det_cfg=resolve_deterministic(base)

    # ────────────────────────────────
    # 1. MODEL OVERVIEW
    # ────────────────────────────────
    st.subheader("🏗️ Model Structure")
    st.markdown(f"""
This is a **bottom-up venture P&L model** that projects monthly financials for ARGUS, a drilling rig monitoring SaaS platform.
The model supports **{len(cnames)} rig classes** ({', '.join(cn.title() for cn in cnames)}), each with independent market parameters,
timelines, and revenue models. Shared costs (compensation, cloud, IT) scale on total combined rigs.

The model operates in two modes:
- **Deterministic** — uses the most likely (mode) value for each parameter, deploys rigs at a steady rate
- **Monte Carlo** — samples from probability distributions for market parameters, uses Poisson process for rig deployment and binomial churn
""")

    st.code("""
  Month loop (for each month 1..N):
    ┌─ Phase check: MVP or Production? ─┐
    │                                     │
    │  For each rig class (onshore, offshore):
    │    1. Rig deployment                │
    │       Deterministic: accumulator    │
    │       Stochastic: Poisson(λ)        │
    │       λ = total_rigs / prod_months  │
    │    2. Churn                          │
    │       Deterministic: round(E[loss]) │
    │       Stochastic: Binomial(n, p)    │
    │    3. Revenue per class              │
    │       Service = rigs × util × days × rate
    │       Install = new_rigs × fee      │
    │                                     │
    │  Sum across classes:                 │
    │    total_rigs, total_revenue         │
    │                                     │
    │  Shared costs (on total rigs):       │
    │    4. COGS = new_total × $7,000     │
    │    5. Compensation (staffing CSV)    │
    │    6. Depreciation = rigs × $4,500/24mo
    │    7. G&A = $800/rig + $579/10rigs  │
    │    8. IT = $1,250/10rigs            │
    │                                     │
    │  P&L:                                │
    │    Gross Profit = Revenue − COGS    │
    │    EBITDA = GP − Comp − G&A − IT    │
    │    EBIT = EBITDA − Depreciation     │
    │    Net Income = EBIT (no tax)       │
    │    Cumulative += Net Income          │
    └─────────────────────────────────────┘
""", language="text")

    st.divider()

    # ────────────────────────────────
    # 2. TIMELINE
    # ────────────────────────────────
    st.subheader("📅 Timeline Assumptions")
    tl_rows = [
        ["Projection Horizon", f"{nm} months ({nm//12} years)", "Total simulation length"],
        ["MVP Phase", f"{base['timeline']['mvp_months']} months", "Development & testing — no rigs, no revenue"],
    ]
    for cn in cnames:
        frm = base["rig_classes"][cn]["timeline"]["first_rig_month"]
        tl_rows.append([f"{cn.title()} First Rig", f"Month {frm}", f"First {cn} rig goes live"])
    for cn in cnames:
        frm = base["rig_classes"][cn]["timeline"]["first_rig_month"]
        tl_rows.append([f"{cn.title()} Prod Months", f"{nm - frm + 1} months", f"Months {frm}–{nm}"])
    st.dataframe(pd.DataFrame(tl_rows, columns=["Parameter", "Value", "Description"]),
                 hide_index=True, use_container_width=True)

    st.divider()

    # ────────────────────────────────
    # 3. MARKET PARAMETERS + DISTRIBUTIONS
    # ────────────────────────────────
    st.subheader("📊 Market Parameters & Distributions")
    st.caption("Each parameter is modeled as a probability distribution. The mode (peak) is used for deterministic runs; the full distribution is sampled in Monte Carlo.")

    param_descriptions = {
        "daily_rate": {
            "name": "Daily Service Rate",
            "unit": "$/day/rig",
            "desc": "Revenue charged per rig per active day. Based on competitive analysis of drilling data service pricing.",
            "formula": "service_revenue = rigs × utilization × days_per_month × daily_rate",
        },
        "utilization_rate": {
            "name": "Utilization Rate",
            "unit": "fraction (0–1)",
            "desc": "Fraction of calendar days a deployed rig is actively drilling. Accounts for rig moves, maintenance, weather, contract gaps.",
            "formula": "active_days = utilization_rate × days_per_month",
        },
        "total_rigs_added": {
            "name": "Total Rigs Added",
            "unit": "rigs over projection",
            "desc": "Cumulative new rig deployments over the full horizon. Drives the Poisson arrival rate in Monte Carlo.",
            "formula": "avg_rigs_per_month = total_rigs_added / production_months",
        },
        "rigs_lost_per_year": {
            "name": "Rigs Lost Per Year (Churn)",
            "unit": "rigs/year",
            "desc": "Expected annual rig losses from cancellation or decommissioning. Converted to monthly per-rig churn probability.",
            "formula": "monthly_churn_prob = rigs_lost_per_year / 12 / expected_avg_fleet",
        },
    }

    chart_idx = 0  # unique key counter for plotly charts
    for cn in cnames:
        st.markdown(f"### {cn.upper()}")
        mkt_raw = base["rig_classes"][cn]["market"]
        rev_cfg = base["rig_classes"][cn]["revenue"]
        frm = base["rig_classes"][cn]["timeline"]["first_rig_month"]

        st.caption(f"First rig: Month {frm} · Install fee: ${rev_cfg['installation_fee']:,} · Days/month: {rev_cfg['days_per_month']}")

        for key, val in mkt_raw.items():
            if not (isinstance(val, dict) and val.get("distribution") == "triangular"):
                continue
            p = val["params"]
            info = param_descriptions.get(key, {"name": key, "unit": "", "desc": "", "formula": ""})

            st.markdown("---")
            col_info, col_plot = st.columns([1, 1])

            with col_info:
                st.markdown(f"#### {info['name']}")
                st.markdown(f"**Distribution:** Triangular · **Unit:** {info['unit']}")

                is_pct = "utilization" in key
                is_dollar = "daily" in key
                if is_pct:
                    fmt = lambda v: f"{v:.0%}"
                elif is_dollar:
                    fmt = lambda v: f"${v:.0f}"
                else:
                    fmt = lambda v: f"{v:.1f}"

                st.markdown(f"""
| | Value |
|---|---|
| **Low** | {fmt(p['low'])} |
| **Mode (most likely)** | {fmt(p['mode'])} |
| **High** | {fmt(p['high'])} |
""")
                st.markdown(f"**Rationale:** {info['desc']}")
                st.markdown(f"**Formula:** `{info['formula']}`")

            with col_plot:
                lo, mo, hi = p["low"], p["mode"], p["high"]
                c = (mo - lo) / (hi - lo)
                x = np.linspace(lo - 0.05 * (hi - lo), hi + 0.05 * (hi - lo), 300)
                pdf = sp_stats.triang.pdf(x, c, loc=lo, scale=hi - lo)

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=x, y=pdf, fill="tozeroy",
                    fillcolor="rgba(37,99,235,0.15)",
                    line=dict(color="#2563EB", width=2), hoverinfo="skip"))

                # Mark mode
                mpdf = sp_stats.triang.pdf(mo, c, loc=lo, scale=hi - lo)
                fig.add_trace(go.Scatter(x=[mo], y=[mpdf], mode="markers+text",
                    marker=dict(size=10, color="#DC2626"),
                    text=[f"Mode: {fmt(mo)}"], textposition="top center",
                    textfont=dict(size=11, color="#DC2626"),
                    showlegend=False))

                # Mark lo/hi
                fig.add_vline(x=lo, line_dash="dot", line_color="#64748b", line_width=1,
                    annotation_text=f"Low: {fmt(lo)}", annotation_position="bottom left",
                    annotation_font_size=10)
                fig.add_vline(x=hi, line_dash="dot", line_color="#64748b", line_width=1,
                    annotation_text=f"High: {fmt(hi)}", annotation_position="bottom right",
                    annotation_font_size=10)

                fig.update_layout(
                    height=280,
                    margin=dict(l=40, r=10, t=20, b=40),
                    showlegend=False,
                    plot_bgcolor="#fff", paper_bgcolor="#fff",
                    xaxis=dict(showgrid=False, title=info["unit"], linecolor="#1e293b"),
                    yaxis=dict(showgrid=False, title="Density", linecolor="#1e293b"),
                )
                st.plotly_chart(fig, use_container_width=True, key=f"dist_{cn}_{key}_{chart_idx}")
                chart_idx += 1

        st.divider()

    # ────────────────────────────────
    # 4. REVENUE MODEL
    # ────────────────────────────────
    st.subheader("💰 Revenue Model")
    rev_rows = []
    for cn in cnames:
        m = det_cfg["rig_classes"][cn]["market"]
        rv = det_cfg["rig_classes"][cn]["revenue"]
        monthly = m["daily_rate"] * m["utilization_rate"] * rv["days_per_month"]
        rev_rows.append([f"{cn.title()} Service Revenue",
            f"rigs × {m['utilization_rate']:.0%} × {rv['days_per_month']}d × ${m['daily_rate']:.0f}",
            f"${monthly:,.0f}/rig/month"])
        rev_rows.append([f"{cn.title()} Installation Fee",
            f"${rv['installation_fee']:,} per new rig (one-time)",
            "Charged upon deployment"])
    st.dataframe(pd.DataFrame(rev_rows, columns=["Component", "Formula", "Value at Mode"]),
                 hide_index=True, use_container_width=True)

    st.divider()

    # ────────────────────────────────
    # 5. COST STRUCTURE
    # ────────────────────────────────
    st.subheader("🏭 Cost Structure")

    st.markdown("#### Staffing (from staffing.csv)")
    comp_cfg = base["compensation"]
    st.caption(f"Benefits multiplier starts at {comp_cfg['benefits_base']:.0%}, increases {comp_cfg['benefits_quarterly_increase']:.1%} per quarter")
    staff_df = pd.read_csv(base["files"]["staffing"])
    display_staff = staff_df.copy()
    if "annual_salary" in display_staff.columns:
        display_staff["annual_salary"] = display_staff["annual_salary"].apply(lambda x: f"${x:,.0f}")
    st.dataframe(display_staff, hide_index=True, use_container_width=True)

    st.markdown("#### Running Costs (from running_costs.csv)")
    st.dataframe(pd.read_csv(base["files"]["running_costs"]), hide_index=True, use_container_width=True)

    st.markdown("#### Cost Scaling Logic")
    cost_explain = [
        ["COGS (Installation)", "$7,000", "One-time per new rig", "Tablet ($3,200) + Dock ($600) + Speedcast Install ($3,000) + Materials ($200)"],
        ["Depreciation", "$188/mo per rig", "Monthly per rig", "Edge Device ($4,500) / 24 months useful life"],
        ["Field Internet", "$800/mo per rig", "Monthly per active rig", "Speedcast managed connectivity"],
        ["Cloud Service", "$579/mo per 10 rigs", "Monthly, steps of 10 rigs", "Operations DB, UI, API hosting"],
        ["Cloud (MVP)", "$28/mo", "Fixed, MVP phase only", "Dev/test environment"],
        ["IT Services", "$1,250/mo per 10 rigs", "Monthly, steps of 10 rigs", "Software support & maintenance"],
    ]
    st.dataframe(pd.DataFrame(cost_explain, columns=["Item", "Amount", "Scaling", "Description"]),
                 hide_index=True, use_container_width=True)

    # Installation economics
    st.markdown("#### Installation Unit Economics")
    for cn in cnames:
        rv = det_cfg["rig_classes"][cn]["revenue"]
        inst_rev = rv["installation_fee"]
        inst_cogs = 7000
        inst_margin = inst_rev - inst_cogs
        c1, c2, c3 = st.columns(3)
        c1.metric(f"{cn.title()} Client Pays", f"${inst_rev:,}")
        c2.metric("COGS", f"${inst_cogs:,}")
        c3.metric("Net per Install", f"${inst_margin:,}")

    st.divider()

    # ────────────────────────────────
    # 6. UNIT ECONOMICS AT MODE
    # ────────────────────────────────
    st.subheader("📐 Unit Economics (at Mode Values)")

    for cn in cnames:
        st.markdown(f"#### {cn.title()}")
        m = det_cfg["rig_classes"][cn]["market"]
        rv = det_cfg["rig_classes"][cn]["revenue"]
        monthly_rev = m["daily_rate"] * m["utilization_rate"] * rv["days_per_month"]
        monthly_internet = 800
        monthly_depr = 4500 / 24
        monthly_cloud = 579 / 10
        monthly_it = 1250 / 10
        monthly_cost = monthly_internet + monthly_depr + monthly_cloud + monthly_it
        monthly_margin = monthly_rev - monthly_cost

        ue_data = [
            ["Service Revenue / rig / month", f"${m['daily_rate']:.0f} × {m['utilization_rate']:.0%} × {rv['days_per_month']}d", f"${monthly_rev:,.0f}"],
            ["Field Internet", "Speedcast", f"-${monthly_internet:,.0f}"],
            ["Depreciation", "$4,500 / 24 mo", f"-${monthly_depr:,.0f}"],
            ["Cloud (at scale)", "$579 / 10 rigs", f"-${monthly_cloud:,.0f}"],
            ["IT (at scale)", "$1,250 / 10 rigs", f"-${monthly_it:,.0f}"],
            ["**Variable Margin / rig / month**", "", f"**${monthly_margin:,.0f}**"],
            ["**Annual margin / rig**", "× 12", f"**${monthly_margin*12:,.0f}**"],
        ]
        st.dataframe(pd.DataFrame(ue_data, columns=["Line Item", "Calculation", "Amount"]),
                     hide_index=True, use_container_width=True)

    # Breakeven rigs
    det_rng = np.random.default_rng(0)
    det_res = run_single_trial(det_cfg, ro, ci, det_rng, deterministic=True)
    det_df = to_df(det_res)
    avg_monthly_comp = det_df["total_compensation"].mean()

    # Use blended margin (weighted by rig count)
    st.markdown("#### Breakeven Analysis")
    st.caption("How many rigs are needed to cover fixed compensation costs?")

    for cn in cnames:
        m = det_cfg["rig_classes"][cn]["market"]
        rv = det_cfg["rig_classes"][cn]["revenue"]
        mr = m["daily_rate"] * m["utilization_rate"] * rv["days_per_month"]
        vc = 800 + 4500/24 + 579/10 + 1250/10
        vm = mr - vc
        if vm > 0:
            be = avg_monthly_comp / vm
            st.metric(f"{cn.title()} Rigs to Cover Comp (alone)", f"{be:.0f} rigs",
                      delta=f"${vm:,.0f}/rig/mo margin")
        else:
            st.metric(f"{cn.title()} Rigs to Cover Comp", "N/A — negative margin")

    st.caption(f"Average monthly compensation: {money(avg_monthly_comp)}")

    st.divider()

    # ────────────────────────────────
    # 7. LIMITATIONS
    # ────────────────────────────────
    st.subheader("⚠️ Key Assumptions & Limitations")
    st.markdown("""
**Included in the model:**
- All compensation with quarterly benefits escalation
- Hardware COGS and depreciation (shared across classes)
- Managed internet connectivity (Speedcast)
- Cloud hosting and IT services
- Installation revenue and costs (per-class fees)
- Rig churn / contract loss (per-class rates)
- Multiple rig classes with independent markets

**NOT included (potential additions):**
- Income tax / corporate tax
- Working capital / accounts receivable timing
- Debt service / interest expense
- Equity dilution / fundraising rounds
- Office & camp costs
- Travel & entertainment
- Legal & professional fees
- Training costs for field technicians
- Field pick-up truck costs (currently disabled)
- Discount rate / NPV / IRR calculations
- Currency risk (all USD)
- Customer concentration risk
- Seasonal utilization patterns
- Offshore-specific costs (crew boats, platform access fees)
- Different COGS per rig class (currently shared $7K)

**Key modeling choices:**
- Revenue recognized monthly (no lag between deployment and first invoice)
- Installation fee recognized in the deployment month
- No capacity constraints on deployment rate
- Compensation scales by start month, not by rig count
- Benefits multiplier increases quarterly
- Cloud and IT costs scale in steps of 10 rigs (not linear)
- COGS and depreciation are the same for onshore and offshore rigs
""")

st.divider();st.caption(f"ARGUS · {nm} months · {', '.join(cn.title() for cn in cnames)} · {cp}")
