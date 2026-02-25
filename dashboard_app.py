# dashboard_app.py — ARGUS v2 (multi-class rigs)
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

    h='<div class="pnl-box"><table><thead><tr><th class="lbl">P&L Line Item</th>'
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
def _xt(mx):
    v=[0]+list(range(12,mx+1,12));t=["M0"]+[f"Y{m//12}"for m in range(12,mx+1,12)]
    return dict(tickvals=v,ticktext=t)

def _layout(title, yt, margin=None):
    return dict(
        title=dict(text=title, font=dict(size=14, color="#1e293b")),
        font=dict(family="IBM Plex Sans", size=12, color="#334155"),
        plot_bgcolor="#fff",
        paper_bgcolor="#fff",
        margin=margin if margin is not None else dict(l=60, r=20, t=50, b=50),
        xaxis=dict(showgrid=False, linecolor="#1e293b", linewidth=1.5,
                   tickfont=dict(color="#475569", size=11)),
        yaxis=dict(title=yt, gridcolor="#e2e8f0", gridwidth=.5,
                   linecolor="#1e293b", linewidth=1.5,
                   zeroline=True, zerolinecolor="#94a3b8", zerolinewidth=1,
                   tickfont=dict(color="#475569", size=11)),
        legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="left", x=0,
                    font=dict(size=11, color="#334155"),
                    bgcolor="rgba(255,255,255,.95)", bordercolor="#cbd5e1", borderwidth=1)
    )

def _fan(bd, lo, mid, hi, title, yt):
    """
    bd: DataFrame returned by _pband() with columns: month, p5/p25/p50/...
    lo/mid/hi: column names like 'p5', 'p50', 'p95'
    """
    fig = go.Figure()
    m = bd["month"]

    # Upper bound line (invisible), then fill down to lower bound
    fig.add_trace(go.Scatter(
        x=m, y=bd[hi],
        line=dict(width=0),
        showlegend=False,
        hoverinfo="skip"
    ))
    fig.add_trace(go.Scatter(
        x=m, y=bd[lo],
        fill="tonexty",
        line=dict(width=0),
        fillcolor="rgba(37,99,235,.18)",
        name=f"{lo.upper()}–{hi.upper()}",
        hoverinfo="skip"
    ))

    # Median
    fig.add_trace(go.Scatter(
        x=m, y=bd[mid],
        line=dict(color="#2563EB", width=3),
        name=mid.upper()
    ))

    fig.update_layout(**_layout(title, yt))
    fig.update_xaxes(**_xt(int(m.max())))
    return fig

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
    for t in range(nt):
        rng=np.random.default_rng(sd+t)
        res=run_single_trial(resolve_stochastic(cfg,rng),ro,ci,rng);df=to_df(res)
        for _,r in df.iterrows():
            row={"trial":t,"month":int(r["month"]),"rigs":float(r["rig_count"]),
                "revenue":float(r["total_revenue"]),"ebitda":float(r["ebitda"]),
                "profit":float(r["profit"]),"cumulative_profit":float(r["cumulative_profit"])}
            for cn in cnames:
                bc=r["by_class"].get(cn,{})if isinstance(r.get("by_class"),dict)else{}
                row[f"rigs_{cn}"]=float(bc.get("rigs",0))
                row[f"rev_{cn}"]=float(bc.get("total_revenue",0))
            rows.append(row)
        b=be_month(df)
        if b:bel.append(b)
        cum[t]=float(df["profit"].sum())
    dfm=pd.DataFrame(rows);vb=np.array(bel)
    return{"dfm":dfm,"bel":bel,"cnames":cnames,
        "s":{"n":nt,"be%":100*len(vb)/nt,"bep50":float(np.median(vb))if len(vb)else None,
             "cp5":float(np.percentile(cum,5)),"cp50":float(np.percentile(cum,50)),"cp95":float(np.percentile(cum,95))}}

# ═══════════════════════════════════════════════
# APP
# ═══════════════════════════════════════════════
st.set_page_config(page_title="ARGUS v2",layout="wide")
st.markdown(CSS,unsafe_allow_html=True)
c1,c2=st.columns([1,8])
with c1:
    if LOGO.exists():st.image(str(LOGO),width=70)
with c2:st.title("ARGUS v2 — Venture P&L Dashboard")

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
    if"Sensit"in mode:
        st.header("Sensitivity")
        sa_steps=st.number_input("Steps",5,30,11,2)
        sa_output=st.selectbox("Metric",["Cumulative Profit","Breakeven Month","Total Revenue"])
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
    st.subheader("📋 P&L Statement")
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

# ─── SENSITIVITY ───
elif"Sensit"in mode:
    det_cfg=resolve_deterministic(base)

    # Gather all params across classes
    all_params={}
    for cn in cnames:
        mkt=base["rig_classes"][cn]["market"]
        for key,val in mkt.items():
            if isinstance(val,dict)and val.get("distribution")=="triangular":
                p=val["params"];uid=f"{cn}.{key}"
                all_params[uid]={"class":cn,"key":key,"name":f"{cn.title()} {key.replace('_',' ').title()}",
                    "low":p["low"],"mode":p["mode"],"high":p["high"]}

    def run_scenario(overrides,metric):
        cfg=deepcopy(det_cfg)
        for uid,v in overrides.items():
            cn,key=uid.split(".",1);cfg["rig_classes"][cn]["market"][key]=v
        rng=np.random.default_rng(0)
        res=run_single_trial(cfg,ro,ci,rng,deterministic=True);df=to_df(res)
        if metric=="Cumulative Profit":return df["cumulative_profit"].iloc[-1]
        elif metric=="Breakeven Month":b=be_month(df);return b if b else nm+1
        elif metric=="Total Revenue":return df["total_revenue"].sum()
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

    st.subheader("🏗️ Model Structure")
    st.markdown("""
**Bottom-up P&L model** with multiple rig classes (onshore, offshore). Each class has independent:
market parameters, timelines, revenue models. Shared costs scale on total combined rigs.
""")
    st.code("""
  Month loop:
    For each rig class:
      1. Rig deployment (Poisson / accumulator)
      2. Churn (Binomial / expected value)
      3. Revenue = rigs × util × days × rate + new × install_fee
    Sum across classes → total rigs, total revenue
    4. COGS = new_total × $7K
    5. Compensation from staffing CSV
    6. G&A, IT, Depreciation scale on total rigs
    7. P&L: Gross Profit → EBITDA → EBIT → Net Income → Cumulative
""",language="text")

    st.divider();st.subheader("📅 Timeline")
    tl_rows=[["MVP Phase",f"{base['timeline']['mvp_months']} months","No rigs, no revenue"],
             ["Projection",f"{nm} months ({nm//12} years)","Total horizon"]]
    for cn in cnames:
        frm=base["rig_classes"][cn]["timeline"]["first_rig_month"]
        tl_rows.append([f"{cn.title()} First Rig",f"Month {frm}",f"First {cn} deployment"])
    st.dataframe(pd.DataFrame(tl_rows,columns=["Parameter","Value","Description"]),hide_index=True,use_container_width=True)

    st.divider();st.subheader("📊 Market Parameters & Distributions")

    for cn in cnames:
        st.markdown(f"### {cn.upper()}")
        mkt_raw=base["rig_classes"][cn]["market"]
        mkt_det=det_cfg["rig_classes"][cn]["market"]
        rev_cfg=base["rig_classes"][cn]["revenue"]

        param_info={"daily_rate":("Daily Rate","$/day/rig"),
            "utilization_rate":("Utilization","fraction"),
            "total_rigs_added":("Total Rigs Added","rigs"),
            "rigs_lost_per_year":("Rigs Lost/Year","rigs/yr")}

        for key,val in mkt_raw.items():
            if not(isinstance(val,dict)and val.get("distribution")=="triangular"):continue
            p=val["params"];nm_,unit=param_info.get(key,(key,""))
            ci_,cp_=st.columns([1,1])
            with ci_:
                st.markdown(f"**{nm_}** ({unit})")
                st.markdown(f"Low: `{p['low']}` · Mode: `{p['mode']}` · High: `{p['high']}`")
            with cp_:
                lo,mo,hi=p["low"],p["mode"],p["high"]
                c=(mo-lo)/(hi-lo);x=np.linspace(lo-.05*(hi-lo),hi+.05*(hi-lo),300)
                pdf=sp_stats.triang.pdf(x,c,loc=lo,scale=hi-lo)
                fig=go.Figure()
                fig.add_trace(go.Scatter(x=x,y=pdf,fill="tozeroy",fillcolor="rgba(37,99,235,.15)",
                    line=dict(color="#2563EB",width=2),hoverinfo="skip"))
                mpdf=sp_stats.triang.pdf(mo,c,loc=lo,scale=hi-lo)
                fig.add_trace(go.Scatter(x=[mo],y=[mpdf],mode="markers",marker=dict(size=10,color="#DC2626"),
                    showlegend=False))
                fig.update_layout(**_layout("", "", margin=dict(l=30, r=10, t=10, b=30)),
                                height=180,
                                showlegend=False
                                 )
                st.plotly_chart(fig, use_container_width=True, key=f"dist_{cn}_{key}")

        st.markdown(f"**Install fee:** ${rev_cfg['installation_fee']:,} · **Days/month:** {rev_cfg['days_per_month']}")
        st.divider()

    st.subheader("🏭 Cost Structure")
    st.markdown("**Staffing**")
    st.dataframe(pd.read_csv(base["files"]["staffing"]),hide_index=True,use_container_width=True)
    st.markdown("**Running Costs**")
    st.dataframe(pd.read_csv(base["files"]["running_costs"]),hide_index=True,use_container_width=True)

    st.subheader("📐 Unit Economics (Mode)")
    for cn in cnames:
        m=det_cfg["rig_classes"][cn]["market"];rv=det_cfg["rig_classes"][cn]["revenue"]
        mr=m["daily_rate"]*m["utilization_rate"]*rv["days_per_month"]
        vc=800+4500/24+579/10+1250/10;vm=mr-vc
        st.markdown(f"**{cn.title()}:** ${m['daily_rate']:.0f}/day × {m['utilization_rate']:.0%} × {rv['days_per_month']}d = **${mr:,.0f}/rig/mo** → margin **${vm:,.0f}/rig/mo**")

    st.subheader("⚠️ Limitations")
    st.markdown("No tax, no working capital, no debt, no NPV/IRR, no currency risk, no seasonal patterns, no capacity constraints.")

st.divider();st.caption(f"ARGUS v2 · {nm} months · {', '.join(cn.title() for cn in cnames)} · {cp}")
