# src/app_api.py
import io
import csv
import os
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

from prophet import Prophet
from fpdf import FPDF

# Optional OpenAI
try:
    from openai import OpenAI
except Exception:
    OpenAI = None


# =========================
# Flask
# =========================
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

print("=== InventoryAI backend (Prophet) running ===")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
client = None
if OPENAI_API_KEY and OpenAI is not None:
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        print("OpenAI client: OK")
    except Exception as e:
        print("OpenAI client: FAILED", e)
        client = None


# =========================
# Robust POS CSV parsing
# =========================
def _sniff_delimiter(first_line: str) -> str:
    try:
        return csv.Sniffer().sniff(first_line).delimiter
    except csv.Error:
        return ","


def _find_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = list(df.columns)
    lower_map = {c.lower().strip(): c for c in cols}

    for cand in candidates:
        key = cand.lower().strip()
        if key in lower_map:
            return lower_map[key]

    # fuzzy contains
    for c in cols:
        lc = c.lower().strip()
        for cand in candidates:
            if cand.lower().strip() in lc:
                return c
    return None


def _to_number(s: pd.Series) -> pd.Series:
    return pd.to_numeric(
        s.astype(str).str.replace(r"[^0-9.\-]", "", regex=True),
        errors="coerce"
    )


def parse_pos_csv(csv_text: str) -> pd.DataFrame:
    """
    Returns daily item units:
      ds, item, y, category

    Supports:
      - Date/Datetime + Item/SKU + Quantity
      - Date/Datetime + Item/SKU + Revenue + Unit Price (quantity derived)

    Notes:
      - Aggregates to daily per item
      - Requires >= 30 distinct days total coverage in file
    """
    raw_lines = [ln for ln in (csv_text or "").splitlines() if ln.strip()]
    if len(raw_lines) < 2:
        raise ValueError("CSV must have a header and at least 1 data row.")

    delim = _sniff_delimiter(raw_lines[0])
    df = pd.read_csv(io.StringIO("\n".join(raw_lines)), delimiter=delim)
    if df.empty:
        raise ValueError("CSV appears empty or unreadable.")

    df.columns = [c.strip() for c in df.columns]

    col_dt = _find_column(df, ["datetime", "timestamp", "sold_at", "created_at"])
    col_date = _find_column(df, ["date", "day", "ds", "sale_date", "order_date", "transaction_date"])

    col_item = _find_column(df, ["item", "product", "product_name", "item_name", "description", "name"])
    col_sku = _find_column(df, ["sku", "upc", "barcode", "plu", "item_code"])

    col_qty = _find_column(df, ["qty", "quantity", "units", "units_sold", "count"])
    col_rev = _find_column(df, ["revenue", "sales", "net_sales", "amount", "total"])
    col_price = _find_column(df, ["unit_price", "price", "selling_price"])

    col_cat = _find_column(df, ["category", "department", "dept", "group"])

    # date
    if col_dt:
        dt = pd.to_datetime(df[col_dt], errors="coerce")
        df["ds"] = pd.to_datetime(dt.dt.date, errors="coerce")
    elif col_date:
        df["ds"] = pd.to_datetime(df[col_date], errors="coerce")
    else:
        raise ValueError("Could not find a Date/Datetime column (e.g., date, timestamp).")

    # item
    if col_item:
        df["item"] = df[col_item].astype(str).str.strip()
    elif col_sku:
        df["item"] = df[col_sku].astype(str).str.strip()
    else:
        raise ValueError("Could not find Item/SKU column (e.g., item, product_name, sku).")

    df = df.dropna(subset=["ds"]).copy()
    df = df[df["item"].str.len() > 0].copy()

    # category
    if col_cat:
        df["category"] = df[col_cat].astype(str).str.strip()
    else:
        df["category"] = "General"

    # units
    if col_qty:
        df["y"] = _to_number(df[col_qty]).fillna(0.0)
    else:
        if not col_rev or not col_price:
            raise ValueError("Missing Quantity/Units. Provide either Quantity OR (Revenue + Unit Price).")
        rev = _to_number(df[col_rev]).fillna(0.0)
        price = _to_number(df[col_price]).replace(0, np.nan)
        df["y"] = (rev / price).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    df["y"] = np.clip(df["y"].astype(float), 0, None)

    daily = (
        df.groupby(["ds", "item"], as_index=False)
          .agg(y=("y", "sum"), category=("category", "first"))
          .sort_values(["item", "ds"])
          .reset_index(drop=True)
    )

    if int(daily["ds"].nunique()) < 30:
        raise ValueError("Need at least 30 days of data coverage.")

    return daily


# =========================
# Prophet per-item forecast
# =========================
def prophet_forecast_units_per_item(daily: pd.DataFrame, horizon_days: int) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    horizon_days = int(horizon_days)
    if horizon_days not in (7, 14, 30):
        horizon_days = 7

    for item, g in daily.groupby("item"):
        g = g.sort_values("ds").reset_index(drop=True)
        if int(g["ds"].nunique()) < 30:
            continue

        df_item = g[["ds", "y"]].copy()
        df_item["ds"] = pd.to_datetime(df_item["ds"])
        df_item["y"] = np.clip(df_item["y"].astype(float), 0, None)

        # Prophet model for this item
        m = Prophet(
            seasonality_mode="multiplicative",
            daily_seasonality=False,
            weekly_seasonality=True,
            yearly_seasonality=False
        )
        m.fit(df_item)

        future = m.make_future_dataframe(periods=horizon_days)
        fc = m.predict(future)

        yhat_next = np.clip(fc.tail(horizon_days)["yhat"].to_numpy(), 0, None)
        total_units = float(np.sum(yhat_next))

        results.append({
            "item_name": str(item),
            "category": str(g["category"].iloc[-1]) if "category" in g.columns else "General",
            "reorder_qty": round(total_units, 1)
        })

    results.sort(key=lambda r: float(r.get("reorder_qty", 0.0)), reverse=True)
    return results


# =========================
# Trend + Insights (legit)
# =========================
def build_trend(daily: pd.DataFrame, max_days: int = 60) -> List[Dict[str, Any]]:
    trend = daily.groupby("ds", as_index=False)["y"].sum().sort_values("ds")
    if len(trend) > max_days:
        trend = trend.tail(max_days)
    return [{"date": pd.to_datetime(d).date().isoformat(), "total_units": round(float(u), 1)}
            for d, u in zip(trend["ds"], trend["y"])]


def _recent_baseline(daily: pd.DataFrame, window_days: int = 14) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for item, g in daily.groupby("item"):
        g = g.sort_values("ds")
        recent = g.tail(window_days)
        if len(recent) == 0:
            out[item] = {"avg_daily": 0.0, "std_daily": 0.0}
            continue
        out[item] = {
            "avg_daily": float(recent["y"].mean()),
            "std_daily": float(recent["y"].std(ddof=0) if len(recent) > 1 else 0.0),
        }
    return out


def _trend_slope(arr: np.ndarray) -> float:
    if arr is None or len(arr) < 5:
        return 0.0
    y = np.asarray(arr, dtype=float)
    x = np.arange(len(y), dtype=float)
    x = x - x.mean()
    denom = (x**2).sum()
    if denom == 0:
        return 0.0
    return float((x * (y - y.mean())).sum() / denom)


def build_insights(daily: pd.DataFrame, forecast_rows: List[Dict[str, Any]], horizon_days: int) -> Dict[str, Any]:
    horizon_days = int(horizon_days)
    base = _recent_baseline(daily, window_days=14)

    last14_map: Dict[str, np.ndarray] = {}
    for item, g in daily.groupby("item"):
        g = g.sort_values("ds").tail(14)
        last14_map[item] = g["y"].to_numpy(dtype=float)

    stockout_risks: List[Dict[str, Any]] = []
    slow_movers: List[Dict[str, Any]] = []
    rising_demand: List[Dict[str, Any]] = []
    declining_demand: List[Dict[str, Any]] = []

    cat_totals: Dict[str, float] = {}
    total_all = 0.0

    for r in forecast_rows:
        item = r["item_name"]
        cat = r.get("category", "General")
        f_total = float(r.get("reorder_qty", 0.0))
        f_daily = f_total / float(horizon_days) if horizon_days > 0 else 0.0

        cat_totals[cat] = cat_totals.get(cat, 0.0) + f_total
        total_all += f_total

        b = base.get(item, {"avg_daily": 0.0, "std_daily": 0.0})
        b_daily = float(b["avg_daily"])
        b_std = float(b["std_daily"])
        slope = _trend_slope(last14_map.get(item, np.array([])))

        growth = (f_daily / b_daily) if b_daily > 0 else None

        # Stockout risk: strong acceleration + volume guard
        if growth is not None and growth >= 1.5 and f_total >= 8 and b_daily >= 0.5:
            stockout_risks.append({
                "item_name": item,
                "category": cat,
                "forecast_total_units": round(f_total, 1),
                "forecast_daily_units": round(f_daily, 2),
                "baseline_daily_units": round(b_daily, 2),
                "growth_factor": round(growth, 2),
                "trend_slope": round(slope, 3)
            })

        # Rising demand
        if growth is not None and growth >= 1.25 and f_total >= 6:
            rising_demand.append({
                "item_name": item,
                "category": cat,
                "forecast_total_units": round(f_total, 1),
                "baseline_daily_units": round(b_daily, 2),
                "growth_factor": round(growth, 2)
            })

        # Declining demand
        if growth is not None and growth <= 0.8 and b_daily >= 0.8:
            declining_demand.append({
                "item_name": item,
                "category": cat,
                "forecast_total_units": round(f_total, 1),
                "baseline_daily_units": round(b_daily, 2),
                "growth_factor": round(growth, 2)
            })

        # Slow movers: consistently low velocity + low forecast
        if b_daily <= 0.5 and f_total <= max(4.0, 0.25 * horizon_days):
            slow_movers.append({
                "item_name": item,
                "category": cat,
                "forecast_total_units": round(f_total, 1),
                "baseline_daily_units": round(b_daily, 2),
                "volatility": round(b_std, 2)
            })

    stockout_risks.sort(key=lambda x: x["growth_factor"], reverse=True)
    rising_demand.sort(key=lambda x: x["growth_factor"], reverse=True)
    declining_demand.sort(key=lambda x: x["growth_factor"])
    slow_movers.sort(key=lambda x: x["forecast_total_units"])

    category_demand = []
    for cat, v in cat_totals.items():
        share = (v / total_all) if total_all > 0 else 0.0
        category_demand.append({
            "category": cat,
            "forecast_units": round(v, 1),
            "share": round(share * 100, 1)
        })
    category_demand.sort(key=lambda x: x["forecast_units"], reverse=True)

    return {
        "labels": {
            "stockout_risks": "Stockout Risk",
            "slow_movers": "Slow Movers",
            "rising_demand": "Rising Demand",
            "declining_demand": "Declining Demand",
            "category_demand": "Category Demand Share"
        },
        "descriptions": {
            "stockout_risks": "Items forecasting much higher daily demand than your recent average. Consider ordering earlier or adding buffer stock.",
            "slow_movers": "Items with low recent velocity and low forecast. Consider reducing orders or running promotions.",
            "rising_demand": "Items trending up vs your recent average (growth factor).",
            "declining_demand": "Items trending down vs your recent average (growth factor).",
            "category_demand": "Which categories will consume the most units over the selected horizon."
        },
        "stockout_risks": stockout_risks[:8],
        "slow_movers": slow_movers[:8],
        "rising_demand": rising_demand[:8],
        "declining_demand": declining_demand[:8],
        "category_demand": category_demand[:10],
    }


# =========================
# LLM (optional)
# =========================
def llm_summary(metrics: Dict[str, Any], forecast: List[Dict[str, Any]], insights: Dict[str, Any], horizon_days: int) -> str:
    # fallback (no key)
    if not client:
        top = forecast[0] if forecast else None
        if not top:
            return "Forecast complete."
        return (
            f"Forecast complete for the next {horizon_days} days. "
            f"Top reorder: {top['item_name']} ({top['reorder_qty']} units)."
        )

    top10 = forecast[:10]
    rows = "\n".join([f"- {r['item_name']}: {r['reorder_qty']} units" for r in top10])

    prompt = f"""
You are an inventory analyst for a small business owner.
Write a short, clear summary (max 6 bullet points).

Horizon: {horizon_days} days
Records: {metrics.get('TOTAL_RECORDS')}
Items: {metrics.get('NUM_ITEMS')}

Top reorder list:
{rows}

Stockout risks: {len(insights.get('stockout_risks', []))}
Slow movers: {len(insights.get('slow_movers', []))}

Rules:
- Be practical, not technical.
- Mention the top 1-2 items and what action to take.
- Mention stockout risks / slow movers counts if any.
"""
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Be concise, practical, and business-friendly."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.25
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return "Forecast complete. Review reorder recommendations and insights."


def llm_chat(question: str, metrics: Dict[str, Any], forecast: List[Dict[str, Any]], insights: Dict[str, Any], horizon_days: int) -> str:
    if not client:
        return "AI Analyst is not available (OPENAI_API_KEY not configured)."

    top10 = forecast[:10]
    rows = "\n".join([f"- {r['item_name']}: {r['reorder_qty']} units" for r in top10])

    prompt = f"""
User question: {question}

Context:
Horizon: {horizon_days} days
Records: {metrics.get('TOTAL_RECORDS')}
Items: {metrics.get('NUM_ITEMS')}

Top reorder list:
{rows}

Stockout risks: {len(insights.get('stockout_risks', []))}
Slow movers: {len(insights.get('slow_movers', []))}

Answer in simple terms. If you reference an item, use the exact item name.
"""
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Answer like a helpful inventory analyst. Short, clear, actionable."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )
    return resp.choices[0].message.content.strip()


# =========================
# PDF helpers (robust)
# =========================
def _break_long_tokens(text: str, max_token: int = 28) -> str:
    parts = str(text or "").split(" ")
    fixed = []
    for p in parts:
        if len(p) > max_token:
            chunks = [p[i:i+max_token] for i in range(0, len(p), max_token)]
            fixed.append(" ".join(chunks))
        else:
            fixed.append(p)
    return " ".join(fixed)


def _safe_text(text: str) -> str:
    repl = {"’": "'", "‘": "'", "“": '"', "”": '"', "—": "-", "–": "-", "…": "..."}
    out = str(text or "")
    for k, v in repl.items():
        out = out.replace(k, v)
    out = _break_long_tokens(out, max_token=28)
    return out.encode("latin-1", errors="ignore").decode("latin-1")


def _pdf_bytes(pdf: FPDF) -> bytes:
    out = pdf.output(dest="S")
    if isinstance(out, (bytes, bytearray)):
        return bytes(out)
    return out.encode("latin-1", errors="ignore")


# =========================
# API routes
# =========================
@app.route("/api/health", methods=["GET"])
def api_health():
    return jsonify({"ok": True, "service": "inventoryai-backend", "model": "prophet"})


@app.route("/api/forecast", methods=["POST"])
def api_forecast():
    try:
        body = request.get_json(force=True)
        csv_text = body.get("csv", "")
        horizon_days = int(body.get("horizon_days", 7))

        daily = parse_pos_csv(csv_text)
        forecast_rows = prophet_forecast_units_per_item(daily, horizon_days)
        if not forecast_rows:
            raise ValueError("No items had enough history (need 30+ days per item).")

        metrics = {
            "TOTAL_RECORDS": int(len(daily)),
            "NUM_ITEMS": int(daily["item"].nunique()),
            "DAYS_COVERED": int(daily["ds"].nunique()),
            "HORIZON_DAYS": int(horizon_days),
        }

        trend = build_trend(daily, max_days=60)
        insights = build_insights(daily, forecast_rows, horizon_days)
        ai_insight = llm_summary(metrics, forecast_rows, insights, horizon_days)

        return jsonify({
            "forecast": forecast_rows,
            "metrics": metrics,
            "insights": insights,
            "trend": trend,
            "ai_insight": ai_insight
        })

    except Exception as e:
        print("FORECAST ERROR:", e)
        return jsonify({"error": str(e)}), 400


@app.route("/api/chat", methods=["POST"])
def api_chat():
    try:
        body = request.get_json(force=True)
        question = str(body.get("question", "")).strip()
        forecast = body.get("forecast", []) or []
        metrics = body.get("metrics", {}) or {}
        insights = body.get("insights", {}) or {}
        horizon_days = int(body.get("horizon_days", metrics.get("HORIZON_DAYS", 7)))

        if not question:
            return jsonify({"response": "Ask a question about your forecast results."})

        answer = llm_chat(question, metrics, forecast, insights, horizon_days)
        return jsonify({"response": answer})

    except Exception as e:
        print("CHAT ERROR:", e)
        return jsonify({"error": str(e)}), 400


@app.route("/api/report/pdf", methods=["POST"])
def api_pdf():
    try:
        payload = request.get_json(force=True)
        forecast = payload.get("forecast", []) or []
        metrics = payload.get("metrics", {}) or {}
        insights = payload.get("insights", {}) or {}
        note = payload.get("insight", "") or ""

        pdf = FPDF()
        pdf.set_auto_page_break(True, 15)
        pdf.add_page()

        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 10, _safe_text("InventoryAI Reorder Report"), ln=True, align="C")
        pdf.ln(4)

        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, _safe_text("Summary"), ln=True)
        pdf.set_font("Helvetica", "", 11)
        pdf.cell(0, 6, _safe_text(f"Records processed: {metrics.get('TOTAL_RECORDS', 'N/A')}"), ln=True)
        pdf.cell(0, 6, _safe_text(f"Items detected: {metrics.get('NUM_ITEMS', 'N/A')}"), ln=True)
        pdf.cell(0, 6, _safe_text(f"Days covered: {metrics.get('DAYS_COVERED', 'N/A')}"), ln=True)
        pdf.cell(0, 6, _safe_text(f"Horizon: {metrics.get('HORIZON_DAYS', 'N/A')} days"), ln=True)
        pdf.ln(4)

        top = sorted(forecast, key=lambda x: float(x.get("reorder_qty", 0)), reverse=True)[:15]

        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, _safe_text("Top Reorder Recommendations (Units)"), ln=True)
        pdf.ln(1)

        # Table header
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(110, 7, _safe_text("Item"), border=1)
        pdf.cell(30, 7, _safe_text("Category"), border=1)
        pdf.cell(25, 7, _safe_text("Units"), border=1, align="R")
        pdf.ln()

        # Table rows (robust: wrap item names with multi_cell)
        pdf.set_font("Helvetica", "", 10)
        for r in top:
            name = _safe_text(str(r.get("item_name", "")))
            cat = _safe_text(str(r.get("category", "General")))[:18]
            units = float(r.get("reorder_qty", 0))

            row_h = 6
            x0 = pdf.get_x()
            y0 = pdf.get_y()

            # Item (wrap)
            pdf.set_xy(x0, y0)
            pdf.multi_cell(110, row_h, name, border=1)

            y1 = pdf.get_y()
            used_h = y1 - y0
            if used_h < row_h:
                used_h = row_h

            # Category
            pdf.set_xy(x0 + 110, y0)
            pdf.cell(30, used_h, cat, border=1)

            # Units
            pdf.set_xy(x0 + 140, y0)
            pdf.cell(25, used_h, f"{units:.1f}", border=1, align="R")

            # Next row
            pdf.set_xy(x0, y0 + used_h)

        # Insight sections
        def _write_list(title: str, rows: List[Dict[str, Any]], fmt: str):
            if not rows:
                return
            pdf.ln(4)
            pdf.set_font("Helvetica", "B", 12)
            pdf.cell(0, 7, _safe_text(title), ln=True)
            pdf.set_font("Helvetica", "", 10)
            for s in rows[:6]:
                line = fmt.format(**s)
                pdf.multi_cell(0, 5, _safe_text("- " + line))

        _write_list(
            "Stockout Risk (Top)",
            (insights.get("stockout_risks") or []),
            "{item_name} — {forecast_total_units} units (growth {growth_factor}x vs baseline)"
        )
        _write_list(
            "Slow Movers (Top)",
            (insights.get("slow_movers") or []),
            "{item_name} — {forecast_total_units} units (baseline {baseline_daily_units}/day)"
        )

        if str(note).strip():
            pdf.ln(3)
            pdf.set_font("Helvetica", "B", 12)
            pdf.cell(0, 7, _safe_text("AI Analyst Summary"), ln=True)
            pdf.set_font("Helvetica", "", 10)
            pdf.multi_cell(0, 5, _safe_text(note))

        stream = io.BytesIO(_pdf_bytes(pdf))
        stream.seek(0)
        return send_file(
            stream,
            mimetype="application/pdf",
            as_attachment=True,
            download_name="InventoryAI_Report.pdf"
        )

    except Exception as e:
        print("PDF ERROR:", e)
        return jsonify({"error": "PDF generation failed: " + str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True, threaded=True)
