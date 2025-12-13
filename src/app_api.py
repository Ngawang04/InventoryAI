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
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
except Exception:
    client = None


# =========================
# Flask App
# =========================
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

print("=== InventoryAI Backend Running (Prophet Model) ===")


# =========================
# Robust CSV Parsing
# =========================
def sniff_delimiter(first_line: str) -> str:
    try:
        return csv.Sniffer().sniff(first_line).delimiter
    except:
        return ","


def find_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = {c.lower().strip(): c for c in df.columns}

    for cand in candidates:
        if cand.lower().strip() in cols:
            return cols[cand.lower().strip()]

    # fuzzy match
    for c in df.columns:
        lc = c.lower().strip()
        for cand in candidates:
            if cand.lower().strip() in lc:
                return c
    return None


def parse_pos_csv(csv_text: str):
    lines = [ln for ln in csv_text.splitlines() if ln.strip()]
    if len(lines) < 2:
        raise ValueError("CSV must contain header + at least 1 row.")

    delim = sniff_delimiter(lines[0])
    df = pd.read_csv(io.StringIO("\n".join(lines)), delimiter=delim)

    df.columns = [c.strip() for c in df.columns]

    col_dt  = find_column(df, ["datetime", "timestamp", "sold_at", "created_at"])
    col_date = find_column(df, ["date", "day", "ds", "sale_date", "order_date"])
    col_item = find_column(df, ["item", "product", "product_name", "item_name", "description"])
    col_qty  = find_column(df, ["qty", "quantity", "units", "units_sold"])
    col_rev  = find_column(df, ["revenue", "sales", "amount", "total"])
    col_price = find_column(df, ["unit_price", "price"])
    col_cat = find_column(df, ["category", "department"])

    # date handling
    if col_dt:
        df["ds"] = pd.to_datetime(df[col_dt]).dt.date
    elif col_date:
        df["ds"] = pd.to_datetime(df[col_date]).dt.date
    else:
        raise ValueError("No Date column found.")

    if not col_item:
        raise ValueError("No Item column found.")
    df["item"] = df[col_item].astype(str).str.strip()

    df["category"] = df[col_cat] if col_cat else "General"

    # derive units
    if col_qty:
        df["y"] = pd.to_numeric(df[col_qty], errors="coerce").fillna(0)
    else:
        if not (col_rev and col_price):
            raise ValueError("Missing Quantity column OR (Revenue + Unit Price).")
        rev = pd.to_numeric(df[col_rev], errors="coerce").fillna(0)
        price = pd.to_numeric(df[col_price], errors="coerce").replace(0, np.nan)
        df["y"] = (rev / price).fillna(0)

    grouped = df.groupby(["ds", "item"], as_index=False).agg(
        y=("y", "sum"),
        category=("category", "first"),
    )

    if grouped["ds"].nunique() < 30:
        raise ValueError("Need at least 30 days of data coverage.")

    return grouped


# =========================
# Prophet Forecast
# =========================
def forecast_items(daily: pd.DataFrame, horizon: int):
    results = []

    for item, g in daily.groupby("item"):
        if g["ds"].nunique() < 30:
            continue

        df = g[["ds", "y"]].copy()
        df["ds"] = pd.to_datetime(df["ds"])

        model = Prophet(
            seasonality_mode="multiplicative",
            weekly_seasonality=True,
            daily_seasonality=False
        )

        model.fit(df)
        future = model.make_future_dataframe(periods=horizon)
        fc = model.predict(future)

        values = np.clip(fc.tail(horizon)["yhat"].to_numpy(), 0, None)
        total_units = float(np.sum(values))

        results.append({
            "item_name": item,
            "category": g["category"].iloc[-1],
            "reorder_qty": round(total_units, 1),
            "est_revenue": round(total_units * 2.5, 2)  # optional display
        })

    return sorted(results, key=lambda x: x["reorder_qty"], reverse=True)


# =========================
# Fast Trend
# =========================
def build_trend(daily):
    t = daily.groupby("ds")["y"].sum().reset_index()
    t = t.tail(60)
    return [{"date": str(row.ds), "total_units": float(row.y)} for _, row in t.iterrows()]


# =========================
# Insights
# =========================
def build_insights(daily, forecast, horizon):
    insights = {
        "stockout_risks": [],
        "slow_movers": [],
        "rising_demand": [],
        "declining_demand": [],
        "category_demand": []
    }

    base = daily.groupby("item")["y"].mean().to_dict()

    cat_totals = {}

    for r in forecast:
        item = r["item_name"]
        f_total = r["reorder_qty"]
        cat = r["category"]
        b = base.get(item, 0)

        cat_totals[cat] = cat_totals.get(cat, 0) + f_total

        # example logic
        if b > 0 and f_total / b >= 1.5:
            insights["stockout_risks"].append(r)
        if b <= 0.5 and f_total <= 4:
            insights["slow_movers"].append(r)

    insights["category_demand"] = [
        {"category": k, "forecast_units": v, "share": round(100 * v / sum(cat_totals.values()), 1)}
        for k, v in cat_totals.items()
    ]

    return insights


# =========================
# API Routes
# =========================
@app.route("/api/health")
def health():
    return jsonify({"ok": True, "model": "prophet"})


@app.route("/api/forecast", methods=["POST"])
def forecast_api():
    try:
        body = request.get_json(force=True)
        csv_text = body["csv"]
        horizon = int(body.get("horizon_days", 7))

        daily = parse_pos_csv(csv_text)
        forecast = forecast_items(daily, horizon)
        trend = build_trend(daily)
        insights = build_insights(daily, forecast, horizon)

        return jsonify({
            "forecast": forecast,
            "metrics": {
                "TOTAL_RECORDS": len(daily),
                "NUM_ITEMS": daily["item"].nunique(),
                "DAYS_COVERED": daily["ds"].nunique(),
                "HORIZON_DAYS": horizon
            },
            "trend": trend,
            "insights": insights,
            "ai_insight": "Upload data to get AI insights."
        })
    except Exception as e:
        print("FORECAST ERROR:", e)
        return jsonify({"error": str(e)}), 400


@app.route("/api/report/pdf", methods=["POST"])
def report_pdf():
    payload = request.get_json(force=True)
    forecast = payload["forecast"]
    metrics = payload["metrics"]

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, "InventoryAI Report", ln=True)

    for r in forecast[:20]:
        pdf.cell(0, 8, f"{r['item_name']} — {r['reorder_qty']} units", ln=True)

    stream = io.BytesIO(pdf.output(dest="S").encode("latin1"))
    return send_file(stream, download_name="report.pdf", mimetype="application/pdf")



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
