# InventoryAI

**AI-Powered Inventory Forecasting for Small Businesses**

---

## Overview

**InventoryAI** is a lightweight, data-driven web application designed to help small businesses—such as bodegas, delis, and neighborhood retail stores—make smarter inventory decisions.

🔗 **Live App:** https://inventoryaipro.netlify.app/

By uploading POS (point-of-sale) CSV files, users instantly receive:
- Accurate **unit-based demand forecasts**
- Clear **reorder quantity recommendations**
- Actionable **inventory insights**
- Plain-English, **AI-powered explanations**

The product emphasizes **simplicity, speed, and practicality**, empowering non-technical business owners to transition from intuition-based ordering to data-driven planning.

---

## Core Features

### 📁 Flexible Data Upload
- Accepts standard POS CSV formats  
- Supports both:
  - **Date + Item + Quantity**
  - **Revenue + Unit Price** (auto-derives quantity)
- Automatic validation and cleaning

### 📈 Demand Forecasting (Prophet)
- Forecasts per-item unit demand  
- Captures seasonality and trends  
- Selectable horizons: **7, 14, or 30 days**

### 📦 Reorder Recommendations
- Predicts required reorder quantities  
- Ranks items by future demand priority  

### 🤖 AI Analyst (GPT-4o mini)
- Generates easy-to-understand summaries  
- Explains demand spikes, drops, and risks  
- Answers natural-language questions  

### 🔍 Inventory Insights
- Stockout-risk detection  
- Slow-mover identification  
- Rising/declining demand flags  
- Category-level demand breakdown  

### 📊 Trend Visualization
- Recent demand trend line chart  
- Category-share summaries  

### 📄 PDF Report Export
- Clean, professional report including:
  - Summary metrics  
  - Reorder recommendations  
  - Stockout/slow-mover flags  
  - AI summary  

### 🖥️ Clean, Responsive Dashboard
- Built with **HTML + TailwindCSS + JS**  
- Fast loading and mobile-friendly  

---

## Technology Stack

### Frontend
- HTML  
- TailwindCSS  
- JavaScript  
- Chart.js  
- Netlify (deployment)

### Backend
- Python + Flask  
- Prophet  
- Pandas / NumPy  
- FPDF  
- OpenAI GPT-4o mini  
- Render (deployment)

---

## Forecasting Approach

Each product is modeled as **its own time series**. Prophet detects:
- Weekly patterns  
- Recent demand trends  
- Longer-term behavior  

Reorder recommendation:
> **Total units predicted over the next 7, 14, or 30 days**

This mirrors real small-business restocking cycles.

---

## Business Value

InventoryAI helps owners:
- Prevent **stockouts**  
- Avoid **over-ordering**  
- Reduce cash tied up in inventory  
- Save hours spent on manual analysis  
- Feel confident with clear AI explanations  

---

## Project Tracking

We use a Notion board for progress tracking:

👉 **InventoryAI Project Tracker (Notion)**  
[https://www.notion.so/27d4d5b4186f8013aefbfa84767d86d4](https://www.notion.so/2c8164d716e08095a9c6efb0b32e9477?v=e1897bd4722c425a8204c7b3087035c1&source=copy_link)

---

## How to Run the Project (Local)

### 1. Set environment variables

export OPENAI_API_KEY=your_openai_api_key_here

###  2. Start the backend
python3 src/app_api.py


### Backend runs at:

http://127.0.0.1:5000

### 3. Start the frontend
python3 -m http.server 5500


Open the app:

http://localhost:5500


Upload a CSV → choose forecast horizon → view insights, AI explanations, and download reports.

#### Authors & Acknowledgements

 **Authors**
- Ngawang Choega
- Dhruv Mane
- Divij Acharya

**Acknowledgements**
- Prof. Darsh Joshi
- Prophet, OpenAI, Flask, Render, Netlify, and the open-source community

For questions or issues, open a GitHub issue or reach out to a project author.


