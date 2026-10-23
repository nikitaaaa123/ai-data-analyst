"""Capture README screenshots + a short demo video with Playwright (offline demo mode).

    streamlit run app.py --server.port 8601 &
    python scripts/capture_demo.py http://localhost:8601
"""
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8601"
OUT = Path(__file__).resolve().parent.parent / "docs"
(OUT / "screenshots").mkdir(exist_ok=True, parents=True)
SHOT = OUT / "screenshots"


def bottom(page):
    page.locator('[data-testid="stMain"]').evaluate("e => e.scrollTo(0, e.scrollHeight)")
    page.wait_for_timeout(800)


def ask(page, q, wait=5):
    box = page.get_by_placeholder("Ask about your data…")
    box.fill(q)
    box.press("Enter")
    page.wait_for_timeout(wait * 1000)


with sync_playwright() as p:
    b = p.chromium.launch()
    ctx = b.new_context(viewport={"width": 1440, "height": 900}, record_video_dir=str(OUT / "_video"),
                        record_video_size={"width": 1440, "height": 900})
    page = ctx.new_page()
    page.goto(URL)
    page.wait_for_timeout(3000)
    page.screenshot(path=SHOT / "01_home.png")
    page.get_by_role("button", name="Load sample data (sales + customers)").click()
    page.wait_for_timeout(2500)
    page.screenshot(path=SHOT / "02_loaded.png")
    ask(page, "Which region generated the highest revenue?")
    page.screenshot(path=SHOT / "03_ranking.png")
    ask(page, "Show monthly sales trends.", 6)
    page.locator(".js-plotly-plot").last.scroll_into_view_if_needed()
    page.wait_for_timeout(800)
    page.screenshot(path=SHOT / "04_trend_chart.png")
    ask(page, "Which products are underperforming?", 6)
    bottom(page)
    page.get_by_text("Reasoning trail: queries & agent steps").last.click()
    bottom(page)
    page.screenshot(path=SHOT / "05_reasoning_sql.png")
    ask(page, "Detect anomalies in the dataset.", 6)
    bottom(page)
    page.screenshot(path=SHOT / "06_anomalies.png")
    page.get_by_role("tab", name="🗂 Data explorer").click()
    page.wait_for_timeout(1500)
    page.get_by_role("button", name="🩺 Run data-quality check").click()
    page.wait_for_timeout(2000)
    page.screenshot(path=SHOT / "07_data_quality.png")
    ctx.close()
    b.close()

vid = next((OUT / "_video").glob("*.webm"))
vid.rename(OUT / "demo.webm")
(OUT / "_video").rmdir()
