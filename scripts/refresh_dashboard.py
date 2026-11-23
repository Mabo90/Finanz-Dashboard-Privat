#!/usr/bin/env python3
import json
import math
import re
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data.json"
TZ = ZoneInfo("Europe/Berlin")
FIB_RATIOS = (0.5, 0.618, 0.716, 0.882, 0.941)

def clamp(v, lo=0, hi=100):
    return max(lo, min(hi, v))

def parse_zone(value):
    nums = re.findall(r"\d+(?:\.\d+)?", str(value or "").replace(",", ""))
    if not nums:
        return None
    a = float(nums[0])
    b = float(nums[1]) if len(nums) > 1 else a
    return (min(a, b), max(a, b))

def price_text(v):
    if v is None or not math.isfinite(v):
        return "—"
    return "~$" + (f"{v:.2f}" if v < 100 else f"{v:.1f}")

def money(v):
    if v is None or not math.isfinite(v):
        return "—"
    return "$" + (f"{v:.2f}" if v < 100 else f"{v:.1f}")

def rsi_wilder(close, length=14):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/length, adjust=False, min_periods=length).mean()
    avg_loss = loss.ewm(alpha=1/length, adjust=False, min_periods=length).mean()
    rs = avg_gain / avg_loss.replace(0, pd.NA)
    return 100 - (100 / (1 + rs))

def technical_trend(price, ma50, ma200):
    if all(x is not None and math.isfinite(x) for x in (price, ma50, ma200)):
        if price > ma50 > ma200:
            return "bullish"
        if price < ma50 < ma200:
            return "bearish"
    return "neutral"

def entry_score(price, l1):
    z = parse_zone(l1)
    if not z or not price:
        return 55
    lo, hi = z
    if lo <= price <= hi:
        return 96
    if price > hi:
        dist = (price / hi - 1) * 100
        return round(clamp(96 - 6.5 * dist, 25, 96))
    dist = (lo / price - 1) * 100
    return round(clamp(82 - 3.0 * dist, 40, 82))

def momentum_score(rsi):
    if rsi is None or not math.isfinite(rsi):
        return 55
    if 50 <= rsi <= 65:
        return 88
    if 42 <= rsi < 50:
        return 76
    if 65 < rsi <= 72:
        return 72
    if 35 <= rsi < 42:
        return 62
    if 72 < rsi <= 80:
        return 52
    return 38

def risk_score(rsi):
    if rsi is None or not math.isfinite(rsi):
        return 65
    if 38 <= rsi <= 68:
        return 82
    if 30 <= rsi < 38 or 68 < rsi <= 75:
        return 68
    return 52

def live_status(item, snap, income=False):
    price = snap["price"]
    z = parse_zone(item.get("l1"))
    research = str(item.get("researchStatus") or item.get("status") or "").upper()
    if z:
        lo, hi = z
        if lo <= price <= hi:
            if income:
                return "BUY/WATCH"
            return "L1 WATCH / REVERSAL" if snap["trend"] == "bearish" else "BUY ZONE"
        if price > hi:
            dist = (price / hi - 1) * 100
            if dist <= 3:
                return "PREPARE"
            if snap["trend"] == "bullish" and dist >= 8:
                return "WATCH PULLBACK"
    if "TURNAROUND" in research:
        return "TURNAROUND WATCH"
    if "CORE" in research:
        return "CORE WATCH"
    return "INCOME WATCH" if income else "WATCH"

def score_item(item, snap, score_key):
    b = item.setdefault("breakdown", {})
    b.setdefault("quality", 75)
    b.setdefault("valuation", 70)
    b.setdefault("growth", 70)
    b["entry"] = entry_score(snap["price"], item.get("l1"))
    b["momentum"] = momentum_score(snap["rsi14"])
    b["risk"] = risk_score(snap["rsi14"])
    base = (
        0.30 * b["quality"] +
        0.25 * b["valuation"] +
        0.20 * b["growth"] +
        0.15 * b["entry"] +
        0.10 * b["momentum"]
    )
    penalty = max(0, 60 - b["risk"]) * 0.10
    item[score_key] = round(clamp(base - penalty))

def weekly_fib_setup(ticker, current_price):
    """Weekly long-retracement setup from the last confirmed swing low to the next confirmed swing high."""
    hist = ticker.history(period="3y", interval="1wk", auto_adjust=False, actions=False)
    if hist is None or hist.empty or len(hist) < 12 or "High" not in hist or "Low" not in hist:
        return None
    hist = hist.dropna(subset=["High", "Low"]).copy()
    if len(hist) < 12:
        return None

    # Ignore the still-open current weekly candle.
    now = datetime.now(TZ)
    current_monday = now.date() - timedelta(days=now.weekday())
    try:
        if hist.index[-1].date() >= current_monday:
            hist = hist.iloc[:-1]
    except Exception:
        pass
    if len(hist) < 10:
        return None

    highs = hist["High"].astype(float)
    lows = hist["Low"].astype(float)
    pivot_highs, pivot_lows = [], []
    left = right = 2
    for i in range(left, len(hist) - right):
        h = float(highs.iloc[i])
        l = float(lows.iloc[i])
        if h >= float(highs.iloc[i-left:i+right+1].max()):
            pivot_highs.append(i)
        if l <= float(lows.iloc[i-left:i+right+1].min()):
            pivot_lows.append(i)

    if pivot_highs:
        high_i = pivot_highs[-1]
    else:
        start = max(0, len(hist) - 26)
        high_i = start + int(highs.iloc[start:].to_numpy().argmax())

    prior_lows = [i for i in pivot_lows if i < high_i]
    if prior_lows:
        low_i = prior_lows[-1]
    else:
        start = max(0, high_i - 26)
        if high_i <= start:
            return None
        low_i = start + int(lows.iloc[start:high_i].to_numpy().argmin())

    swing_low = float(lows.iloc[low_i])
    swing_high = float(highs.iloc[high_i])
    if not math.isfinite(swing_low) or not math.isfinite(swing_high) or swing_high <= swing_low:
        return None

    span = swing_high - swing_low
    levels = {f"{r:g}": swing_high - r * span for r in FIB_RATIOS}
    p05 = levels["0.5"]
    p0941 = levels["0.941"]
    coverage_pct = max(0.0, (p05 - p0941) / p05 * 100)
    theoretical_lev = (100 / coverage_pct) if coverage_pct > 0 else 20.0
    # Reserve beyond 0.941: at least 2 percentage points or 25% of the 0.5→0.941 range.
    safety_buffer_pct = max(2.0, coverage_pct * 0.25)
    buffered_lev = 100 / (coverage_pct + safety_buffer_pct) if (coverage_pct + safety_buffer_pct) > 0 else 20.0
    buffered_lev = max(1.0, min(20.0, math.floor(buffered_lev * 2) / 2))

    next_ratio, next_price, next_dist = None, None, None
    candidates = []
    for r in FIB_RATIOS:
        lp = levels[f"{r:g}"]
        if current_price >= lp:
            candidates.append((r, lp, (current_price - lp) / current_price * 100))
    if candidates:
        next_ratio, next_price, next_dist = min(candidates, key=lambda x: x[2])

    if current_price < p0941:
        status = "BELOW 0.941"
        next_ratio, next_price, next_dist = 0.941, p0941, (p0941 - current_price) / current_price * 100
    elif next_ratio is None:
        status = "WAIT"
    elif next_dist <= 0.5:
        status = f"FIB {next_ratio:g} HIT"
    elif next_dist <= 2.0:
        status = f"PREPARE {next_ratio:g}"
    elif next_dist <= 5.0:
        status = f"WATCH {next_ratio:g}"
    else:
        status = f"WAIT {next_ratio:g}"

    def dt_text(i):
        try:
            return hist.index[i].strftime("%d.%m.%Y")
        except Exception:
            return str(hist.index[i])[:10]

    return {
        "anchorLow": round(swing_low, 4),
        "anchorLowDate": dt_text(low_i),
        "anchorHigh": round(swing_high, 4),
        "anchorHighDate": dt_text(high_i),
        "levels": {k: round(v, 4) for k, v in levels.items()},
        "coveragePct": round(coverage_pct, 2),
        "maxLeverageTheoretical": round(theoretical_lev, 2),
        "maxLeverageBuffered": round(buffered_lev, 2),
        "safetyBufferPct": round(safety_buffer_pct, 2),
        "status": status,
        "nextRatio": None if next_ratio is None else f"{next_ratio:g}",
        "nextPrice": None if next_price is None else round(next_price, 4),
        "distanceToNextPct": None if next_dist is None else round(next_dist, 2),
        "method": "confirmed-weekly-swing-2x2"
    }

def fetch_snapshot(symbol, need_info=False):
    ticker = yf.Ticker(symbol)
    hist = ticker.history(period="1y", interval="1d", auto_adjust=False, actions=False)
    if hist is None or hist.empty or "Close" not in hist:
        raise RuntimeError("no daily history")
    close = hist["Close"].dropna()
    if len(close) < 20:
        raise RuntimeError("insufficient history")

    price = float(close.iloc[-1])
    ma50 = float(close.tail(50).mean()) if len(close) >= 50 else float(close.mean())
    ma200 = float(close.tail(200).mean()) if len(close) >= 200 else float(close.mean())
    rsi_series = rsi_wilder(close)
    rsi = float(rsi_series.dropna().iloc[-1]) if not rsi_series.dropna().empty else None
    high52 = float(hist["High"].dropna().max()) if "High" in hist else float(close.max())
    low52 = float(hist["Low"].dropna().min()) if "Low" in hist else float(close.min())

    info = {}
    if need_info:
        try:
            info = ticker.get_info() or {}
        except Exception:
            info = {}

    return {
        "price": price,
        "ma50": ma50,
        "ma200": ma200,
        "rsi14": rsi,
        "high52": high52,
        "low52": low52,
        "trend": technical_trend(price, ma50, ma200),
        "target": info.get("targetMeanPrice"),
        "dividendYield": info.get("dividendYield"),
        "weeklyFib": weekly_fib_setup(ticker, price),
    }

def apply_market_fields(item, snap):
    item["price"] = price_text(snap["price"])
    item["technicalTrend"] = snap["trend"]
    item["weeklyFib"] = snap.get("weeklyFib")
    item["market"] = {
        "price": round(snap["price"], 4),
        "sma50": round(snap["ma50"], 4),
        "sma200": round(snap["ma200"], 4),
        "rsi14": None if snap["rsi14"] is None else round(snap["rsi14"], 2),
        "high52": round(snap["high52"], 4),
        "low52": round(snap["low52"], 4),
    }
    if snap["rsi14"] is not None:
        item["autoNote"] = (
            f"Auto: RSI14 {snap['rsi14']:.1f} · 50DMA {money(snap['ma50'])} · "
            f"200DMA {money(snap['ma200'])} · 52W High {money(snap['high52'])}"
        )
    else:
        item["autoNote"] = f"Auto: 50DMA {money(snap['ma50'])} · 200DMA {money(snap['ma200'])}"
    target = snap.get("target")
    if isinstance(target, (int, float)) and target and target > 0:
        item["expectedUpside"] = round((target / snap["price"] - 1) * 100, 1)

def update_opportunity(item, snap):
    item.setdefault("researchStatus", item.get("status"))
    item.setdefault("researchFlag", item.get("flag"))
    apply_market_fields(item, snap)
    item["status"] = live_status(item, snap, income=False)
    score_item(item, snap, "opportunity")

def update_income(item, snap, today_text):
    item.setdefault("researchStatus", item.get("status"))
    apply_market_fields(item, snap)
    item["status"] = live_status(item, snap, income=True)
    score_item(item, snap, "score")
    dy = snap.get("dividendYield")
    if isinstance(dy, (int, float)) and math.isfinite(dy) and dy > 0:
        pct = dy * 100 if dy < 1 else dy
        item["yield"] = f"{pct:.2f}%"
    if snap["rsi14"] is not None:
        item["trend"] = f"{snap['trend'].capitalize()}; RSI14 {snap['rsi14']:.1f}, Kurs vs. 50/200DMA wird automatisch aktualisiert."
    else:
        item["trend"] = snap["trend"].capitalize()
    hist = item.setdefault("history", [])
    last_status = hist[-1].get("status") if hist and isinstance(hist[-1], dict) else None
    if last_status != item["status"]:
        hist.append({
            "date": today_text,
            "status": item["status"],
            "score": item["score"],
            "note": "Automatischer Nacht-Refresh: Kurs, Trend, RSI und Entry-Score aktualisiert."
        })
        item["history"] = hist[-20:]

def update_short(item, snap):
    item.setdefault("researchStatus", item.get("status"))
    item.setdefault("researchReason", item.get("reason"))
    p = snap["price"]
    rsi = snap["rsi14"] or 50
    dist_high = max(0.0, (snap["high52"] / p - 1) * 100)
    ext50 = (p / snap["ma50"] - 1) * 100 if snap["ma50"] else 0
    risk = clamp(35 + max(0, rsi - 50) * 1.6 + max(0, 5 - dist_high) * 5 + max(0, ext50) * 1.8)
    if rsi >= 75 and dist_high <= 2:
        status = "ARMED"
    elif rsi >= 68 and dist_high <= 4:
        status = "TRIGGER WATCH"
    elif rsi >= 60:
        status = "WATCH"
    elif p < snap["ma50"]:
        status = "RETEST ONLY"
    else:
        status = "WATCH"

    item["price"] = price_text(p)
    item["rsi14"] = f"{rsi:.1f}"
    item["technicalTrend"] = snap["trend"]
    item["riskScore"] = round(risk)
    item["status"] = status
    item["extension"] = f"{dist_high:.1f}% unter 52W-High · {ext50:+.1f}% vs 50DMA"
    lo = snap["high52"] * 0.985
    hi = snap["high52"] * 1.005
    item["shortZone"] = f"{money(lo)}–{money(hi)}"
    item["invalidation"] = f">{money(snap['high52'] * 1.01)} hält"
    item["tp1"] = money(p * 0.96)
    item["tp2"] = money(p * 0.92)
    item["tp3"] = money(p * 0.86)
    item["trigger"] = "Failed retest der Short-Zone + 1H Lower High/Strukturbruch; kein Blind-Short."
    item["reason"] = f"Auto-Check: RSI14 {rsi:.1f}, Abstand 52W-High {dist_high:.1f}%, Trend {snap['trend']}."

def update_discovery(item, snap, date_text):
    item.setdefault("researchReason", item.get("reason"))
    item["snapshot"] = date_text
    item["weeklyFib"] = snap.get("weeklyFib")
    if snap["rsi14"] is not None:
        item["reason"] = (
            f"Auto-Check: {price_text(snap['price'])}, RSI14 {snap['rsi14']:.1f}, "
            f"Trend {snap['trend']}. Research: {item.get('researchReason','')}"
        )

def main():
    data = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    meta = data.setdefault("meta", {})
    meta.setdefault("researchUpdated", meta.get("updated"))
    now = datetime.now(TZ)
    today_text = now.strftime("%d.%m.%Y %H:%M")

    need_info = {x["ticker"] for x in data.get("opportunities", []) + data.get("income", []) if x.get("ticker")}
    symbols = []
    for group in ("opportunities", "income", "shortWatch", "discovery"):
        for item in data.get(group, []):
            s = item.get("ticker")
            if s and s not in symbols:
                symbols.append(s)

    snapshots = {}
    failures = {}
    for symbol in symbols:
        try:
            snapshots[symbol] = fetch_snapshot(symbol, need_info=symbol in need_info)
        except Exception as exc:
            failures[symbol] = str(exc)
        time.sleep(0.15)

    minimum = max(1, math.ceil(len(symbols) * 0.70))
    if len(snapshots) < minimum:
        raise RuntimeError(f"Refresh aborted: only {len(snapshots)}/{len(symbols)} symbols updated. Failures: {failures}")

    for item in data.get("opportunities", []):
        snap = snapshots.get(item.get("ticker"))
        if snap:
            update_opportunity(item, snap)

    for item in data.get("income", []):
        snap = snapshots.get(item.get("ticker"))
        if snap:
            update_income(item, snap, today_text)

    for item in data.get("shortWatch", []):
        snap = snapshots.get(item.get("ticker"))
        if snap:
            update_short(item, snap)

    date_text = now.strftime("%d %b %Y")
    for item in data.get("discovery", []):
        snap = snapshots.get(item.get("ticker"))
        if snap:
            update_discovery(item, snap, date_text)

    meta["version"] = "1.4-weekly-fib"
    meta["updated"] = now.isoformat(timespec="seconds")
    meta["refreshMode"] = "nightly-market"
    meta["symbolsUpdated"] = len(snapshots)
    meta["symbolsFailed"] = failures
    research = meta.get("researchUpdated", "unbekannt")
    meta["note"] = (
        f"Automatisches Marktupdate: {len(snapshots)}/{len(symbols)} Symbole aktualisiert. "
        f"Kurse, RSI14, 50/200DMA, Weekly-Fib-Level, technischer Status und dynamische Entry/Momentum-Scores sind aktuell. "
        f"Fundamentale These, qualitative Flags und Buy-Zonen stammen aus dem Research-Stand {research}."
    )

    tmp = DATA_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(DATA_PATH)
    print(meta["note"])

if __name__ == "__main__":
    main()
