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
FIB_RATIOS = (0.236, 0.382, 0.5, 0.618, 0.786, 0.882, 0.941)
ENTRY_FIB_RATIOS = (0.5, 0.618, 0.786, 0.882, 0.941)

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
    """Structural macro weekly impulse: last major reset low -> subsequent macro high."""
    hist = ticker.history(period="5y", interval="1wk", auto_adjust=False, actions=False)
    if hist is None or hist.empty or len(hist) < 30 or "High" not in hist or "Low" not in hist:
        return None
    hist = hist.dropna(subset=["Open", "High", "Low", "Close"]).copy()
    if len(hist) < 30:
        return None

    now = datetime.now(TZ)
    current_monday = now.date() - timedelta(days=now.weekday())
    try:
        if hist.index[-1].date() >= current_monday:
            hist = hist.iloc[:-1]
    except Exception:
        pass
    if len(hist) < 30:
        return None

    highs = hist["High"].astype(float).to_numpy()
    lows = hist["Low"].astype(float).to_numpy()
    closes = hist["Close"].astype(float).to_numpy()

    # Major weekly pivot lows. A 4/4-week window filters out the small local swings.
    pivot_lows = []
    w = 4
    for i in range(w, len(hist) - w):
        lo = lows[i]
        if math.isfinite(lo) and lo <= min(lows[i-w:i+w+1]):
            pivot_lows.append(i)

    candidates = []
    for low_i in pivot_lows:
        # Need enough history to establish the prior cycle high and enough room for the new impulse.
        if low_i < 12 or low_i > len(hist) - 8:
            continue

        prior_start = max(0, low_i - 52)
        prior_end = max(prior_start + 1, low_i - 2)
        prior_slice = highs[prior_start:prior_end]
        if len(prior_slice) == 0:
            continue
        prior_rel = int(prior_slice.argmax())
        prior_high_i = prior_start + prior_rel
        prior_high = float(highs[prior_high_i])
        low = float(lows[low_i])
        if not (math.isfinite(prior_high) and math.isfinite(low) and prior_high > low > 0):
            continue

        reset_drawdown = (prior_high / low - 1) * 100

        # The user's examples are real "reset" lows, not ordinary 10–20% pullbacks.
        rolling_start = max(0, low_i - 52)
        rolling_low = float(min(lows[rolling_start:low_i+1]))
        near_52w_low = low <= rolling_low * 1.035

        # Macro high is the strongest subsequent high after that reset low.
        future = highs[low_i + 6:]
        if len(future) == 0:
            continue
        high_i = low_i + 6 + int(future.argmax())
        high = float(highs[high_i])
        if not (math.isfinite(high) and high > low):
            continue

        impulse_gain = (high / low - 1) * 100
        if impulse_gain < 50:
            continue

        # Structural confirmation: the move must reclaim/break the prior cycle high.
        reclaim_slice = closes[low_i + 1:high_i + 1]
        reclaimed_prior_high = len(reclaim_slice) > 0 and float(max(reclaim_slice)) >= prior_high * 1.01

        # Primary definition: deep reset + major low + confirmed breakout.
        strong_reset = reset_drawdown >= 30 and near_52w_low and reclaimed_prior_high

        # Fallback for calmer stocks: still require a meaningful reset and breakout.
        fallback_reset = reset_drawdown >= 22 and near_52w_low and reclaimed_prior_high and impulse_gain >= 80

        if not (strong_reset or fallback_reset):
            continue

        # Prefer the most recent qualified structural reset. Gain/prominence only break close ties.
        score = (
            low_i * 100000
            + min(reset_drawdown, 100) * 100
            + min(impulse_gain, 500)
        )
        candidates.append({
            "score": score,
            "low_i": low_i,
            "high_i": high_i,
            "prior_high_i": prior_high_i,
            "prior_high": prior_high,
            "reset_drawdown": reset_drawdown,
            "impulse_gain": impulse_gain,
        })

    if candidates:
        best = max(candidates, key=lambda c: c["score"])
        low_i = best["low_i"]
        high_i = best["high_i"]
        prior_high_i = best["prior_high_i"]
        prior_high = best["prior_high"]
        reset_drawdown = best["reset_drawdown"]
        macro_gain = best["impulse_gain"] / 100
        anchor_confidence = "HIGH"
    else:
        # Conservative fallback: use the largest 2Y impulse, but mark confidence lower.
        start = max(0, len(hist) - 104)
        best_pair = None
        running_low_i = start
        running_low = lows[start]
        for high_i in range(start + 8, len(hist)):
            candidate_i = high_i - 8
            if lows[candidate_i] < running_low:
                running_low = lows[candidate_i]
                running_low_i = candidate_i
            high = highs[high_i]
            if running_low > 0 and high > running_low:
                gain = high / running_low - 1
                if best_pair is None or gain > best_pair[0]:
                    best_pair = (gain, running_low_i, high_i)
        if best_pair is None:
            return None
        macro_gain, low_i, high_i = best_pair
        prior_high_i = max(0, low_i - 1)
        prior_high = float(highs[prior_high_i])
        reset_drawdown = (prior_high / lows[low_i] - 1) * 100 if prior_high > lows[low_i] else 0
        anchor_confidence = "FALLBACK"

    swing_low = float(lows[low_i])
    swing_high = float(highs[high_i])
    if not math.isfinite(swing_low) or not math.isfinite(swing_high) or swing_high <= swing_low:
        return None

    span = swing_high - swing_low
    levels = {f"{r:g}": swing_high - r * span for r in FIB_RATIOS}
    p05 = levels["0.5"]
    p0941 = levels["0.941"]
    coverage_pct = max(0.0, (p05 - p0941) / p05 * 100)
    theoretical_lev = (100 / coverage_pct) if coverage_pct > 0 else 20.0
    safety_buffer_pct = max(2.0, coverage_pct * 0.25)
    buffered_lev = 100 / (coverage_pct + safety_buffer_pct) if (coverage_pct + safety_buffer_pct) > 0 else 20.0
    buffered_lev = max(1.0, min(20.0, math.floor(buffered_lev * 2) / 2))

    next_ratio, next_price, next_dist = None, None, None
    candidates_next = []
    for r in ENTRY_FIB_RATIOS:
        lp = levels[f"{r:g}"]
        if current_price >= lp:
            candidates_next.append((r, lp, (current_price - lp) / current_price * 100))
    if candidates_next:
        next_ratio, next_price, next_dist = min(candidates_next, key=lambda x: x[2])

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

    chart_start = max(0, min(prior_high_i, low_i) - 8)
    chart = []
    for i in range(chart_start, len(hist)):
        row = hist.iloc[i]
        chart.append({
            "d": dt_text(i),
            "o": round(float(row["Open"]), 4),
            "h": round(float(row["High"]), 4),
            "l": round(float(row["Low"]), 4),
            "c": round(float(row["Close"]), 4),
        })

    return {
        "anchorLow": round(swing_low, 4),
        "anchorLowDate": dt_text(low_i),
        "anchorHigh": round(swing_high, 4),
        "anchorHighDate": dt_text(high_i),
        "priorCycleHigh": round(float(prior_high), 4),
        "priorCycleHighDate": dt_text(prior_high_i),
        "resetDrawdownPct": round(reset_drawdown, 1),
        "macroGainPct": round(macro_gain * 100, 1),
        "anchorConfidence": anchor_confidence,
        "levels": {k: round(v, 4) for k, v in levels.items()},
        "coveragePct": round(coverage_pct, 2),
        "maxLeverageTheoretical": round(theoretical_lev, 2),
        "maxLeverageBuffered": round(buffered_lev, 2),
        "safetyBufferPct": round(safety_buffer_pct, 2),
        "status": status,
        "nextRatio": None if next_ratio is None else f"{next_ratio:g}",
        "nextPrice": None if next_price is None else round(next_price, 4),
        "distanceToNextPct": None if next_dist is None else round(next_dist, 2),
        "method": "structural-macro-reset-weekly",
        "chart": chart
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

    meta["version"] = "1.6-structural-macro-fib"
    meta["updated"] = now.isoformat(timespec="seconds")
    meta["refreshMode"] = "nightly-market"
    meta["symbolsUpdated"] = len(snapshots)
    meta["symbolsFailed"] = failures
    research = meta.get("researchUpdated", "unbekannt")
    meta["note"] = (
        f"Automatisches Marktupdate: {len(snapshots)}/{len(symbols)} Symbole aktualisiert. "
        f"Kurse, RSI14, 50/200DMA, Macro-Weekly-Fib-Level, technischer Status und dynamische Entry/Momentum-Scores sind aktuell. "
        f"Fundamentale These, qualitative Flags und Buy-Zonen stammen aus dem Research-Stand {research}."
    )

    tmp = DATA_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(DATA_PATH)
    print(meta["note"])

if __name__ == "__main__":
    main()
