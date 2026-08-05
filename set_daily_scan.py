#!/usr/bin/env python3
"""
SET Daily Scanner
=================
ดึงราคาหุ้น SET50 (แก้ไขรายชื่อได้) รายวันแบบ delayed/end-of-day ฟรีผ่าน yfinance
แล้วสรุป:
  - Top 20 Gainers / Top 20 Losers (เปลี่ยนแปลง % รายวัน)
  - สัญญาณทางเทคนิค (SMA20/50 crossover, RSI14, MACD) -> จัดกลุ่มเป็น ซื้อ/ขาย/ถือ

⚠️ นี่คือสัญญาณเชิงเทคนิคจากกฎที่ตั้งไว้เท่านั้น ไม่ใช่คำแนะนำการลงทุน
   ใช้ประกอบการตัดสินใจของคุณเอง ไม่ใช่คำตอบสำเร็จรูป

ติดตั้งก่อนใช้งาน:
    pip install yfinance pandas --break-system-packages   # (บนเครื่อง Mac/Linux ปกติไม่ต้องมี flag นี้)

รันครั้งเดียว:
    python3 set_daily_scan.py

ตั้งให้รันทุกวันอัตโนมัติ (ดูท้ายไฟล์ สำหรับ cron / Task Scheduler)
"""

import sys
import datetime as dt
from pathlib import Path

import pandas as pd

try:
    import yfinance as yf
except ImportError:
    print("ต้องติดตั้ง yfinance ก่อน: pip install yfinance pandas")
    sys.exit(1)

# ----------------------------------------------------------------------------
# 1) รายชื่อหุ้น — ค่าเริ่มต้นคือ SET100 (อิงประกาศ SET รอบ H1/2025)
#    ครอบคลุมหุ้นใหญ่และมีสภาพคล่องสูงสุด ~100 ตัว ซึ่งคิดเป็นสัดส่วนมูลค่าซื้อขาย
#    ส่วนใหญ่ของทั้งตลาดจริงๆ (สแกน "ทุกหลักทรัพย์" ราว 800 ตัวรวม warrant/DR/กองทุน
#    ไม่แนะนำ เพราะ Yahoo Finance ฟรีจะเริ่ม rate-limit และหุ้นเล็กจำนวนมากไม่มีวอลุ่มเทรด
#    ทำให้สัญญาณไม่มีความหมาย)
#
#    อยากเพิ่มหุ้นตัวอื่นเอง: เปิดหน้า https://www.set.or.th/en/market/index/set100/overview
#    หรือค้นชื่อหุ้นที่ set.or.th แล้วเติม symbol (ไม่ต้องมี .BK) ต่อท้าย list ด้านล่างได้เลย
# ----------------------------------------------------------------------------
TICKERS = [
    "AAV", "ADVANC", "AEONTS", "AMATA", "AOT", "AP", "AWC", "BA", "BAM", "BANPU",
    "BBL", "BCH", "BCP", "BCPG", "BDMS", "BEM", "BGRIM", "BH", "BJC", "BLA",
    "BTG", "BTS", "CBG", "CCET", "CENTEL", "CHG", "CK", "CKP", "COCOCO", "COM7",
    "CPALL", "CPF", "CPN", "CRC", "DELTA", "DOHOME", "EA", "EGCO", "ERW", "GLOBAL",
    "GPSC", "GULF", "GUNKUL", "HANA", "HMPRO", "ICHI", "IRPC", "ITC", "IVL", "JAS",
    "JMART", "JMT", "KBANK", "KCE", "KKP", "KTB", "KTC", "LH", "M", "MEGA",
    "MINT", "MOSHI", "MTC", "OR", "OSP", "PLANB", "PR9", "PRM", "PTT", "PTTEP",
    "PTTGC", "QH", "RATCH", "RCL", "ROJNA", "SAPPE", "SAWAD", "SCB", "SCC", "SCGP",
    "SIRI", "SISB", "SJWD", "SKY", "SNNP", "SPALI", "SPRC", "STA", "STGT", "TASCO",
    "TCAP", "TIDLOR", "TISCO", "TLI", "TOP", "TRUE", "TTB", "TU", "VGI", "WHA",
]

OUTPUT_DIR = Path(__file__).parent / "set_scan_output"
OUTPUT_DIR.mkdir(exist_ok=True)

DOCS_DIR = Path(__file__).parent / "docs"
DOCS_DIR.mkdir(exist_ok=True)

LOOKBACK_DAYS = 180  # ต้องพอสำหรับ SMA50/MACD warm-up


# ----------------------------------------------------------------------------
# 2) Indicator math
# ----------------------------------------------------------------------------
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["sma20"] = df["Close"].rolling(20).mean()
    df["sma50"] = df["Close"].rolling(50).mean()

    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-9)
    df["rsi14"] = 100 - (100 / (1 + rs))

    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    return df


def compute_signal_series(df: pd.DataFrame) -> pd.Series:
    """Compute BUY/SELL/HOLD for every row (not just the latest), using the same
    3-rule scoring as before, so we can tell how long the current signal has held."""
    sma20, sma50, rsi14, hist_ = df["sma20"], df["sma50"], df["rsi14"], df["macd_hist"]

    trend = pd.Series(1, index=df.index).where(sma20 > sma50, -1)

    cross_up = (sma20.shift(1) <= sma50.shift(1)) & (sma20 > sma50)
    cross_down = (sma20.shift(1) >= sma50.shift(1)) & (sma20 < sma50)
    trend_bonus = pd.Series(0, index=df.index)
    trend_bonus[cross_up] = 2
    trend_bonus[cross_down] = -2

    rsi_score = pd.Series(0, index=df.index)
    rsi_score[rsi14 < 30] = 1
    rsi_score[rsi14 > 70] = -1

    prev_hist = hist_.shift(1)
    macd_up = (prev_hist <= 0) & (hist_ > 0)
    macd_down = (prev_hist >= 0) & (hist_ < 0)
    macd_score = pd.Series(0, index=df.index)
    macd_score[macd_up] = 1
    macd_score[macd_down] = -1

    score = trend + trend_bonus + rsi_score + macd_score

    signal = pd.Series("HOLD", index=df.index)
    signal[score >= 2] = "BUY"
    signal[score <= -2] = "SELL"

    # not enough warm-up data yet (SMA50/RSI/MACD need history) -> N/A
    warmup_ok = df[["sma20", "sma50", "rsi14", "macd_hist"]].notna().all(axis=1) & \
                df[["sma20", "sma50", "rsi14", "macd_hist"]].shift(1).notna().all(axis=1)
    signal[~warmup_ok] = "N/A"
    return signal


def current_signal_info(df: pd.DataFrame):
    """Return (signal, since_date, trading_days_held) for the most recent row,
    by walking backward while the signal stays the same."""
    if len(df) < 51:
        return "N/A", None, 0

    sig = compute_signal_series(df)
    current = sig.iloc[-1]
    since_date = sig.index[-1]
    days = 0
    for date in sig.index[::-1]:
        if sig.loc[date] == current:
            days += 1
            since_date = date
        else:
            break
    return current, since_date, days


# ----------------------------------------------------------------------------
# 3) Main scan
# ----------------------------------------------------------------------------
def scan(tickers=TICKERS):
    end = dt.date.today()
    start = end - dt.timedelta(days=LOOKBACK_DAYS)
    yf_symbols = [f"{t}.BK" for t in tickers]

    print(f"กำลังดึงข้อมูล {len(tickers)} หุ้น ({start} ถึง {end}) แบบ batch เดียว ...")
    raw = yf.download(
        yf_symbols, start=start, end=end, progress=False,
        auto_adjust=False, group_by="ticker", threads=True,
    )

    rows = []
    for code, yf_symbol in zip(tickers, yf_symbols):
        try:
            if yf_symbol not in raw.columns.get_level_values(0):
                print(f"  {code}: ไม่มีข้อมูล ข้าม")
                continue
            hist = raw[yf_symbol].dropna(how="all")
            if hist.empty or len(hist) < 5:
                print(f"  {code}: ไม่มีข้อมูลเพียงพอ ข้าม")
                continue
            hist = compute_indicators(hist)
            last, prev = hist.iloc[-1], hist.iloc[-2]
            pct_change = (last["Close"] - prev["Close"]) / prev["Close"] * 100
            signal, since_date, days_held = current_signal_info(hist)
            rows.append({
                "Symbol": code,
                "Date": hist.index[-1].strftime("%Y-%m-%d"),
                "Close": round(float(last["Close"]), 2),
                "ChangePct": round(float(pct_change), 2),
                "Volume": int(last["Volume"]),
                "RSI14": round(float(last["rsi14"]), 1) if pd.notna(last["rsi14"]) else None,
                "SMA20": round(float(last["sma20"]), 2) if pd.notna(last["sma20"]) else None,
                "SMA50": round(float(last["sma50"]), 2) if pd.notna(last["sma50"]) else None,
                "Signal": signal,
                "SignalSince": since_date.strftime("%Y-%m-%d") if since_date is not None else None,
                "SignalDays": days_held,
            })
        except Exception as e:
            print(f"  {code}: error {e}")

    print(f"ดึงข้อมูลสำเร็จ {len(rows)}/{len(tickers)} ตัว")
    return pd.DataFrame(rows)


def _row_html(r, cols):
    cells = "".join(f"<td>{r[c]}</td>" for c in cols)
    return f"<tr>{cells}</tr>"


def _table_html(sub_df, cols, headers):
    head = "".join(f"<th>{h}</th>" for h in headers)
    body = "\n".join(_row_html(r, cols) for _, r in sub_df.iterrows())
    if not body:
        body = f"<tr><td colspan='{len(cols)}' class='empty'>ไม่มีข้อมูล</td></tr>"
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def write_html(df: pd.DataFrame):
    """Write a shareable static HTML report to docs/index.html (for GitHub Pages)."""
    today = dt.date.today().isoformat()
    now_th = dt.datetime.utcnow() + dt.timedelta(hours=7)  # GitHub Actions runs in UTC; convert to Thailand time
    now_str = now_th.strftime("%Y-%m-%d %H:%M")

    if df.empty:
        gainers = losers = buys = sells = df
        data_date = "—"
    else:
        gainers = df.sort_values("ChangePct", ascending=False).head(20)
        losers = df.sort_values("ChangePct", ascending=True).head(20)
        buys = df[df["Signal"] == "BUY"].sort_values("SignalDays", ascending=False)
        sells = df[df["Signal"] == "SELL"].sort_values("SignalDays", ascending=False)
        # Most common last-trading-date across tickers = the actual data date
        data_date = df["Date"].mode().iloc[0] if "Date" in df.columns and not df["Date"].empty else "—"

    def fmt_pct(v):
        return f"+{v:.2f}%" if v >= 0 else f"{v:.2f}%"

    def fmt_since(row):
        if row.get("SignalSince") is None:
            return "—"
        return f"{row['SignalSince']} ({int(row['SignalDays'])} วันทำการ)"

    def with_fmt(sub, add_since=False):
        sub = sub.copy()
        if not sub.empty:
            sub["ChangePct"] = sub["ChangePct"].apply(fmt_pct)
            if add_since:
                sub["Since"] = sub.apply(fmt_since, axis=1)
        return sub

    gainers_html = _table_html(with_fmt(gainers), ["Symbol", "Close", "ChangePct", "Volume", "Signal"],
                                ["Symbol", "ราคาปิด", "% เปลี่ยนแปลง", "วอลุ่ม", "สัญญาณ"])
    losers_html = _table_html(with_fmt(losers), ["Symbol", "Close", "ChangePct", "Volume", "Signal"],
                               ["Symbol", "ราคาปิด", "% เปลี่ยนแปลง", "วอลุ่ม", "สัญญาณ"])
    buys_html = _table_html(with_fmt(buys, add_since=True), ["Symbol", "Close", "ChangePct", "RSI14", "SMA20", "SMA50", "Since"],
                             ["Symbol", "ราคาปิด", "% เปลี่ยนแปลง", "RSI14", "SMA20", "SMA50", "เป็นสัญญาณนี้มาตั้งแต่"])
    sells_html = _table_html(with_fmt(sells, add_since=True), ["Symbol", "Close", "ChangePct", "RSI14", "SMA20", "SMA50", "Since"],
                              ["Symbol", "ราคาปิด", "% เปลี่ยนแปลง", "RSI14", "SMA20", "SMA50", "เป็นสัญญาณนี้มาตั้งแต่"])

    html = f"""<!DOCTYPE html>
<html lang="th">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SET Daily Scan — {today}</title>
<style>
  body {{ font-family: -apple-system, 'Segoe UI', sans-serif; background:#12181f; color:#e8e4d8; margin:0; padding:24px 16px 60px; }}
  .wrap {{ max-width: 960px; margin: 0 auto; }}
  h1 {{ font-size: 22px; margin-bottom:2px; }}
  .updated {{ color:#9aa5b1; font-size:13px; margin-bottom:24px; }}
  h2 {{ font-size:16px; color:#C9A227; border-bottom:1px solid #2a333e; padding-bottom:6px; margin-top:32px; }}
  table {{ width:100%; border-collapse: collapse; font-size:13px; margin-top:8px; }}
  th, td {{ text-align:left; padding:6px 10px; border-bottom:1px solid #232c36; }}
  th {{ color:#9aa5b1; font-weight:600; }}
  td.empty {{ color:#5c6773; text-align:center; }}
  .note {{ font-size:12px; color:#9aa5b1; margin-top:36px; border-top:1px solid #2a333e; padding-top:12px; }}
</style>
</head>
<body>
<div class="wrap">
  <h1>SET Daily Scan</h1>
  <div class="updated">ข้อมูลราคาปิดล่าสุด ณ วันที่: <strong style="color:#e8e4d8">{data_date}</strong> &nbsp;•&nbsp; ระบบรันล่าสุดเมื่อ: {now_str} น. (เวลาไทย)</div>

  <h2>Top 20 Gainers</h2>
  {gainers_html}

  <h2>Top 20 Losers</h2>
  {losers_html}

  <h2>สัญญาณ BUY ({len(buys)} ตัว)</h2>
  {buys_html}

  <h2>สัญญาณ SELL ({len(sells)} ตัว)</h2>
  {sells_html}

  <div class="note">
    ⚠️ สัญญาณ BUY/SELL คำนวณจากกฎเชิงเทคนิค (SMA20/50 crossover, RSI14, MACD) เท่านั้น
    ไม่ใช่คำแนะนำการลงทุน ไม่ได้พิจารณาข่าวหรือปัจจัยพื้นฐาน ใช้ประกอบการตัดสินใจของคุณเอง
  </div>
</div>
</body>
</html>"""

    (DOCS_DIR / "index.html").write_text(html, encoding="utf-8")
    print(f"บันทึกหน้าเว็บสรุปไว้ที่: {DOCS_DIR / 'index.html'}")


def print_report(df: pd.DataFrame):
    if df.empty:
        print("ไม่มีข้อมูลให้แสดง")
        return

    today = dt.date.today().isoformat()
    print(f"\n===== SET Daily Scan — {today} =====\n")

    gainers = df.sort_values("ChangePct", ascending=False).head(20)
    losers = df.sort_values("ChangePct", ascending=True).head(20)

    print("--- Top 20 Gainers ---")
    print(gainers[["Symbol", "Close", "ChangePct", "Volume", "Signal"]].to_string(index=False))

    print("\n--- Top 20 Losers ---")
    print(losers[["Symbol", "Close", "ChangePct", "Volume", "Signal"]].to_string(index=False))

    buys = df[df["Signal"] == "BUY"].sort_values("SignalDays", ascending=False)
    sells = df[df["Signal"] == "SELL"].sort_values("SignalDays", ascending=False)

    print(f"\n--- สัญญาณ BUY ({len(buys)} ตัว) ---")
    if buys.empty:
        print("(ไม่มีหุ้นเข้าเงื่อนไข BUY วันนี้)")
    else:
        print(buys[["Symbol", "Close", "ChangePct", "RSI14", "SMA20", "SMA50", "SignalSince", "SignalDays"]].to_string(index=False))

    print(f"\n--- สัญญาณ SELL ({len(sells)} ตัว) ---")
    if sells.empty:
        print("(ไม่มีหุ้นเข้าเงื่อนไข SELL วันนี้)")
    else:
        print(sells[["Symbol", "Close", "ChangePct", "RSI14", "SMA20", "SMA50", "SignalSince", "SignalDays"]].to_string(index=False))

    print("\n⚠️  สัญญาณข้างต้นคำนวณจากกฎ SMA/RSI/MACD ล้วนๆ ไม่ใช่คำแนะนำการลงทุน")
    print("    ควรตรวจสอบข้อมูลพื้นฐาน ข่าว และความเสี่ยงส่วนตัวก่อนตัดสินใจทุกครั้ง\n")

    out_file = OUTPUT_DIR / f"set_scan_{today}.csv"
    df.sort_values("ChangePct", ascending=False).to_csv(out_file, index=False, encoding="utf-8-sig")
    print(f"บันทึกผลทั้งหมดไว้ที่: {out_file}")

    write_html(df)


if __name__ == "__main__":
    result = scan()
    print_report(result)
