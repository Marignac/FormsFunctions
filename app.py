import datetime as dt
from typing import Dict, List, Tuple

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf


PAIR_UNIVERSE: Dict[str, str] = {
    # G10 majors and liquid crosses
    "EUR/USD": "EURUSD=X",
    "USD/JPY": "JPY=X",
    "GBP/USD": "GBPUSD=X",
    "AUD/USD": "AUDUSD=X",
    "USD/CAD": "CAD=X",
    "USD/CHF": "CHF=X",
    "NZD/USD": "NZDUSD=X",
    "EUR/JPY": "EURJPY=X",
    "EUR/GBP": "EURGBP=X",
    "EUR/CHF": "EURCHF=X",
    "GBP/JPY": "GBPJPY=X",
    # Europe / additional EUR crosses
    "EUR/CZK": "EURCZK=X",
    "EUR/PLN": "EURPLN=X",
    "EUR/NOK": "EURNOK=X",
    "EUR/SEK": "EURSEK=X",
    "EUR/HUF": "EURHUF=X",
    "EUR/RON": "EURRON=X",
    "EUR/TRY": "EURTRY=X",
    # Asia + Middle East
    "USD/CNH": "USDCNH=X",
    "USD/SGD": "USDSGD=X",
    "USD/HKD": "HKD=X",
    "USD/KRW": "KRW=X",
    "USD/INR": "INR=X",
    "USD/IDR": "IDR=X",
    "USD/THB": "THB=X",
    "USD/TWD": "TWD=X",
    "USD/MYR": "MYR=X",
    "USD/PHP": "PHP=X",
    "USD/ILS": "ILS=X",
    "USD/TRY": "TRY=X",
    # LATAM
    "USD/MXN": "MXN=X",
    "USD/BRL": "BRL=X",
    "USD/CLP": "CLP=X",
    "USD/COP": "COP=X",
    "USD/PEN": "PEN=X",
    "USD/ARS": "ARS=X",
    # Other Europe
    "USD/NOK": "NOK=X",
    "USD/SEK": "SEK=X",
    "USD/PLN": "PLN=X",
    "USD/CZK": "CZK=X",
    "USD/HUF": "HUF=X",
    "USD/RON": "RON=X",
    "USD/DKK": "DKK=X",
}


INTERVAL_PERIOD_OPTIONS: Dict[str, List[str]] = {
    "1d": ["3mo", "6mo", "1y", "2y", "5y"],
    "1h": ["7d", "30d", "60d", "730d"],
    "30m": ["7d", "30d", "60d"],
    "15m": ["7d", "30d", "60d"],
}


@st.cache_data(ttl=1800)
def download_pair_history(ticker: str, period: str, interval: str) -> pd.DataFrame:
    df = yf.download(
        tickers=ticker,
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False,
    )
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    keep_cols = ["Open", "High", "Low", "Close"]
    existing = [c for c in keep_cols if c in df.columns]
    cleaned = df[existing].dropna(how="all")
    return cleaned


@st.cache_data(ttl=1800)
def download_universe_snapshot(period: str = "1mo", interval: str = "1d") -> pd.DataFrame:
    rows: List[dict] = []
    for name, ticker in PAIR_UNIVERSE.items():
        hist = download_pair_history(ticker=ticker, period=period, interval=interval)
        if hist.empty:
            continue
        latest = hist.tail(1).reset_index().iloc[0]
        rows.append(
            {
                "Pair": name,
                "Ticker": ticker,
                "Date": latest["Date"],
                "Open": float(latest["Open"]),
                "High": float(latest["High"]),
                "Low": float(latest["Low"]),
                "Close": float(latest["Close"]),
            }
        )
    return pd.DataFrame(rows).sort_values(by="Pair").reset_index(drop=True)


def compute_signals_for_pair(df: pd.DataFrame, pair_name: str) -> pd.DataFrame:
    out = df.copy().dropna()
    if out.empty or len(out) < 60:
        return pd.DataFrame(columns=["Date", "Pair", "Signal", "Bias", "Close"])

    out["EMA_9"] = out["Close"].ewm(span=9, adjust=False).mean()
    out["EMA_21"] = out["Close"].ewm(span=21, adjust=False).mean()
    out["SMA_20"] = out["Close"].rolling(20).mean()
    out["SMA_50"] = out["Close"].rolling(50).mean()

    delta = out["Close"].diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    avg_gain = gains.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    avg_loss = losses.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss
    out["RSI_14"] = 100 - (100 / (1 + rs))

    out["BB_MID"] = out["Close"].rolling(20).mean()
    bb_std = out["Close"].rolling(20).std()
    out["BB_UPPER"] = out["BB_MID"] + (2 * bb_std)
    out["BB_LOWER"] = out["BB_MID"] - (2 * bb_std)

    macd_fast = out["Close"].ewm(span=12, adjust=False).mean()
    macd_slow = out["Close"].ewm(span=26, adjust=False).mean()
    out["MACD"] = macd_fast - macd_slow
    out["MACD_SIGNAL"] = out["MACD"].ewm(span=9, adjust=False).mean()

    out["HIGH_20"] = out["High"].rolling(20).max().shift(1)
    out["LOW_20"] = out["Low"].rolling(20).min().shift(1)

    week_ago = pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(days=7)
    signals: List[dict] = []

    for i in range(1, len(out)):
        ts = out.index[i]
        if ts < week_ago:
            continue

        prev = out.iloc[i - 1]
        cur = out.iloc[i]

        if prev["EMA_9"] <= prev["EMA_21"] and cur["EMA_9"] > cur["EMA_21"]:
            signals.append({"Date": ts, "Pair": pair_name, "Signal": "EMA 9/21 Bullish Crossover", "Bias": "Bullish", "Close": cur["Close"]})
        if prev["EMA_9"] >= prev["EMA_21"] and cur["EMA_9"] < cur["EMA_21"]:
            signals.append({"Date": ts, "Pair": pair_name, "Signal": "EMA 9/21 Bearish Crossover", "Bias": "Bearish", "Close": cur["Close"]})

        if prev["SMA_20"] <= prev["SMA_50"] and cur["SMA_20"] > cur["SMA_50"]:
            signals.append({"Date": ts, "Pair": pair_name, "Signal": "SMA 20/50 Golden Cross", "Bias": "Bullish", "Close": cur["Close"]})
        if prev["SMA_20"] >= prev["SMA_50"] and cur["SMA_20"] < cur["SMA_50"]:
            signals.append({"Date": ts, "Pair": pair_name, "Signal": "SMA 20/50 Death Cross", "Bias": "Bearish", "Close": cur["Close"]})

        if prev["RSI_14"] < 30 and cur["RSI_14"] >= 30:
            signals.append({"Date": ts, "Pair": pair_name, "Signal": "RSI Recovery from Oversold", "Bias": "Bullish", "Close": cur["Close"]})
        if prev["RSI_14"] > 70 and cur["RSI_14"] <= 70:
            signals.append({"Date": ts, "Pair": pair_name, "Signal": "RSI Rejection from Overbought", "Bias": "Bearish", "Close": cur["Close"]})

        if prev["MACD"] <= prev["MACD_SIGNAL"] and cur["MACD"] > cur["MACD_SIGNAL"]:
            signals.append({"Date": ts, "Pair": pair_name, "Signal": "MACD Bullish Cross", "Bias": "Bullish", "Close": cur["Close"]})
        if prev["MACD"] >= prev["MACD_SIGNAL"] and cur["MACD"] < cur["MACD_SIGNAL"]:
            signals.append({"Date": ts, "Pair": pair_name, "Signal": "MACD Bearish Cross", "Bias": "Bearish", "Close": cur["Close"]})

        if cur["Close"] > cur["HIGH_20"]:
            signals.append({"Date": ts, "Pair": pair_name, "Signal": "20-Day Breakout", "Bias": "Bullish", "Close": cur["Close"]})
        if cur["Close"] < cur["LOW_20"]:
            signals.append({"Date": ts, "Pair": pair_name, "Signal": "20-Day Breakdown", "Bias": "Bearish", "Close": cur["Close"]})

        if prev["Close"] < prev["BB_LOWER"] and cur["Close"] > cur["BB_LOWER"]:
            signals.append({"Date": ts, "Pair": pair_name, "Signal": "Bollinger Mean-Reversion Up", "Bias": "Bullish", "Close": cur["Close"]})
        if prev["Close"] > prev["BB_UPPER"] and cur["Close"] < cur["BB_UPPER"]:
            signals.append({"Date": ts, "Pair": pair_name, "Signal": "Bollinger Mean-Reversion Down", "Bias": "Bearish", "Close": cur["Close"]})

    if not signals:
        return pd.DataFrame(columns=["Date", "Pair", "Signal", "Bias", "Close"])

    signals_df = pd.DataFrame(signals).drop_duplicates().sort_values("Date", ascending=False)
    signals_df["Date"] = pd.to_datetime(signals_df["Date"]).dt.strftime("%Y-%m-%d %H:%M")
    return signals_df


@st.cache_data(ttl=1800)
def compute_universe_signals() -> pd.DataFrame:
    all_signals: List[pd.DataFrame] = []
    for pair_name, ticker in PAIR_UNIVERSE.items():
        hist = download_pair_history(ticker=ticker, period="6mo", interval="1d")
        if hist.empty:
            continue
        pair_signals = compute_signals_for_pair(hist, pair_name)
        if not pair_signals.empty:
            all_signals.append(pair_signals)

    if not all_signals:
        return pd.DataFrame(columns=["Date", "Pair", "Signal", "Bias", "Close"])

    merged = pd.concat(all_signals, ignore_index=True)
    return merged.sort_values(by="Date", ascending=False)


def build_candle(df: pd.DataFrame, pair_name: str) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="Candles",
            increasing_line_color="#26a69a",
            increasing_fillcolor="#26a69a",
            decreasing_line_color="#ef5350",
            decreasing_fillcolor="#ef5350",
            whiskerwidth=0.4,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["Close"],
            mode="lines",
            name="Close",
            line={"width": 1.2, "color": "#90caf9"},
            opacity=0.8,
        )
    )

    fig.update_layout(
        title=f"{pair_name} Candlestick",
        xaxis_title="Date",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        height=620,
        hovermode="x unified",
    )
    return fig


def to_csv_bytes(df: pd.DataFrame, include_index: bool = True) -> bytes:
    return df.to_csv(index=include_index).encode("utf-8")


def select_pair_from_signals(universe_signals: pd.DataFrame) -> Tuple[str, int]:
    if universe_signals.empty:
        return list(PAIR_UNIVERSE.keys())[0], 0

    signaled_pairs = sorted(universe_signals["Pair"].unique().tolist())
    default_pair = signaled_pairs[0]
    options = list(PAIR_UNIVERSE.keys())
    default_index = options.index(default_pair) if default_pair in options else 0
    return default_pair, default_index


def main() -> None:
    st.set_page_config(page_title="FX Trader Dashboard", layout="wide")
    st.title("🌍 FX Trader Dashboard (Yahoo Finance)")
    st.caption("Signals-first workflow for G10, Europe crosses, Asia, and LATAM pairs.")

    st.subheader("🔥 Signals in the Last 7 Days (All Pairs)")
    universe_signals = compute_universe_signals()

    if universe_signals.empty:
        st.info("No configured signals were triggered in the last 7 days across tracked pairs.")
    else:
        c1, c2 = st.columns([1, 1])
        with c1:
            bias_filter = st.selectbox("Filter by bias", ["All", "Bullish", "Bearish"], index=0)
        with c2:
            pair_filter = st.selectbox("Filter by pair", ["All"] + sorted(universe_signals["Pair"].unique().tolist()), index=0)

        filtered = universe_signals.copy()
        if bias_filter != "All":
            filtered = filtered[filtered["Bias"] == bias_filter]
        if pair_filter != "All":
            filtered = filtered[filtered["Pair"] == pair_filter]

        st.dataframe(filtered, use_container_width=True, hide_index=True)
        st.download_button(
            label="⬇️ Export weekly signals CSV",
            data=to_csv_bytes(filtered, include_index=False),
            file_name=f"fx_signals_{dt.date.today().isoformat()}.csv",
            mime="text/csv",
        )

    default_pair, default_index = select_pair_from_signals(universe_signals)

    st.markdown("---")
    st.subheader("📊 Inspect Pair Chart and Data")

    with st.sidebar:
        st.header("Chart Controls")
        selected_pair = st.selectbox("Select currency pair", options=list(PAIR_UNIVERSE.keys()), index=default_index)
        interval = st.selectbox("Bar interval", options=list(INTERVAL_PERIOD_OPTIONS.keys()), index=0)
        period = st.selectbox("History period", options=INTERVAL_PERIOD_OPTIONS[interval], index=2 if len(INTERVAL_PERIOD_OPTIONS[interval]) > 2 else 0)

    ticker = PAIR_UNIVERSE[selected_pair]
    hist = download_pair_history(ticker=ticker, period=period, interval=interval)

    if hist.empty:
        st.error("No data returned from Yahoo Finance for this pair/period/interval.")
        return

    st.plotly_chart(build_candle(hist, selected_pair), use_container_width=True)

    latest = hist.iloc[-1]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Open", f"{latest['Open']:,.5f}")
    m2.metric("High", f"{latest['High']:,.5f}")
    m3.metric("Low", f"{latest['Low']:,.5f}")
    m4.metric("Close", f"{latest['Close']:,.5f}")

    st.download_button(
        label=f"⬇️ Export {selected_pair} history CSV",
        data=to_csv_bytes(hist),
        file_name=f"{selected_pair.replace('/', '')}_{period}_{interval}.csv",
        mime="text/csv",
    )

    st.subheader("Latest OHLC Snapshot (Universe)")
    snapshot = download_universe_snapshot(period="1mo", interval="1d")
    st.dataframe(snapshot, use_container_width=True, hide_index=True)
    st.download_button(
        label="⬇️ Export universe OHLC CSV",
        data=to_csv_bytes(snapshot, include_index=False),
        file_name=f"fx_universe_ohlc_{dt.date.today().isoformat()}.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
