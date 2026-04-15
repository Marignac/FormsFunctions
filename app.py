import datetime as dt
from typing import Dict, List

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf


PAIR_UNIVERSE: Dict[str, str] = {
    # G10 and closely traded majors
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
    # Asia
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
    # LATAM
    "USD/MXN": "MXN=X",
    "USD/BRL": "BRL=X",
    "USD/CLP": "CLP=X",
    "USD/COP": "COP=X",
    "USD/PEN": "PEN=X",
    "USD/ARS": "ARS=X",
}


@st.cache_data(ttl=3600)
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
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna(how="all")
    return df


@st.cache_data(ttl=3600)
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
                "Volume": float(latest["Volume"]),
            }
        )
    return pd.DataFrame(rows).sort_values(by="Pair").reset_index(drop=True)


def compute_signals(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["SMA_20"] = out["Close"].rolling(20).mean()
    out["SMA_50"] = out["Close"].rolling(50).mean()

    delta = out["Close"].diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    rs = gains.rolling(14).mean() / losses.rolling(14).mean()
    out["RSI_14"] = 100 - (100 / (1 + rs))

    out["AvgVol_20"] = out["Volume"].rolling(20).mean()
    out["High_20"] = out["High"].rolling(20).max().shift(1)
    out["Low_20"] = out["Low"].rolling(20).min().shift(1)

    signals: List[dict] = []
    week_ago = pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(days=7)

    for i in range(1, len(out)):
        ts = out.index[i]
        if ts < week_ago:
            continue

        prev = out.iloc[i - 1]
        cur = out.iloc[i]

        if pd.notna(prev["SMA_20"]) and pd.notna(prev["SMA_50"]) and pd.notna(cur["SMA_20"]) and pd.notna(cur["SMA_50"]):
            if prev["SMA_20"] <= prev["SMA_50"] and cur["SMA_20"] > cur["SMA_50"]:
                signals.append({"Date": ts, "Signal": "Bullish SMA crossover (20 > 50)", "Close": cur["Close"]})
            if prev["SMA_20"] >= prev["SMA_50"] and cur["SMA_20"] < cur["SMA_50"]:
                signals.append({"Date": ts, "Signal": "Bearish SMA crossover (20 < 50)", "Close": cur["Close"]})

        if pd.notna(cur["RSI_14"]):
            if cur["RSI_14"] > 70:
                signals.append({"Date": ts, "Signal": "RSI overbought (>70)", "Close": cur["Close"]})
            if cur["RSI_14"] < 30:
                signals.append({"Date": ts, "Signal": "RSI oversold (<30)", "Close": cur["Close"]})

        if pd.notna(cur["High_20"]) and cur["Close"] > cur["High_20"]:
            signals.append({"Date": ts, "Signal": "20-day breakout (close above prior high)", "Close": cur["Close"]})

        if pd.notna(cur["Low_20"]) and cur["Close"] < cur["Low_20"]:
            signals.append({"Date": ts, "Signal": "20-day breakdown (close below prior low)", "Close": cur["Close"]})

        if pd.notna(cur["AvgVol_20"]) and cur["Volume"] > 1.5 * cur["AvgVol_20"]:
            signals.append({"Date": ts, "Signal": "Volume spike (>1.5x 20-day avg)", "Close": cur["Close"]})

    if not signals:
        return pd.DataFrame(columns=["Date", "Signal", "Close"])

    signal_df = pd.DataFrame(signals).drop_duplicates().sort_values("Date", ascending=False)
    signal_df["Date"] = signal_df["Date"].dt.strftime("%Y-%m-%d")
    return signal_df


def build_candle(df: pd.DataFrame, pair_name: str) -> go.Figure:
    fig = go.Figure(
        data=[
            go.Candlestick(
                x=df.index,
                open=df["Open"],
                high=df["High"],
                low=df["Low"],
                close=df["Close"],
                name=pair_name,
            )
        ]
    )
    fig.update_layout(
        title=f"{pair_name} Candlestick",
        xaxis_title="Date",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        height=600,
    )
    return fig


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=True).encode("utf-8")


def main() -> None:
    st.set_page_config(page_title="FX Trader Dashboard", layout="wide")
    st.title("🌍 FX Trader Dashboard (Yahoo Finance)")
    st.caption("Major G10 + Asia + LATAM pairs with OHLCV, candlesticks, exports, and recent signals.")

    with st.sidebar:
        st.header("Controls")
        selected_pair = st.selectbox("Select currency pair", options=list(PAIR_UNIVERSE.keys()), index=0)
        period = st.selectbox("History period", options=["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3)
        interval = st.selectbox("Bar interval", options=["1d", "4h", "1h"], index=0)
        st.markdown("---")
        st.write("Universe includes majors plus Asia and LATAM crosses quoted by Yahoo Finance.")

    snapshot = download_universe_snapshot(period="1mo", interval="1d")
    st.subheader("Latest OHLCV Snapshot (Universe)")
    st.dataframe(snapshot, use_container_width=True, hide_index=True)

    snapshot_csv = snapshot.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Export snapshot CSV",
        data=snapshot_csv,
        file_name=f"fx_universe_snapshot_{dt.date.today().isoformat()}.csv",
        mime="text/csv",
    )

    ticker = PAIR_UNIVERSE[selected_pair]
    hist = download_pair_history(ticker=ticker, period=period, interval=interval)

    if hist.empty:
        st.error("No data returned from Yahoo Finance for this pair/period/interval.")
        return

    st.subheader(f"Candlestick — {selected_pair} ({ticker})")
    st.plotly_chart(build_candle(hist, selected_pair), use_container_width=True)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.metric("Last Close", f"{hist['Close'].iloc[-1]:,.5f}")
    with col2:
        last_vol = hist["Volume"].iloc[-1]
        st.metric("Last Volume", f"{last_vol:,.0f}")

    st.download_button(
        label=f"⬇️ Export {selected_pair} history CSV",
        data=to_csv_bytes(hist),
        file_name=f"{selected_pair.replace('/', '')}_{period}_{interval}.csv",
        mime="text/csv",
    )

    st.subheader("Signals from the past 7 days")
    signals = compute_signals(hist)
    if signals.empty:
        st.info("No configured signals triggered in the last 7 days for this pair.")
    else:
        st.dataframe(signals, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
