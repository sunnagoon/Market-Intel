# Market Pulse Intelligence

Real-time style market dashboard built on `yfinance` proxies. It explains market action with:

- Market move narrative ("why the tape is moving")
- Intraday volume pressure vs expected volume
- Best and worst sectors across daily, weekly, and monthly windows
- Sentiment indicators (VIX, breadth, trend, headline tone)
- CTA positioning proxy model
- Yield curve state with steepening/flattening/inversion trend (1W/1M)
- Treasury futures focus table (`ZT=F`, `ZN=F`, `ZB=F`, `UB=F`)
- Downside capitulation / upside exhaustion scores (0-100)
- Trigger + confirmation model (extreme today, reversal in next 1-3 bars)

## What This App Uses

- Price/volume data from Yahoo Finance (`yfinance`)
- Headline feed from Yahoo-linked ticker news via `yfinance`
- Proxy calculations when institutional datasets are unavailable

## Install

```bash
python -m pip install -r requirements.txt
```

## Run

```bash
python -m streamlit run app.py
```

## Notes

- Data can be delayed depending on Yahoo feed availability.
- The CTA panel is a trend-following proxy, not official CFTC positioning.
- Intraday volume is estimated from liquid ETF proxies (`SPY`, `QQQ`, `DIA`, `IWM`).
- Front-end curve uses 13W T-bill (`^IRX`) as a front-end proxy when a direct Yahoo 2Y cash-yield index is unavailable.
