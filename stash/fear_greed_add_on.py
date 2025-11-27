import pandas as pd
import numpy as np
import yfinance as yf
import fear_and_greed as fg
import smtplib
from datetime import datetime, timedelta
import json
import os
import time

# =====================================================
# CONFIGURATION
# =====================================================
TICKER = 'QQQ'
TRADE_TICKER = 'TQQQ'
TOTAL_CAPITAL = 10000
UPDATE_INTERVAL_HOURS = 12  # Send update every 12 hours

# Email settings
EMAIL_ADDRESS = "vikramatmuri01@gmail.com"
EMAIL_PASSWORD = "vyxs fpgn egzf owop"
TO_EMAIL = "vikramatmuri01@gmail.com"

STATE_FILE = "trade_state.json"

# =====================================================
# EMAIL FUNCTION
# =====================================================
def send_email(subject, body):
    email_text = f"Subject: {subject}\n\n{body}"
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            server.sendmail(EMAIL_ADDRESS, TO_EMAIL, email_text)
        print(f"Email sent: {subject}")
    except Exception as e:
        print(f"Email failed: {e}")

# =====================================================
# STATE MANAGEMENT
# =====================================================
def load_state():
    default_state = {
        'status': 'waiting',        # waiting, entered, holding
        'entry_date': None,
        'shares': 75,               # Your real shares
        'avg_cost': 47.34,          # Your real average cost
        'cash': TOTAL_CAPITAL - 3500,  # Remaining cash
        'total_invested_pct': 35     # Currently 35% invested
    }
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, 'r') as f:
                content = f.read().strip()
                if not content:
                    return default_state
                state = json.loads(content)
                # Safety checks
                if state.get('avg_cost') in [0, None]:
                    state['avg_cost'] = 47.34
                if state.get('shares') in [0, None]:
                    state['shares'] = 75
                if state.get('total_invested_pct') in [0, None]:
                    state['total_invested_pct'] = 35
                return state
        except (json.JSONDecodeError, ValueError):
            return default_state
    else:
        return default_state

def save_state(state):
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2)

# =====================================================
# GET CURRENT DATA
# =====================================================
def get_current_data():
    end = datetime.today()
    start = end - timedelta(days=300)

    qqq = yf.download(TICKER, start=start, end=end, auto_adjust=True, progress=False)
    tqqq = yf.download(TRADE_TICKER, start=start, end=end, auto_adjust=True, progress=False)

    qqq_price = qqq['Close'].iloc[-1].item()
    tqqq_price = tqqq['Close'].iloc[-1].item()
    ma200 = qqq['Close'].rolling(200).mean().iloc[-1].item()

    try:
        fg_value = fg.get().value
    except:
        fg_value = None

    return {
        'date': datetime.today().strftime('%Y-%m-%d'),
        'qqq_price': qqq_price,
        'tqqq_price': tqqq_price,
        'ma200': ma200,
        'above_ma200': qqq_price > ma200,
        'fg': fg_value
    }

# =====================================================
# PERCENT CHANGE CALC
# =====================================================
def pct_change(current, reference):
    if reference in [0, None]:
        return 0.0
    return (current - reference) / reference * 100

# =====================================================
# MAIN MONITOR LOGIC
# =====================================================
def run_monitor():
    print("="*60)
    print(f"TQQQ STRATEGY MONITOR - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("="*60)

    state = load_state()
    data = get_current_data()

    shares = state['shares']
    avg_cost = state['avg_cost']
    current_value = shares * data['tqqq_price']
    unrealized_return = pct_change(data['tqqq_price'], avg_cost)

    # Handle Fear & Greed safely
    fg_text = f"{data['fg']:.1f}" if data['fg'] is not None else "N/A"

    # Print info
    print(f"\nCurrent Conditions:")
    print(f"  QQQ Price: ${data['qqq_price']:.2f}")
    print(f"  TQQQ Price: ${data['tqqq_price']:.2f}")
    print(f"  MA200: ${data['ma200']:.2f}")
    print(f"  Above MA200: {data['above_ma200']}")
    print(f"  Fear & Greed: {fg_text}")
    print(f"\nMerged Strategy Position:")
    print(f"  Shares: {shares}")
    print(f"  Avg cost: ${avg_cost:.2f}")
    print(f"  Current value: ${current_value:.2f}")
    print(f"  Unrealized return: {unrealized_return:.2f}%")

    # ================== ENTRY LOGIC ==================
    if state['status'] == 'waiting' and data['fg'] is not None and data['fg'] < 10 and data['above_ma200']:
        subject = "TQQQ ENTRY SIGNAL - BUY OPPORTUNITY"
        body = f"""
ENTRY SIGNAL TRIGGERED!

Position:
- Shares: {shares}
- Avg cost: ${avg_cost:.2f}
- Current value: ${current_value:.2f}
- Unrealized return: {unrealized_return:.2f}%

Action: Monitor for next week to add more if needed.
"""
        send_email(subject, body)
        state['status'] = 'entered'
        state['entry_date'] = data['date']
        save_state(state)
        print("Entry signal email sent.")

    # ================== WEEKLY CHECK ==================
    if state['status'] == 'entered':
        entry_date = datetime.strptime(state['entry_date'], '%Y-%m-%d')
        if (datetime.today() - entry_date).days >= 7:
            price_change = pct_change(data['qqq_price'], state.get('entry_price') or data['qqq_price'])
            subject = "TQQQ WEEKLY CHECK"
            body = f"""
1-Week Check:

QQQ Price Change: {price_change:.2f}%

No additional shares are needed as position is fully merged with real position.
"""
            send_email(subject, body)
            state['status'] = 'holding'
            save_state(state)

    # ================== EXIT LOGIC ==================
    if state['status'] == 'holding' and data['fg'] is not None and data['fg'] > 70:
        total_return = pct_change(data['qqq_price'], state.get('entry_price') or data['qqq_price'])
        subject = "TQQQ EXIT SIGNAL - SELL POSITION"
        body = f"""
EXIT SIGNAL TRIGGERED!

Entry QQQ Price: ${state.get('entry_price', 0):.2f}
Current QQQ Price: ${data['qqq_price']:.2f}
Return: {total_return:.2f}%

Action: Sell TQQQ to exit strategy.
"""
        send_email(subject, body)
        # Reset state
        state['status'] = 'waiting'
        state['entry_date'] = None
        save_state(state)

    # ================== STATUS UPDATE EMAIL ==================
    subject = "TQQQ POSITION UPDATE"
    body = f"""
Current Conditions:
- QQQ Price: ${data['qqq_price']:.2f}
- TQQQ Price: ${data['tqqq_price']:.2f}
- MA200: ${data['ma200']:.2f}
- Above MA200: {data['above_ma200']}
- Fear & Greed: {fg_text}

Position:
- Shares: {shares}
- Avg cost: ${avg_cost:.2f}
- Current value: ${current_value:.2f}
- Unrealized return: {unrealized_return:.2f}%
- Status: {state['status']}
"""
    send_email(subject, body)
    print("Status update email sent.\nMonitor check complete.")

# =====================================================
# RUN LOOP
# =====================================================
CHECK_INTERVAL_SECONDS = UPDATE_INTERVAL_HOURS * 3600  # Convert hours to seconds

def monitor_loop():
    while True:
        run_monitor()
        print(f"Waiting {UPDATE_INTERVAL_HOURS} hours until next check...\n")
        time.sleep(CHECK_INTERVAL_SECONDS)

if __name__ == "__main__":
    monitor_loop()
