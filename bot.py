"""Minimal Telegram bot. Only /status command. Run: python bot.py"""
import json, os, yaml
from pathlib import Path
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler

BASE = Path(__file__).resolve().parent
EXPERIMENTS = BASE / "experiments"

def load_config():
    with open(BASE / "user_config.yaml") as f:
        return yaml.safe_load(f)

async def status(update: Update, context):
    if not EXPERIMENTS.is_dir():
        await update.message.reply_text("No experiments yet.")
        return
    exps = []
    for f in sorted(EXPERIMENTS.glob("*.json")):
        with open(f) as fh:
            exps.append(json.load(fh))
    total = len(exps)
    ok = sum(1 for e in exps if e.get("status") == "ok")
    failed = total - ok
    best, best_num = None, None
    for e in exps:
        s = e.get("score")
        if s is not None and str(s) != "-inf":
            if best is None or float(s) > best:
                best, best_num = float(s), e["experiment_num"]
    msg = f"Experiments: {total} ({ok} ok, {failed} failed)\n"
    if best is not None:
        msg += f"Best: #{best_num} score={best:.1f}\n"
    if exps:
        last = exps[-1]
        ls = last.get("score", "?")
        msg += f"Latest: #{last['experiment_num']} {last.get('status','')} score={ls}"
    await update.message.reply_text(msg)

if __name__ == "__main__":
    tg = load_config().get("telegram", {})
    app = ApplicationBuilder().token(tg["bot_token"]).build()
    app.add_handler(CommandHandler("status", status))
    print("Bot running. /status only.")
    app.run_polling()
