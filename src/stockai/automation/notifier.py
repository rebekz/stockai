"""Notification System.

Send trading alerts via Telegram, Email, or custom webhooks.
"""

import asyncio
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any
import httpx

import logging

logger = logging.getLogger(__name__)


@dataclass
class TradingAlert:
    """A trading alert message."""
    title: str
    message: str
    signal: str  # BUY, SELL, HOLD, ALERT
    symbol: str | None = None
    price: float | None = None
    target: float | None = None
    stop_loss: float | None = None
    urgency: str = "normal"  # low, normal, high, critical

    def to_text(self) -> str:
        """Convert to plain text."""
        lines = [f"🔔 {self.title}", ""]

        if self.symbol:
            emoji = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}.get(self.signal, "⚪")
            lines.append(f"{emoji} {self.signal}: {self.symbol}")

        lines.append("")
        lines.append(self.message)

        if self.price:
            lines.append(f"\n💰 Price: Rp {self.price:,.0f}")
        if self.target:
            lines.append(f"🎯 Target: Rp {self.target:,.0f}")
        if self.stop_loss:
            lines.append(f"🛑 Stop Loss: Rp {self.stop_loss:,.0f}")

        return "\n".join(lines)

    def to_markdown(self) -> str:
        """Convert to Markdown format."""
        emoji = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}.get(self.signal, "⚪")

        md = f"## {emoji} {self.title}\n\n"

        if self.symbol:
            md += f"**Signal:** {self.signal} {self.symbol}\n\n"

        md += f"{self.message}\n\n"

        if any([self.price, self.target, self.stop_loss]):
            md += "| Metric | Value |\n|--------|-------|\n"
            if self.price:
                md += f"| Price | Rp {self.price:,.0f} |\n"
            if self.target:
                md += f"| Target | Rp {self.target:,.0f} |\n"
            if self.stop_loss:
                md += f"| Stop Loss | Rp {self.stop_loss:,.0f} |\n"

        return md


class Notifier(ABC):
    """Base class for notification providers."""

    @abstractmethod
    async def send(self, alert: TradingAlert) -> bool:
        """Send a trading alert.

        Returns:
            True if sent successfully
        """
        pass

    @abstractmethod
    async def test_connection(self) -> bool:
        """Test the notification connection."""
        pass


class TelegramNotifier(Notifier):
    """Send alerts via Telegram bot.

    Setup:
        1. Create bot via @BotFather
        2. Get bot token
        3. Get chat_id by messaging bot and visiting:
           https://api.telegram.org/bot<TOKEN>/getUpdates
    """

    def __init__(self, bot_token: str, chat_id: str):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.api_url = f"https://api.telegram.org/bot{bot_token}"

    async def send(self, alert: TradingAlert) -> bool:
        """Send alert via Telegram."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.api_url}/sendMessage",
                    json={
                        "chat_id": self.chat_id,
                        "text": alert.to_text(),
                        "parse_mode": "HTML",
                    },
                    timeout=10.0,
                )
                response.raise_for_status()
                logger.info(f"Telegram alert sent: {alert.title}")
                return True

        except Exception as e:
            logger.error(f"Telegram send failed: {e}")
            return False

    async def test_connection(self) -> bool:
        """Test Telegram bot connection."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.api_url}/getMe",
                    timeout=10.0,
                )
                response.raise_for_status()
                data = response.json()
                logger.info(f"Telegram connected: @{data['result']['username']}")
                return True
        except Exception as e:
            logger.error(f"Telegram connection test failed: {e}")
            return False


class EmailNotifier(Notifier):
    """Send alerts via email (SMTP)."""

    def __init__(
        self,
        smtp_host: str,
        smtp_port: int,
        username: str,
        password: str,
        from_email: str,
        to_emails: list[str],
    ):
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.from_email = from_email
        self.to_emails = to_emails

    async def send(self, alert: TradingAlert) -> bool:
        """Send alert via email."""
        import smtplib
        from email.mime.text import MIMEText
        from email.mime.multipart import MIMEMultipart

        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = f"[StockAI] {alert.signal}: {alert.title}"
            msg["From"] = self.from_email
            msg["To"] = ", ".join(self.to_emails)

            # Plain text version
            text_part = MIMEText(alert.to_text(), "plain")
            msg.attach(text_part)

            # HTML version
            html_content = f"""
            <html>
            <body style="font-family: Arial, sans-serif;">
                <h2>{alert.title}</h2>
                <p><strong>Signal:</strong> {alert.signal}</p>
                <p>{alert.message}</p>
                {"<p><strong>Price:</strong> Rp {:,.0f}</p>".format(alert.price) if alert.price else ""}
                {"<p><strong>Target:</strong> Rp {:,.0f}</p>".format(alert.target) if alert.target else ""}
                {"<p><strong>Stop Loss:</strong> Rp {:,.0f}</p>".format(alert.stop_loss) if alert.stop_loss else ""}
            </body>
            </html>
            """
            html_part = MIMEText(html_content, "html")
            msg.attach(html_part)

            # Send
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls()
                server.login(self.username, self.password)
                server.sendmail(self.from_email, self.to_emails, msg.as_string())

            logger.info(f"Email alert sent: {alert.title}")
            return True

        except Exception as e:
            logger.error(f"Email send failed: {e}")
            return False

    async def test_connection(self) -> bool:
        """Test SMTP connection."""
        import smtplib

        try:
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls()
                server.login(self.username, self.password)
            logger.info("Email SMTP connection successful")
            return True
        except Exception as e:
            logger.error(f"Email connection test failed: {e}")
            return False


class WebhookNotifier(Notifier):
    """Send alerts via webhook (Discord, Slack, custom)."""

    def __init__(self, webhook_url: str, platform: str = "custom"):
        self.webhook_url = webhook_url
        self.platform = platform

    async def send(self, alert: TradingAlert) -> bool:
        """Send alert via webhook."""
        try:
            payload = self._format_payload(alert)

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.webhook_url,
                    json=payload,
                    timeout=10.0,
                )
                response.raise_for_status()
                logger.info(f"Webhook alert sent: {alert.title}")
                return True

        except Exception as e:
            logger.error(f"Webhook send failed: {e}")
            return False

    def _format_payload(self, alert: TradingAlert) -> dict:
        """Format payload based on platform."""
        if self.platform == "discord":
            return {
                "content": alert.to_text(),
                "username": "StockAI Bot",
            }
        elif self.platform == "slack":
            return {
                "text": alert.to_text(),
            }
        else:
            return {
                "title": alert.title,
                "message": alert.message,
                "signal": alert.signal,
                "symbol": alert.symbol,
                "price": alert.price,
                "target": alert.target,
                "stop_loss": alert.stop_loss,
            }

    async def test_connection(self) -> bool:
        """Test webhook by sending a test message."""
        test_alert = TradingAlert(
            title="Connection Test",
            message="StockAI notification system connected successfully.",
            signal="ALERT",
        )
        return await self.send(test_alert)


class MultiNotifier(Notifier):
    """Send alerts to multiple channels."""

    def __init__(self, notifiers: list[Notifier]):
        self.notifiers = notifiers

    async def send(self, alert: TradingAlert) -> bool:
        """Send to all notifiers."""
        results = await asyncio.gather(
            *[n.send(alert) for n in self.notifiers],
            return_exceptions=True,
        )
        return any(r is True for r in results)

    async def test_connection(self) -> bool:
        """Test all notifier connections."""
        results = await asyncio.gather(
            *[n.test_connection() for n in self.notifiers],
            return_exceptions=True,
        )
        return any(r is True for r in results)
