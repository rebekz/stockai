"""StockAI CLI - Main Entry Point.

AI-Powered Indonesian Stock Analysis CLI.
Think "Claude Code for IDX investing."
"""

from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from stockai import __version__
from stockai.config import get_settings
from stockai.core.predictor import EnsemblePredictor
from stockai.data.database import init_database, get_db
from stockai.data.sources.yahoo import YahooFinanceSource
from stockai.data.sources.idx import IDXIndexSource, get_idx30, get_lq45

# Initialize Typer app
app = typer.Typer(
    name="stock",
    help="StockAI - AI-Powered Indonesian Stock Analysis CLI",
    add_completion=False,
    rich_markup_mode="rich",
)

console = Console()


def version_callback(value: bool) -> None:
    """Display version and exit."""
    if value:
        console.print(f"[bold blue]StockAI[/bold blue] version [green]{__version__}[/green]")
        raise typer.Exit()


@app.callback()
def main(
    version: Optional[bool] = typer.Option(
        None,
        "--version",
        "-v",
        help="Show version and exit.",
        callback=version_callback,
        is_eager=True,
    ),
) -> None:
    """StockAI - AI-Powered Indonesian Stock Analysis.

    An autonomous financial research agent for Indonesian stock market (IDX).
    Think "Claude Code for IDX investing."

    Examples:
        stock analyze BBCA
        stock predict TLKM --days 7
        stock portfolio add BBRI 100 14500
    """
    pass


@app.command("list")
def list_stocks(
    index: str = typer.Option("IDX30", "--index", "-i", help="Index to list (IDX30, LQ45)"),
    prices: bool = typer.Option(False, "--prices", "-p", help="Include current prices"),
) -> None:
    """List stocks in an index.

    Examples:
        stock list
        stock list --index LQ45
        stock list --prices
    """
    index = index.upper()
    console.print(f"\n[bold]Listing {index} stocks...[/bold]\n")

    idx_source = IDXIndexSource()

    if index == "IDX30":
        stocks = idx_source.get_idx30_stocks(include_prices=prices)
    elif index == "LQ45":
        stocks = idx_source.get_lq45_stocks(include_prices=prices)
    else:
        console.print(f"[red]Error:[/red] Unknown index {index}. Use IDX30 or LQ45.")
        raise typer.Exit(1)

    table = Table(title=f"📊 {index} Stocks ({len(stocks)} total)", show_header=True)
    table.add_column("#", style="dim", width=3)
    table.add_column("Symbol", style="cyan")

    if prices:
        table.add_column("Price", justify="right")
        table.add_column("Change", justify="right")
        table.add_column("Volume", justify="right", style="dim")

    for i, stock in enumerate(stocks, 1):
        row = [str(i), stock["symbol"]]

        if prices:
            price = stock.get("price")
            change_pct = stock.get("change_percent")

            if price:
                row.append(f"Rp {price:,.0f}")
            else:
                row.append("-")

            if change_pct is not None:
                color = "green" if change_pct >= 0 else "red"
                sign = "+" if change_pct >= 0 else ""
                row.append(f"[{color}]{sign}{change_pct:.2f}%[/{color}]")
            else:
                row.append("-")

            volume = stock.get("volume")
            row.append(f"{volume:,}" if volume else "-")

        table.add_row(*row)

    console.print(table)


@app.command("init")
def init_db() -> None:
    """Initialize the database.

    Creates all required tables if they don't exist.
    """
    try:
        init_database()
        settings = get_settings()
        console.print(f"[green]✓[/green] Database initialized at: {settings.db_full_path}")
    except Exception as e:
        console.print(f"[red]✗[/red] Failed to initialize database: {e}")
        raise typer.Exit(1)


@app.command()
def config() -> None:
    """Show current configuration status."""
    settings = get_settings()

    table = Table(title="StockAI Configuration", show_header=True)
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")
    table.add_column("Status", style="yellow")

    # API Keys status
    table.add_row(
        "Google API",
        "***" + settings.google_api_key[-4:] if settings.has_google_api else "Not set",
        "✓" if settings.has_google_api else "✗",
    )
    table.add_row(
        "Firecrawl API",
        "***" + settings.firecrawl_api_key[-4:] if settings.has_firecrawl_api else "Not set",
        "✓" if settings.has_firecrawl_api else "✗",
    )
    table.add_row(
        "OpenAI API",
        "***" + settings.openai_api_key[-4:] if settings.has_openai_api else "Not set",
        "✓" if settings.has_openai_api else "○",
    )
    table.add_row(
        "Anthropic API",
        "***" + settings.anthropic_api_key[-4:] if settings.has_anthropic_api else "Not set",
        "✓" if settings.has_anthropic_api else "○",
    )

    # Model settings
    table.add_row("Model", settings.model, "✓")
    table.add_row("Default Index", settings.default_index, "✓")
    table.add_row("Log Level", settings.log_level, "✓")
    table.add_row("Cache TTL", f"{settings.cache_ttl}s", "✓")
    table.add_row("Database", str(settings.db_full_path), "✓")

    console.print(table)


@app.command()
def tools() -> None:
    """List all available agent tools.

    Shows tools that the AI agent can use for research and analysis.

    Examples:
        stock tools
    """
    from stockai.tools import get_registry, register_stock_tools

    # Ensure tools are registered
    register_stock_tools()
    registry = get_registry()

    tool_list = registry.list_tools()

    if not tool_list:
        console.print("[yellow]No tools registered.[/yellow]")
        return

    table = Table(title="🔧 Available Agent Tools", show_header=True)
    table.add_column("Tool", style="cyan")
    table.add_column("Category", style="yellow")
    table.add_column("Permission", style="green")
    table.add_column("Description")

    for tool in sorted(tool_list, key=lambda x: (x.get("category", ""), x.get("name", ""))):
        perm = tool.get("permission", "safe")
        perm_icon = {"safe": "✅", "elevated": "⚠️", "dangerous": "🚫"}.get(perm, "")
        table.add_row(
            tool.get("name", ""),
            tool.get("category", "general"),
            f"{perm_icon} {perm}",
            tool.get("description", "")[:60],
        )

    console.print(table)
    console.print(f"\n[dim]Total: {len(tool_list)} tools[/dim]")


@app.command()
def info(
    symbol: str = typer.Argument(..., help="Stock symbol (e.g., BBCA, TLKM)"),
) -> None:
    """Get detailed information about a stock.

    Examples:
        stock info BBCA
        stock info TLKM
    """
    symbol = symbol.upper()
    console.print(f"\n[bold]Fetching info for {symbol}...[/bold]\n")

    idx_source = IDXIndexSource()
    stock_info = idx_source.get_stock_details(symbol)

    if not stock_info:
        console.print(f"[red]Error:[/red] Could not find stock {symbol}")
        raise typer.Exit(1)

    # Create info panel
    info_lines = []
    info_lines.append(f"[bold cyan]Company:[/bold cyan] {stock_info.get('name', 'N/A')}")
    info_lines.append(f"[bold cyan]Sector:[/bold cyan] {stock_info.get('sector', 'N/A')}")
    info_lines.append(f"[bold cyan]Industry:[/bold cyan] {stock_info.get('industry', 'N/A')}")
    info_lines.append("")

    # Index membership
    indices = []
    if stock_info.get("is_idx30"):
        indices.append("IDX30")
    if stock_info.get("is_lq45"):
        indices.append("LQ45")
    info_lines.append(f"[bold cyan]Index Membership:[/bold cyan] {', '.join(indices) if indices else 'None'}")
    info_lines.append("")

    # Price info
    current_price = stock_info.get("current_price")
    prev_close = stock_info.get("previous_close")
    if current_price:
        info_lines.append(f"[bold green]Current Price:[/bold green] Rp {current_price:,.0f}")
        if prev_close:
            change = current_price - prev_close
            change_pct = (change / prev_close) * 100
            color = "green" if change >= 0 else "red"
            sign = "+" if change >= 0 else ""
            info_lines.append(f"[bold {color}]Change:[/bold {color}] {sign}Rp {change:,.0f} ({sign}{change_pct:.2f}%)")
    info_lines.append("")

    # Volume
    volume = stock_info.get("volume")
    if volume:
        info_lines.append(f"[bold cyan]Volume:[/bold cyan] {volume:,}")

    # Market cap
    market_cap = stock_info.get("market_cap")
    if market_cap:
        if market_cap >= 1e12:
            cap_str = f"Rp {market_cap/1e12:.2f}T"
        else:
            cap_str = f"Rp {market_cap/1e9:.2f}B"
        info_lines.append(f"[bold cyan]Market Cap:[/bold cyan] {cap_str}")
    info_lines.append("")

    # 52-week range
    w52_high = stock_info.get("fifty_two_week_high")
    w52_low = stock_info.get("fifty_two_week_low")
    if w52_high and w52_low:
        info_lines.append(f"[bold cyan]52-Week Range:[/bold cyan] Rp {w52_low:,.0f} - Rp {w52_high:,.0f}")

    # Key metrics
    pe = stock_info.get("pe_ratio")
    pb = stock_info.get("pb_ratio")
    div_yield = stock_info.get("dividend_yield")

    metrics = []
    if pe:
        metrics.append(f"P/E: {pe:.2f}")
    if pb:
        metrics.append(f"P/B: {pb:.2f}")
    if div_yield:
        metrics.append(f"Div Yield: {div_yield*100:.2f}%")
    if metrics:
        info_lines.append(f"[bold cyan]Key Metrics:[/bold cyan] {' | '.join(metrics)}")

    console.print(Panel("\n".join(info_lines), title=f"📊 {symbol}", border_style="blue"))


@app.command()
def analyze(
    symbol: str = typer.Argument(..., help="Stock symbol to analyze"),
    deep: bool = typer.Option(False, "--deep", "-d", help="Perform deep analysis"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed execution"),
) -> None:
    """Analyze a stock using AI agent.

    The agent will research the stock, analyze fundamentals,
    technicals, and sentiment to provide insights.

    Examples:
        stock analyze BBCA
        stock analyze TLKM --deep
        stock analyze BBRI --verbose
    """
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.markdown import Markdown

    from stockai.agent import create_agent
    from stockai.tools import get_all_tools, register_stock_tools
    from stockai.config import get_settings

    symbol = symbol.upper()
    settings = get_settings()

    # Check API key
    if not settings.has_google_api:
        console.print("[red]Error:[/red] Google API key not configured.")
        console.print("Set GOOGLE_API_KEY in your .env file or environment.")
        raise typer.Exit(1)

    # Register tools
    register_stock_tools()
    tools = get_all_tools()

    mode = "deep" if deep else "standard"
    query = f"Analyze {symbol} stock with {'comprehensive technical and fundamental analysis' if deep else 'key metrics and current status'}"

    console.print(f"\n[bold]🤖 Analyzing {symbol}[/bold] ({mode} mode)\n")

    if verbose:
        console.print(f"[dim]Model: {settings.model}[/dim]")
        console.print(f"[dim]Tools: {', '.join(tools.keys())}[/dim]\n")

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            # Planning phase
            task = progress.add_task("Planning research...", total=None)

            agent = create_agent(tools=tools)

            progress.update(task, description="Executing analysis...")

            result = agent.run(query, symbol=symbol)

        if result.get("success"):
            answer = result.get("answer", "No analysis generated.")

            # Display as markdown
            console.print()
            md = Markdown(answer)
            console.print(md)

            # Show stats if verbose
            if verbose and result.get("duration"):
                console.print(f"\n[dim]Completed in {result['duration']:.1f}s[/dim]")
                console.print(f"[dim]Tool calls: {len(result.get('tool_results', []))}[/dim]")
        else:
            error = result.get("error", "Unknown error")
            console.print(f"[red]Analysis failed:[/red] {error}")
            raise typer.Exit(1)

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        if verbose:
            import traceback
            console.print(f"[dim]{traceback.format_exc()}[/dim]")
        raise typer.Exit(1)


@app.command()
def predict(
    symbol: str = typer.Argument(..., help="Stock symbol to predict"),
    horizon: int = typer.Option(3, "--horizon", "-h", help="Prediction horizon in days (1, 3, 7)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed model info"),
    show_accuracy: bool = typer.Option(
        False, "--show-accuracy", "-a", help="Show historical accuracy for this stock"
    ),
) -> None:
    """Predict stock price movement (UP/DOWN).

    Uses ML ensemble (XGBoost + LSTM + Sentiment) to
    predict stock direction.

    Examples:
        stock predict BBCA
        stock predict TLKM --horizon 7
        stock predict BBRI --verbose
        stock predict BBCA --show-accuracy
    """
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from pathlib import Path

    symbol = symbol.upper()
    settings = get_settings()

    # Validate horizon
    if horizon not in [1, 3, 7]:
        console.print(f"[yellow]Warning:[/yellow] Horizon {horizon} adjusted to nearest valid value (1, 3, or 7)")
        horizon = min([1, 3, 7], key=lambda x: abs(x - horizon))

    console.print(f"\n[bold]🔮 Predicting {symbol}[/bold] ({horizon}-day horizon)\n")

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task("Fetching price data...", total=None)

            # Get historical data
            yahoo = YahooFinanceSource()
            df = yahoo.get_price_history(symbol, period="6mo")

            if df.empty or len(df) < 50:
                console.print(f"[red]Error:[/red] Insufficient price data for {symbol}")
                console.print("Need at least 50 trading days for prediction.")
                raise typer.Exit(1)

            progress.update(task, description="Loading prediction models...")

            # Load ensemble predictor
            model_dir = settings.project_root / "data" / "models"
            xgb_path = model_dir / "xgboost_v1.json"
            lstm_path = model_dir / "lstm_v1.pt"

            ensemble = EnsemblePredictor(
                xgboost_path=xgb_path,
                lstm_path=lstm_path,
            )

            # Load models
            loaded = ensemble.load_models()
            active_models = sum(loaded.values())

            if active_models == 0:
                console.print("[yellow]Warning:[/yellow] No trained models found.")
                console.print("Run 'stock train' to train prediction models first.")
                console.print(
                    Panel(
                        f"[dim]Expected model locations:[/dim]\n"
                        f"  • XGBoost: {xgb_path}\n"
                        f"  • LSTM: {lstm_path}\n\n"
                        "[yellow]Showing placeholder prediction...[/yellow]",
                        title="⚠️ Models Not Trained",
                    )
                )
                # Show placeholder for demo purposes
                _show_placeholder_prediction(symbol, horizon)
                return

            progress.update(task, description="Generating prediction...")

            # Generate prediction
            result = ensemble.predict(df)

        # Display results
        _display_prediction_result(symbol, horizon, result, verbose)

        if verbose:
            # Show model status
            console.print("\n[bold]Model Status:[/bold]")
            for model, loaded_status in loaded.items():
                status = "[green]✓ Loaded[/green]" if loaded_status else "[red]✗ Not found[/red]"
                console.print(f"  • {model.upper()}: {status}")

        # Show historical accuracy if requested
        if show_accuracy:
            _display_stock_accuracy(symbol)

    except ImportError as e:
        console.print(f"[red]Error:[/red] Missing required package: {e}")
        console.print("Install with: pip install torch xgboost")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        if verbose:
            import traceback
            console.print(f"[dim]{traceback.format_exc()}[/dim]")
        raise typer.Exit(1)


def _show_placeholder_prediction(symbol: str, horizon: int) -> None:
    """Show placeholder when models aren't trained."""
    console.print(
        Panel(
            f"[bold cyan]Direction:[/bold cyan] [yellow]UNKNOWN[/yellow] (no models)\n"
            f"[bold cyan]Confidence:[/bold cyan] N/A\n"
            f"[bold cyan]Horizon:[/bold cyan] {horizon} days\n\n"
            "[dim]Train models to get real predictions:[/dim]\n"
            "  stock train --symbol BBCA",
            title=f"🔮 Prediction for {symbol}",
            border_style="yellow",
        )
    )


def _display_prediction_result(
    symbol: str,
    horizon: int,
    result: dict,
    verbose: bool,
) -> None:
    """Display prediction result with formatting."""
    direction = result.get("direction", "UNKNOWN")
    probability = result.get("probability", 0.5)
    confidence = result.get("confidence", 0)
    confidence_level = result.get("confidence_level", "LOW")
    agreement = result.get("model_agreement", False)

    # Direction styling
    if direction == "UP":
        dir_color = "green"
        dir_icon = "📈"
    else:
        dir_color = "red"
        dir_icon = "📉"

    # Confidence styling
    if confidence_level == "HIGH":
        conf_color = "green"
        conf_icon = "🟢"
    elif confidence_level == "MEDIUM":
        conf_color = "yellow"
        conf_icon = "🟡"
    else:
        conf_color = "red"
        conf_icon = "🔴"

    # Build result lines
    lines = []
    lines.append(f"[bold cyan]Direction:[/bold cyan] [{dir_color}]{dir_icon} {direction}[/{dir_color}]")
    lines.append(f"[bold cyan]Probability:[/bold cyan] {probability:.1%}")
    lines.append(f"[bold cyan]Confidence:[/bold cyan] [{conf_color}]{conf_icon} {confidence:.1%} ({confidence_level})[/{conf_color}]")
    lines.append(f"[bold cyan]Horizon:[/bold cyan] {horizon} days")
    lines.append(f"[bold cyan]Model Agreement:[/bold cyan] {'✓ Yes' if agreement else '✗ No'}")

    # Add contributions if verbose
    if verbose:
        contributions = result.get("contributions", {})
        lines.append("\n[bold]Model Contributions:[/bold]")
        for model, contrib in contributions.items():
            if "error" in contrib:
                lines.append(f"  • {model}: [red]Error - {contrib['error'][:30]}[/red]")
            elif "probability" in contrib:
                prob = contrib["probability"]
                weight = contrib.get("weight", 0)
                model_dir = "UP" if prob > 0.5 else "DOWN"
                lines.append(f"  • {model}: {model_dir} ({prob:.1%}) [dim]weight={weight:.0%}[/dim]")
            elif model == "sentiment" and contrib.get("score") is None:
                lines.append(f"  • {model}: [dim]Not available[/dim]")

    # Warning for low confidence
    if confidence_level == "LOW":
        lines.append("\n[yellow]⚠️ Low confidence prediction - use with caution[/yellow]")

    console.print(
        Panel(
            "\n".join(lines),
            title=f"🔮 Prediction for {symbol}",
            border_style=dir_color,
        )
    )


def _display_stock_accuracy(symbol: str) -> None:
    """Display historical accuracy for a stock.

    Shows accuracy metrics if available, warns if accuracy is low.
    """
    from stockai.data.database import init_database
    from stockai.core.predictor import PredictionAccuracyTracker

    # Low accuracy threshold for warning
    LOW_ACCURACY_THRESHOLD = 50.0

    init_database()
    tracker = PredictionAccuracyTracker()
    accuracy = tracker.get_stock_accuracy(symbol)

    total = accuracy.get("total_predictions", 0)
    correct = accuracy.get("correct_predictions", 0)
    accuracy_rate = accuracy.get("accuracy_rate", 0.0)

    # Check if we have any evaluated predictions
    if total == 0:
        console.print(
            Panel(
                "[dim]No historical accuracy data available for this stock.[/dim]\n\n"
                "Accuracy data is populated after predictions pass their target date.\n"
                "[dim]Run backfill to update:[/dim]\n"
                "  stock predictions backfill",
                title="📊 Historical Accuracy",
                border_style="dim",
            )
        )
        return

    # Determine styling based on accuracy
    if accuracy_rate >= 70:
        acc_color = "green"
        acc_icon = "🟢"
    elif accuracy_rate >= 50:
        acc_color = "yellow"
        acc_icon = "🟡"
    else:
        acc_color = "red"
        acc_icon = "🔴"

    # Build accuracy display lines
    lines = []
    lines.append(
        f"[bold cyan]Accuracy Rate:[/bold cyan] [{acc_color}]{acc_icon} {accuracy_rate:.1f}%[/{acc_color}]"
    )
    lines.append(f"[bold cyan]Predictions:[/bold cyan] {correct}/{total} correct")

    # Show direction breakdown if available
    by_direction = accuracy.get("by_direction", {})
    if by_direction:
        dir_parts = []
        for direction in ["UP", "DOWN", "NEUTRAL"]:
            stats = by_direction.get(direction, {})
            if stats.get("total", 0) > 0:
                dir_parts.append(f"{direction}: {stats['accuracy_rate']:.0f}%")
        if dir_parts:
            lines.append(f"[bold cyan]By Direction:[/bold cyan] {', '.join(dir_parts)}")

    # Show confidence breakdown if available
    by_confidence = accuracy.get("by_confidence", {})
    if by_confidence:
        conf_parts = []
        for level in ["HIGH", "MEDIUM", "LOW"]:
            stats = by_confidence.get(level, {})
            if stats.get("total", 0) > 0:
                conf_parts.append(f"{level}: {stats['accuracy_rate']:.0f}%")
        if conf_parts:
            lines.append(f"[bold cyan]By Confidence:[/bold cyan] {', '.join(conf_parts)}")

    # Add warning if accuracy is low
    if accuracy_rate < LOW_ACCURACY_THRESHOLD:
        lines.append(
            f"\n[red]⚠️ Warning: Historical accuracy is below {LOW_ACCURACY_THRESHOLD:.0f}%[/red]"
        )
        lines.append("[red]Consider this when making investment decisions.[/red]")

    console.print(
        Panel(
            "\n".join(lines),
            title=f"📊 Historical Accuracy for {symbol}",
            border_style=acc_color,
        )
    )


@app.command()
def volume(
    symbol: str = typer.Argument(..., help="Stock symbol to analyze"),
    period: str = typer.Option("3mo", "--period", "-p", help="Data period (1mo, 3mo, 6mo, 1y)"),
) -> None:
    """Analyze volume patterns and generate trading signals.

    Uses volume spikes, money flow, accumulation/distribution, and
    OBV trends to identify buy/sell signals.

    Examples:
        stock volume BBCA
        stock volume TLKM --period 6mo
    """
    from stockai.tools.stock_tools import get_volume_signals, get_volume_analysis

    symbol = symbol.upper()
    console.print(f"\n[bold]📊 Volume Analysis: {symbol}[/bold]\n")

    with console.status("[bold cyan]Analyzing volume patterns..."):
        # Get volume signals
        signals = get_volume_signals(symbol, period=period)

        if "error" in signals:
            console.print(f"[red]Error:[/red] {signals['error']}")
            raise typer.Exit(1)

        # Get detailed volume analysis
        analysis = get_volume_analysis(symbol, period=period)

    # Display overall signal
    overall = signals.get("overall_signal", {})
    direction = overall.get("direction", "neutral")
    confidence = overall.get("confidence", 0)

    dir_color = {"buy": "green", "sell": "red", "neutral": "yellow"}.get(direction, "white")
    dir_emoji = {"buy": "🟢", "sell": "🔴", "neutral": "🟡"}.get(direction, "⚪")

    # Signal summary panel
    lines = [
        f"{dir_emoji} [bold {dir_color}]{direction.upper()}[/bold {dir_color}] Signal",
        f"Confidence: [bold]{confidence:.0%}[/bold]",
        "",
        f"[dim]{overall.get('description', '')}[/dim]",
    ]

    console.print(Panel("\n".join(lines), title="Volume Signal", border_style=dir_color))

    # Volume metrics table
    table = Table(title="📈 Volume Metrics", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")
    table.add_column("Interpretation", style="dim")

    current_price = signals.get("current_price", 0)
    current_volume = signals.get("current_volume", 0)
    volume_ratio = signals.get("volume_ratio_20d", 1.0)

    table.add_row("Current Price", f"Rp {current_price:,.0f}", "")
    table.add_row("Current Volume", f"{current_volume:,.0f}", "")

    vol_interp = "High" if volume_ratio > 1.5 else ("Low" if volume_ratio < 0.7 else "Normal")
    table.add_row("Volume Ratio (20d)", f"{volume_ratio:.2f}x", vol_interp)

    if "error" not in analysis:
        vwap = analysis.get("vwap", 0)
        mfi = analysis.get("money_flow_index", 0)
        obv_trend = analysis.get("obv_trend", "neutral")

        table.add_row("VWAP", f"Rp {vwap:,.0f}", "Above = bullish" if current_price > vwap else "Below = bearish")
        mfi_interp = "Overbought" if mfi > 80 else ("Oversold" if mfi < 20 else "Neutral")
        table.add_row("Money Flow Index", f"{mfi:.1f}", mfi_interp)
        table.add_row("OBV Trend", obv_trend.capitalize(), "")

    console.print(table)

    # Individual signals
    individual = signals.get("individual_signals", [])
    if individual:
        console.print("\n[bold]Detected Signals:[/bold]")
        for sig in individual:
            sig_dir = sig.get("direction", "neutral")
            sig_emoji = {"bullish": "🟢", "bearish": "🔴", "neutral": "🟡"}.get(sig_dir, "⚪")
            sig_conf = sig.get("confidence", 0)
            console.print(f"  {sig_emoji} {sig.get('type', '')}: {sig.get('description', '')} ({sig_conf:.0%})")

    # Signal summary
    summary = signals.get("signal_summary", {})
    console.print(f"\n[dim]Bullish: {summary.get('bullish', 0)} | Bearish: {summary.get('bearish', 0)} | Neutral: {summary.get('neutral', 0)}[/dim]")


@app.command()
def suggest(
    index: str = typer.Option("IDX30", "--index", "-i", help="Index to scan (IDX30, LQ45)"),
    min_score: float = typer.Option(0.6, "--min-score", "-m", help="Minimum signal score (0-1)"),
    top: int = typer.Option(10, "--top", "-t", help="Number of top suggestions"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed analysis"),
) -> None:
    """Find stocks with buy signals using technical analysis.

    Scans index stocks and identifies those with bullish indicators
    based on RSI, MACD, and Bollinger Bands analysis.

    Examples:
        stock suggest
        stock suggest --index LQ45
        stock suggest --min-score 0.7 --top 5
    """
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
    import pandas as pd
    import numpy as np

    index = index.upper()
    console.print(f"\n[bold]🔍 Scanning {index} for Buy Signals[/bold]\n")

    # Get stock list
    idx_source = IDXIndexSource()
    if index == "IDX30":
        stocks = idx_source.get_idx30_stocks()
    elif index == "LQ45":
        stocks = idx_source.get_lq45_stocks()
    else:
        console.print(f"[red]Error:[/red] Unknown index {index}. Use IDX30 or LQ45.")
        raise typer.Exit(1)

    if not stocks:
        console.print(f"[red]Error:[/red] Could not fetch {index} stocks")
        raise typer.Exit(1)

    yahoo = YahooFinanceSource()
    results = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console,
    ) as progress:
        task = progress.add_task(f"Analyzing {len(stocks)} stocks...", total=len(stocks))

        for stock in stocks:
            symbol = stock["symbol"]
            try:
                # Get price history
                df = yahoo.get_price_history(symbol, period="3mo")
                if df.empty or len(df) < 20:
                    progress.advance(task)
                    continue

                # Calculate technical indicators
                signals = _calculate_buy_signals(df)
                signals["symbol"] = symbol
                signals["name"] = stock.get("name", symbol)

                if signals["score"] >= min_score:
                    results.append(signals)

            except Exception as e:
                if verbose:
                    console.print(f"[dim]Skipping {symbol}: {e}[/dim]")

            progress.advance(task)

    # Sort by score and take top N
    results = sorted(results, key=lambda x: x["score"], reverse=True)[:top]

    if not results:
        console.print(f"[yellow]No stocks found with signal score >= {min_score:.0%}[/yellow]")
        console.print("[dim]Try lowering --min-score or using a different index[/dim]")
        return

    # Display results
    console.print(f"\n[bold green]📈 Top {len(results)} Buy Signals[/bold green]\n")

    table = Table(show_header=True, title=f"🎯 {index} Buy Suggestions")
    table.add_column("#", style="dim", width=3)
    table.add_column("Symbol", style="cyan")
    table.add_column("Score", justify="center")
    table.add_column("RSI", justify="right")
    table.add_column("MACD", justify="center")
    table.add_column("BB", justify="center")
    table.add_column("Price", justify="right")
    table.add_column("Signal", justify="center")

    for i, r in enumerate(results, 1):
        # Score color
        score = r["score"]
        if score >= 0.8:
            score_str = f"[green]{score:.0%}[/green]"
            signal = "[bold green]STRONG BUY[/bold green]"
        elif score >= 0.6:
            score_str = f"[yellow]{score:.0%}[/yellow]"
            signal = "[yellow]BUY[/yellow]"
        else:
            score_str = f"[dim]{score:.0%}[/dim]"
            signal = "[dim]HOLD[/dim]"

        # RSI signal
        rsi = r.get("rsi", 50)
        if rsi < 30:
            rsi_str = f"[green]{rsi:.0f}[/green]"
        elif rsi > 70:
            rsi_str = f"[red]{rsi:.0f}[/red]"
        else:
            rsi_str = f"{rsi:.0f}"

        # MACD signal
        macd_signal = r.get("macd_signal", "NEUTRAL")
        if macd_signal == "BULLISH":
            macd_str = "[green]↑[/green]"
        elif macd_signal == "BEARISH":
            macd_str = "[red]↓[/red]"
        else:
            macd_str = "[dim]−[/dim]"

        # BB signal
        bb_signal = r.get("bb_signal", "NEUTRAL")
        if bb_signal == "OVERSOLD":
            bb_str = "[green]↑[/green]"
        elif bb_signal == "OVERBOUGHT":
            bb_str = "[red]↓[/red]"
        else:
            bb_str = "[dim]−[/dim]"

        price = r.get("current_price", 0)
        price_str = f"Rp {price:,.0f}" if price else "-"

        table.add_row(
            str(i),
            r["symbol"],
            score_str,
            rsi_str,
            macd_str,
            bb_str,
            price_str,
            signal,
        )

    console.print(table)

    # Legend
    console.print("\n[dim]Legend: RSI (Relative Strength Index), MACD (Moving Average Convergence Divergence), BB (Bollinger Bands)[/dim]")
    console.print("[dim]↑ = Bullish signal, ↓ = Bearish signal, − = Neutral[/dim]")

    # Detailed analysis for top picks
    if verbose and results:
        console.print("\n[bold]📊 Detailed Analysis[/bold]")
        for r in results[:3]:
            _display_stock_signals(r)


def _calculate_buy_signals(df: "pd.DataFrame") -> dict:
    """Calculate technical buy signals for a stock.

    Returns dict with score and individual indicator signals.
    """
    import numpy as np

    result = {
        "score": 0.0,
        "rsi": 50,
        "macd_signal": "NEUTRAL",
        "bb_signal": "NEUTRAL",
        "current_price": 0,
        "signals": [],
    }

    if len(df) < 20:
        return result

    close = df["close"].values
    result["current_price"] = float(close[-1])

    signals_count = 0
    total_weight = 0

    # 1. RSI (14-day)
    try:
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)

        avg_gain = np.convolve(gain, np.ones(14)/14, mode='valid')
        avg_loss = np.convolve(loss, np.ones(14)/14, mode='valid')

        if len(avg_gain) > 0 and len(avg_loss) > 0 and avg_loss[-1] != 0:
            rs = avg_gain[-1] / avg_loss[-1]
            rsi = 100 - (100 / (1 + rs))
        else:
            rsi = 50

        result["rsi"] = float(rsi)

        # RSI signal scoring
        if rsi < 30:  # Oversold - bullish
            signals_count += 1.0
            result["signals"].append("RSI oversold (<30)")
        elif rsi < 40:  # Near oversold - slightly bullish
            signals_count += 0.5
            result["signals"].append("RSI near oversold")
        elif rsi > 70:  # Overbought - bearish
            signals_count -= 0.5
        total_weight += 1.0
    except Exception:
        pass

    # 2. MACD
    try:
        ema12 = _ema(close, 12)
        ema26 = _ema(close, 26)

        if len(ema12) > 0 and len(ema26) > 0:
            macd_line = ema12[-1] - ema26[-1]
            macd_prev = ema12[-2] - ema26[-2] if len(ema12) > 1 else macd_line

            # Signal line (9-day EMA of MACD)
            macd_values = ema12[-9:] - ema26[-9:] if len(ema12) >= 9 else np.array([macd_line])
            signal_line = np.mean(macd_values)

            # MACD crossover detection
            if macd_line > signal_line and macd_prev <= signal_line:
                result["macd_signal"] = "BULLISH"
                signals_count += 1.0
                result["signals"].append("MACD bullish crossover")
            elif macd_line > signal_line:
                result["macd_signal"] = "BULLISH"
                signals_count += 0.5
            elif macd_line < signal_line and macd_prev >= signal_line:
                result["macd_signal"] = "BEARISH"
                signals_count -= 0.5
            elif macd_line < signal_line:
                result["macd_signal"] = "BEARISH"
                signals_count -= 0.25

            total_weight += 1.0
    except Exception:
        pass

    # 3. Bollinger Bands
    try:
        sma20 = np.convolve(close, np.ones(20)/20, mode='valid')
        if len(sma20) > 0:
            std20 = np.std(close[-20:])
            upper_band = sma20[-1] + (2 * std20)
            lower_band = sma20[-1] - (2 * std20)

            current = close[-1]

            if current <= lower_band:
                result["bb_signal"] = "OVERSOLD"
                signals_count += 1.0
                result["signals"].append("Price at lower Bollinger Band")
            elif current >= upper_band:
                result["bb_signal"] = "OVERBOUGHT"
                signals_count -= 0.5
            else:
                # Position within bands
                bb_pos = (current - lower_band) / (upper_band - lower_band)
                if bb_pos < 0.3:
                    signals_count += 0.5
                    result["signals"].append("Price near lower Bollinger Band")

            total_weight += 1.0
    except Exception:
        pass

    # 4. Price momentum (optional bonus)
    try:
        if len(close) >= 5:
            momentum = (close[-1] - close[-5]) / close[-5]
            if -0.05 < momentum < 0:  # Small recent dip - potential reversal
                signals_count += 0.3
            elif momentum > 0.02:  # Positive momentum
                signals_count += 0.2
    except Exception:
        pass

    # Calculate final score (normalized 0-1)
    if total_weight > 0:
        raw_score = (signals_count / total_weight + 1) / 2  # Normalize to 0-1
        result["score"] = max(0, min(1, raw_score))

    return result


def _ema(data: "np.ndarray", period: int) -> "np.ndarray":
    """Calculate Exponential Moving Average."""
    import numpy as np
    if len(data) < period:
        return np.array([])
    multiplier = 2 / (period + 1)
    ema = np.zeros(len(data))
    ema[period-1] = np.mean(data[:period])
    for i in range(period, len(data)):
        ema[i] = (data[i] - ema[i-1]) * multiplier + ema[i-1]
    return ema[period-1:]


def _display_stock_signals(signals: dict) -> None:
    """Display detailed signal analysis for a stock."""
    symbol = signals["symbol"]
    name = signals.get("name", symbol)

    lines = []
    lines.append(f"[bold cyan]Symbol:[/bold cyan] {symbol} ({name})")
    lines.append(f"[bold cyan]Price:[/bold cyan] Rp {signals.get('current_price', 0):,.0f}")
    lines.append(f"[bold cyan]Overall Score:[/bold cyan] {signals['score']:.0%}")
    lines.append("")
    lines.append("[bold]Indicators:[/bold]")
    lines.append(f"  • RSI: {signals.get('rsi', 50):.1f}")
    lines.append(f"  • MACD: {signals.get('macd_signal', 'NEUTRAL')}")
    lines.append(f"  • Bollinger: {signals.get('bb_signal', 'NEUTRAL')}")

    if signals.get("signals"):
        lines.append("")
        lines.append("[bold]Buy Signals:[/bold]")
        for sig in signals["signals"]:
            lines.append(f"  [green]✓[/green] {sig}")

    console.print(Panel("\n".join(lines), title=f"📊 {symbol}", border_style="blue"))


@app.command()
def train(
    symbol: str = typer.Option(None, "--symbol", "-s", help="Train on specific stock (default: IDX30)"),
    horizon: int = typer.Option(3, "--horizon", "-h", help="Prediction horizon in days"),
    force: bool = typer.Option(False, "--force", "-f", help="Force retrain even if models exist"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed training info"),
) -> None:
    """Train prediction models.

    Trains XGBoost and LSTM models on historical stock data.
    By default, trains on IDX30 stocks for generalization.

    Examples:
        stock train
        stock train --symbol BBCA
        stock train --horizon 7 --force
    """
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
    from pathlib import Path
    import time

    settings = get_settings()
    model_dir = settings.project_root / "data" / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    xgb_path = model_dir / "xgboost_v1.json"
    lstm_path = model_dir / "lstm_v1.pt"

    # Check if models exist
    if not force and xgb_path.exists() and lstm_path.exists():
        console.print("[green]✓[/green] Models already trained.")
        console.print(f"  XGBoost: {xgb_path}")
        console.print(f"  LSTM: {lstm_path}")
        console.print("\nUse --force to retrain.")
        return

    console.print(f"\n[bold]🏋️ Training Prediction Models[/bold]\n")
    console.print(f"  Horizon: {horizon} days")
    console.print(f"  Target: {'Single stock (' + symbol.upper() + ')' if symbol else 'IDX30 index (combined)'}")
    console.print()

    try:
        start_time = time.time()

        # Collect training data
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            if symbol:
                # Train on single stock
                symbols = [symbol.upper()]
            else:
                # Train on IDX30
                idx_source = IDXIndexSource()
                idx30_stocks = idx_source.get_idx30_stocks()
                symbols = [s["symbol"] for s in idx30_stocks[:10]]  # Use top 10 for faster training

            # Fetch historical data
            task = progress.add_task("Fetching historical data...", total=len(symbols))
            yahoo = YahooFinanceSource()

            all_data = []
            for sym in symbols:
                try:
                    df = yahoo.get_price_history(sym, period="2y")
                    if len(df) >= 200:  # Need at least 200 days
                        all_data.append(df)
                        if verbose:
                            console.print(f"  [dim]Fetched {sym}: {len(df)} days[/dim]")
                except Exception as e:
                    if verbose:
                        console.print(f"  [dim]Skipped {sym}: {e}[/dim]")
                progress.advance(task)

            if not all_data:
                console.print("[red]Error:[/red] No valid training data found.")
                raise typer.Exit(1)

            # Combine data
            import pandas as pd
            combined_df = pd.concat(all_data, ignore_index=True)
            console.print(f"\n[dim]Training on {len(combined_df)} samples from {len(all_data)} stocks[/dim]\n")

            # Initialize and train ensemble
            progress.update(task, description="Loading predictor...", completed=0, total=None)

            ensemble = EnsemblePredictor(
                xgboost_path=xgb_path,
                lstm_path=lstm_path,
            )

            # Train models
            progress.update(task, description="Training XGBoost model...")

            results = ensemble.train_all(
                combined_df,
                horizon=horizon,
                xgboost_params={"n_estimators": 100, "max_depth": 6},
                lstm_params={"epochs": 50, "patience": 10},
            )

            progress.update(task, description="Saving models...")

            # Save models
            save_results = ensemble.save_all()

        elapsed = time.time() - start_time

        # Display results
        console.print("\n[bold green]✓ Training Complete![/bold green]\n")

        # XGBoost results
        xgb_result = results.get("xgboost", {})
        if "error" not in xgb_result:
            console.print("[bold]XGBoost Results:[/bold]")
            console.print(f"  Train Accuracy: {xgb_result.get('train_accuracy', 0):.1%}")
            console.print(f"  Val Accuracy: {xgb_result.get('val_accuracy', 0):.1%}")
            if "val_auc" in xgb_result:
                console.print(f"  Val AUC: {xgb_result.get('val_auc', 0):.3f}")
            console.print(f"  Saved: {xgb_path}")
        else:
            console.print(f"[red]XGBoost training failed:[/red] {xgb_result['error']}")

        console.print()

        # LSTM results
        lstm_result = results.get("lstm", {})
        if "error" not in lstm_result:
            console.print("[bold]LSTM Results:[/bold]")
            console.print(f"  Train Accuracy: {lstm_result.get('train_accuracy', 0):.1%}")
            console.print(f"  Val Accuracy: {lstm_result.get('val_accuracy', 0):.1%}")
            console.print(f"  Epochs Trained: {lstm_result.get('epochs_trained', 0)}")
            console.print(f"  Saved: {lstm_path}")
        else:
            console.print(f"[red]LSTM training failed:[/red] {lstm_result['error']}")

        console.print(f"\n[dim]Completed in {elapsed:.1f}s[/dim]")

    except ImportError as e:
        console.print(f"[red]Error:[/red] Missing required package: {e}")
        console.print("Install with: pip install torch xgboost scikit-learn")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        if verbose:
            import traceback
            console.print(f"[dim]{traceback.format_exc()}[/dim]")
        raise typer.Exit(1)


@app.command()
def history(
    symbol: str = typer.Argument(..., help="Stock symbol"),
    period: str = typer.Option("1mo", "--period", "-p", help="Time period (1d,5d,1mo,3mo,6mo,1y)"),
    limit: int = typer.Option(20, "--limit", "-l", help="Number of rows to show"),
) -> None:
    """Show price history for a stock.

    Examples:
        stock history BBCA
        stock history TLKM --period 3mo
        stock history BBRI --limit 10
    """
    symbol = symbol.upper()
    console.print(f"\n[bold]Fetching price history for {symbol} ({period})...[/bold]\n")

    yahoo = YahooFinanceSource()
    df = yahoo.get_price_history(symbol, period=period)

    if df.empty:
        console.print(f"[red]Error:[/red] No price data found for {symbol}")
        raise typer.Exit(1)

    # Create table
    table = Table(title=f"📈 {symbol} Price History ({period})", show_header=True)
    table.add_column("Date", style="cyan")
    table.add_column("Open", justify="right")
    table.add_column("High", justify="right", style="green")
    table.add_column("Low", justify="right", style="red")
    table.add_column("Close", justify="right", style="bold")
    table.add_column("Volume", justify="right", style="dim")

    # Show most recent first, limited rows
    df_display = df.sort_values("date", ascending=False).head(limit)

    for _, row in df_display.iterrows():
        date_str = row["date"].strftime("%Y-%m-%d")
        table.add_row(
            date_str,
            f"Rp {row['open']:,.0f}",
            f"Rp {row['high']:,.0f}",
            f"Rp {row['low']:,.0f}",
            f"Rp {row['close']:,.0f}",
            f"{row['volume']:,}",
        )

    console.print(table)

    # Summary stats
    if len(df) > 1:
        first_close = df.iloc[0]["close"]
        last_close = df.iloc[-1]["close"]
        change = last_close - first_close
        change_pct = (change / first_close) * 100

        avg_vol = df["volume"].mean()
        high_price = df["high"].max()
        low_price = df["low"].min()

        console.print(f"\n[bold]Summary ({len(df)} days):[/bold]")
        color = "green" if change >= 0 else "red"
        sign = "+" if change >= 0 else ""
        console.print(f"  Period Change: [{color}]{sign}Rp {change:,.0f} ({sign}{change_pct:.2f}%)[/{color}]")
        console.print(f"  High: [green]Rp {high_price:,.0f}[/green] | Low: [red]Rp {low_price:,.0f}[/red]")
        console.print(f"  Avg Volume: {avg_vol:,.0f}")


# Portfolio subcommand group
portfolio_app = typer.Typer(help="Manage your stock portfolio")
app.add_typer(portfolio_app, name="portfolio")


@portfolio_app.command("list")
def portfolio_list(
    prices: bool = typer.Option(True, "--prices/--no-prices", "-p/-P", help="Include current prices and P&L"),
) -> None:
    """List all stocks in portfolio.

    Shows holdings with current values, P&L, and allocation.

    Examples:
        stock portfolio list
        stock portfolio list --no-prices
    """
    from stockai.data.database import init_database
    from stockai.core.portfolio import PnLCalculator

    init_database()
    pnl_calc = PnLCalculator()

    if prices:
        # Get full P&L summary
        summary = pnl_calc.get_portfolio_summary()
        positions = summary.get("positions", [])
    else:
        # Just get positions without prices
        from stockai.core.portfolio import PortfolioManager
        manager = PortfolioManager()
        positions = manager.get_positions()

    if not positions:
        console.print(
            Panel(
                "[dim]No positions in portfolio.[/dim]\n\n"
                "Add positions with:\n"
                "  stock portfolio add BBCA 100 9500",
                title="💼 Portfolio",
            )
        )
        return

    table = Table(title=f"💼 Portfolio ({len(positions)} positions)", show_header=True)
    table.add_column("Symbol", style="cyan")
    table.add_column("Shares", justify="right")
    table.add_column("Avg Cost", justify="right")
    table.add_column("Cost Basis", justify="right")

    if prices:
        table.add_column("Price", justify="right")
        table.add_column("Value", justify="right")
        table.add_column("P&L", justify="right")
        table.add_column("%", justify="right")
        table.add_column("Alloc", justify="right", style="dim")

    total_cost = 0
    total_value = 0
    total_pnl = 0

    for pos in positions:
        row = [
            pos.get("symbol"),
            f"{pos.get('shares', 0):,}",
            f"Rp {pos.get('avg_cost', 0):,.0f}",
            f"Rp {pos.get('cost_basis', 0):,.0f}",
        ]

        if prices:
            current_price = pos.get("current_price")
            market_value = pos.get("market_value")
            unrealized_pnl = pos.get("unrealized_pnl")
            pnl_percent = pos.get("pnl_percent", 0)
            allocation = pos.get("allocation_percent", 0)

            if current_price:
                row.append(f"Rp {current_price:,.0f}")
            else:
                row.append("[dim]-[/dim]")

            if market_value:
                row.append(f"Rp {market_value:,.0f}")
                total_value += market_value
            else:
                row.append("[dim]-[/dim]")

            if unrealized_pnl is not None:
                color = "green" if unrealized_pnl >= 0 else "red"
                sign = "+" if unrealized_pnl >= 0 else ""
                row.append(f"[{color}]{sign}Rp {unrealized_pnl:,.0f}[/{color}]")
                row.append(f"[{color}]{sign}{pnl_percent:.1f}%[/{color}]")
                total_pnl += unrealized_pnl
            else:
                row.extend(["[dim]-[/dim]", "[dim]-[/dim]"])

            row.append(f"{allocation:.1f}%")

        total_cost += pos.get("cost_basis", 0)
        table.add_row(*row)

    console.print(table)

    # Summary
    if prices and total_cost > 0:
        pnl_pct = (total_pnl / total_cost) * 100
        color = "green" if total_pnl >= 0 else "red"
        sign = "+" if total_pnl >= 0 else ""

        console.print(f"\n[bold]Summary:[/bold]")
        console.print(f"  Total Cost:  Rp {total_cost:,.0f}")
        console.print(f"  Total Value: Rp {total_value:,.0f}")
        console.print(f"  Total P&L:   [{color}]{sign}Rp {total_pnl:,.0f} ({sign}{pnl_pct:.1f}%)[/{color}]")


@portfolio_app.command("add")
def portfolio_add(
    symbol: str = typer.Argument(..., help="Stock symbol"),
    shares: int = typer.Argument(..., help="Number of shares"),
    price: float = typer.Argument(..., help="Purchase price per share"),
    notes: str = typer.Option(None, "--notes", "-n", help="Transaction notes"),
) -> None:
    """Add shares to portfolio (buy).

    If you already own the stock, this adds to your position
    and updates the average cost.

    Examples:
        stock portfolio add BBCA 100 9500
        stock portfolio add TLKM 500 3400 --notes "DCA"
    """
    from stockai.data.database import init_database
    from stockai.core.portfolio import PortfolioManager

    init_database()
    manager = PortfolioManager()

    try:
        result = manager.add_position(
            symbol=symbol,
            shares=shares,
            price=price,
            notes=notes,
        )

        console.print(f"\n[green]✓ Added position[/green]\n")
        console.print(f"  Symbol:       {result['symbol']}")
        console.print(f"  Shares Added: {result['shares']:,}")
        console.print(f"  Price:        Rp {result['price']:,.0f}")
        console.print(f"  Total Cost:   Rp {result['total_cost']:,.0f}")
        console.print()
        console.print(f"  [bold]Total Shares:[/bold] {result['total_shares']:,}")
        console.print(f"  [bold]Avg Price:[/bold]    Rp {result['avg_price']:,.0f}")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@portfolio_app.command("sell")
def portfolio_sell(
    symbol: str = typer.Argument(..., help="Stock symbol"),
    shares: int = typer.Option(None, "--shares", "-s", help="Shares to sell (all if not specified)"),
    price: float = typer.Option(None, "--price", "-p", help="Sale price per share"),
    notes: str = typer.Option(None, "--notes", "-n", help="Transaction notes"),
) -> None:
    """Sell shares from portfolio.

    If shares not specified, sells entire position.

    Examples:
        stock portfolio sell BBCA --shares 50 --price 10000
        stock portfolio sell TLKM  # Sells all
    """
    from stockai.data.database import init_database
    from stockai.core.portfolio import PortfolioManager

    init_database()
    manager = PortfolioManager()

    try:
        result = manager.remove_position(
            symbol=symbol,
            shares=shares,
            price=price,
            notes=notes,
        )

        console.print(f"\n[green]✓ Sold position[/green]\n")
        console.print(f"  Symbol:     {result['symbol']}")
        console.print(f"  Shares:     {result['shares']:,}")
        console.print(f"  Price:      Rp {result['price']:,.0f}")
        console.print(f"  Sale Value: Rp {result['sale_value']:,.0f}")
        console.print()

        # P&L
        pnl = result.get("realized_pnl", 0)
        pnl_pct = result.get("pnl_percent", 0)
        color = "green" if pnl >= 0 else "red"
        sign = "+" if pnl >= 0 else ""
        console.print(f"  [bold]Realized P&L:[/bold] [{color}]{sign}Rp {pnl:,.0f} ({sign}{pnl_pct:.1f}%)[/{color}]")

        if result.get("position_closed"):
            console.print(f"\n  [dim]Position closed[/dim]")
        else:
            console.print(f"\n  Remaining: {result.get('remaining_shares', 0):,} shares")

    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@portfolio_app.command("remove")
def portfolio_remove(
    symbol: str = typer.Argument(..., help="Stock symbol to remove"),
) -> None:
    """Remove entire position from portfolio (alias for sell all).

    Examples:
        stock portfolio remove BBCA
    """
    # Delegate to sell with no shares specified
    portfolio_sell(symbol=symbol, shares=None, price=None, notes="Position removed")


@portfolio_app.command("pnl")
def portfolio_pnl(
    symbol: str = typer.Option(None, "--symbol", "-s", help="Filter by symbol"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed breakdown"),
) -> None:
    """Show portfolio P&L (profit/loss).

    Examples:
        stock portfolio pnl
        stock portfolio pnl --symbol BBCA
        stock portfolio pnl --verbose
    """
    from stockai.data.database import init_database
    from stockai.core.portfolio import PnLCalculator

    init_database()
    pnl_calc = PnLCalculator()

    if symbol:
        # Single position P&L
        result = pnl_calc.calculate_position_pnl(symbol)

        if result.get("error"):
            console.print(f"[red]Error:[/red] {result['error']}")
            raise typer.Exit(1)

        console.print(
            Panel(
                f"[bold cyan]Symbol:[/bold cyan] {result['symbol']}\n"
                f"[bold cyan]Shares:[/bold cyan] {result.get('shares', 0):,}\n"
                f"[bold cyan]Avg Cost:[/bold cyan] Rp {result.get('avg_cost', 0):,.0f}\n"
                f"[bold cyan]Cost Basis:[/bold cyan] Rp {result.get('cost_basis', 0):,.0f}\n\n"
                f"[bold cyan]Current Price:[/bold cyan] Rp {result.get('current_price', 0):,.0f}\n"
                f"[bold cyan]Market Value:[/bold cyan] Rp {result.get('market_value', 0):,.0f}\n\n"
                f"[bold {'green' if result.get('is_profit') else 'red'}]"
                f"P&L: {'+' if result.get('is_profit') else ''}Rp {result.get('unrealized_pnl', 0):,.0f} "
                f"({'+' if result.get('is_profit') else ''}{result.get('pnl_percent', 0):.1f}%)"
                f"[/bold {'green' if result.get('is_profit') else 'red'}]",
                title=f"💰 P&L for {symbol.upper()}",
            )
        )
    else:
        # Full portfolio P&L
        summary = pnl_calc.get_portfolio_summary()

        s = summary.get("summary", {})
        positions = summary.get("positions", [])

        if not positions:
            console.print("[dim]No positions in portfolio.[/dim]")
            return

        # Summary panel
        pnl = s.get("total_unrealized_pnl", 0)
        pnl_pct = s.get("total_pnl_percent", 0)
        color = "green" if s.get("is_profit") else "red"
        sign = "+" if pnl >= 0 else ""

        console.print(
            Panel(
                f"[bold]Positions:[/bold] {s.get('position_count', 0)}\n"
                f"[bold]Total Cost:[/bold] Rp {s.get('total_cost_basis', 0):,.0f}\n"
                f"[bold]Market Value:[/bold] Rp {s.get('total_market_value', 0):,.0f}\n\n"
                f"[bold {color}]Unrealized P&L: {sign}Rp {pnl:,.0f} ({sign}{pnl_pct:.1f}%)[/bold {color}]\n"
                f"[bold cyan]Realized P&L:[/bold cyan] Rp {s.get('total_realized_pnl', 0):,.0f}",
                title="💰 Portfolio P&L",
            )
        )

        if verbose:
            # Best/worst performers
            best = summary.get("best_performer")
            worst = summary.get("worst_performer")

            if best:
                console.print(f"\n[green]Best:[/green] {best.get('symbol')} (+{best.get('pnl_percent', 0):.1f}%)")
            if worst:
                console.print(f"[red]Worst:[/red] {worst.get('symbol')} ({worst.get('pnl_percent', 0):.1f}%)")


@portfolio_app.command("transactions")
def portfolio_transactions(
    symbol: str = typer.Option(None, "--symbol", "-s", help="Filter by symbol"),
    limit: int = typer.Option(20, "--limit", "-l", help="Max transactions to show"),
) -> None:
    """Show transaction history.

    Examples:
        stock portfolio transactions
        stock portfolio transactions --symbol BBCA
    """
    from stockai.data.database import init_database
    from stockai.core.portfolio import PortfolioManager

    init_database()
    manager = PortfolioManager()

    transactions = manager.get_transactions(symbol=symbol, limit=limit)

    if not transactions:
        console.print("[dim]No transactions found.[/dim]")
        return

    table = Table(title="📋 Transaction History", show_header=True)
    table.add_column("Date", style="dim")
    table.add_column("Symbol", style="cyan")
    table.add_column("Type")
    table.add_column("Shares", justify="right")
    table.add_column("Price", justify="right")
    table.add_column("Total", justify="right")

    for txn in transactions:
        txn_type = txn.get("type")
        type_style = "green" if txn_type == "BUY" else "red"

        # Format date
        date_str = txn.get("date", "")[:10]

        table.add_row(
            date_str,
            txn.get("symbol"),
            f"[{type_style}]{txn_type}[/{type_style}]",
            f"{txn.get('shares', 0):,}",
            f"Rp {txn.get('price', 0):,.0f}",
            f"Rp {txn.get('total', 0):,.0f}",
        )

    console.print(table)


@portfolio_app.command("analyze")
def portfolio_analyze(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show full analysis"),
) -> None:
    """AI-powered portfolio analysis.

    Analyzes concentration, sector allocation, volatility,
    and provides recommendations.

    Examples:
        stock portfolio analyze
        stock portfolio analyze --verbose
    """
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from stockai.data.database import init_database
    from stockai.core.portfolio import PortfolioAnalytics

    init_database()
    analytics = PortfolioAnalytics()

    console.print("\n[bold]🔍 Analyzing Portfolio...[/bold]\n")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Running analysis...", total=None)

        analysis = analytics.get_full_analysis()
        insights = analytics.generate_ai_insights(analysis)

    # Display health score
    score = analysis.get("overall_score", 0)
    health = analysis.get("health_status", "UNKNOWN")

    if health == "EXCELLENT":
        health_color = "green"
        health_icon = "🟢"
    elif health == "GOOD":
        health_color = "cyan"
        health_icon = "🔵"
    elif health == "NEEDS_ATTENTION":
        health_color = "yellow"
        health_icon = "🟡"
    else:
        health_color = "red"
        health_icon = "🔴"

    console.print(
        Panel(
            f"[bold]Overall Score:[/bold] {score:.0%}\n"
            f"[bold]Health Status:[/bold] [{health_color}]{health_icon} {health}[/{health_color}]",
            title="📊 Portfolio Health",
        )
    )

    # Concentration
    conc = analysis.get("concentration", {})
    console.print(f"\n[bold]Concentration Risk:[/bold] {conc.get('risk_level', 'N/A')}")
    console.print(f"  HHI Index: {conc.get('hhi_index', 0):.0f}")

    top_holdings = conc.get("top_holdings", [])[:3]
    if top_holdings:
        console.print("  Top Holdings:")
        for h in top_holdings:
            console.print(f"    • {h.get('symbol')}: {h.get('allocation', 0):.1f}%")

    # Sectors
    sectors = analysis.get("sector_allocation", {})
    console.print(f"\n[bold]Sector Diversification:[/bold] {sectors.get('diversification_level', 'N/A')}")
    console.print(f"  Sector Count: {sectors.get('sector_count', 0)}")

    # Volatility
    vol = analysis.get("volatility", {})
    console.print(f"\n[bold]Portfolio Volatility:[/bold] {vol.get('risk_level', 'N/A')}")
    console.print(f"  Annual Vol: {vol.get('portfolio_volatility', 0):.1f}%")

    # AI Insights
    if insights:
        console.print("\n[bold]🤖 AI Insights:[/bold]")
        for insight in insights[:5]:
            console.print(f"  • {insight}")

    # Recommendations
    recs = analysis.get("recommendations", [])
    if recs:
        console.print("\n[bold]📋 Recommendations:[/bold]")
        for rec in recs[:5]:
            console.print(f"  [yellow]→[/yellow] {rec}")


# Sentiment subcommand group
sentiment_app = typer.Typer(help="Stock sentiment analysis")
app.add_typer(sentiment_app, name="sentiment")


@sentiment_app.command("analyze")
def sentiment_analyze(
    symbol: str = typer.Argument(..., help="Stock symbol to analyze"),
    days: int = typer.Option(7, "--days", "-d", help="Days of news to analyze"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed analysis"),
) -> None:
    """Analyze news sentiment for a stock.

    Fetches recent news and analyzes sentiment using
    multilingual transformer models.

    Examples:
        stock sentiment analyze BBCA
        stock sentiment analyze TLKM --days 14
    """
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from stockai.core.sentiment import get_sentiment_analyzer, NewsAggregator

    symbol = symbol.upper()
    console.print(f"\n[bold]🎯 Analyzing sentiment for {symbol}[/bold]\n")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Fetching news...", total=None)

        # Fetch news
        news_agg = NewsAggregator()
        articles = news_agg.fetch_all(symbol, max_articles=15, days_back=days)

        if not articles:
            console.print(f"[yellow]No recent news found for {symbol}[/yellow]")
            console.print("[dim]Try a major stock like BBCA, BBRI, or TLKM[/dim]")
            return

        progress.update(task, description="Analyzing sentiment...")

        # Analyze sentiment (uses Gemini if available, otherwise fallback)
        analyzer = get_sentiment_analyzer()
        aggregated = analyzer.aggregate_sentiment(articles, symbol)

    # Display results
    score = aggregated.avg_sentiment_score
    if score > 0.2:
        sentiment_color = "green"
        sentiment_icon = "📈"
    elif score < -0.2:
        sentiment_color = "red"
        sentiment_icon = "📉"
    else:
        sentiment_color = "yellow"
        sentiment_icon = "➡️"

    # Build summary
    lines = []
    lines.append(f"[bold cyan]Symbol:[/bold cyan] {aggregated.symbol}")
    lines.append(f"[bold cyan]Articles Analyzed:[/bold cyan] {aggregated.article_count}")
    lines.append("")
    lines.append(f"[bold cyan]Sentiment Score:[/bold cyan] [{sentiment_color}]{sentiment_icon} {score:+.2f}[/{sentiment_color}]")
    lines.append(f"[bold cyan]Dominant Sentiment:[/bold cyan] [{sentiment_color}]{aggregated.dominant_label.value}[/{sentiment_color}]")
    lines.append(f"[bold cyan]Signal Strength:[/bold cyan] {aggregated.signal_strength}")
    lines.append(f"[bold cyan]Confidence:[/bold cyan] {aggregated.confidence:.0%}")
    lines.append("")
    lines.append(f"[green]Bullish:[/green] {aggregated.bullish_count} | [red]Bearish:[/red] {aggregated.bearish_count} | [dim]Neutral:[/dim] {aggregated.neutral_count}")

    console.print(
        Panel(
            "\n".join(lines),
            title=f"🎯 Sentiment Analysis: {symbol}",
            border_style=sentiment_color,
        )
    )

    # Show articles if verbose
    if verbose and articles:
        console.print("\n[bold]📰 Recent News:[/bold]")
        table = Table(show_header=True)
        table.add_column("Sentiment", width=8)
        table.add_column("Title")
        table.add_column("Source", style="dim")

        for article in articles[:10]:
            if article.sentiment:
                label = article.sentiment.label.value
                if label == "BULLISH":
                    sent_str = "[green]BULLISH[/green]"
                elif label == "BEARISH":
                    sent_str = "[red]BEARISH[/red]"
                else:
                    sent_str = "[dim]NEUTRAL[/dim]"
            else:
                sent_str = "[dim]?[/dim]"

            title = article.title[:60] + "..." if len(article.title) > 60 else article.title
            table.add_row(sent_str, title, article.source)

        console.print(table)


@sentiment_app.command("news")
def sentiment_news(
    symbol: str = typer.Argument(..., help="Stock symbol"),
    limit: int = typer.Option(10, "--limit", "-l", help="Max news articles"),
) -> None:
    """Fetch recent news for a stock.

    Examples:
        stock sentiment news BBCA
        stock sentiment news TLKM --limit 20
    """
    from stockai.core.sentiment import NewsAggregator

    symbol = symbol.upper()
    console.print(f"\n[bold]📰 Fetching news for {symbol}...[/bold]\n")

    news_agg = NewsAggregator()
    articles = news_agg.fetch_all(symbol, max_articles=limit)

    if not articles:
        console.print(f"[yellow]No recent news found for {symbol}[/yellow]")
        return

    table = Table(title=f"📰 News for {symbol} ({len(articles)} articles)", show_header=True)
    table.add_column("Date", style="dim", width=10)
    table.add_column("Title")
    table.add_column("Source", style="cyan", width=15)

    for article in articles:
        date_str = article.published_at.strftime("%Y-%m-%d") if article.published_at else "Unknown"
        title = article.title[:70] + "..." if len(article.title) > 70 else article.title
        table.add_row(date_str, title, article.source)

    console.print(table)


@sentiment_app.command("market")
def sentiment_market() -> None:
    """Analyze overall market sentiment.

    Fetches and analyzes general IHSG/IDX market news.

    Examples:
        stock sentiment market
    """
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from stockai.core.sentiment import get_sentiment_analyzer, NewsAggregator

    console.print(f"\n[bold]🏛️ Analyzing Market Sentiment (IHSG)...[/bold]\n")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Fetching market news...", total=None)

        news_agg = NewsAggregator()
        articles = news_agg.get_market_news(max_articles=10)

        if not articles:
            console.print("[yellow]No recent market news found[/yellow]")
            return

        progress.update(task, description="Analyzing sentiment...")

        # Uses Gemini if available, otherwise fallback
        analyzer = get_sentiment_analyzer()
        aggregated = analyzer.aggregate_sentiment(articles, "IHSG")

    # Display
    score = aggregated.avg_sentiment_score
    if score > 0.2:
        color = "green"
        icon = "📈"
    elif score < -0.2:
        color = "red"
        icon = "📉"
    else:
        color = "yellow"
        icon = "➡️"

    console.print(
        Panel(
            f"[bold cyan]Market:[/bold cyan] IHSG (Indonesian Composite Index)\n"
            f"[bold cyan]Articles:[/bold cyan] {aggregated.article_count}\n\n"
            f"[bold cyan]Sentiment:[/bold cyan] [{color}]{icon} {aggregated.dominant_label.value}[/{color}]\n"
            f"[bold cyan]Score:[/bold cyan] [{color}]{score:+.2f}[/{color}]\n"
            f"[bold cyan]Signal:[/bold cyan] {aggregated.signal_strength}",
            title="🏛️ Market Sentiment",
            border_style=color,
        )
    )


# Multi-Agent Trading subcommand group
agents_app = typer.Typer(help="Multi-agent trading analysis system")
app.add_typer(agents_app, name="agents")


@agents_app.command("analyze")
def agents_analyze(
    symbol: str = typer.Argument(..., help="Stock symbol to analyze"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed agent outputs"),
) -> None:
    """Run full multi-agent analysis on a stock.

    Uses 7 specialized AI agents to analyze:
    - Fundamentals (research agent)
    - Technical indicators (technical analyst)
    - News sentiment (sentiment analyst)
    - Portfolio fit (portfolio manager)
    - Risk assessment (risk manager)
    - Final trading signal (trading execution)

    Examples:
        stock agents analyze BBCA
        stock agents analyze TLKM --verbose
    """
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.markdown import Markdown

    from stockai.agents import create_trading_orchestrator
    from stockai.config import get_settings

    symbol = symbol.upper()
    settings = get_settings()

    # Check API key
    if not settings.has_google_api:
        console.print("[red]Error:[/red] Google API key not configured.")
        console.print("Set GOOGLE_API_KEY in your .env file or environment.")
        raise typer.Exit(1)

    console.print(f"\n[bold]🤖 Multi-Agent Analysis: {symbol}[/bold]\n")
    console.print("[dim]Running 7 specialized agents for comprehensive analysis...[/dim]\n")

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=not verbose,
        ) as progress:
            task = progress.add_task("Initializing orchestrator...", total=None)

            orchestrator = create_trading_orchestrator()

            progress.update(task, description="Running multi-agent analysis...")

            result = orchestrator.run(
                query=f"Provide comprehensive trading analysis for {symbol}",
                symbol=symbol,
            )

        if result.get("success"):
            # Display recommendation
            recommendation = result.get("recommendation")
            score = result.get("composite_score", 0)

            if recommendation:
                if "BUY" in recommendation:
                    rec_color = "green"
                elif "SELL" in recommendation:
                    rec_color = "red"
                else:
                    rec_color = "yellow"

                console.print(
                    Panel(
                        f"[bold {rec_color}]{recommendation}[/bold {rec_color}]\n\n"
                        f"[bold cyan]Composite Score:[/bold cyan] {score:.1f}/10\n"
                        f"[bold cyan]Agents Executed:[/bold cyan] {len(result.get('agents_executed', []))}",
                        title=f"🎯 {symbol} Trading Signal",
                        border_style=rec_color,
                    )
                )

            # Display full analysis
            answer = result.get("answer", "")
            if answer:
                console.print()
                md = Markdown(answer)
                console.print(md)

            # Stats
            if verbose:
                console.print(f"\n[dim]Started: {result.get('started_at')}[/dim]")
                console.print(f"[dim]Completed: {result.get('completed_at')}[/dim]")
                console.print(f"[dim]Agents: {', '.join(result.get('agents_executed', []))}[/dim]")
        else:
            error = result.get("error", "Unknown error")
            console.print(f"[red]Analysis failed:[/red] {error}")
            raise typer.Exit(1)

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        if verbose:
            import traceback
            console.print(f"[dim]{traceback.format_exc()}[/dim]")
        raise typer.Exit(1)


@agents_app.command("scan")
def agents_scan(
    index: str = typer.Option("IDX30", "--index", "-i", help="Index to scan (IDX30, LQ45)"),
    top: int = typer.Option(5, "--top", "-t", help="Number of top opportunities"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed outputs"),
) -> None:
    """Scan market for trading opportunities.

    Uses Market Scanner agent to identify:
    - Volume spikes
    - Breakout patterns
    - Sector rotations
    - Momentum plays

    Examples:
        stock agents scan
        stock agents scan --index LQ45 --top 10
    """
    from rich.progress import Progress, SpinnerColumn, TextColumn

    from stockai.agents import create_trading_orchestrator
    from stockai.config import get_settings

    index = index.upper()
    settings = get_settings()

    if not settings.has_google_api:
        console.print("[red]Error:[/red] Google API key not configured.")
        raise typer.Exit(1)

    console.print(f"\n[bold]🔍 Scanning {index} for Opportunities[/bold]\n")

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task("Running Market Scanner...", total=None)

            orchestrator = create_trading_orchestrator()

            result = orchestrator.run(
                query=f"Scan {index} for top {top} trading opportunities with volume spikes, breakouts, or momentum signals",
            )

        if result.get("success"):
            answer = result.get("answer", "No opportunities found.")
            from rich.markdown import Markdown
            console.print(Markdown(answer))
        else:
            console.print(f"[red]Scan failed:[/red] {result.get('error', 'Unknown error')}")
            raise typer.Exit(1)

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@agents_app.command("recommend")
def agents_recommend(
    symbol: str = typer.Argument(..., help="Stock symbol"),
) -> None:
    """Get buy/sell recommendation for a stock.

    Runs full agent pipeline and provides:
    - Clear BUY/SELL/HOLD recommendation
    - Entry and exit points
    - Position sizing suggestion
    - Stop-loss levels

    Examples:
        stock agents recommend BBCA
        stock agents recommend TLKM
    """
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.markdown import Markdown

    from stockai.agents import create_trading_orchestrator
    from stockai.config import get_settings

    symbol = symbol.upper()
    settings = get_settings()

    if not settings.has_google_api:
        console.print("[red]Error:[/red] Google API key not configured.")
        raise typer.Exit(1)

    console.print(f"\n[bold]💡 Getting Trading Recommendation: {symbol}[/bold]\n")

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task("Analyzing with trading agents...", total=None)

            orchestrator = create_trading_orchestrator()

            result = orchestrator.run(
                query=f"Should I buy or sell {symbol}? Provide specific entry/exit points, position size, and stop-loss.",
                symbol=symbol,
            )

        if result.get("success"):
            recommendation = result.get("recommendation")
            score = result.get("composite_score", 0)

            if recommendation:
                if "BUY" in recommendation:
                    rec_color = "green"
                    rec_icon = "📈"
                elif "SELL" in recommendation:
                    rec_color = "red"
                    rec_icon = "📉"
                else:
                    rec_color = "yellow"
                    rec_icon = "➡️"

                console.print(
                    Panel(
                        f"[bold {rec_color}]{rec_icon} {recommendation}[/bold {rec_color}]\n\n"
                        f"[bold]Composite Score:[/bold] {score:.1f}/10",
                        title=f"💡 Recommendation: {symbol}",
                        border_style=rec_color,
                    )
                )

            answer = result.get("answer", "")
            if answer:
                console.print()
                console.print(Markdown(answer))
        else:
            console.print(f"[red]Failed:[/red] {result.get('error', 'Unknown error')}")
            raise typer.Exit(1)

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@agents_app.command("risk")
def agents_risk(
    symbol: str = typer.Argument(..., help="Stock symbol"),
) -> None:
    """Assess risk for a stock position.

    Uses Risk Manager agent to evaluate:
    - Volatility metrics
    - Value at Risk (VaR)
    - Maximum drawdown
    - Stop-loss recommendations

    Examples:
        stock agents risk BBCA
    """
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.markdown import Markdown

    from stockai.agents import create_trading_orchestrator
    from stockai.config import get_settings

    symbol = symbol.upper()
    settings = get_settings()

    if not settings.has_google_api:
        console.print("[red]Error:[/red] Google API key not configured.")
        raise typer.Exit(1)

    console.print(f"\n[bold]⚠️ Risk Assessment: {symbol}[/bold]\n")

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task("Running Risk Manager...", total=None)

            orchestrator = create_trading_orchestrator()

            result = orchestrator.run(
                query=f"What are the risks of investing in {symbol}? Provide detailed risk assessment with stop-loss recommendations.",
                symbol=symbol,
            )

        if result.get("success"):
            answer = result.get("answer", "")
            if answer:
                console.print(Markdown(answer))
        else:
            console.print(f"[red]Failed:[/red] {result.get('error', 'Unknown error')}")
            raise typer.Exit(1)

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@agents_app.command("daily")
def agents_daily(
    capital: int = typer.Argument(..., help="Investment capital in Rupiah (e.g., 1000000)"),
    horizon: str = typer.Option("both", "--horizon", "-h", help="Investment horizon: short, long, or both"),
    holdings: str = typer.Option(None, "--holdings", help="Current holdings to check for sell signals (e.g., 'BBCA,TLKM')"),
    index: str = typer.Option("IDX30", "--index", "-i", help="Index to scan (IDX30, LQ45)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed analysis"),
) -> None:
    """Daily trading recommendations based on your capital.

    Scans the market and provides personalized portfolio allocation
    with specific buy/sell recommendations for both short-term
    and long-term strategies.

    Examples:
        stock agents daily 1000000
        stock agents daily 5000000 --horizon short
        stock agents daily 10000000 --holdings "BBCA,TLKM"
        stock agents daily 50000000 --horizon long --index LQ45
    """
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.markdown import Markdown

    from stockai.agents import create_trading_orchestrator
    from stockai.config import get_settings

    settings = get_settings()

    if not settings.has_google_api:
        console.print("[red]Error:[/red] Google API key not configured.")
        raise typer.Exit(1)

    # Format capital for display
    if capital >= 1_000_000_000:
        capital_str = f"Rp {capital/1_000_000_000:.1f}B"
    elif capital >= 1_000_000:
        capital_str = f"Rp {capital/1_000_000:.1f}M"
    else:
        capital_str = f"Rp {capital:,}"

    console.print(f"\n[bold]📊 Daily Trading Recommendations[/bold]")
    console.print(f"[bold cyan]Capital:[/bold cyan] {capital_str}")
    console.print(f"[bold cyan]Horizon:[/bold cyan] {horizon.upper()}")
    if holdings:
        console.print(f"[bold cyan]Holdings:[/bold cyan] {holdings.upper()}")
    console.print()

    # Build the query for the orchestrator
    query_parts = [
        f"I have {capital_str} to invest in Indonesian stocks.",
        f"Scan {index.upper()} and provide specific portfolio recommendations.",
    ]

    if horizon == "short":
        query_parts.append("Focus on SHORT-TERM trades (1-2 weeks) for quick gains.")
    elif horizon == "long":
        query_parts.append("Focus on LONG-TERM investments (3-12 months) for steady growth.")
    else:
        query_parts.append("Provide BOTH short-term (1-2 weeks) and long-term (3-12 months) recommendations.")

    query_parts.extend([
        "For each stock recommendation, provide:",
        "1. Specific number of LOTS to buy (1 lot = 100 shares)",
        "2. Total investment amount in Rupiah",
        "3. Entry price zone",
        "4. Stop-loss level",
        "5. Target prices",
        "6. Risk-reward ratio",
    ])

    if holdings:
        holdings_list = [h.strip().upper() for h in holdings.split(",")]
        query_parts.append(f"\nAlso analyze my current holdings: {', '.join(holdings_list)}")
        query_parts.append("Tell me which to HOLD, which to SELL, and recommended exit points.")

    query_parts.append(f"\nEnsure total recommended investment does not exceed {capital_str}.")
    query_parts.append("Include a summary table with all positions and allocation percentages.")

    query = "\n".join(query_parts)

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=not verbose,
        ) as progress:
            task = progress.add_task("Running multi-agent analysis...", total=None)

            orchestrator = create_trading_orchestrator()

            result = orchestrator.run(query=query)

        if result.get("success"):
            answer = result.get("answer", "")

            # Display header with recommendation summary
            recommendation = result.get("recommendation")
            score = result.get("composite_score", 0)

            if recommendation or score:
                console.print(
                    Panel(
                        f"[bold]Market Outlook:[/bold] {recommendation or 'See analysis below'}\n"
                        f"[bold]Confidence Score:[/bold] {score:.1f}/10",
                        title=f"📊 Daily Portfolio ({capital_str})",
                        border_style="blue",
                    )
                )

            # Display the full analysis
            if answer:
                console.print()
                console.print(Markdown(answer))

            # Quick reference table
            console.print(
                Panel(
                    "[bold]Quick Actions:[/bold]\n\n"
                    f"• [cyan]stockai agents analyze <SYMBOL>[/cyan] - Deep dive on specific stock\n"
                    f"• [cyan]stockai agents risk <SYMBOL>[/cyan] - Risk assessment\n"
                    f"• [cyan]stockai portfolio add <SYMBOL> <LOTS> <PRICE>[/cyan] - Record purchase\n\n"
                    "[dim]Note: 1 lot = 100 shares. Minimum transaction on IDX is 1 lot.[/dim]",
                    title="📌 Next Steps",
                    border_style="dim",
                )
            )
        else:
            console.print(f"[red]Analysis failed:[/red] {result.get('error', 'Unknown error')}")
            raise typer.Exit(1)

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        if verbose:
            import traceback
            console.print(f"[dim]{traceback.format_exc()}[/dim]")
        raise typer.Exit(1)


@agents_app.command("signal")
def agents_signal(
    symbol: str = typer.Argument(..., help="Stock symbol"),
    format: str = typer.Option("table", "--format", "-f", help="Output format: table, markdown, json"),
) -> None:
    """Get quick trading signal for a stock.

    Provides entry/exit points without full analysis.

    Examples:
        stock agents signal BBCA
        stock agents signal TLKM --format markdown
    """
    from rich.progress import Progress, SpinnerColumn, TextColumn

    from stockai.agents import create_trading_orchestrator
    from stockai.config import get_settings

    symbol = symbol.upper()
    settings = get_settings()

    if not settings.has_google_api:
        console.print("[red]Error:[/red] Google API key not configured.")
        raise typer.Exit(1)

    console.print(f"\n[bold]⚡ Quick Signal: {symbol}[/bold]\n")

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task("Generating signal...", total=None)

            orchestrator = create_trading_orchestrator()

            result = orchestrator.run(
                query=f"""Provide a quick trading signal for {symbol}:
                - Current price and trend
                - BUY/SELL/HOLD recommendation
                - Entry zone (price range)
                - Stop-loss level
                - Target 1 and Target 2
                - Risk-reward ratio
                - Confidence level

                Format as a clear, concise signal card.""",
                symbol=symbol,
            )

        if result.get("success"):
            recommendation = result.get("recommendation", "HOLD")
            score = result.get("composite_score", 5.0)

            # Color based on recommendation
            if "BUY" in recommendation:
                color = "green"
                icon = "📈"
            elif "SELL" in recommendation:
                color = "red"
                icon = "📉"
            else:
                color = "yellow"
                icon = "➡️"

            if format == "table":
                console.print(
                    Panel(
                        f"[bold {color}]{icon} {recommendation}[/bold {color}]\n\n"
                        f"[bold]Score:[/bold] {score:.1f}/10\n\n"
                        f"{result.get('answer', 'See details below')}",
                        title=f"⚡ {symbol} Signal",
                        border_style=color,
                    )
                )
            elif format == "markdown":
                from rich.markdown import Markdown
                console.print(Markdown(result.get("answer", "")))
            else:
                import json
                output = {
                    "symbol": symbol,
                    "recommendation": recommendation,
                    "score": score,
                    "analysis": result.get("answer", ""),
                }
                console.print(json.dumps(output, indent=2))
        else:
            console.print(f"[red]Failed:[/red] {result.get('error', 'Unknown error')}")
            raise typer.Exit(1)

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@agents_app.command("list")
def agents_list() -> None:
    """List all available trading agents.

    Shows the 7 specialized agents in the system.
    """
    from stockai.agents import get_all_subagents

    agents = get_all_subagents()

    table = Table(title="🤖 StockAI Trading Agents", show_header=True)
    table.add_column("#", style="dim", width=3)
    table.add_column("Agent", style="cyan")
    table.add_column("Description")
    table.add_column("Tools", justify="right", style="dim")

    for i, agent in enumerate(agents, 1):
        table.add_row(
            str(i),
            agent["name"],
            agent["description"][:70] + "..." if len(agent["description"]) > 70 else agent["description"],
            str(len(agent["tools"])),
        )

    console.print(table)
    console.print(f"\n[dim]Total: {len(agents)} agents[/dim]")


# Predictions subcommand group
predictions_app = typer.Typer(help="Prediction accuracy tracking and management")
app.add_typer(predictions_app, name="predictions")


@predictions_app.command("accuracy")
def predictions_accuracy(
    symbol: str = typer.Option(None, "--symbol", "-s", help="Filter by stock symbol"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed model analysis"),
) -> None:
    """Show prediction accuracy metrics.

    Displays overall accuracy statistics including breakdowns by
    direction (UP/DOWN/NEUTRAL) and confidence level (HIGH/MEDIUM/LOW).

    Examples:
        stock predictions accuracy
        stock predictions accuracy --symbol BBCA
        stock predictions accuracy --verbose
    """
    from stockai.data.database import init_database
    from stockai.core.predictor import PredictionAccuracyTracker

    init_database()
    tracker = PredictionAccuracyTracker()

    if symbol:
        # Show stock-specific accuracy
        symbol = symbol.upper()
        console.print(f"\n[bold]📊 Prediction Accuracy for {symbol}[/bold]\n")

        metrics = tracker.get_stock_accuracy(symbol)

        # Handle no predictions found
        if metrics.get("message") and metrics.get("total_predictions", 0) == 0:
            console.print(
                Panel(
                    f"[yellow]{metrics.get('message', 'No predictions found')}[/yellow]\n\n"
                    "[dim]Run predictions first with:[/dim]\n"
                    f"  stock predict {symbol}",
                    title=f"📊 {symbol}",
                    border_style="yellow",
                )
            )
            return

        # Display overall metrics
        accuracy = metrics.get("accuracy_rate", 0)
        if accuracy >= 60:
            acc_color = "green"
            acc_icon = "🟢"
        elif accuracy >= 40:
            acc_color = "yellow"
            acc_icon = "🟡"
        else:
            acc_color = "red"
            acc_icon = "🔴"

        stock_name = metrics.get("stock_name", symbol)
        console.print(
            Panel(
                f"[bold cyan]Stock:[/bold cyan] {symbol} ({stock_name})\n"
                f"[bold cyan]Total Predictions:[/bold cyan] {metrics.get('total_predictions', 0)}\n"
                f"[bold cyan]Correct:[/bold cyan] {metrics.get('correct_predictions', 0)}\n"
                f"[bold cyan]Accuracy Rate:[/bold cyan] [{acc_color}]{acc_icon} {accuracy:.1f}%[/{acc_color}]",
                title=f"📊 Accuracy Summary",
                border_style="blue",
            )
        )

        # Display accuracy by direction
        by_direction = metrics.get("by_direction", {})
        if any(by_direction.get(d, {}).get("total", 0) > 0 for d in ["UP", "DOWN", "NEUTRAL"]):
            dir_table = Table(title="📈 Accuracy by Direction", show_header=True)
            dir_table.add_column("Direction", style="cyan")
            dir_table.add_column("Total", justify="right")
            dir_table.add_column("Correct", justify="right")
            dir_table.add_column("Accuracy", justify="right")

            for direction in ["UP", "DOWN", "NEUTRAL"]:
                stats = by_direction.get(direction, {})
                total = stats.get("total", 0)
                if total > 0:
                    correct = stats.get("correct", 0)
                    acc_rate = stats.get("accuracy_rate", 0)
                    acc_str = f"{acc_rate:.1f}%"

                    if direction == "UP":
                        dir_icon = "📈"
                    elif direction == "DOWN":
                        dir_icon = "📉"
                    else:
                        dir_icon = "➡️"

                    dir_table.add_row(
                        f"{dir_icon} {direction}",
                        str(total),
                        str(correct),
                        acc_str,
                    )

            console.print(dir_table)

        # Display accuracy by confidence level
        by_confidence = metrics.get("by_confidence", {})
        if any(by_confidence.get(level, {}).get("total", 0) > 0 for level in ["HIGH", "MEDIUM", "LOW"]):
            conf_table = Table(title="🎯 Accuracy by Confidence Level", show_header=True)
            conf_table.add_column("Confidence", style="cyan")
            conf_table.add_column("Total", justify="right")
            conf_table.add_column("Correct", justify="right")
            conf_table.add_column("Accuracy", justify="right")

            for level in ["HIGH", "MEDIUM", "LOW"]:
                stats = by_confidence.get(level, {})
                total = stats.get("total", 0)
                if total > 0:
                    correct = stats.get("correct", 0)
                    acc_rate = stats.get("accuracy_rate", 0)
                    acc_str = f"{acc_rate:.1f}%"

                    if level == "HIGH":
                        level_icon = "🟢"
                    elif level == "MEDIUM":
                        level_icon = "🟡"
                    else:
                        level_icon = "🔴"

                    conf_table.add_row(
                        f"{level_icon} {level}",
                        str(total),
                        str(correct),
                        acc_str,
                    )

            console.print(conf_table)

        # Display recent predictions
        recent = metrics.get("recent_predictions", [])
        if recent:
            console.print("\n[bold]📋 Recent Predictions[/bold]")
            recent_table = Table(show_header=True)
            recent_table.add_column("Date", style="dim", width=10)
            recent_table.add_column("Predicted", justify="center")
            recent_table.add_column("Actual", justify="center")
            recent_table.add_column("Return", justify="right")
            recent_table.add_column("Result", justify="center")

            for pred in recent[:5]:
                target_date = pred.get("target_date", "")[:10]

                # Predicted direction
                predicted = pred.get("direction", "?")
                if predicted == "UP":
                    pred_str = "[green]📈 UP[/green]"
                elif predicted == "DOWN":
                    pred_str = "[red]📉 DOWN[/red]"
                else:
                    pred_str = "[dim]➡️ NEUTRAL[/dim]"

                # Actual direction
                actual = pred.get("actual_direction", "?")
                if actual == "UP":
                    actual_str = "[green]📈 UP[/green]"
                elif actual == "DOWN":
                    actual_str = "[red]📉 DOWN[/red]"
                else:
                    actual_str = "[dim]➡️ NEUTRAL[/dim]"

                # Return
                actual_return = pred.get("actual_return")
                if actual_return is not None:
                    return_color = "green" if actual_return >= 0 else "red"
                    return_str = f"[{return_color}]{actual_return:+.2f}%[/{return_color}]"
                else:
                    return_str = "[dim]-[/dim]"

                # Result
                is_correct = pred.get("is_correct")
                if is_correct is True:
                    result_str = "[green]✓ Correct[/green]"
                elif is_correct is False:
                    result_str = "[red]✗ Wrong[/red]"
                else:
                    result_str = "[dim]?[/dim]"

                recent_table.add_row(target_date, pred_str, actual_str, return_str, result_str)

            console.print(recent_table)

        # Display accuracy trend if verbose
        if verbose:
            trend = metrics.get("accuracy_trend", [])
            if trend:
                console.print("\n[bold]📈 Monthly Accuracy Trend[/bold]")
                trend_table = Table(show_header=True)
                trend_table.add_column("Month", style="cyan")
                trend_table.add_column("Total", justify="right")
                trend_table.add_column("Correct", justify="right")
                trend_table.add_column("Accuracy", justify="right")

                for month_data in trend[-6:]:  # Last 6 months
                    acc_rate = month_data.get("accuracy_rate", 0)
                    if acc_rate >= 60:
                        acc_str = f"[green]{acc_rate:.1f}%[/green]"
                    elif acc_rate >= 40:
                        acc_str = f"[yellow]{acc_rate:.1f}%[/yellow]"
                    else:
                        acc_str = f"[red]{acc_rate:.1f}%[/red]"

                    trend_table.add_row(
                        month_data.get("month", ""),
                        str(month_data.get("total", 0)),
                        str(month_data.get("correct", 0)),
                        acc_str,
                    )

                console.print(trend_table)

    else:
        # Show overall accuracy
        console.print("\n[bold]📊 Overall Prediction Accuracy[/bold]\n")

        metrics = tracker.get_accuracy_metrics()

        # Handle no predictions found
        if metrics.get("message") and metrics.get("total_predictions", 0) == 0:
            console.print(
                Panel(
                    f"[yellow]{metrics.get('message', 'No predictions found')}[/yellow]\n\n"
                    "[dim]Run predictions first:[/dim]\n"
                    "  stock predict BBCA\n"
                    "  stock predict TLKM\n\n"
                    "[dim]Then run backfill to update accuracy:[/dim]\n"
                    "  stock predictions backfill",
                    title="📊 Prediction Accuracy",
                    border_style="yellow",
                )
            )
            return

        # Display overall metrics
        accuracy = metrics.get("accuracy_rate", 0)
        if accuracy >= 60:
            acc_color = "green"
            acc_icon = "🟢"
        elif accuracy >= 40:
            acc_color = "yellow"
            acc_icon = "🟡"
        else:
            acc_color = "red"
            acc_icon = "🔴"

        console.print(
            Panel(
                f"[bold cyan]Total Predictions:[/bold cyan] {metrics.get('total_predictions', 0)}\n"
                f"[bold cyan]Correct:[/bold cyan] {metrics.get('correct_predictions', 0)}\n"
                f"[bold cyan]Accuracy Rate:[/bold cyan] [{acc_color}]{acc_icon} {accuracy:.1f}%[/{acc_color}]",
                title="📊 Overall Accuracy",
                border_style="blue",
            )
        )

        # Display accuracy by direction
        by_direction = metrics.get("by_direction", {})
        dir_table = Table(title="📈 Accuracy by Direction", show_header=True)
        dir_table.add_column("Direction", style="cyan")
        dir_table.add_column("Total", justify="right")
        dir_table.add_column("Correct", justify="right")
        dir_table.add_column("Accuracy", justify="right")

        for direction in ["UP", "DOWN", "NEUTRAL"]:
            stats = by_direction.get(direction, {})
            total = stats.get("total", 0)
            correct = stats.get("correct", 0)
            acc_rate = stats.get("accuracy_rate", 0)
            acc_str = f"{acc_rate:.1f}%"

            if direction == "UP":
                dir_icon = "📈"
            elif direction == "DOWN":
                dir_icon = "📉"
            else:
                dir_icon = "➡️"

            dir_table.add_row(
                f"{dir_icon} {direction}",
                str(total),
                str(correct),
                acc_str,
            )

        console.print(dir_table)

        # Display accuracy by confidence level
        by_confidence = metrics.get("by_confidence", {})
        conf_table = Table(title="🎯 Accuracy by Confidence Level", show_header=True)
        conf_table.add_column("Confidence", style="cyan")
        conf_table.add_column("Total", justify="right")
        conf_table.add_column("Correct", justify="right")
        conf_table.add_column("Accuracy", justify="right")

        for level in ["HIGH", "MEDIUM", "LOW"]:
            stats = by_confidence.get(level, {})
            total = stats.get("total", 0)
            correct = stats.get("correct", 0)
            acc_rate = stats.get("accuracy_rate", 0)
            acc_str = f"{acc_rate:.1f}%"

            if level == "HIGH":
                level_icon = "🟢"
            elif level == "MEDIUM":
                level_icon = "🟡"
            else:
                level_icon = "🔴"

            conf_table.add_row(
                f"{level_icon} {level}",
                str(total),
                str(correct),
                acc_str,
            )

        console.print(conf_table)

        # Show model analysis if verbose
        if verbose:
            console.print("\n[bold]🔬 Model Component Analysis[/bold]")

            model_analysis = tracker.get_accuracy_by_model()

            # Display insights
            insights = model_analysis.get("insights", [])
            if insights:
                console.print("\n[bold]💡 Insights:[/bold]")
                for insight in insights:
                    console.print(f"  • {insight}")

            # Display correlation summary
            corr_summary = model_analysis.get("correlation_summary", {})
            corr_table = Table(title="📊 Model Correlation with Accuracy", show_header=True)
            corr_table.add_column("Model", style="cyan")
            corr_table.add_column("Correlation", justify="center")
            corr_table.add_column("Low Bins Acc", justify="right")
            corr_table.add_column("High Bins Acc", justify="right")
            corr_table.add_column("Difference", justify="right")

            for model_name, corr in [
                ("XGBoost", corr_summary.get("xgboost", {})),
                ("LSTM", corr_summary.get("lstm", {})),
                ("Sentiment", corr_summary.get("sentiment", {})),
            ]:
                has_corr = corr.get("has_correlation")
                direction = corr.get("direction", "?")

                if has_corr is True:
                    if direction == "positive":
                        corr_str = "[green]↑ Positive[/green]"
                    else:
                        corr_str = "[red]↓ Negative[/red]"
                elif has_corr is False:
                    corr_str = "[dim]~ Weak[/dim]"
                else:
                    corr_str = "[dim]N/A[/dim]"

                low_acc = corr.get("low_accuracy")
                high_acc = corr.get("high_accuracy")
                diff = corr.get("difference")

                low_str = f"{low_acc:.1f}%" if low_acc is not None else "[dim]-[/dim]"
                high_str = f"{high_acc:.1f}%" if high_acc is not None else "[dim]-[/dim]"

                if diff is not None:
                    if diff > 0:
                        diff_str = f"[green]+{diff:.1f}%[/green]"
                    elif diff < 0:
                        diff_str = f"[red]{diff:.1f}%[/red]"
                    else:
                        diff_str = "0.0%"
                else:
                    diff_str = "[dim]-[/dim]"

                corr_table.add_row(model_name, corr_str, low_str, high_str, diff_str)

            console.print(corr_table)

    console.print()


@predictions_app.command("backfill")
def predictions_backfill(
    dry_run: bool = typer.Option(
        False, "--dry-run", "-n", help="Show what would be updated without making changes"
    ),
) -> None:
    """Backfill prediction accuracy data.

    Updates past predictions with actual outcomes by comparing
    predicted direction with actual price movements.

    Examples:
        stock predictions backfill
        stock predictions backfill --dry-run
    """
    from rich.progress import Progress, SpinnerColumn, TextColumn

    from stockai.data.database import init_database
    from stockai.core.predictor import PredictionAccuracyTracker

    init_database()
    tracker = PredictionAccuracyTracker()

    console.print("\n[bold]🔄 Prediction Accuracy Backfill[/bold]\n")

    if dry_run:
        console.print("[yellow]🔍 Dry run mode - no changes will be made[/yellow]\n")

        # Get pending predictions to show what would be updated
        pending = tracker.get_pending_predictions()

        if not pending:
            console.print(
                Panel(
                    "[green]✓ No pending predictions to update[/green]\n\n"
                    "All predictions with past target dates have already been evaluated.",
                    title="📋 Dry Run Results",
                    border_style="green",
                )
            )
            return

        console.print(f"[cyan]Found {len(pending)} predictions to update:[/cyan]\n")

        # Show sample of pending predictions
        pending_table = Table(title="📋 Pending Predictions", show_header=True)
        pending_table.add_column("Symbol", style="cyan")
        pending_table.add_column("Target Date", style="dim")
        pending_table.add_column("Direction", justify="center")
        pending_table.add_column("Confidence", justify="right")

        # Group by symbol for summary
        symbol_counts: dict[str, int] = {}
        for pred in pending:
            symbol = pred.get("symbol", "?")
            symbol_counts[symbol] = symbol_counts.get(symbol, 0) + 1

        # Show first 10 predictions
        for pred in pending[:10]:
            symbol = pred.get("symbol", "?")
            target_date = pred.get("target_date")
            date_str = target_date.strftime("%Y-%m-%d") if target_date else "?"

            direction = pred.get("direction", "?")
            if direction == "UP":
                dir_str = "[green]📈 UP[/green]"
            elif direction == "DOWN":
                dir_str = "[red]📉 DOWN[/red]"
            else:
                dir_str = "[dim]➡️ NEUTRAL[/dim]"

            confidence = pred.get("confidence")
            conf_str = f"{confidence:.1%}" if confidence else "[dim]-[/dim]"

            pending_table.add_row(symbol, date_str, dir_str, conf_str)

        console.print(pending_table)

        if len(pending) > 10:
            console.print(f"\n[dim]... and {len(pending) - 10} more predictions[/dim]")

        # Show summary by symbol
        console.print("\n[bold]📊 Summary by Symbol:[/bold]")
        for symbol, count in sorted(symbol_counts.items(), key=lambda x: -x[1]):
            console.print(f"  • {symbol}: {count} predictions")

        console.print(
            f"\n[yellow]Run without --dry-run to update these {len(pending)} predictions[/yellow]"
        )
        return

    # Perform actual backfill
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Fetching pending predictions...", total=None)

        # First get count of pending predictions
        pending = tracker.get_pending_predictions()
        pending_count = len(pending)

        if pending_count == 0:
            console.print(
                Panel(
                    "[green]✓ No pending predictions to update[/green]\n\n"
                    "All predictions with past target dates have already been evaluated.",
                    title="📋 Backfill Complete",
                    border_style="green",
                )
            )
            return

        progress.update(task, description=f"Updating {pending_count} predictions...")

        # Perform backfill
        result = tracker.update_past_predictions()

    # Display results
    updated = result.get("updated_count", 0)
    skipped = result.get("skipped_count", 0)
    errors = result.get("error_count", 0)
    total = result.get("total_pending", 0)

    # Determine overall status
    if errors == 0 and skipped == 0:
        status_color = "green"
        status_icon = "✓"
        status_text = "Backfill completed successfully"
    elif errors > 0:
        status_color = "yellow"
        status_icon = "⚠"
        status_text = "Backfill completed with some errors"
    else:
        status_color = "blue"
        status_icon = "ℹ"
        status_text = "Backfill completed with some skipped predictions"

    console.print(
        Panel(
            f"[{status_color}]{status_icon} {status_text}[/{status_color}]\n\n"
            f"[bold cyan]Total Processed:[/bold cyan] {total}\n"
            f"[bold green]Updated:[/bold green] {updated}\n"
            f"[bold yellow]Skipped:[/bold yellow] {skipped} (missing price data)\n"
            f"[bold red]Errors:[/bold red] {errors}",
            title="📋 Backfill Results",
            border_style=status_color,
        )
    )

    # Show errors if any
    error_list = result.get("errors", [])
    if error_list:
        console.print("\n[bold red]Errors encountered:[/bold red]")
        for err in error_list:
            console.print(f"  [red]• {err}[/red]")

    # Show next steps
    if updated > 0:
        console.print(
            "\n[dim]View accuracy metrics with:[/dim]\n"
            "  stock predictions accuracy"
        )

    console.print()


# Watchlist subcommand group
watchlist_app = typer.Typer(help="Manage stock watchlist")
app.add_typer(watchlist_app, name="watchlist")


@watchlist_app.command("list")
def watchlist_list() -> None:
    """List all stocks in watchlist.

    Examples:
        stock watchlist list
    """
    from stockai.web.services.watchlist import get_watchlist_items
    from stockai.data.sources.yahoo import YahooFinanceSource

    try:
        items = get_watchlist_items()
    except Exception as e:
        console.print(f"[red]Error loading watchlist:[/red] {e}")
        raise typer.Exit(1)

    if not items:
        console.print(
            Panel(
                "[yellow]Your watchlist is empty.[/yellow]\n\n"
                "Add stocks with:\n"
                "  [cyan]stock watchlist add BBCA[/cyan]",
                title="👀 Watchlist",
            )
        )
        return

    table = Table(title="👀 Watchlist", show_header=True)
    table.add_column("#", style="dim", width=3)
    table.add_column("Symbol", style="cyan bold")
    table.add_column("Price", justify="right")
    table.add_column("Change", justify="right")
    table.add_column("Alert ↑", justify="right", style="green")
    table.add_column("Alert ↓", justify="right", style="red")
    table.add_column("Notes", style="dim")

    yahoo = YahooFinanceSource()
    symbols = [item.stock.symbol for item in items if item.stock]

    # Fetch current prices
    prices = {}
    for symbol in symbols:
        try:
            quote = yahoo.get_quote(symbol)
            if quote:
                prices[symbol] = quote
        except Exception:
            pass

    for i, item in enumerate(items, 1):
        symbol = item.stock.symbol if item.stock else "N/A"
        quote = prices.get(symbol, {})

        price_str = f"Rp {quote.get('price', 0):,.0f}" if quote.get('price') else "-"
        change = quote.get('change_percent', 0)
        change_str = f"[{'green' if change >= 0 else 'red'}]{change:+.2f}%[/]" if change else "-"

        alert_above = f"Rp {item.alert_price_above:,.0f}" if item.alert_price_above else "-"
        alert_below = f"Rp {item.alert_price_below:,.0f}" if item.alert_price_below else "-"
        notes = (item.notes[:20] + "...") if item.notes and len(item.notes) > 20 else (item.notes or "-")

        table.add_row(str(i), symbol, price_str, change_str, alert_above, alert_below, notes)

    console.print(table)
    console.print(f"\n[dim]Total: {len(items)} stocks in watchlist[/dim]")


@watchlist_app.command("add")
def watchlist_add(
    symbol: str = typer.Argument(..., help="Stock symbol to watch"),
    alert_above: Optional[float] = typer.Option(None, "--alert-above", "-a", help="Alert when price goes above this value"),
    alert_below: Optional[float] = typer.Option(None, "--alert-below", "-b", help="Alert when price goes below this value"),
    notes: Optional[str] = typer.Option(None, "--notes", "-n", help="Notes for this watchlist item"),
) -> None:
    """Add a stock to watchlist.

    Examples:
        stock watchlist add BBCA
        stock watchlist add TLKM --alert-above 4000 --alert-below 3500
        stock watchlist add BBRI --notes "Wait for dip"
    """
    from stockai.web.services.watchlist import (
        add_to_watchlist,
        WatchlistItemExistsError,
    )

    symbol = symbol.upper()

    try:
        item = add_to_watchlist(
            symbol=symbol,
            alert_price_above=alert_above,
            alert_price_below=alert_below,
            notes=notes,
        )
        console.print(f"[green]✓ Added {symbol} to watchlist[/green]")

        if alert_above or alert_below:
            if alert_above:
                console.print(f"  Alert above: Rp {alert_above:,.0f}")
            if alert_below:
                console.print(f"  Alert below: Rp {alert_below:,.0f}")

    except WatchlistItemExistsError:
        console.print(f"[yellow]{symbol} is already in your watchlist[/yellow]")
    except Exception as e:
        console.print(f"[red]Error adding {symbol}:[/red] {e}")
        raise typer.Exit(1)


@watchlist_app.command("remove")
def watchlist_remove(
    symbol: str = typer.Argument(..., help="Stock symbol to remove"),
) -> None:
    """Remove a stock from watchlist.

    Examples:
        stock watchlist remove BBCA
    """
    from stockai.web.services.watchlist import (
        remove_from_watchlist_by_symbol,
        WatchlistItemNotFoundError,
    )

    symbol = symbol.upper()

    try:
        remove_from_watchlist_by_symbol(symbol)
        console.print(f"[green]✓ Removed {symbol} from watchlist[/green]")
    except WatchlistItemNotFoundError:
        console.print(f"[yellow]{symbol} is not in your watchlist[/yellow]")
    except Exception as e:
        console.print(f"[red]Error removing {symbol}:[/red] {e}")
        raise typer.Exit(1)


@app.command()
def web(
    host: str = typer.Option("127.0.0.1", "--host", "-h", help="Host to bind to"),
    port: int = typer.Option(8000, "--port", "-p", help="Port to run on"),
    reload: bool = typer.Option(False, "--reload", "-r", help="Enable auto-reload for development"),
) -> None:
    """Start the StockAI web dashboard.

    Launches a web server with interactive stock analysis,
    portfolio management, and AI-powered insights.

    Examples:
        stock web
        stock web --port 3000
        stock web --reload
    """
    import uvicorn

    console.print(
        Panel(
            f"[bold]Starting StockAI Web Dashboard[/bold]\n\n"
            f"Server: http://{host}:{port}\n"
            f"API Docs: http://{host}:{port}/api/docs\n\n"
            "[dim]Press Ctrl+C to stop[/dim]",
            title="🌐 StockAI Web",
            border_style="blue",
        )
    )

    uvicorn.run(
        "stockai.web.app:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )


# Automation subcommand group
auto_app = typer.Typer(help="Automated trading system")
app.add_typer(auto_app, name="auto")


@auto_app.command("start")
def auto_start(
    capital: int = typer.Argument(..., help="Investment capital in Rupiah"),
    index: str = typer.Option("IDX30", "--index", "-i", help="Index to scan (IDX30, LQ45)"),
    holdings: str = typer.Option(None, "--holdings", "-h", help="Current holdings (comma-separated)"),
    telegram_token: str = typer.Option(None, "--telegram-token", envvar="STOCKAI_TELEGRAM_TOKEN", help="Telegram bot token"),
    telegram_chat: str = typer.Option(None, "--telegram-chat", envvar="STOCKAI_TELEGRAM_CHAT", help="Telegram chat ID"),
    output_dir: str = typer.Option(None, "--output", "-o", help="Output directory for reports"),
) -> None:
    """Start automated trading scheduler.

    Runs continuously with scheduled tasks:
    - 8:30 AM: Morning scan & daily recommendations
    - 9:15 AM: Post-market-open check
    - 12:00 PM: Mid-day review
    - 3:45 PM: Pre-close signals
    - 4:15 PM: End of day summary

    Examples:
        stockai auto start 10000000
        stockai auto start 50000000 --holdings "BBRI,TLKM,BBCA"
        stockai auto start 10000000 --telegram-token "BOT_TOKEN" --telegram-chat "CHAT_ID"
    """
    import asyncio
    from stockai.automation.runner import run_automated_trading

    holdings_list = [h.strip().upper() for h in holdings.split(",")] if holdings else []

    console.print(
        Panel(
            f"[bold]Starting Automated Trading[/bold]\n\n"
            f"💰 Capital: Rp {capital:,}\n"
            f"📊 Index: {index}\n"
            f"📁 Holdings: {', '.join(holdings_list) if holdings_list else 'None'}\n"
            f"📱 Telegram: {'Enabled' if telegram_token else 'Disabled'}\n\n"
            "[dim]Press Ctrl+C to stop[/dim]",
            title="🤖 StockAI Automation",
            border_style="green",
        )
    )

    try:
        asyncio.run(
            run_automated_trading(
                capital=capital,
                index=index,
                holdings=holdings_list,
                telegram_token=telegram_token,
                telegram_chat_id=telegram_chat,
            )
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]Automation stopped by user[/yellow]")


@auto_app.command("run")
def auto_run(
    task: str = typer.Argument(..., help="Task to run: scan, daily, signals, portfolio, summary"),
    capital: int = typer.Option(10_000_000, "--capital", "-c", help="Investment capital"),
    holdings: str = typer.Option(None, "--holdings", "-h", help="Current holdings (comma-separated)"),
    horizon: str = typer.Option("both", "--horizon", help="Investment horizon: short, long, both"),
    output: str = typer.Option(None, "--output", "-o", help="Save output to file"),
) -> None:
    """Run a single automated task.

    Tasks:
        scan      - Market scan for opportunities
        daily     - Daily portfolio recommendations
        signals   - Quick signals for holdings
        portfolio - Check current portfolio
        summary   - End of day summary

    Examples:
        stockai auto run scan
        stockai auto run daily --capital 5000000 --horizon short
        stockai auto run signals --holdings "BBRI,TLKM"
        stockai auto run portfolio --holdings "BBCA,ASII,TLKM"
    """
    from pathlib import Path
    from stockai.automation.runner import AutomatedTrader

    holdings_list = [h.strip().upper() for h in holdings.split(",")] if holdings else []

    trader = AutomatedTrader(
        capital=capital,
        holdings=holdings_list,
        output_dir=output,
    )

    task_lower = task.lower()

    with console.status(f"[bold blue]Running {task}...[/bold blue]"):
        if task_lower == "scan":
            result = trader.run_market_scan()
        elif task_lower == "daily":
            result = trader.run_daily_recommendations(horizon=horizon)
        elif task_lower == "signals":
            result = trader.run_quick_signals()
        elif task_lower == "portfolio":
            result = trader.run_portfolio_check()
        elif task_lower == "summary":
            result = trader.run_daily_summary()
        else:
            console.print(f"[red]Unknown task:[/red] {task}")
            console.print("Available: scan, daily, signals, portfolio, summary")
            raise typer.Exit(1)

    # Display result
    if result.success:
        console.print(
            Panel(
                result.raw_output[:2000] + "..." if len(result.raw_output) > 2000 else result.raw_output,
                title=f"✅ {task.title()} Complete",
                border_style="green",
            )
        )

        if result.recommendations:
            console.print(f"\n[bold]Recommendations:[/bold] {len(result.recommendations)}")
        if result.signals:
            console.print(f"[bold]Signals:[/bold] {len(result.signals)}")

        # Show saved file location
        console.print(f"\n[dim]Report saved to: {trader.output_dir}[/dim]")
    else:
        console.print(
            Panel(
                "\n".join(result.errors) if result.errors else "Unknown error",
                title=f"❌ {task.title()} Failed",
                border_style="red",
            )
        )
        raise typer.Exit(1)


@auto_app.command("test-notify")
def auto_test_notify(
    telegram_token: str = typer.Option(..., "--telegram-token", "-t", help="Telegram bot token"),
    telegram_chat: str = typer.Option(..., "--telegram-chat", "-c", help="Telegram chat ID"),
) -> None:
    """Test notification setup.

    Examples:
        stockai auto test-notify --telegram-token "BOT_TOKEN" --telegram-chat "CHAT_ID"
    """
    import asyncio
    from stockai.automation.notifier import TelegramNotifier, TradingAlert

    async def test():
        notifier = TelegramNotifier(telegram_token, telegram_chat)

        console.print("[bold]Testing Telegram connection...[/bold]")

        if await notifier.test_connection():
            console.print("[green]✓ Connection successful![/green]")

            alert = TradingAlert(
                title="Test Alert",
                message="StockAI automation is configured correctly!",
                signal="ALERT",
            )

            if await notifier.send(alert):
                console.print("[green]✓ Test message sent![/green]")
            else:
                console.print("[red]✗ Failed to send test message[/red]")
        else:
            console.print("[red]✗ Connection failed[/red]")
            raise typer.Exit(1)

    asyncio.run(test())


@auto_app.command("schedule")
def auto_schedule() -> None:
    """Show automation schedule.

    Displays the default trading schedule for IDX market hours.
    """
    table = Table(title="📅 StockAI Automation Schedule (Asia/Jakarta)", show_header=True)
    table.add_column("Time", style="cyan")
    table.add_column("Task", style="bold")
    table.add_column("Description")

    schedule = [
        ("08:30", "Morning Scan", "Daily recommendations and market opportunities"),
        ("09:15", "Post-Open Check", "Market scan after opening bell"),
        ("12:00", "Mid-Day Review", "Portfolio status and position checks"),
        ("15:45", "Pre-Close Signals", "Quick signals for holdings"),
        ("16:15", "EOD Summary", "End of day summary and outlook"),
    ]

    for time, task, desc in schedule:
        table.add_row(time, task, desc)

    console.print(table)
    console.print("\n[dim]Schedule runs Monday-Friday (IDX trading days)[/dim]")
    console.print("\n[bold]Quick Start:[/bold]")
    console.print("  stockai auto start 10000000")
    console.print("  stockai auto start 10000000 --telegram-token TOKEN --telegram-chat CHAT_ID")


# ============================================================================
# TUTORIAL & LEARNING
# ============================================================================

learn_app = typer.Typer(help="Learn stock trading step by step")
app.add_typer(learn_app, name="learn")


@learn_app.command("start")
def learn_start() -> None:
    """Start the beginner tutorial.

    Interactive lessons covering:
    - Stock market basics
    - How to analyze stocks
    - Risk management
    - Using StockAI effectively

    Examples:
        stockai learn start
    """
    from stockai.tutorial.lessons import get_all_lessons, LessonProgress, LessonCategory
    from pathlib import Path

    lessons = get_all_lessons()
    progress_path = Path.home() / ".stockai" / "learning_progress.json"
    progress = LessonProgress.load(progress_path) if progress_path.exists() else LessonProgress()

    # Welcome message
    console.print(
        Panel(
            "[bold]Selamat Datang di StockAI Learning![/bold]\n\n"
            "Tutorial interaktif untuk belajar investasi saham Indonesia.\n\n"
            f"📚 Total Pelajaran: {len(lessons)}\n"
            f"✅ Selesai: {len(progress.completed_lessons)}\n"
            f"📈 Progress: {progress.get_progress_percent(len(lessons)):.0f}%\n\n"
            "[dim]Ketik 'stockai learn list' untuk melihat daftar pelajaran[/dim]",
            title="📖 StockAI Learning",
            border_style="blue",
        )
    )

    # Show categories
    console.print("\n[bold]Kategori Pelajaran:[/bold]\n")

    categories = {
        LessonCategory.BASICS: ("📘 Dasar-Dasar", "Apa itu saham, cara untung, lot & biaya"),
        LessonCategory.ANALYSIS: ("📊 Analisis", "Fundamental & teknikal analysis"),
        LessonCategory.RISK: ("⚠️ Manajemen Risiko", "Stop-loss, position sizing, diversifikasi"),
        LessonCategory.STOCKAI: ("🤖 Menggunakan StockAI", "Command dan workflow"),
    }

    for cat, (icon, desc) in categories.items():
        cat_lessons = [l for l in lessons if l.category == cat]
        completed = len([l for l in cat_lessons if l.id in progress.completed_lessons])
        console.print(f"  {icon}: {desc} ({completed}/{len(cat_lessons)})")

    console.print("\n[bold]Mulai Belajar:[/bold]")
    console.print("  stockai learn lesson basics_01_what_is_stock")
    console.print("  stockai learn list")


@learn_app.command("list")
def learn_list() -> None:
    """List all available lessons."""
    from stockai.tutorial.lessons import get_all_lessons, LessonProgress
    from pathlib import Path

    lessons = get_all_lessons()
    progress_path = Path.home() / ".stockai" / "learning_progress.json"
    progress = LessonProgress.load(progress_path) if progress_path.exists() else LessonProgress()

    table = Table(title="📚 Daftar Pelajaran", show_header=True)
    table.add_column("#", style="dim", width=3)
    table.add_column("Status", width=3)
    table.add_column("ID", style="cyan")
    table.add_column("Judul")
    table.add_column("Durasi", justify="right", style="dim")

    for i, lesson in enumerate(lessons, 1):
        status = "✅" if lesson.id in progress.completed_lessons else "⬜"
        table.add_row(
            str(i),
            status,
            lesson.id,
            lesson.title,
            f"{lesson.duration_minutes} menit",
        )

    console.print(table)
    console.print(f"\n[dim]Progress: {len(progress.completed_lessons)}/{len(lessons)} selesai[/dim]")
    console.print("\n[bold]Buka pelajaran:[/bold] stockai learn lesson <ID>")


@learn_app.command("lesson")
def learn_lesson(
    lesson_id: str = typer.Argument(..., help="Lesson ID to view"),
    mark_complete: bool = typer.Option(False, "--complete", "-c", help="Mark lesson as complete"),
) -> None:
    """View a specific lesson.

    Examples:
        stockai learn lesson basics_01_what_is_stock
        stockai learn lesson basics_02_how_to_profit --complete
    """
    from stockai.tutorial.lessons import get_lesson, get_next_lesson, LessonProgress
    from rich.markdown import Markdown
    from pathlib import Path

    lesson = get_lesson(lesson_id)
    if not lesson:
        console.print(f"[red]Pelajaran tidak ditemukan:[/red] {lesson_id}")
        console.print("Gunakan 'stockai learn list' untuk melihat daftar pelajaran.")
        raise typer.Exit(1)

    # Display lesson
    console.print(
        Panel(
            f"[bold]{lesson.title}[/bold]\n"
            f"[dim]Kategori: {lesson.category.value} | Durasi: {lesson.duration_minutes} menit[/dim]",
            border_style="blue",
        )
    )

    # Content
    console.print(Markdown(lesson.content))

    # Key points
    if lesson.key_points:
        console.print("\n[bold yellow]📌 Poin Penting:[/bold yellow]")
        for point in lesson.key_points:
            console.print(f"  • {point}")

    # Practice command
    if lesson.practice_command:
        console.print(f"\n[bold green]💻 Latihan:[/bold green]")
        console.print(f"  {lesson.practice_command}")

    # Quiz
    if lesson.quiz_questions:
        console.print(f"\n[bold cyan]📝 Quiz tersedia![/bold cyan] Jalankan: stockai learn quiz {lesson_id}")

    # Mark complete
    progress_path = Path.home() / ".stockai" / "learning_progress.json"
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    progress = LessonProgress.load(progress_path) if progress_path.exists() else LessonProgress()

    if mark_complete:
        progress.complete_lesson(lesson_id)
        progress.save(progress_path)
        console.print("\n[green]✅ Pelajaran ditandai selesai![/green]")

    # Next lesson
    next_lesson = get_next_lesson(lesson_id)
    if next_lesson:
        console.print(f"\n[dim]Pelajaran berikutnya: stockai learn lesson {next_lesson.id}[/dim]")


@learn_app.command("quiz")
def learn_quiz(
    lesson_id: str = typer.Argument(..., help="Lesson ID to take quiz for"),
) -> None:
    """Take a quiz for a lesson.

    Examples:
        stockai learn quiz basics_01_what_is_stock
    """
    from stockai.tutorial.lessons import get_lesson, LessonProgress
    from stockai.tutorial.quiz import Question, Quiz
    from pathlib import Path

    lesson = get_lesson(lesson_id)
    if not lesson:
        console.print(f"[red]Pelajaran tidak ditemukan:[/red] {lesson_id}")
        raise typer.Exit(1)

    if not lesson.quiz_questions:
        console.print(f"[yellow]Pelajaran ini tidak memiliki quiz.[/yellow]")
        raise typer.Exit(0)

    # Create quiz
    questions = [
        Question(
            text=q["question"],
            options=q["options"],
            correct_index=q["correct"],
        )
        for q in lesson.quiz_questions
    ]
    quiz = Quiz(lesson_id=lesson_id, questions=questions)

    console.print(
        Panel(
            f"[bold]Quiz: {lesson.title}[/bold]\n"
            f"[dim]{len(questions)} pertanyaan[/dim]",
            border_style="cyan",
        )
    )

    # Interactive quiz
    correct = 0
    for i, question in enumerate(questions, 1):
        console.print(f"\n[bold]Pertanyaan {i}/{len(questions)}:[/bold]")
        console.print(f"  {question.text}\n")

        for j, option in enumerate(question.options):
            console.print(f"    {j + 1}. {option}")

        while True:
            try:
                answer = typer.prompt("\nJawaban Anda (1-4)")
                answer_idx = int(answer) - 1
                if 0 <= answer_idx < len(question.options):
                    break
                console.print("[red]Pilih 1-4[/red]")
            except ValueError:
                console.print("[red]Masukkan angka 1-4[/red]")

        if question.check_answer(answer_idx):
            console.print("[green]✅ Benar![/green]")
            correct += 1
        else:
            console.print(f"[red]❌ Salah. Jawaban benar: {question.correct_answer}[/red]")

    # Results
    score = (correct / len(questions)) * 100
    passed = score >= 70

    console.print(
        Panel(
            f"[bold]Hasil Quiz[/bold]\n\n"
            f"Benar: {correct}/{len(questions)}\n"
            f"Score: {score:.0f}%\n"
            f"Status: {'[green]LULUS ✅[/green]' if passed else '[red]BELUM LULUS ❌[/red]'}",
            border_style="green" if passed else "red",
        )
    )

    # Save progress if passed
    if passed:
        progress_path = Path.home() / ".stockai" / "learning_progress.json"
        progress_path.parent.mkdir(parents=True, exist_ok=True)
        progress = LessonProgress.load(progress_path) if progress_path.exists() else LessonProgress()
        progress.complete_lesson(lesson_id)
        progress.set_quiz_score(lesson_id, score)
        progress.save(progress_path)
        console.print("[green]Pelajaran ditandai selesai![/green]")


# ============================================================================
# PAPER TRADING
# ============================================================================

paper_app = typer.Typer(help="Paper trading (simulated) for practice")
app.add_typer(paper_app, name="paper")


@paper_app.command("start")
def paper_start(
    capital: int = typer.Argument(10_000_000, help="Starting capital in Rupiah"),
) -> None:
    """Start a new paper trading account.

    Creates a simulated trading account to practice without real money.

    Examples:
        stockai paper start              # Default Rp 10 million
        stockai paper start 5000000      # Start with Rp 5 million
    """
    from stockai.tutorial.paper_trading import create_paper_account, get_default_paper_path

    path = get_default_paper_path()

    if path.exists():
        if not typer.confirm("Paper account sudah ada. Reset dengan modal baru?"):
            console.print("[yellow]Dibatalkan.[/yellow]")
            raise typer.Exit(0)

    account = create_paper_account(capital=capital, save_path=path)

    console.print(
        Panel(
            f"[bold green]Paper Trading Account Created![/bold green]\n\n"
            f"💰 Modal Awal: Rp {capital:,}\n"
            f"📁 Saved to: {path}\n\n"
            "[dim]Mulai trading dengan:[/dim]\n"
            "  stockai paper buy BBRI 2\n"
            "  stockai paper portfolio",
            title="📝 Paper Trading",
            border_style="green",
        )
    )


@paper_app.command("buy")
def paper_buy(
    symbol: str = typer.Argument(..., help="Stock symbol"),
    lots: int = typer.Argument(..., help="Number of lots to buy"),
    price: float = typer.Option(None, "--price", "-p", help="Buy price (auto-fetch if not specified)"),
    stop_loss: float = typer.Option(None, "--stoploss", "-sl", help="Stop-loss price"),
    target: float = typer.Option(None, "--target", "-t", help="Target price"),
    notes: str = typer.Option("", "--notes", "-n", help="Trade notes"),
) -> None:
    """Execute a paper buy order.

    Examples:
        stockai paper buy BBRI 2
        stockai paper buy BBCA 1 --price 9500 --stoploss 8800 --target 10500
    """
    from stockai.tutorial.paper_trading import PaperTradingAccount, get_default_paper_path
    from stockai.data.sources.yahoo import YahooFinanceSource

    path = get_default_paper_path()
    if not path.exists():
        console.print("[red]Paper account belum dibuat. Jalankan:[/red] stockai paper start")
        raise typer.Exit(1)

    account = PaperTradingAccount.load(path)
    symbol = symbol.upper()

    # Auto-fetch price if not specified
    if price is None:
        with console.status(f"Fetching {symbol} price..."):
            yahoo = YahooFinanceSource()
            ticker = f"{symbol}.JK"
            info = yahoo.get_stock_info(ticker)
            price = info.get("regularMarketPrice") or info.get("previousClose")
            if not price:
                console.print(f"[red]Cannot fetch price for {symbol}. Specify with --price[/red]")
                raise typer.Exit(1)

    result = account.buy(symbol, lots, price, stop_loss, target, notes)

    if isinstance(result, str):
        console.print(f"[red]Error:[/red] {result}")
        raise typer.Exit(1)

    account.save(path)

    total = lots * 100 * price
    console.print(
        Panel(
            f"[bold green]BUY Order Executed[/bold green]\n\n"
            f"📈 {symbol}: {lots} lot @ Rp {price:,.0f}\n"
            f"💵 Total: Rp {total:,.0f} + fee Rp {result.fee:,.0f}\n"
            f"💰 Cash remaining: Rp {account.cash:,.0f}"
            + (f"\n🛑 Stop-loss: Rp {stop_loss:,.0f}" if stop_loss else "")
            + (f"\n🎯 Target: Rp {target:,.0f}" if target else ""),
            title="✅ Paper Trade",
            border_style="green",
        )
    )


@paper_app.command("sell")
def paper_sell(
    symbol: str = typer.Argument(..., help="Stock symbol"),
    lots: int = typer.Argument(..., help="Number of lots to sell"),
    price: float = typer.Option(None, "--price", "-p", help="Sell price (auto-fetch if not specified)"),
    notes: str = typer.Option("", "--notes", "-n", help="Trade notes"),
) -> None:
    """Execute a paper sell order.

    Examples:
        stockai paper sell BBRI 1
        stockai paper sell BBCA 2 --price 10000
    """
    from stockai.tutorial.paper_trading import PaperTradingAccount, get_default_paper_path
    from stockai.data.sources.yahoo import YahooFinanceSource

    path = get_default_paper_path()
    if not path.exists():
        console.print("[red]Paper account belum dibuat. Jalankan:[/red] stockai paper start")
        raise typer.Exit(1)

    account = PaperTradingAccount.load(path)
    symbol = symbol.upper()

    # Auto-fetch price
    if price is None:
        with console.status(f"Fetching {symbol} price..."):
            yahoo = YahooFinanceSource()
            ticker = f"{symbol}.JK"
            info = yahoo.get_stock_info(ticker)
            price = info.get("regularMarketPrice") or info.get("previousClose")
            if not price:
                console.print(f"[red]Cannot fetch price for {symbol}. Specify with --price[/red]")
                raise typer.Exit(1)

    # Calculate P&L before selling
    if symbol in account.positions:
        pos = account.positions[symbol]
        pnl_per_share = price - pos.avg_price
        pnl_total = pnl_per_share * lots * 100
        pnl_pct = (pnl_per_share / pos.avg_price) * 100
    else:
        pnl_total = 0
        pnl_pct = 0

    result = account.sell(symbol, lots, price, notes)

    if isinstance(result, str):
        console.print(f"[red]Error:[/red] {result}")
        raise typer.Exit(1)

    account.save(path)

    pnl_color = "green" if pnl_total >= 0 else "red"
    console.print(
        Panel(
            f"[bold red]SELL Order Executed[/bold red]\n\n"
            f"📉 {symbol}: {lots} lot @ Rp {price:,.0f}\n"
            f"💵 Proceeds: Rp {result.total_value:,.0f} (after fee Rp {result.fee:,.0f})\n"
            f"[{pnl_color}]P&L: Rp {pnl_total:,.0f} ({pnl_pct:+.1f}%)[/{pnl_color}]\n"
            f"💰 Cash: Rp {account.cash:,.0f}",
            title="✅ Paper Trade",
            border_style="red",
        )
    )


@paper_app.command("portfolio")
def paper_portfolio() -> None:
    """View paper trading portfolio.

    Shows current positions, P&L, and account summary.

    Examples:
        stockai paper portfolio
    """
    from stockai.tutorial.paper_trading import PaperTradingAccount, get_default_paper_path
    from stockai.data.sources.yahoo import YahooFinanceSource

    path = get_default_paper_path()
    if not path.exists():
        console.print("[red]Paper account belum dibuat. Jalankan:[/red] stockai paper start")
        raise typer.Exit(1)

    account = PaperTradingAccount.load(path)

    # Update prices
    if account.positions:
        with console.status("Updating prices..."):
            yahoo = YahooFinanceSource()
            prices = {}
            for symbol in account.positions.keys():
                try:
                    info = yahoo.get_stock_info(f"{symbol}.JK")
                    prices[symbol] = info.get("regularMarketPrice") or info.get("previousClose", 0)
                except Exception:
                    pass
            account.update_prices(prices)

    # Summary
    summary = account.get_summary()
    pnl_color = "green" if summary["total_pnl"] >= 0 else "red"

    console.print(
        Panel(
            f"[bold]Paper Trading Portfolio[/bold]\n\n"
            f"💰 Modal Awal: Rp {summary['initial_capital']:,.0f}\n"
            f"💵 Cash: Rp {summary['cash']:,.0f}\n"
            f"📊 Nilai Portfolio: Rp {summary['portfolio_value']:,.0f}\n"
            f"[{pnl_color}]📈 Total P&L: Rp {summary['total_pnl']:,.0f} ({summary['total_pnl_pct']:+.1f}%)[/{pnl_color}]\n"
            f"🎯 Win Rate: {summary['win_rate']:.0f}%\n"
            f"📝 Total Trades: {summary['trades_count']}",
            title="📊 Portfolio Summary",
            border_style="blue",
        )
    )

    # Positions table
    if account.positions:
        table = Table(title="📈 Open Positions", show_header=True)
        table.add_column("Symbol", style="cyan")
        table.add_column("Lots", justify="right")
        table.add_column("Avg Price", justify="right")
        table.add_column("Current", justify="right")
        table.add_column("Value", justify="right")
        table.add_column("P&L", justify="right")
        table.add_column("P&L %", justify="right")

        for symbol, pos in account.positions.items():
            pnl_style = "green" if pos.unrealized_pnl >= 0 else "red"
            table.add_row(
                symbol,
                str(pos.lots),
                f"Rp {pos.avg_price:,.0f}",
                f"Rp {pos.current_price:,.0f}",
                f"Rp {pos.current_value:,.0f}",
                f"[{pnl_style}]Rp {pos.unrealized_pnl:,.0f}[/{pnl_style}]",
                f"[{pnl_style}]{pos.unrealized_pnl_pct:+.1f}%[/{pnl_style}]",
            )

        console.print(table)

        # Check alerts
        warnings = account.check_stop_losses(prices)
        targets = account.check_targets(prices)
        for msg in warnings + targets:
            console.print(msg)
    else:
        console.print("\n[dim]Belum ada posisi. Mulai dengan: stockai paper buy BBRI 2[/dim]")


@paper_app.command("history")
def paper_history(
    limit: int = typer.Option(10, "--limit", "-l", help="Number of trades to show"),
) -> None:
    """View paper trading history.

    Examples:
        stockai paper history
        stockai paper history --limit 20
    """
    from stockai.tutorial.paper_trading import PaperTradingAccount, get_default_paper_path

    path = get_default_paper_path()
    if not path.exists():
        console.print("[red]Paper account belum dibuat.[/red]")
        raise typer.Exit(1)

    account = PaperTradingAccount.load(path)

    if not account.trades:
        console.print("[dim]Belum ada trade history.[/dim]")
        return

    table = Table(title="📜 Trade History", show_header=True)
    table.add_column("Date", style="dim")
    table.add_column("Action")
    table.add_column("Symbol", style="cyan")
    table.add_column("Lots", justify="right")
    table.add_column("Price", justify="right")
    table.add_column("Value", justify="right")
    table.add_column("Fee", justify="right", style="dim")

    for trade in reversed(account.trades[-limit:]):
        action_style = "green" if trade.action.value == "BUY" else "red"
        table.add_row(
            trade.timestamp.strftime("%Y-%m-%d %H:%M"),
            f"[{action_style}]{trade.action.value}[/{action_style}]",
            trade.symbol,
            str(trade.lots),
            f"Rp {trade.price:,.0f}",
            f"Rp {trade.total_value:,.0f}",
            f"Rp {trade.fee:,.0f}",
        )

    console.print(table)


@paper_app.command("reset")
def paper_reset(
    capital: int = typer.Option(10_000_000, "--capital", "-c", help="New starting capital"),
) -> None:
    """Reset paper trading account.

    Clears all positions and history, starts fresh.

    Examples:
        stockai paper reset
        stockai paper reset --capital 5000000
    """
    from stockai.tutorial.paper_trading import create_paper_account, get_default_paper_path

    if not typer.confirm("Reset semua posisi dan history?"):
        console.print("[yellow]Dibatalkan.[/yellow]")
        raise typer.Exit(0)

    path = get_default_paper_path()
    account = create_paper_account(capital=capital, save_path=path)

    console.print(f"[green]Paper account reset dengan modal Rp {capital:,}[/green]")


# =============================================================================
# SCORING COMMANDS - Multi-factor stock scoring
# =============================================================================

score_app = typer.Typer(help="Multi-factor stock scoring system")
app.add_typer(score_app, name="score")


@score_app.command("stock")
def score_stock(
    symbol: str = typer.Argument(..., help="Stock symbol (e.g., BBCA)"),
) -> None:
    """Calculate multi-factor score for a stock.

    Uses hedge fund-style scoring: Value (25%), Quality (30%),
    Momentum (25%), Volatility (20%).

    Examples:
        stockai score stock BBCA
        stockai score stock TLKM
    """
    from stockai.scoring.factors import score_stock, get_score_interpretation
    from stockai.scoring.signals import SignalGenerator, format_signal_for_display
    from stockai.data.sources.yahoo import YahooFinanceSource

    symbol = symbol.upper()
    console.print(f"\n[bold]Calculating score for {symbol}...[/bold]\n")

    source = YahooFinanceSource()

    # Get stock info
    info = source.get_stock_info(symbol)
    if not info:
        console.print(f"[red]Error: Could not fetch data for {symbol}[/red]")
        raise typer.Exit(1)

    # Get price history for technical metrics
    df = source.get_price_history(symbol, period="6mo")

    # Build fundamentals dict
    fundamentals = {
        "pe_ratio": info.get("forwardPE") or info.get("trailingPE"),
        "pb_ratio": info.get("priceToBook"),
        "roe": info.get("returnOnEquity", 0) * 100 if info.get("returnOnEquity") else None,
        "debt_to_equity": info.get("debtToEquity", 0) / 100 if info.get("debtToEquity") else None,
        "profit_margin": info.get("profitMargins", 0) * 100 if info.get("profitMargins") else None,
        "market_cap": info.get("marketCap"),
    }

    # Build price data dict
    price_data = {}
    if not df.empty:
        returns = (df["close"].iloc[-1] / df["close"].iloc[0] - 1) * 100
        price_data["returns_6m"] = returns
        if len(df) >= 60:
            price_data["returns_3m"] = (df["close"].iloc[-1] / df["close"].iloc[-60] - 1) * 100
        if len(df) >= 20:
            price_data["returns_1m"] = (df["close"].iloc[-1] / df["close"].iloc[-20] - 1) * 100
            price_data["std_dev"] = df["close"].pct_change().std() * (252 ** 0.5) * 100

        # Calculate beta (simplified)
        price_data["beta"] = info.get("beta", 1.0)

    # Calculate scores
    scores = score_stock(symbol, fundamentals, price_data)

    # Display results
    console.print(Panel(
        f"[bold cyan]{symbol}[/bold cyan] Multi-Factor Score\n\n"
        f"[bold]Composite Score: {scores.composite_score:.0f}/100[/bold]\n\n"
        f"  📊 Value (25%):      {scores.value_score:5.0f}/100\n"
        f"  ⭐ Quality (30%):    {scores.quality_score:5.0f}/100\n"
        f"  📈 Momentum (25%):   {scores.momentum_score:5.0f}/100\n"
        f"  🛡️ Volatility (20%): {scores.volatility_score:5.0f}/100\n\n"
        f"[dim]Interpretation: {get_score_interpretation(scores.composite_score)}[/dim]",
        title="📊 Factor Scores",
        border_style="blue",
    ))

    # Show metrics
    table = Table(title="📋 Underlying Metrics", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")

    if scores.pe_ratio:
        table.add_row("P/E Ratio", f"{scores.pe_ratio:.1f}")
    if scores.pb_ratio:
        table.add_row("P/B Ratio", f"{scores.pb_ratio:.2f}")
    if scores.roe:
        table.add_row("ROE", f"{scores.roe:.1f}%")
    if scores.debt_to_equity:
        table.add_row("Debt/Equity", f"{scores.debt_to_equity:.2f}")
    if scores.profit_margin:
        table.add_row("Profit Margin", f"{scores.profit_margin:.1f}%")
    if scores.momentum_6m:
        table.add_row("6M Return", f"{scores.momentum_6m:.1f}%")
    if scores.beta:
        table.add_row("Beta", f"{scores.beta:.2f}")
    if scores.std_dev:
        table.add_row("Volatility", f"{scores.std_dev:.1f}%")

    console.print(table)

    # Generate signal
    current_price = info.get("regularMarketPrice") or info.get("previousClose", 0)
    if current_price > 0:
        signal_gen = SignalGenerator()
        signal = signal_gen.generate_signal(
            symbol=symbol,
            current_score=scores.composite_score,
            current_price=current_price,
            momentum_score=scores.momentum_score,
        )

        console.print()
        console.print(format_signal_for_display(signal))


@score_app.command("rank")
def score_rank(
    index: str = typer.Option("IDX30", "--index", "-i", help="Index to rank (IDX30, LQ45)"),
    top: int = typer.Option(10, "--top", "-t", help="Number of top stocks to show"),
) -> None:
    """Rank stocks by composite score.

    Shows top N stocks in an index ranked by multi-factor score.

    Examples:
        stockai score rank
        stockai score rank --index LQ45 --top 15
    """
    from stockai.scoring.factors import score_stock
    from stockai.data.sources.yahoo import YahooFinanceSource
    from stockai.data.sources.idx import IDXIndexSource

    console.print(f"\n[bold]Ranking {index} stocks by composite score...[/bold]\n")
    console.print("[dim]This may take a minute to fetch all data...[/dim]\n")

    # Get stock list
    idx_source = IDXIndexSource()
    if index.upper() == "IDX30":
        stocks = idx_source.get_idx30_stocks()
    else:
        stocks = idx_source.get_lq45_stocks()

    source = YahooFinanceSource()
    rankings = []

    with console.status("[bold green]Fetching stock data..."):
        for stock in stocks:
            symbol = stock["symbol"]
            try:
                info = source.get_stock_info(symbol)
                if not info:
                    continue

                df = source.get_price_history(symbol, period="6mo")

                fundamentals = {
                    "pe_ratio": info.get("forwardPE") or info.get("trailingPE"),
                    "pb_ratio": info.get("priceToBook"),
                    "roe": info.get("returnOnEquity", 0) * 100 if info.get("returnOnEquity") else None,
                    "debt_to_equity": info.get("debtToEquity", 0) / 100 if info.get("debtToEquity") else None,
                    "profit_margin": info.get("profitMargins", 0) * 100 if info.get("profitMargins") else None,
                }

                price_data = {}
                if not df.empty:
                    returns = (df["close"].iloc[-1] / df["close"].iloc[0] - 1) * 100
                    price_data["returns_6m"] = returns
                    price_data["beta"] = info.get("beta", 1.0)
                    if len(df) >= 20:
                        price_data["std_dev"] = df["close"].pct_change().std() * (252 ** 0.5) * 100

                scores = score_stock(symbol, fundamentals, price_data)
                rankings.append({
                    "symbol": symbol,
                    "score": scores.composite_score,
                    "value": scores.value_score,
                    "quality": scores.quality_score,
                    "momentum": scores.momentum_score,
                    "volatility": scores.volatility_score,
                    "price": info.get("regularMarketPrice", 0),
                })
            except Exception as e:
                console.print(f"[dim]Skipped {symbol}: {e}[/dim]")

    # Sort by score
    rankings.sort(key=lambda x: x["score"], reverse=True)

    # Display table
    table = Table(title=f"🏆 Top {top} Stocks by Score ({index})", show_header=True)
    table.add_column("#", style="dim", width=3)
    table.add_column("Symbol", style="cyan")
    table.add_column("Score", justify="right", style="bold")
    table.add_column("Value", justify="right")
    table.add_column("Quality", justify="right")
    table.add_column("Mom.", justify="right")
    table.add_column("Vol.", justify="right")
    table.add_column("Price", justify="right")

    for i, stock in enumerate(rankings[:top], 1):
        score_style = "green" if stock["score"] >= 70 else "yellow" if stock["score"] >= 50 else "red"
        table.add_row(
            str(i),
            stock["symbol"],
            f"[{score_style}]{stock['score']:.0f}[/{score_style}]",
            f"{stock['value']:.0f}",
            f"{stock['quality']:.0f}",
            f"{stock['momentum']:.0f}",
            f"{stock['volatility']:.0f}",
            f"Rp {stock['price']:,.0f}" if stock['price'] else "-",
        )

    console.print(table)


# =============================================================================
# RISK COMMANDS - Position sizing and risk management
# =============================================================================

risk_app = typer.Typer(help="Risk management tools")
app.add_typer(risk_app, name="risk")


@risk_app.command("position")
def risk_position_size(
    symbol: str = typer.Argument(..., help="Stock symbol"),
    capital: int = typer.Option(5_000_000, "--capital", "-c", help="Total capital in Rupiah"),
    stop_loss_pct: float = typer.Option(8.0, "--stop", "-s", help="Stop-loss percentage below entry"),
    target_pct: float = typer.Option(15.0, "--target", "-t", help="Target percentage above entry"),
) -> None:
    """Calculate optimal position size using 2% risk rule.

    Determines how many lots to buy based on your capital and risk tolerance.

    Examples:
        stockai risk position BBCA
        stockai risk position BBRI --capital 10000000 --stop 10
    """
    from stockai.risk.position_sizing import calculate_position_size, format_position_size_for_display
    from stockai.data.sources.yahoo import YahooFinanceSource

    symbol = symbol.upper()
    console.print(f"\n[bold]Calculating position size for {symbol}...[/bold]\n")

    source = YahooFinanceSource()
    info = source.get_stock_info(symbol)

    if not info:
        console.print(f"[red]Error: Could not fetch data for {symbol}[/red]")
        raise typer.Exit(1)

    current_price = info.get("regularMarketPrice") or info.get("previousClose", 0)
    if current_price <= 0:
        console.print(f"[red]Error: No price data for {symbol}[/red]")
        raise typer.Exit(1)

    stop_loss_price = current_price * (1 - stop_loss_pct / 100)
    target_price = current_price * (1 + target_pct / 100)

    pos = calculate_position_size(
        capital=capital,
        entry_price=current_price,
        stop_loss_price=stop_loss_price,
        target_price=target_price,
        symbol=symbol,
    )

    console.print(format_position_size_for_display(pos, capital))


@risk_app.command("diversification")
def risk_diversification() -> None:
    """Check portfolio diversification against limits.

    Ensures you're not too concentrated in any stock or sector.

    Examples:
        stockai risk diversification
    """
    from stockai.risk.diversification import check_diversification, format_diversification_for_display, DiversificationLimits
    from stockai.tutorial.paper_trading import PaperTradingAccount, get_default_paper_path
    from stockai.data import get_stock_sector

    path = get_default_paper_path()
    if not path.exists():
        console.print("[red]Paper account not found. Use 'stockai paper start' first.[/red]")
        raise typer.Exit(1)

    account = PaperTradingAccount.load(path)

    if not account.positions:
        console.print("[yellow]No positions to check.[/yellow]")
        return

    # Build positions list with sector info
    positions = []
    for symbol, pos in account.positions.items():
        sector = get_stock_sector(symbol) or "Unknown"
        positions.append({
            "symbol": symbol,
            "value": pos.current_value or pos.total_cost,
            "sector": sector,
        })

    check = check_diversification(positions)
    console.print(format_diversification_for_display(check))


@risk_app.command("portfolio")
def risk_portfolio() -> None:
    """Analyze portfolio-level risk metrics.

    Shows volatility, VaR, drawdown, and other risk measures.

    Examples:
        stockai risk portfolio
    """
    from stockai.risk.portfolio_risk import calculate_portfolio_risk, format_portfolio_risk_for_display
    from stockai.tutorial.paper_trading import PaperTradingAccount, get_default_paper_path

    path = get_default_paper_path()
    if not path.exists():
        console.print("[red]Paper account not found. Use 'stockai paper start' first.[/red]")
        raise typer.Exit(1)

    account = PaperTradingAccount.load(path)

    positions = []
    for symbol, pos in account.positions.items():
        positions.append({
            "symbol": symbol,
            "value": pos.current_value or pos.total_cost,
            "weight": 0,  # Will be calculated
        })

    risk = calculate_portfolio_risk(positions)
    console.print(format_portfolio_risk_for_display(risk))


# =============================================================================
# BRIEFING COMMANDS - Daily and weekly briefings
# =============================================================================

@app.command("morning")
def briefing_morning() -> None:
    """Get morning briefing before market open.

    Shows critical alerts, portfolio snapshot, and today's focus.
    Designed to be read in 5 minutes.

    Examples:
        stockai morning
    """
    from stockai.briefing.daily import generate_morning_briefing, format_morning_briefing
    from stockai.tutorial.paper_trading import PaperTradingAccount, get_default_paper_path
    from stockai.data.sources.yahoo import YahooFinanceSource

    path = get_default_paper_path()
    portfolio = None

    if path.exists():
        account = PaperTradingAccount.load(path)

        # Update prices
        source = YahooFinanceSource()
        prices = {}
        for symbol in account.positions:
            try:
                info = source.get_stock_info(symbol)
                if info:
                    prices[symbol] = info.get("regularMarketPrice") or info.get("previousClose", 0)
            except Exception:
                pass
        account.update_prices(prices)

        portfolio = {
            "positions": {s: {
                "symbol": s,
                "current_price": p.current_price,
                "avg_price": p.avg_price,
                "shares": p.shares,
                "current_value": p.current_value,
                "stop_loss": p.stop_loss,
                "target": p.target,
            } for s, p in account.positions.items()},
            "cash": account.cash,
            "initial_capital": account.initial_capital,
        }

    briefing = generate_morning_briefing(portfolio=portfolio)
    console.print(format_morning_briefing(briefing))


@app.command("evening")
def briefing_evening() -> None:
    """Get evening briefing after market close.

    Shows today's performance, trades, and tomorrow's focus.
    Designed to be read in 5 minutes.

    Examples:
        stockai evening
    """
    from stockai.briefing.daily import generate_evening_briefing, format_evening_briefing
    from stockai.tutorial.paper_trading import PaperTradingAccount, get_default_paper_path
    from stockai.data.sources.yahoo import YahooFinanceSource

    path = get_default_paper_path()
    portfolio = None
    trades_today = []

    if path.exists():
        account = PaperTradingAccount.load(path)

        # Update prices
        source = YahooFinanceSource()
        prices = {}
        for symbol in account.positions:
            try:
                info = source.get_stock_info(symbol)
                if info:
                    prices[symbol] = info.get("regularMarketPrice") or info.get("previousClose", 0)
            except Exception:
                pass
        account.update_prices(prices)

        portfolio = {
            "positions": {s: {
                "symbol": s,
                "current_price": p.current_price,
                "avg_price": p.avg_price,
                "shares": p.shares,
                "current_value": p.current_value,
            } for s, p in account.positions.items()},
            "cash": account.cash,
            "initial_capital": account.initial_capital,
        }

        # Get today's trades
        from datetime import datetime
        today = datetime.now().date()
        trades_today = [
            {
                "action": t.action.value,
                "symbol": t.symbol,
                "lots": t.lots,
                "price": t.price,
            }
            for t in account.trades
            if t.timestamp.date() == today
        ]

    briefing = generate_evening_briefing(portfolio=portfolio, trades_today=trades_today)
    console.print(format_evening_briefing(briefing))


@app.command("weekly")
def briefing_weekly() -> None:
    """Get weekly performance review.

    Shows week's performance, trade analysis, and lessons learned.
    Designed for 30-minute weekend review.

    Examples:
        stockai weekly
    """
    from stockai.briefing.weekly import generate_weekly_review, format_weekly_review
    from stockai.tutorial.paper_trading import PaperTradingAccount, get_default_paper_path
    from stockai.data.sources.yahoo import YahooFinanceSource
    from datetime import datetime, timedelta

    path = get_default_paper_path()

    if not path.exists():
        console.print("[red]Paper account not found. Use 'stockai paper start' first.[/red]")
        raise typer.Exit(1)

    account = PaperTradingAccount.load(path)

    # Update prices
    source = YahooFinanceSource()
    prices = {}
    for symbol in account.positions:
        try:
            info = source.get_stock_info(symbol)
            if info:
                prices[symbol] = info.get("regularMarketPrice") or info.get("previousClose", 0)
        except Exception:
            pass
    account.update_prices(prices)

    portfolio = {
        "positions": {s: {
            "symbol": s,
            "current_price": p.current_price,
            "avg_price": p.avg_price,
            "shares": p.shares,
            "current_value": p.current_value,
        } for s, p in account.positions.items()},
        "cash": account.cash,
        "initial_capital": account.initial_capital,
    }

    # Get this week's trades
    now = datetime.now()
    week_start = now - timedelta(days=now.weekday())
    trades_this_week = [
        {
            "action": t.action.value,
            "symbol": t.symbol,
            "lots": t.lots,
            "price": t.price,
            "pnl": 0,  # Would need to calculate actual P&L
        }
        for t in account.trades
        if t.timestamp >= week_start
    ]

    # Get IHSG performance (simplified)
    ihsg_return = 0.0
    try:
        ihsg_df = source.get_price_history("^JKSE", period="1wk")
        if not ihsg_df.empty and len(ihsg_df) >= 2:
            ihsg_return = (ihsg_df["close"].iloc[-1] / ihsg_df["close"].iloc[0] - 1) * 100
    except Exception:
        pass

    review = generate_weekly_review(
        portfolio=portfolio,
        trades_this_week=trades_this_week,
        ihsg_weekly_return=ihsg_return,
    )
    console.print(format_weekly_review(review))


if __name__ == "__main__":
    app()
