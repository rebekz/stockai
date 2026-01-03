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
) -> None:
    """Predict stock price movement (UP/DOWN).

    Uses ML ensemble (XGBoost + LSTM + Sentiment) to
    predict stock direction.

    Examples:
        stock predict BBCA
        stock predict TLKM --horizon 7
        stock predict BBRI --verbose
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
            from stockai.core.predictor import EnsemblePredictor

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

            from stockai.core.predictor import EnsemblePredictor

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
    from stockai.core.sentiment import SentimentAnalyzer, NewsAggregator

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

        # Analyze sentiment
        analyzer = SentimentAnalyzer()
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
    from stockai.core.sentiment import SentimentAnalyzer, NewsAggregator

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

        analyzer = SentimentAnalyzer()
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


# Watchlist subcommand group
watchlist_app = typer.Typer(help="Manage stock watchlist")
app.add_typer(watchlist_app, name="watchlist")


@watchlist_app.command("list")
def watchlist_list() -> None:
    """List all stocks in watchlist."""
    console.print(
        Panel(
            "[yellow]Watchlist - Coming soon![/yellow]\n\n"
            "Will show:\n"
            "• Watched stocks with current prices\n"
            "• Daily change %\n"
            "• Alert conditions",
            title="👀 Watchlist",
        )
    )


@watchlist_app.command("add")
def watchlist_add(
    symbol: str = typer.Argument(..., help="Stock symbol to watch"),
) -> None:
    """Add a stock to watchlist."""
    console.print(f"[green]Adding {symbol.upper()} to watchlist[/green] - Coming soon!")


@watchlist_app.command("remove")
def watchlist_remove(
    symbol: str = typer.Argument(..., help="Stock symbol to remove"),
) -> None:
    """Remove a stock from watchlist."""
    console.print(f"[yellow]Removing {symbol.upper()} from watchlist - Coming soon![/yellow]")


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


if __name__ == "__main__":
    app()
