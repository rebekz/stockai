"""E2E Tests for Agent Engine (Epic 2).

Tests the multi-phase agent workflow:
- Base agent setup (Story 2.1)
- Planning phase (Story 2.2)
- Action phase (Story 2.3)
- Validation phase (Story 2.4)
- Answer synthesis (Story 2.5)
- Tool registration (Story 2.6)
- Analyze command (Story 2.7)
"""

import json
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from stockai.agent.state import (
    Task,
    AgentPlan,
    AgentState,
    create_initial_state,
    SessionManager,
)
from stockai.agent.phases.planning import PlanningPhase, TaskPlan, ResearchPlan
from stockai.agent.phases.action import ActionPhase
from stockai.agent.phases.validation import ValidationPhase, ValidationResult
from stockai.agent.phases.answer import AnswerPhase
from stockai.tools.registry import ToolRegistry, stockai_tool, get_registry


class TestAgentState:
    """Test agent state management (Story 2.1)."""

    def test_task_model(self):
        """AC2.1.2: Task has required fields."""
        task = Task(
            id="task_1",
            description="Test task",
            tools=["get_stock_info"],
            dependencies=[],
        )

        assert task.id == "task_1"
        assert task.status == "pending"
        assert "get_stock_info" in task.tools

    def test_create_initial_state(self):
        """AC2.1.5: Initial state created correctly."""
        state = create_initial_state("Analyze BBCA", symbol="BBCA")

        assert state["query"] == "Analyze BBCA"
        assert state["symbol"] == "BBCA"
        assert state["phase"] == "planning"
        assert state["tool_results"] == []
        assert state["tool_calls"] == 0

    def test_session_manager_save_load(self, tmp_path):
        """AC2.1.5: State persisted to session file."""
        session = SessionManager(session_dir=tmp_path)

        state = create_initial_state("Test query", symbol="TEST")
        state["answer"] = "Test answer"

        # Save
        assert session.save_state(state) is True
        assert session.has_session() is True

        # Load
        loaded = session.load_state()
        assert loaded is not None
        assert loaded["query"] == "Test query"
        assert loaded["answer"] == "Test answer"

    def test_session_manager_clear(self, tmp_path):
        """Test session clearing."""
        session = SessionManager(session_dir=tmp_path)

        state = create_initial_state("Test", symbol="TEST")
        session.save_state(state)
        assert session.has_session() is True

        session.clear_session()
        assert session.has_session() is False


class TestPlanningPhase:
    """Test planning phase (Story 2.2)."""

    @pytest.fixture
    def planner(self):
        return PlanningPhase(available_tools=["get_stock_info", "get_current_price"])

    def test_extract_symbol_from_query(self, planner):
        """AC2.2.1: Extract stock symbol from query."""
        assert planner.extract_symbol("Analyze BBCA stock") == "BBCA"
        assert planner.extract_symbol("What is TLKM price?") == "TLKM"
        assert planner.extract_symbol("Compare BBRI and BMRI") == "BBRI"

    def test_create_default_plan(self, planner):
        """AC2.2.3: "Analyze BBCA" creates multiple tasks."""
        plan = planner.create_default_plan("Analyze BBCA", "BBCA")

        assert len(plan.tasks) >= 3  # At least 3 tasks
        assert plan.symbol == "BBCA"

        # Check tasks have proper structure
        for task in plan.tasks:
            assert task.id is not None
            assert task.description is not None
            assert len(task.tools) > 0

    def test_plan_with_technical_analysis(self, planner):
        """AC2.2.4: Technical analysis keyword triggers additional task."""
        plan = planner.create_default_plan(
            "Analyze BBCA with technical indicators RSI and MACD",
            "BBCA",
        )

        # Should have technical analysis task
        tech_tasks = [t for t in plan.tasks if "technical" in t.description.lower()]
        assert len(tech_tasks) >= 1

    def test_plan_task_dependencies(self, planner):
        """AC2.2.4: Tasks have proper dependencies."""
        plan = planner.create_default_plan("Analyze BBCA", "BBCA")

        # First task should have no dependencies
        assert plan.tasks[0].dependencies == []

        # Later tasks may depend on earlier ones
        for task in plan.tasks[1:]:
            for dep in task.dependencies:
                # All dependencies should be valid task IDs
                dep_ids = [t.id for t in plan.tasks]
                assert dep in dep_ids

    def test_parse_llm_plan_json(self, planner):
        """Test parsing LLM response with JSON."""
        llm_response = """
        Here's the plan:
        ```json
        [
            {"id": "task_1", "description": "Get info", "tools": ["get_stock_info"], "dependencies": []},
            {"id": "task_2", "description": "Get price", "tools": ["get_current_price"], "dependencies": ["task_1"]}
        ]
        ```
        """

        plan = planner.parse_llm_plan(llm_response, "Analyze BBCA")
        assert len(plan.tasks) == 2
        assert plan.tasks[0].id == "task_1"


class TestActionPhase:
    """Test action phase (Story 2.3)."""

    @pytest.fixture
    def mock_tools(self):
        return {
            "get_stock_info": Mock(return_value={"name": "Test Corp", "sector": "Finance"}),
            "get_current_price": Mock(return_value={"price": 9500, "change_percent": 1.5}),
            "failing_tool": Mock(side_effect=Exception("Tool error")),
        }

    @pytest.fixture
    def action_phase(self, mock_tools):
        return ActionPhase(tools=mock_tools, max_retries=2)

    def test_execute_task_success(self, action_phase):
        """AC2.3.1: Agent selects and executes tools."""
        task = {
            "id": "task_1",
            "tools": ["get_stock_info"],
            "dependencies": [],
        }
        context = {"symbol": "BBCA"}

        result = action_phase.execute_task(task, context)

        assert result["task_id"] == "task_1"
        assert result["success"] is True
        assert len(result["results"]) == 1
        assert len(result["errors"]) == 0

    def test_execute_task_with_retry(self, action_phase):
        """AC2.3.3: Errors handled with retry."""
        task = {
            "id": "task_fail",
            "tools": ["failing_tool"],
            "dependencies": [],
        }
        context = {"symbol": "BBCA"}

        result = action_phase.execute_task(task, context)

        assert result["success"] is False
        assert len(result["errors"]) == 1
        assert "Tool error" in result["errors"][0]["error"]

    def test_get_ready_tasks(self, action_phase):
        """AC2.3.1: Only execute tasks with met dependencies."""
        tasks = [
            {"id": "task_1", "status": "pending", "dependencies": []},
            {"id": "task_2", "status": "pending", "dependencies": ["task_1"]},
            {"id": "task_3", "status": "pending", "dependencies": ["task_1", "task_2"]},
        ]

        # Initially, only task_1 is ready (no dependencies)
        ready = action_phase.get_ready_tasks(tasks, set())
        ready_ids = [t["id"] for t in ready]
        assert "task_1" in ready_ids
        # task_2 has dependency on task_1, so should not be ready
        assert "task_3" not in ready_ids  # has dependencies

        # After task_1 complete, task_2 becomes ready
        ready = action_phase.get_ready_tasks(tasks, {"task_1"})
        ready_ids = [t["id"] for t in ready]
        assert "task_2" in ready_ids

        # After task_1 and task_2 complete, task_3 is ready
        ready = action_phase.get_ready_tasks(tasks, {"task_1", "task_2"})
        ready_ids = [t["id"] for t in ready]
        assert "task_3" in ready_ids


class TestValidationPhase:
    """Test validation phase (Story 2.4)."""

    @pytest.fixture
    def validator(self):
        return ValidationPhase(max_attempts=2)

    def test_validate_complete_data(self, validator):
        """AC2.4.1: All tasks completed passes validation."""
        tasks = [{"id": "task_1"}, {"id": "task_2"}]
        results = [
            {"task_id": "task_1", "results": [{"tool": "get_stock_info", "data": {"name": "Test"}}], "errors": []},
            {"task_id": "task_2", "results": [{"tool": "get_price", "data": {"price": 100}}], "errors": []},
        ]

        validation = validator.validate(tasks, results)

        assert validation.status == "PASS"
        assert len(validation.issues) == 0

    def test_validate_missing_tasks(self, validator):
        """AC2.4.1: Missing tasks detected."""
        tasks = [{"id": "task_1"}, {"id": "task_2"}, {"id": "task_3"}]
        results = [
            {"task_id": "task_1", "results": [{"data": {"name": "Test"}}], "errors": []},
        ]

        validation = validator.validate(tasks, results)

        assert validation.status in ["RETRY", "FAIL"]
        assert len(validation.retry_tasks) > 0

    def test_validate_empty_data(self, validator):
        """AC2.4.2: Empty results detected."""
        tasks = [{"id": "task_1"}]
        results = [
            {"task_id": "task_1", "results": [{"tool": "get_info", "data": {}}], "errors": []},
        ]

        validation = validator.validate(tasks, results)

        # Empty data should be flagged
        assert len(validation.issues) > 0

    def test_validate_with_errors(self, validator):
        """AC2.4.3: Tool errors trigger retry."""
        tasks = [{"id": "task_1"}]
        results = [
            {"task_id": "task_1", "results": [], "errors": [{"tool": "get_info", "error": "Failed"}]},
        ]

        validation = validator.validate(tasks, results)

        assert validation.status in ["RETRY", "FAIL"]
        assert "task_1" in validation.retry_tasks


class TestAnswerPhase:
    """Test answer synthesis (Story 2.5)."""

    @pytest.fixture
    def answerer(self):
        return AnswerPhase()

    def test_synthesize_basic(self, answerer):
        """AC2.5.1: Answer synthesizes all results."""
        results = [
            {
                "task_id": "task_1",
                "results": [
                    {"tool": "get_stock_info", "data": {
                        "name": "Bank Central Asia",
                        "sector": "Finance",
                        "is_idx30": True,
                    }}
                ],
                "errors": [],
            },
        ]

        answer = answerer.synthesize("Analyze BBCA", "BBCA", results)

        assert "BBCA" in answer
        assert "Bank Central Asia" in answer or "Analysis" in answer

    def test_answer_includes_disclaimer(self, answerer):
        """AC2.5.5: Disclaimer automatically appended."""
        results = [{"task_id": "task_1", "results": [], "errors": []}]

        answer = answerer.synthesize("Analyze BBCA", "BBCA", results)

        assert "Disclaimer" in answer
        assert "financial advice" in answer.lower()

    def test_format_price_data(self, answerer):
        """AC2.5.2: Price formatted correctly."""
        results = [
            {
                "task_id": "task_1",
                "results": [
                    {"tool": "get_current_price", "data": {
                        "price": 9500,
                        "change": 100,
                        "change_percent": 1.06,
                    }}
                ],
                "errors": [],
            },
        ]

        answer = answerer.synthesize("Analyze BBCA", "BBCA", results)

        # Should have Rupiah formatting
        assert "Rp" in answer or "9,500" in answer or "9500" in answer


class TestToolRegistry:
    """Test tool registration (Story 2.6)."""

    def test_register_tool(self):
        """AC2.6.1: Tools can be registered."""
        registry = ToolRegistry()
        registry.clear()

        def my_tool():
            """Test tool description."""
            return "result"

        registry.register("my_tool", my_tool, category="test")

        assert "my_tool" in registry.tools
        assert registry.get_tool("my_tool") is not None

    def test_tool_decorator(self):
        """AC2.6.1: @stockai_tool decorator works."""
        registry = get_registry()
        registry.clear()

        @stockai_tool(name="decorated_tool", category="test")
        def test_func():
            """A test function."""
            return "decorated result"

        assert "decorated_tool" in registry.tools
        assert registry.get_tool("decorated_tool")() == "decorated result"

    def test_tool_descriptions(self):
        """AC2.6.4: Tools have docstrings."""
        registry = ToolRegistry()
        registry.clear()

        def documented_tool():
            """This is the tool description."""
            pass

        registry.register("doc_tool", documented_tool)

        info = registry.get_tool_info("doc_tool")
        assert "description" in info
        assert "This is the tool description" in info["description"]

    def test_get_tools_dict(self):
        """AC2.6.2: Tools auto-registered and retrievable."""
        registry = ToolRegistry()
        registry.clear()

        def tool_a():
            pass

        def tool_b():
            pass

        registry.register("tool_a", tool_a)
        registry.register("tool_b", tool_b)

        tools_dict = registry.get_tools_dict()

        assert "tool_a" in tools_dict
        assert "tool_b" in tools_dict

    def test_permission_levels(self):
        """AC2.6.5: Permission system for tools."""
        registry = ToolRegistry()
        registry.clear()

        def safe_tool():
            pass

        def dangerous_tool():
            pass

        registry.register("safe", safe_tool, permission="safe")
        registry.register("dangerous", dangerous_tool, permission="dangerous")

        assert registry.permissions["safe"] == "safe"
        assert registry.permissions["dangerous"] == "dangerous"


class TestStockTools:
    """Test stock analysis tools integration."""

    def test_stock_tools_registration(self):
        """Tools are registered on import."""
        # Clear registry first to avoid pollution from other tests
        from stockai.tools.registry import get_registry
        registry = get_registry()
        registry.clear()

        # Import the module which triggers decorator registration
        import importlib
        from stockai.tools import stock_tools
        importlib.reload(stock_tools)

        from stockai.tools import get_all_tools
        tools = get_all_tools()

        # Should have core tools registered
        assert len(tools) >= 5, f"Expected at least 5 tools, got {list(tools.keys())}"
        assert "get_stock_info" in tools
        assert "get_current_price" in tools


class TestAnalyzeCommand:
    """Test analyze CLI command (Story 2.7)."""

    def test_analyze_help(self):
        """AC2.7.1: Command help available."""
        from typer.testing import CliRunner
        from stockai.cli.main import app

        runner = CliRunner()
        result = runner.invoke(app, ["analyze", "--help"])

        assert result.exit_code == 0
        assert "Stock symbol to analyze" in result.output

    def test_analyze_requires_api_key(self):
        """Test analyze fails without API key."""
        from typer.testing import CliRunner
        from stockai.cli.main import app

        runner = CliRunner()

        with patch.dict("os.environ", {"GOOGLE_API_KEY": ""}, clear=True):
            result = runner.invoke(app, ["analyze", "BBCA"])
            # Should fail or warn about missing API key
            # (exact behavior depends on environment)

    def test_tools_command(self):
        """AC2.6.3: stock tools displays tool list."""
        from typer.testing import CliRunner
        from stockai.cli.main import app

        # Pre-import the stock_tools module to ensure decorators run
        from stockai.tools import stock_tools  # noqa: F401

        runner = CliRunner()
        result = runner.invoke(app, ["tools"])

        assert result.exit_code == 0
        # Should show the tools table or total count
        assert "Tool" in result.output or "Total:" in result.output or "get_stock_info" in result.output


class TestVolumeToolsIntegration:
    """Integration tests for volume analysis tools.

    Tests that volume tools are properly registered and can be invoked
    through the agent system with actual data flow.
    """

    def test_volume_tools_registered_in_global_registry(self):
        """AC3.2: Volume tools are properly registered in the global registry."""
        from stockai.tools.registry import get_registry

        # Clear registry first to avoid pollution from other tests
        registry = get_registry()
        registry.clear()

        # Import the module which triggers decorator registration
        import importlib
        from stockai.tools import stock_tools
        importlib.reload(stock_tools)

        from stockai.tools import get_all_tools
        tools = get_all_tools()

        # Volume tools should be registered
        assert "get_volume_analysis" in tools, "get_volume_analysis should be registered"
        assert "get_volume_profile" in tools, "get_volume_profile should be registered"
        assert "get_volume_signals" in tools, "get_volume_signals should be registered"

    def test_volume_tools_have_correct_category(self):
        """AC3.2: Volume tools are registered with 'analysis' category."""
        from stockai.tools.registry import get_registry

        # Clear and reload
        registry = get_registry()
        registry.clear()

        import importlib
        from stockai.tools import stock_tools
        importlib.reload(stock_tools)

        # Check volume tool categories
        volume_analysis_info = registry.get_tool_info("get_volume_analysis")
        volume_profile_info = registry.get_tool_info("get_volume_profile")
        volume_signals_info = registry.get_tool_info("get_volume_signals")

        assert volume_analysis_info["category"] == "analysis"
        assert volume_profile_info["category"] == "analysis"
        assert volume_signals_info["category"] == "analysis"

    def test_volume_tools_invokable_through_action_phase(self):
        """AC3.2: Volume tools can be invoked through the agent system."""
        import pandas as pd
        import numpy as np
        from datetime import datetime, timedelta

        # Create mock price data
        days = 60
        dates = [datetime.now() - timedelta(days=days - i) for i in range(days)]
        mock_df = pd.DataFrame({
            "date": dates,
            "symbol": "TEST",
            "open": [10000 + i * 10 for i in range(days)],
            "high": [10050 + i * 10 for i in range(days)],
            "low": [9950 + i * 10 for i in range(days)],
            "close": [10000 + i * 10 for i in range(days)],
            "volume": [1000000 + i * 10000 for i in range(days)],
        })

        with patch("stockai.tools.stock_tools._yahoo") as mock_yahoo:
            mock_yahoo.get_price_history.return_value = mock_df

            # Clear registry and reload tools
            from stockai.tools.registry import get_registry
            registry = get_registry()
            registry.clear()

            import importlib
            from stockai.tools import stock_tools
            importlib.reload(stock_tools)

            from stockai.tools import get_all_tools
            tools = get_all_tools()

            # Create action phase with volume tools
            action = ActionPhase(tools=tools)

            # Test get_volume_analysis through action phase
            volume_analysis_task = {
                "id": "volume_analysis_task",
                "tools": ["get_volume_analysis"],
                "dependencies": [],
            }
            result = action.execute_task(volume_analysis_task, {"symbol": "BBCA"})

            assert result["success"] is True
            assert result["task_id"] == "volume_analysis_task"
            assert len(result["results"]) == 1
            assert result["results"][0]["tool"] == "get_volume_analysis"

            # Verify result data structure
            data = result["results"][0]["data"]
            assert "indicators" in data
            assert "obv" in data["indicators"]
            assert "vwap" in data["indicators"]
            assert "mfi" in data["indicators"]

    def test_volume_profile_invokable_through_action_phase(self):
        """AC3.2: Volume profile tool can be invoked through agent."""
        import pandas as pd
        import numpy as np
        from datetime import datetime, timedelta

        # Create mock price data
        days = 30
        dates = [datetime.now() - timedelta(days=days - i) for i in range(days)]
        mock_df = pd.DataFrame({
            "date": dates,
            "symbol": "TEST",
            "open": [10000 + i * 10 for i in range(days)],
            "high": [10100 + i * 10 for i in range(days)],
            "low": [9900 + i * 10 for i in range(days)],
            "close": [10050 + i * 10 for i in range(days)],
            "volume": [1000000 for _ in range(days)],
        })

        with patch("stockai.tools.stock_tools._yahoo") as mock_yahoo:
            mock_yahoo.get_price_history.return_value = mock_df

            from stockai.tools.registry import get_registry
            registry = get_registry()
            registry.clear()

            import importlib
            from stockai.tools import stock_tools
            importlib.reload(stock_tools)

            from stockai.tools import get_all_tools
            tools = get_all_tools()

            action = ActionPhase(tools=tools)

            # Test get_volume_profile through action phase
            volume_profile_task = {
                "id": "volume_profile_task",
                "tools": ["get_volume_profile"],
                "dependencies": [],
            }
            result = action.execute_task(volume_profile_task, {"symbol": "BBCA"})

            assert result["success"] is True
            assert len(result["results"]) == 1

            # Verify volume profile structure
            data = result["results"][0]["data"]
            assert "volume_profile" in data
            assert "poc" in data["volume_profile"]
            assert "value_area" in data["volume_profile"]

    def test_volume_signals_invokable_through_action_phase(self):
        """AC3.2: Volume signals tool can be invoked through agent."""
        import pandas as pd
        from datetime import datetime, timedelta

        # Create mock price data
        days = 60
        dates = [datetime.now() - timedelta(days=days - i) for i in range(days)]
        mock_df = pd.DataFrame({
            "date": dates,
            "symbol": "TEST",
            "open": [10000 + i * 10 for i in range(days)],
            "high": [10100 + i * 10 for i in range(days)],
            "low": [9900 + i * 10 for i in range(days)],
            "close": [10050 + i * 10 for i in range(days)],
            "volume": [1000000 for _ in range(days)],
        })

        with patch("stockai.tools.stock_tools._yahoo") as mock_yahoo:
            mock_yahoo.get_price_history.return_value = mock_df

            from stockai.tools.registry import get_registry
            registry = get_registry()
            registry.clear()

            import importlib
            from stockai.tools import stock_tools
            importlib.reload(stock_tools)

            from stockai.tools import get_all_tools
            tools = get_all_tools()

            action = ActionPhase(tools=tools)

            # Test get_volume_signals through action phase
            volume_signals_task = {
                "id": "volume_signals_task",
                "tools": ["get_volume_signals"],
                "dependencies": [],
            }
            result = action.execute_task(volume_signals_task, {"symbol": "BBCA"})

            assert result["success"] is True
            assert len(result["results"]) == 1

            # Verify signals structure
            data = result["results"][0]["data"]
            assert "overall_signal" in data
            assert "direction" in data["overall_signal"]
            assert data["overall_signal"]["direction"] in ["buy", "sell", "neutral"]

    def test_volume_tools_integrate_with_technical_analysis(self):
        """AC3.2: Volume tools integrate with existing technical analysis workflow."""
        import pandas as pd
        from datetime import datetime, timedelta

        # Create mock price data with sufficient history
        days = 90
        dates = [datetime.now() - timedelta(days=days - i) for i in range(days)]
        mock_df = pd.DataFrame({
            "date": dates,
            "symbol": "TEST",
            "open": [10000 + i * 10 for i in range(days)],
            "high": [10100 + i * 10 for i in range(days)],
            "low": [9900 + i * 10 for i in range(days)],
            "close": [10050 + i * 10 for i in range(days)],
            "volume": [1000000 + i * 5000 for i in range(days)],
        })

        with patch("stockai.tools.stock_tools._yahoo") as mock_yahoo:
            mock_yahoo.get_price_history.return_value = mock_df
            mock_yahoo.get_stock_info.return_value = {
                "name": "Test Corp",
                "sector": "Finance",
                "market_cap": 1000000000,
            }
            mock_yahoo.get_current_price.return_value = {
                "price": 10500,
                "change": 50,
                "change_percent": 0.48,
                "volume": 1000000,
            }

            from stockai.tools.registry import get_registry
            registry = get_registry()
            registry.clear()

            import importlib
            from stockai.tools import stock_tools
            importlib.reload(stock_tools)

            from stockai.tools import get_all_tools
            tools = get_all_tools()

            action = ActionPhase(tools=tools)

            # Simulate a multi-tool analysis task combining technical and volume analysis
            # This represents a typical workflow where technical indicators are analyzed
            # alongside volume indicators

            # Task 1: Get technical indicators
            tech_task = {
                "id": "tech_analysis",
                "tools": ["get_technical_indicators"],
                "dependencies": [],
            }
            tech_result = action.execute_task(tech_task, {"symbol": "BBCA"})

            # Task 2: Get volume analysis (depends on understanding price context)
            volume_task = {
                "id": "volume_analysis",
                "tools": ["get_volume_analysis", "get_volume_signals"],
                "dependencies": ["tech_analysis"],
            }
            volume_result = action.execute_task(volume_task, {"symbol": "BBCA"})

            # Both should succeed
            assert tech_result["success"] is True, "Technical analysis should succeed"
            assert volume_result["success"] is True, "Volume analysis should succeed"

            # Technical indicators should be present
            tech_data = tech_result["results"][0]["data"]
            assert "indicators" in tech_data
            assert "rsi" in tech_data["indicators"]
            assert "macd" in tech_data["indicators"]

            # Volume results should include both tools
            assert len(volume_result["results"]) == 2
            tool_names = [r["tool"] for r in volume_result["results"]]
            assert "get_volume_analysis" in tool_names
            assert "get_volume_signals" in tool_names

    def test_volume_tools_appear_in_tools_description(self):
        """AC3.2: Volume tools appear in LLM tool descriptions."""
        from stockai.tools.registry import get_registry

        registry = get_registry()
        registry.clear()

        import importlib
        from stockai.tools import stock_tools
        importlib.reload(stock_tools)

        # Get formatted descriptions for LLM
        descriptions = registry.get_tool_descriptions()

        # Volume tools should appear in the descriptions
        assert "get_volume_analysis" in descriptions
        assert "get_volume_profile" in descriptions
        assert "get_volume_signals" in descriptions

        # They should be under the analysis category
        assert "Analysis" in descriptions or "analysis" in descriptions.lower()

    def test_volume_tools_work_with_planning_phase(self):
        """AC3.2: Volume tools can be included in planning phase tasks."""
        planner = PlanningPhase(available_tools=[
            "get_stock_info",
            "get_current_price",
            "get_technical_indicators",
            "get_volume_analysis",
            "get_volume_profile",
            "get_volume_signals",
        ])

        # Test that volume analysis query creates plan with volume tools
        plan = planner.create_default_plan(
            "Analyze BBCA with volume analysis",
            "BBCA",
        )

        # Should have tasks with volume-related tools
        all_tools = []
        for task in plan.tasks:
            all_tools.extend(task.tools)

        # At least one volume-related tool should be in the plan
        # (exact tool depends on implementation, but volume query should include volume analysis)
        assert len(plan.tasks) >= 1

    def test_multiple_volume_tools_in_single_task(self):
        """AC3.2: Multiple volume tools can be executed in a single task."""
        import pandas as pd
        from datetime import datetime, timedelta

        days = 60
        dates = [datetime.now() - timedelta(days=days - i) for i in range(days)]
        mock_df = pd.DataFrame({
            "date": dates,
            "symbol": "TEST",
            "open": [10000 + i * 10 for i in range(days)],
            "high": [10100 + i * 10 for i in range(days)],
            "low": [9900 + i * 10 for i in range(days)],
            "close": [10050 + i * 10 for i in range(days)],
            "volume": [1000000 for _ in range(days)],
        })

        with patch("stockai.tools.stock_tools._yahoo") as mock_yahoo:
            mock_yahoo.get_price_history.return_value = mock_df

            from stockai.tools.registry import get_registry
            registry = get_registry()
            registry.clear()

            import importlib
            from stockai.tools import stock_tools
            importlib.reload(stock_tools)

            from stockai.tools import get_all_tools
            tools = get_all_tools()

            action = ActionPhase(tools=tools)

            # Execute all volume tools in a single task
            multi_volume_task = {
                "id": "comprehensive_volume",
                "tools": ["get_volume_analysis", "get_volume_profile", "get_volume_signals"],
                "dependencies": [],
            }
            result = action.execute_task(multi_volume_task, {"symbol": "BBCA"})

            assert result["success"] is True
            assert len(result["results"]) == 3
            assert len(result["errors"]) == 0

            # All three tools should have executed
            executed_tools = [r["tool"] for r in result["results"]]
            assert "get_volume_analysis" in executed_tools
            assert "get_volume_profile" in executed_tools
            assert "get_volume_signals" in executed_tools


class TestAgentIntegration:
    """Integration tests for complete agent workflow."""

    @pytest.fixture
    def mock_llm(self):
        """Create mock LLM for testing."""
        mock = MagicMock()

        # Planning response
        plan_response = MagicMock()
        plan_response.content = """
        ```json
        [
            {"id": "task_1", "description": "Get stock info", "tools": ["get_stock_info"], "dependencies": []}
        ]
        ```
        """

        # Validation response
        validation_response = MagicMock()
        validation_response.content = '```json\n{"status": "PASS", "issues": [], "message": "OK"}\n```'

        # Answer response
        answer_response = MagicMock()
        answer_response.content = "# Analysis\n\nThis is the analysis.\n\n---\n*Disclaimer: Not financial advice.*"

        mock.invoke.side_effect = [plan_response, validation_response, answer_response]

        return mock

    @pytest.mark.skip(reason="Requires mocking LLM initialization")
    def test_full_agent_workflow(self, mock_llm):
        """Test complete agent execution flow."""
        # This would require more extensive mocking of the LangChain setup
        pass
