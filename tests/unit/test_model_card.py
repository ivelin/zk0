"""Tests for model card generation."""


from src.server.server_utils import generate_model_card


class TestGenerateModelCard:
    """Test model card generation functionality."""


    def test_generate_model_card_excludes_na_insights(self):
        """Test that N/A insights are excluded from the model card."""
        hyperparams = {"num_server_rounds": 2, "local_epochs": 20}
        train_datasets = [{"name": "test_dataset", "description": "Test dataset"}]
        eval_datasets = [{"name": "eval_dataset", "description": "Eval dataset"}]
        metrics = {"composite_eval_loss": 0.5}
        insights = {
            "convergence_trend": "N/A",
            "avg_client_loss_trend": "N/A",
            "param_update_norm_trend": "N/A",
            "lr_mu_adjustments": "N/A",
            "client_participation_rate": "N/A",
            "anomalies": []
        }
        other_info = {
            "timestamp": "2025-10-28T15:56:25",
            "version": "0.3.7",
            "federation": "local-simulation-serialized-gpu"
        }

        result = generate_model_card(
            hyperparams, train_datasets, eval_datasets,
            metrics, insights, other_info
        )

        # Should not contain N/A insight lines
        assert "- **Convergence Trend**: N/A" not in result
        assert "- **Avg Client Loss Trend**: N/A" not in result
        assert "- **Param Update Norm Trend**: N/A" not in result
        assert "- **LR/μ Adjustments**: N/A" not in result
        assert "- **Client Participation Rate**: N/A" not in result

        # Should still contain anomalies (even if empty)
        assert "- **Anomalies**: None detected" in result

    def test_generate_model_card_includes_valid_insights(self):
        """Test that valid insights are included in the model card."""
        hyperparams = {"num_server_rounds": 2, "local_epochs": 20}
        train_datasets = [{"name": "test_dataset", "description": "Test dataset"}]
        eval_datasets = [{"name": "eval_dataset", "description": "Eval dataset"}]
        metrics = {"composite_eval_loss": 0.5}
        insights = {
            "convergence_trend": "Policy loss: 1.0 → 0.5",
            "avg_client_loss_trend": "Started at 0.8, ended at 0.6",
            "param_update_norm_trend": "Average 0.001",
            "lr_mu_adjustments": "Final LR: 0.0001, μ: 0.01",
            "client_participation_rate": "Average 2.0 clients per round",
            "anomalies": ["Client dropouts in rounds: 5"]
        }
        other_info = {
            "timestamp": "2025-10-28T15:56:25",
            "version": "0.3.7",
            "federation": "local-simulation-serialized-gpu"
        }

        result = generate_model_card(
            hyperparams, train_datasets, eval_datasets,
            metrics, insights, other_info
        )

        # Should contain valid insight lines
        assert "- **Convergence Trend**: Policy loss: 1.0 → 0.5" in result
        assert "- **Avg Client Loss Trend**: Started at 0.8, ended at 0.6" in result
        assert "- **Param Update Norm Trend**: Average 0.001" in result
        assert "- **LR/μ Adjustments**: Final LR: 0.0001, μ: 0.01" in result
        assert "- **Client Participation Rate**: Average 2.0 clients per round" in result
        assert "- **Anomalies**: Client dropouts in rounds: 5" in result