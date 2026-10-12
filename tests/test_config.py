"""Tests for LEREDD configuration singleton and lifecycle management (ISSUE-012)."""

import pytest
from leredd.config import LEREDDConfig, get_config, reset_config


def test_get_config_returns_same_instance_when_not_reset():
    """Verify get_config() returns the same singleton instance across repeated calls."""
    reset_config()
    try:
        cfg1 = get_config()
        cfg2 = get_config()
        assert cfg1 is cfg2
    finally:
        reset_config()


def test_reset_config_creates_new_instance():
    """Verify reset_config() causes the subsequent get_config() call to return a new instance."""
    reset_config()
    try:
        cfg1 = get_config()
        reset_config()
        cfg2 = get_config()
        assert cfg1 is not cfg2
        assert isinstance(cfg2, LEREDDConfig)
    finally:
        reset_config()


def test_env_variable_reflected_after_reset(monkeypatch):
    """Verify environment variable changes are reflected when configuration is initialized after reset."""
    reset_config()
    try:
        # Initial config has default mock provider
        cfg_initial = get_config()
        assert cfg_initial.llm_provider == "mock"

        # Change environment variable and reset cached singleton
        monkeypatch.setenv("LLM_PROVIDER", "ollama")
        reset_config()

        # Newly created config must reflect the environment variable
        cfg_reloaded = get_config()
        assert cfg_reloaded.llm_provider == "ollama"
        assert cfg_reloaded is not cfg_initial
    finally:
        reset_config()


def test_reset_does_not_mutate_existing_reference(monkeypatch):
    """Verify that resetting and reinitializing does not mutate an already-held reference to the previous configuration object."""
    reset_config()
    try:
        cfg_old = get_config()
        original_k = cfg_old.k_examples
        assert original_k == 6

        # Reset and load new configuration with different env var
        reset_config()
        monkeypatch.setenv("K_EXAMPLES", "3")
        cfg_new = get_config()

        assert cfg_new.k_examples == 3
        # cfg_old must retain its original state without mutation
        assert cfg_old.k_examples == original_k
        assert cfg_old is not cfg_new
    finally:
        reset_config()
