#!/usr/bin/env python3
"""
AltLAS ConfigLoader - Module for loading and parsing configuration files.
"""

import configparser
import logging
from pathlib import Path

log = logging.getLogger(__name__)

class ConfigLoader:
    """
    Handles loading and parsing of configuration files for AltLAS components.
    """
    def __init__(self, config_path=None):
        """
        Initialize the config loader with a path to the configuration file.
        
        Args:
            config_path (str or Path, optional): Path to the configuration file.
                If None, the default config.ini in the project root will be used.
        """
        if config_path is None:
            # Default to config.ini in the project root
            self.config_path = Path(__file__).parent.parent / "config.ini"
        else:
            self.config_path = Path(config_path)
            
        self.config = configparser.ConfigParser(inline_comment_prefixes=(';', '#'))
        self._load_config()
    
    def _load_config(self):
        """Load the configuration file."""
        if not self.config_path.exists():
            log.error(f"Configuration file not found at {self.config_path}")
            raise FileNotFoundError(f"Configuration file not found at {self.config_path}")
        
        try:
            self.config.read(self.config_path)
            log.debug(f"Configuration loaded from {self.config_path}")
        except Exception as e:
            log.error(f"Error reading configuration file: {e}")
            raise
    
    def get_section(self, section_name):
        """
        Get a specific configuration section.
        
        Args:
            section_name (str): Name of the configuration section.
            
        Returns:
            configparser.SectionProxy: The requested section.
            
        Raises:
            KeyError: If the section doesn't exist.
        """
        if section_name not in self.config:
            log.error(f"Configuration section '{section_name}' not found")
            raise KeyError(f"Configuration section '{section_name}' not found")
        
        return self.config[section_name]
    
    def get_runner_config(self):
        """
        Get the Runner configuration section with parsed values.
        
        Returns:
            dict: Dictionary containing the Runner configuration values.
        """
        try:
            runner_config = self.get_section('Runner')
            
            return {
                'max_attempts': runner_config.getint('MaxAttempts', 1000),
                'stuck_check_window': runner_config.getint('StuckCheckWindow', 15),
                'stuck_threshold': runner_config.getfloat('StuckThreshold', 0.01),
                'hint_probability_on_stuck': runner_config.getfloat('HintProbabilityOnStuck', 1.0),
                'max_consecutive_stuck_checks': runner_config.getint('MaxConsecutiveStuckChecks', 3),
                'log_frequency': runner_config.getint('LogFrequency', 500),
                'top_tokens_to_log': runner_config.getint('TopTokensToLog', 10),
                'report_frequency': runner_config.getint('ReportFrequency', 1000),
                'report_on_success': runner_config.getboolean('ReportOnSuccess', True),
            }
        except (KeyError, ValueError) as e:
            log.error(f"Error parsing Runner configuration: {e}")
            raise
    
    def get_scorer_config(self):
        """
        Get the Scorer configuration section with parsed values.
        
        Returns:
            dict: Dictionary containing the Scorer configuration values.
        """
        try:
            scorer_config = self.get_section('Scorer')
            
            return {
                'success_threshold': scorer_config.getfloat('SuccessThreshold', 0.99),
                # Add other scorer-specific configuration values as needed
            }
        except (KeyError, ValueError) as e:
            log.error(f"Error parsing Scorer configuration: {e}")
            raise
    
    def get_llm_config(self):
        """
        Get the LLM configuration section with parsed values.
        
        Returns:
            dict: Dictionary containing the LLM configuration values.
        """
        try:
            llm_config = self.get_section('LLM')
            
            # Parse model preference list
            model_pref_str = llm_config.get('ModelPreference', 'wizardcoder,codellama,code-llama,stable-code,starcoder,llama')
            model_preference = [m.strip() for m in model_pref_str.split(',') if m.strip()]
            
            return {
                'provider': llm_config.get('Provider', 'vllm'),
                'vllm_base_url': llm_config.get('vLLMBaseURL', 'http://localhost:8000/v1'),
                'lmstudio_base_url': llm_config.get('LMStudioBaseURL', 'http://localhost:1234/v1'),
                'model_preference': model_preference,
                'timeout': llm_config.getint('Timeout', 120),
                'temperature': llm_config.getfloat('Temperature', 0.7),
                'max_tokens': llm_config.getint('MaxTokens', 500),
            }
        except (KeyError, ValueError) as e:
            log.error(f"Error parsing LLM configuration: {e}")
            # Return defaults or raise? For now, return defaults to be robust
            return {
                'provider': 'vllm',
                'vllm_base_url': 'http://localhost:8000/v1',
                'lmstudio_base_url': 'http://localhost:1234/v1',
                'model_preference': ['wizardcoder', 'codellama', 'code-llama', 'stable-code', 'starcoder', 'llama'],
                'timeout': 120,
                'temperature': 0.7,
                'max_tokens': 500,
            }
    
    def get_config_path(self):
        """
        Get the path to the configuration file.
        
        Returns:
            Path: The path to the configuration file.
        """
        return self.config_path