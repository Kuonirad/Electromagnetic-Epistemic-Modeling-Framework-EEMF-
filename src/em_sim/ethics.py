"""Ethics monitoring for consciousness simulations."""

import logging
from typing import Dict, Any, Optional
import numpy as np

class ConsciousnessEthics:
    """Monitors ethical considerations in consciousness simulations."""
    
    # Perturbational Complexity Index threshold based on human consciousness
    HUMAN_PCI = 0.44
    
    def __init__(
        self,
        alert_callback: Optional[callable] = None,
        log_file: str = "consciousness_ethics.log"
    ):
        """Initialize ethics monitor.
        
        Args:
            alert_callback: Function to call when thresholds are exceeded
            log_file: Path to log file for ethics monitoring
        """
        self.alert_callback = alert_callback or self._default_alert
        self.logger = self._setup_logger(log_file)
        
    def evaluate(self, pci: float, additional_metrics: Optional[Dict[str, Any]] = None) -> bool:
        """Evaluate consciousness metrics against ethical thresholds.
        
        Args:
            pci: Perturbational Complexity Index value
            additional_metrics: Optional additional consciousness metrics
            
        Returns:
            True if within ethical bounds, False if thresholds exceeded
        """
        if pci >= self.HUMAN_PCI:
            message = f"Potential sentience threshold crossed: PCI = {pci:.3f}"
            self.logger.warning(message)
            self.alert_callback(message)
            return False
            
        if additional_metrics:
            self._evaluate_additional_metrics(additional_metrics)
            
        return True
    
    def _evaluate_additional_metrics(self, metrics: Dict[str, Any]) -> None:
        """Evaluate additional consciousness metrics.
        
        Args:
            metrics: Dictionary of additional metrics to evaluate
        """
        # Information Integration Theory (IIT) Phi value
        if "phi" in metrics and metrics["phi"] > 0.5:
            message = f"High information integration detected: Phi = {metrics['phi']:.3f}"
            self.logger.warning(message)
            self.alert_callback(message)
        
        # Neural Complexity
        if "neural_complexity" in metrics and metrics["neural_complexity"] > 0.8:
            message = f"High neural complexity: {metrics['neural_complexity']:.3f}"
            self.logger.warning(message)
            self.alert_callback(message)
    
    def _setup_logger(self, log_file: str) -> logging.Logger:
        """Set up logging configuration.
        
        Args:
            log_file: Path to log file
            
        Returns:
            Configured logger instance
        """
        logger = logging.getLogger("consciousness_ethics")
        logger.setLevel(logging.INFO)
        
        handler = logging.FileHandler(log_file)
        handler.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    def _default_alert(self, message: str) -> None:
        """Default alert mechanism when no callback is provided.
        
        Args:
            message: Alert message to log
        """
        self.logger.critical(f"CONSCIOUSNESS ALERT: {message}")
