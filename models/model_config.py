# models/model_config.py
"""
Model configuration and factory for loading different inference models.
"""

class ModelConfig:
    """Configuration for available models."""
    
    # Model registry mapping model names to their classes and parameters
    MODEL_REGISTRY = {
        'gpt-4o': {
            'class': 'GPTInferencer',
            'module': 'models.gpt',
            'requires_model_param': True,
            'model_param_value': 'gpt-4o'
        },
        'gpt-4v': {
            'class': 'GPTInferencer',
            'module': 'models.gpt',
            'requires_model_param': True,
            'model_param_value': 'gpt-4v'
        },
        'cogvlm': {
            'class': 'CogVLMInferencer',
            'module': 'models.cogvlm',
            'requires_model_param': False,
            'model_param_value': None
        }
    }
    
    @classmethod
    def get_available_models(cls):
        """
        Get list of all available model names.
        
        Returns:
            list: List of model names
        """
        return list(cls.MODEL_REGISTRY.keys())
    
    @classmethod
    def get_model_inferencer(cls, model_name):
        """
        Factory method to get the appropriate model inferencer.
        
        Args:
            model_name (str): Name of the model
            
        Returns:
            tuple: (inferencer_instance, model_param_value)
                - inferencer_instance: Instance of the model inferencer
                - model_param_value: Model parameter value (None if not required)
                
        Raises:
            ValueError: If model_name is not supported
        """
        if model_name not in cls.MODEL_REGISTRY:
            available = ', '.join(cls.get_available_models())
            raise ValueError(
                f"Unsupported model: {model_name}. "
                f"Available models: {available}"
            )
        
        model_info = cls.MODEL_REGISTRY[model_name]
        
        # Dynamically import the module
        import importlib
        module = importlib.import_module(model_info['module'])
        
        # Get the class from the module
        inferencer_class = getattr(module, model_info['class'])
        
        # Instantiate the inferencer
        inferencer = inferencer_class()
        
        # Return inferencer and model parameter (if needed)
        model_param = model_info['model_param_value'] if model_info['requires_model_param'] else None
        
        return inferencer, model_param
    
    @classmethod
    def requires_model_param(cls, model_name):
        """
        Check if a model requires a model parameter during inference.
        
        Args:
            model_name (str): Name of the model
            
        Returns:
            bool: True if model parameter is required
        """
        if model_name not in cls.MODEL_REGISTRY:
            return False
        return cls.MODEL_REGISTRY[model_name]['requires_model_param']