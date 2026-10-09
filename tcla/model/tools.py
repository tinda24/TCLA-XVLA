class Registry:
    def __init__(self, name):
        self.name = name
        self._registry = {}

    def register(self, key):
        def decorator(module_class):
            self._registry[key] = module_class
            return module_class

        return decorator

    def __getitem__(self, key):
        return self._registry[key]

    def list(self):
        return {key: value for key, value in self._registry.items()}


FRAMEWORK_REGISTRY = Registry("frameworks")
