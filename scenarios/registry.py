import importlib
import pkgutil
import logging
from typing import List, Type
from scenarios.base_scenario import BaseScenario

logger = logging.getLogger(__name__)

class ScenarioRegistry:
    """
    Dynamic Plugin Registry for Analytics Scenarios.
    Validates scenario unique names and raises exceptions on failure or collisions.
    """
    _registered_scenarios: List[Type[BaseScenario]] = []

    @classmethod
    def register(cls, scenario_cls: Type[BaseScenario]):
        if scenario_cls not in cls._registered_scenarios:
            cls._registered_scenarios.append(scenario_cls)
            logger.info(f"Registered scenario class: {scenario_cls.__name__}")
        return scenario_cls

    @classmethod
    def discover_scenarios(cls):
        """
        Dynamically imports all modules in the `scenarios` package starting with `scenario_`.
        """
        import scenarios
        package = scenarios
        for _, module_name, is_pkg in pkgutil.iter_modules(package.__path__):
            if not is_pkg and module_name.startswith("scenario_"):
                full_module_name = f"scenarios.{module_name}"
                importlib.import_module(full_module_name)

    @classmethod
    def get_all_scenarios(cls) -> List[BaseScenario]:
        """
        Instantiates registered scenario plugins.
        Raises ValueError on scenario_name collision or instantiation failure.
        """
        cls.discover_scenarios()
        instances: List[BaseScenario] = []
        registered_names = set()

        for scenario_cls in cls._registered_scenarios:
            try:
                instance = scenario_cls()
                sc_name = instance.scenario_name
                if sc_name in registered_names:
                    raise ValueError(f"Scenario name collision detected: '{sc_name}' is registered multiple times.")
                registered_names.add(sc_name)
                instances.append(instance)
            except Exception as e:
                logger.error(f"Failed to instantiate scenario class [{scenario_cls.__name__}]: {e}")
                raise e

        return instances
