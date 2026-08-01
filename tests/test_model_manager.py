import os
import sys
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.model_manager import ModelManager

def test_model_manager_singleton():
    manager1 = ModelManager.get_instance()
    manager2 = ModelManager.get_instance()
    assert manager1 is manager2
