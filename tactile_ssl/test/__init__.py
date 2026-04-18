# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC-BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

<<<<<<< Updated upstream
from importlib import import_module

_EXPORTS = {
    "TestForceSL": (".test_t1_force", "TestForceSL"),
    "TestSlipSL": (".test_t2_slip", "TestSlipSL"),
    "TestPoseSL": (".test_t3_pose", "TestPoseSL"),
    "TestGraspSL": (".test_t4_grasp", "TestGraspSL"),
    "TestTextileSL": (".test_t6_textile", "TestTextileSL"),
    "DemoForceField": (".demo_t1_forcefield", "DemoForceField"),
    "DemoEncoderRerun": (".demo_encoder_rerun", "DemoEncoderRerun"),
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
