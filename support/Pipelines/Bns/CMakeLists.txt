# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_python_add_module(
  Bns
  MODULE_PATH Pipelines
  PYTHON_FILES
  __init__.py
  ComputeTrajectories.py
)

# Create a target to compile all executables for this pipeline
set(PIPELINE_TARGET bns)
add_custom_target(${PIPELINE_TARGET})
