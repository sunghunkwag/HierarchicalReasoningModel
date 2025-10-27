# Changelog

All notable changes to the Hierarchical Reasoning Model project will be documented in this file.

## [Unreleased] - 2024-10-27

### Fixed
- **Critical Bug**: Fixed shape mismatch in `examples/basic_usage.py` dataset creation (line 53)
  - Issue: Inconsistent use of `keepdim=True` caused broadcasting error
  - Impact: Example script was unable to run training
  - Solution: Ensured consistent dimensionality in tensor operations

### Added
- **Testing Infrastructure**
  - Created comprehensive test suite in `tests/test_hierarchical_model.py`
  - Tests cover all major components and functionality
  - Includes unit tests for configuration, modules, training, and edge cases
  
- **Utilities Module** (`src/utils.py`)
  - Model checkpoint saving and loading
  - Model weight management
  - Parameter counting and model summary
  - Evaluation utilities
  - Early stopping implementation
  - Metrics tracking system
  
- **Advanced Training Example** (`examples/advanced_training.py`)
  - Train/validation split demonstration
  - Model checkpointing
  - Early stopping
  - Comprehensive metrics tracking
  - Checkpoint loading and inference
  
- **Project Configuration**
  - Added `requirements.txt` for dependency management
  - Added `.gitignore` for Python projects
  - Added `CHANGELOG.md` for tracking changes
  - Added `ISSUES_FOUND.md` documenting all identified issues

### Improved
- **Documentation**
  - Enhanced code comments and docstrings
  - Added comprehensive testing documentation
  - Improved example clarity

### Changed
- **Code Quality**
  - Fixed tensor shape handling in dataset creation
  - Improved error handling in various components
  - Enhanced type safety with better function signatures

## [1.0.0] - Initial Release

### Added
- Core hierarchical reasoning model implementation
- Adaptive Computation Time (ACT) mechanism
- Meta-reinforcement learning controller
- Batch-wise ACT processing
- Basic usage examples
- README with comprehensive documentation

