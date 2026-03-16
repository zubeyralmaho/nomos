# Contributing to Nomos

Thank you for your interest in contributing to Nomos! This guide will help you get started.

## Table of Contents

- [Development Environment](#development-environment)
- [Project Structure](#project-structure)
- [Code Standards](#code-standards)
- [Pull Request Process](#pull-request-process)
- [Adding New Features](#adding-new-features)
- [Testing](#testing)
- [Reporting Issues](#reporting-issues)

---

## Development Environment

### Prerequisites

| Requirement | Version | Purpose |
|-------------|---------|---------|
| Rust | 1.82+ | Core development |
| Cargo | Latest | Package management |
| Git | Latest | Version control |
| Python | 3.8+ | Test suite |

### Optional (for advanced features)

| Requirement | Purpose |
|-------------|---------|
| Nightly Rust | eBPF development |
| bpf-linker | eBPF compilation |
| wasm32-wasip1 target | WASM healer development |

### Setup

```bash
# Clone the repository
git clone https://github.com/zubeyralmaho/nomos.git
cd nomos

# Build the project
cargo build

# Run tests to verify setup
cargo test -p nomos-core

# Install test dependencies
cd tests && pip install -r requirements.txt
```

---

## Project Structure

```
nomos/
├── nomos-core/               # Main library and binary
│   └── src/
│       ├── lib.rs            # Library entry point
│       ├── main.rs           # Binary entry point
│       ├── proxy.rs          # HTTP proxy server
│       ├── middleware.rs     # Response transformation
│       ├── schema.rs         # Schema store
│       ├── control.rs        # Control plane API
│       ├── ebpf.rs           # eBPF integration
│       ├── wasm_host.rs      # WASM healer host
│       │
│       ├── nlp/              # NLP algorithms
│       │   ├── levenshtein.rs
│       │   ├── jaro.rs
│       │   ├── tfidf.rs
│       │   ├── synonym.rs
│       │   └── combined.rs
│       │
│       └── engine/           # Healing engine
│           ├── simd.rs       # SIMD acceleration
│           ├── embedding.rs  # Field embeddings
│           ├── lsh.rs        # LSH index
│           ├── confidence.rs # Match scoring
│           └── healer.rs     # Main healer
│
├── nomos-ebpf/               # eBPF/XDP kernel program
├── nomos-ebpf-common/        # Shared eBPF types
├── nomos-healer-guest/       # WASM healer module
├── dashboard/                # Web monitoring UI
├── docs/                     # Documentation
└── tests/                    # Chaos & stress tests
```

---

## Code Standards

### Rust Style

1. **Format with rustfmt**
   ```bash
   cargo fmt
   ```

2. **Lint with Clippy**
   ```bash
   cargo clippy -p nomos-core -- -D warnings
   ```

3. **Document all public APIs**
   ```rust
   /// Compares two field names for similarity.
   ///
   /// # Arguments
   /// * `source` - The source field name
   /// * `target` - The target field name to match against
   ///
   /// # Returns
   /// A similarity score between 0.0 (no match) and 1.0 (exact match)
   ///
   /// # Example
   /// ```
   /// let score = compare("user_name", "userName");
   /// assert!(score > 0.8);
   /// ```
   pub fn compare(source: &str, target: &str) -> f64 {
       // Implementation
   }
   ```

4. **Follow memory discipline**
   - No `.clone()` on hot path without justification
   - Prefer borrowed references over owned data
   - Use arena allocation for per-request data

### Commit Messages

Use [Conventional Commits](https://www.conventionalcommits.org/) format:

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

**Types:**
| Type | Description |
|------|-------------|
| `feat` | New feature |
| `fix` | Bug fix |
| `docs` | Documentation only |
| `test` | Adding or updating tests |
| `refactor` | Code change that neither fixes a bug nor adds a feature |
| `perf` | Performance improvement |
| `chore` | Build process or auxiliary tool changes |

**Examples:**
```
feat(nlp): add metaphone algorithm for phonetic matching

fix(proxy): handle empty request bodies correctly

docs(readme): update installation instructions

test(jaro): add edge cases for unicode strings

perf(simd): optimize dot product with AVX2 intrinsics

refactor(engine): extract embedding logic to separate module
```

---

## Pull Request Process

### Before Starting

1. Check existing issues and PRs for related work
2. For large changes, open an issue first to discuss the approach
3. Read the [Architecture Guide](docs/ARCHITECTURE.md) to understand the codebase

### Workflow

1. **Fork** the repository
2. **Create a branch** from `main`:
   ```bash
   git checkout -b feat/your-feature-name
   ```

3. **Make your changes**
   - Write tests for new functionality
   - Update documentation if needed
   - Keep commits atomic and well-described

4. **Verify your changes**
   ```bash
   # Format code
   cargo fmt
   
   # Run linter
   cargo clippy -p nomos-core -- -D warnings
   
   # Run tests
   cargo test -p nomos-core
   
   # Run benchmarks (ensure no regression)
   ./bench.sh
   ```

5. **Push and create PR**
   ```bash
   git push origin feat/your-feature-name
   ```

### PR Checklist

- [ ] Code follows the project style guidelines
- [ ] All tests pass
- [ ] New functionality has tests
- [ ] Documentation updated (if applicable)
- [ ] `cargo fmt` has been run
- [ ] `cargo clippy` shows no warnings
- [ ] No benchmark regressions
- [ ] Commit messages follow convention

---

## Adding New Features

### Adding a New NLP Algorithm

1. **Create the algorithm file**
   ```rust
   // nomos-core/src/nlp/your_algorithm.rs
   
   /// Your algorithm description.
   pub struct YourAlgorithm;
   
   impl YourAlgorithm {
       /// Calculate similarity between two strings.
       pub fn similarity(s1: &str, s2: &str) -> f64 {
           // Your implementation
           0.0
       }
   }
   
   #[cfg(test)]
   mod tests {
       use super::*;
   
       #[test]
       fn test_exact_match() {
           assert_eq!(YourAlgorithm::similarity("test", "test"), 1.0);
       }
   
       #[test]
       fn test_completely_different() {
           assert!(YourAlgorithm::similarity("abc", "xyz") < 0.3);
       }
   
       #[test]
       fn test_similar_strings() {
           let score = YourAlgorithm::similarity("user_name", "userName");
           assert!(score > 0.7);
       }
   }
   ```

2. **Export in mod.rs**
   ```rust
   // nomos-core/src/nlp/mod.rs
   pub mod your_algorithm;
   pub use your_algorithm::YourAlgorithm;
   ```

3. **Add to ensemble** (if applicable)
   ```rust
   // nomos-core/src/engine/confidence.rs
   // Add to the weighted ensemble
   ```

4. **Write comprehensive tests**

5. **Update documentation**

### Adding a New Healing Operation

1. Define the operation in `engine/healing.rs`
2. Implement the transformation logic
3. Add tests with various input cases
4. Update the API documentation

---

## Testing

### Unit Tests

```bash
# Run all unit tests
cargo test -p nomos-core

# Run specific module tests
cargo test -p nomos-core nlp
cargo test -p nomos-core engine

# Run with output
cargo test -p nomos-core -- --nocapture
```

### Integration Tests

```bash
# Start test upstream
python3 tests/upstream_server.py &

# Start Nomos
cargo run --release -p nomos-core &

# Run integration tests
python3 tests/chaos_suite.py
```

### Stress Tests

```bash
# Run load test
python3 tests/stress_test.py --rps 5000 --duration 30

# Run boundary tests
python3 tests/boundary_test.py
```

### Benchmarks

```bash
# Run all benchmarks
./bench.sh

# Run specific benchmark
cargo bench --bench proxy_throughput
```

### Test Coverage

Aim for at least 80% code coverage on new code:

```bash
# Install cargo-tarpaulin
cargo install cargo-tarpaulin

# Run with coverage
cargo tarpaulin -p nomos-core
```

---

## Reporting Issues

### Bug Reports

Please include:
- **Rust version**: `rustc --version`
- **OS**: e.g., Ubuntu 22.04, macOS 14
- **Steps to reproduce**: Minimal example to trigger the bug
- **Expected behavior**: What should happen
- **Actual behavior**: What actually happens
- **Logs**: Relevant error messages or debug output

### Feature Requests

Please include:
- **Use case**: Why you need this feature
- **Proposed solution**: How you'd like it to work
- **Alternatives**: Other approaches you've considered

### Security Issues

For security vulnerabilities, please **do not** open a public issue. Instead, contact the maintainers directly.

---

## Community

- **GitHub Issues**: For bugs and feature requests
- **GitHub Discussions**: For questions and general discussion
- **Pull Requests**: For code contributions

---

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to Nomos! 🚀
