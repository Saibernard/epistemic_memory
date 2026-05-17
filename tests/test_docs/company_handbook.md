# Nexora Technologies — Employee Handbook 2026

## Company Overview

Nexora Technologies was founded on March 14, 2019 by Dr. Amara Okonkwo and Rajesh Patel in Austin, Texas. The company specializes in quantum-resistant cryptography and post-quantum secure communication protocols. Our headquarters is located at 4720 Bee Caves Road, Suite 310, Austin, TX 78746.

The company currently has 287 employees across 4 offices: Austin (HQ), Berlin, Singapore, and São Paulo. Annual revenue for FY2025 was $43.7 million, a 62% increase year-over-year.

## Engineering Standards

All production code must be written in Rust or Go. Python is approved only for internal tooling, data analysis, and ML pipelines. JavaScript/TypeScript is used exclusively for frontend applications built with SolidJS (not React — we migrated away from React in Q3 2024).

Code review requires approval from at least 2 senior engineers before merge. The CI pipeline runs on Buildkite (not GitHub Actions) and must complete within 12 minutes. Test coverage must exceed 85% for any new module.

Our primary database is CockroachDB for distributed transactional workloads. Redis Cluster is used for caching with a 15-minute TTL for most keys. ClickHouse handles all analytics and event storage. We explicitly do NOT use MongoDB or DynamoDB in production.

## Security Protocols

All internal services communicate over mTLS using certificates issued by our internal CA named "Nexora Vault CA". API keys rotate every 72 hours automatically via the KeyRotator service. Employee laptops must use full-disk encryption with FileVault (macOS) or LUKS (Linux). Windows machines are not permitted in engineering.

The bug bounty program pays between $500 and $25,000 depending on severity. Critical vulnerabilities (CVSS 9.0+) trigger an automatic "Code Red" protocol where the on-call SRE team has 30 minutes to acknowledge and 4 hours to deploy a fix.

## Benefits & PTO

Employees receive 28 days of PTO per year (not including public holidays). The parental leave policy is 16 weeks fully paid for all parents regardless of gender. The 401(k) match is 6% of salary. Health insurance is provided through Anthem Blue Cross with a $500 annual deductible.

The annual company retreat is held in the second week of October. The 2025 retreat was in Queenstown, New Zealand. The 2026 retreat is planned for Reykjavik, Iceland.

## Unique Internal Tools

- **Prism**: Our internal observability platform built on top of OpenTelemetry. All services must emit traces to Prism.
- **Lighthouse**: The internal code search engine (think Sourcegraph but custom-built). Indexes 2.3 million files.
- **Meridian**: Our proprietary build system that replaced Bazel in early 2025.
- **Compass**: The internal LLM gateway that routes between Claude, GPT-4, and our fine-tuned Mistral model called "Nexora-7B-Sec" which specializes in security code review.
