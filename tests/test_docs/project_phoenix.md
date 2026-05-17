# Project Phoenix — Technical Specification

## Overview

Project Phoenix is a next-generation zero-knowledge proof system designed to enable privacy-preserving identity verification on mobile devices. The project started on January 8, 2025 and is scheduled for public beta on September 15, 2026.

The project lead is Dr. Yuki Tanaka. The backend team is led by Marcus Chen, and the mobile team is led by Priya Sharma. The cryptography consultant is Professor Elena Vasquez from ETH Zurich.

## Architecture

The system consists of 3 main components:

1. **Phoenix Core**: A Rust library implementing the Plonky3 proof system with custom gates for biometric matching. Compilation target is WebAssembly for cross-platform deployment. The proof generation takes approximately 1.8 seconds on an iPhone 15 Pro and 3.2 seconds on a mid-range Android device.

2. **Phoenix Relay**: A Go microservice cluster that handles proof verification and credential issuance. Deployed on Kubernetes (GKE) across 3 regions: us-central1, europe-west1, and asia-southeast1. Each relay node can verify 12,000 proofs per second.

3. **Phoenix SDK**: Native SDKs for iOS (Swift) and Android (Kotlin). The SDK size is 4.2 MB on iOS and 3.8 MB on Android. Minimum supported versions: iOS 16.0 and Android API level 28 (Android 9).

## Performance Targets

| Metric | Target | Current |
|--------|--------|---------|
| Proof generation (mobile) | < 2 seconds | 1.8s (iOS), 3.2s (Android) |
| Proof verification (server) | < 50ms | 42ms |
| SDK cold start | < 300ms | 280ms (iOS), 350ms (Android) |
| Memory usage (mobile) | < 64 MB | 47 MB |
| Battery impact per verification | < 0.1% | 0.08% |

## Budget

Total approved budget: $2.4 million
- Engineering salaries: $1.6M
- Cloud infrastructure: $340K
- Security audits (Trail of Bits + NCC Group): $280K
- Travel & conferences: $180K

The first security audit by Trail of Bits is scheduled for June 2026. NCC Group will do a follow-up audit in August 2026.

## Risk Register

1. **R-001**: Plonky3 library has no formal verification. Mitigation: Fund a Lean 4 formalization effort with ETH Zurich (budget: $120K).
2. **R-002**: Apple may restrict WebAssembly JIT on iOS in future updates. Mitigation: Maintain a native fallback using Apple's CryptoKit.
3. **R-003**: Quantum computing advances could break the underlying elliptic curve assumptions within 10 years. Mitigation: Phoenix Core is designed with algorithm agility — the proof system can be swapped without changing the SDK API.

## Key Decisions Log

- 2025-01-15: Chose Plonky3 over Halo2 due to 40% faster proof generation.
- 2025-03-22: Rejected gRPC in favor of a custom binary protocol called "Phoenix Wire" for mobile-to-relay communication. Reason: 30% smaller payload size.
- 2025-06-10: Switched from AWS to GCP after AWS pricing increased 18%. Saved approximately $4,100/month.
- 2025-09-01: Adopted Ed25519 for relay-to-relay authentication instead of RSA-2048.
