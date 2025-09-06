# Architect Mode Guidelines

## MANDATORY CONSTRAINTS SUMMARY
⚠️ **CRITICAL**: Before any work:
- Work only in project root directory (no external changes)
- Use conda environment "zk0"
- Focus: SmolVLA + Flower + SO-100 datasets
- Reference quickstart-lerobot structure
- See [.kilocode/rules/memory-bank/project-constraints.md](.kilocode/rules/memory-bank/project-constraints.md) for full details

## Overview
Architect mode focuses on planning, design, and strategizing before implementation.

## Key Responsibilities
- Create detailed technical specifications
- Design system architecture
- Document architectural decisions
- Prepare implementation handoffs

## Architecture Planning Guidelines
See [.kilocode/rules/memory-bank/architecture.md](.kilocode/rules/memory-bank/architecture.md) for detailed system architecture.
Key principles:
- Define clear system boundaries and interfaces
- Plan for modularity and component reusability
- Document assumptions and constraints early

## Technical Decision Frameworks
- Evaluate trade-offs (performance vs. complexity)
- Consider long-term maintainability
- Document decision rationale

## Federated Learning Architecture Best Practices
- Design client-server topology optimized for Flower framework
- Plan secure parameter exchange between SmolVLA clients and server
- Establish strategies for handling heterogeneous client environments
- Design fault-tolerant mechanisms for client failures and disconnections
- Plan for efficient model aggregation and parameter compression
- Consider privacy-preserving techniques for model updates
- Design monitoring for communication efficiency and resource usage
- Establish simulation environments for testing federated scenarios