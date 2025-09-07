# Architect Mode Guidelines

## MANDATORY CONSTRAINTS SUMMARY
⚠️ **CRITICAL**: Before any work:
- Work only in project root directory (no external changes)
- Use conda environment "zk0"
- Focus: SmolVLA + Flower + SO-100 datasets

## Overview
Architect mode focuses on planning, design, and strategizing before implementation.

## Key Responsibilities
- Create detailed technical specifications
- Design system architecture
- Document architectural decisions
- Prepare implementation handoffs

## Architecture Planning Guidelines
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