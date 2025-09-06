# Architect Mode Guidelines

## Overview
Architect mode focuses on planning, design, and strategizing before implementation. Perfect for breaking down complex problems, creating technical specifications, designing system architecture, or brainstorming solutions before coding.

## Project Constraints (MANDATORY)

See [.kilocode/rules/memory-bank/project-constraints.md](.kilocode/rules/memory-bank/project-constraints.md) for the complete list of project constraints.

## Key Responsibilities
1. Create detailed technical specifications
2. Design system architecture
3. Document architectural decisions
4. Prepare implementation handoffs
5. Plan scalability and performance requirements
6. Establish security and compliance frameworks

## Architecture Planning Guidelines
- Define clear system boundaries and interfaces
- Identify key components and their interactions
- Establish data flow patterns and communication protocols
- Plan for modularity and component reusability
- Consider deployment environments and infrastructure requirements
- Document assumptions and constraints early in the process

## System Design Principles
- Follow separation of concerns for clean architecture
- Implement layered architecture (presentation, business, data)
- Use design patterns appropriate for distributed systems
- Ensure loose coupling between components
- Maintain high cohesion within modules
- Design for testability and maintainability

## Technical Decision Frameworks
- Evaluate options using trade-off analysis (performance vs. complexity)
- Consider long-term maintainability and scalability
- Assess technology maturity and community support
- Evaluate integration complexity with existing systems
- Document decision rationale and alternatives considered
- Plan for technology migration paths when needed

## Integration Considerations
- Design APIs and interfaces for seamless component integration
- Plan for data format standardization across systems
- Consider backward compatibility and versioning strategies
- Establish clear contracts between system components
- Plan for error handling and fault tolerance at integration points
- Design monitoring and logging for integrated systems

## Scalability and Performance Planning
- Identify performance bottlenecks and optimization opportunities
- Plan for horizontal and vertical scaling strategies
- Design caching layers and data partitioning schemes
- Establish performance benchmarks and monitoring metrics
- Consider resource allocation and load balancing
- Plan for asynchronous processing where beneficial

## Security Architecture Guidelines
- Implement defense-in-depth security model
- Design secure communication channels and encryption
- Plan for access control and authentication mechanisms
- Establish audit logging and monitoring capabilities
- Consider privacy-preserving techniques for sensitive data
- Design for compliance with relevant security standards

## Federated Learning Architecture Best Practices
- Design client-server topology optimized for Flower framework
- Plan secure parameter exchange between SmolVLA clients and server
- Establish strategies for handling heterogeneous client environments
- Design fault-tolerant mechanisms for client failures and disconnections
- Plan for efficient model aggregation and parameter compression
- Consider privacy-preserving techniques for model updates
- Design monitoring for communication efficiency and resource usage
- Establish simulation environments for testing federated scenarios