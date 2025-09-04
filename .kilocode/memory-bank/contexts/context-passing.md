# Context Passing Mechanism

## Overview

The context passing mechanism ensures that critical information flows seamlessly between tasks, subtasks, and mode transitions without loss or degradation.

## Core Components

### 1. Context Capture
**What to capture:**
- Project constraints (always)
- Task objectives and success criteria
- Technical context and dependencies
- Key decisions and assumptions
- Risk assessments and mitigation

### 2. Context Packaging
**How to package:**
- Use standardized templates
- Include validation checklists
- Add metadata (timestamps, versions)
- Compress for efficiency when needed

### 3. Context Transfer
**How to transfer:**
- Direct inheritance for subtasks
- Mode transition protocols
- Documentation handoffs
- Memory bank storage

### 4. Context Validation
**How to validate:**
- Completeness checks
- Consistency verification
- Relevance assessment
- Integrity confirmation

## Passing Protocols

### Subtask Creation Protocol

1. **Context Extraction**
   - Extract relevant context from parent task
   - Identify inheritance requirements
   - Determine validation needs

2. **Context Packaging**
   - Use task context template
   - Include all mandatory constraints
   - Add subtask-specific context

3. **Context Transfer**
   - Embed context in subtask description
   - Reference memory bank for details
   - Include validation checkpoints

4. **Context Validation**
   - Verify completeness
   - Confirm constraints included
   - Validate success criteria

### Mode Transition Protocol

1. **Transition Assessment**
   - Evaluate context compatibility
   - Identify transition requirements
   - Assess risk of context loss

2. **Context Preservation**
   - Extract current context state
   - Package for target mode
   - Include transition metadata

3. **Handover Execution**
   - Transfer to target mode
   - Document handoff details
   - Include rollback procedures

4. **Transition Validation**
   - Verify context integrity
   - Confirm mode compatibility
   - Validate success criteria

## Context Formats

### Template-Based Format
```markdown
## Project Constraints (MANDATORY)
- [ ] Work ONLY within the local project repository root directory
- [ ] No changes to sibling or parent directories
- [ ] Use conda environment "zk0"
- [ ] Focus on SmolVLA model and SO-100 real-world datasets

## Task Context
[Task-specific context]

## Success Criteria
[Measurable outcomes]
```

### Structured Format
```json
{
  "constraints": {
    "workDirectory": "local project repository root directory",
    "environment": "zk0",
    "focus": "SmolVLA + SO-100"
  },
  "context": {
    "taskId": "TASK-001",
    "successCriteria": [...],
    "dependencies": [...]
  },
  "validation": {
    "checklist": [...],
    "metadata": {
      "created": "2025-09-03",
      "version": "1.0"
    }
  }
}
```

## Validation Mechanisms

### Automated Validation
- **Constraint Checkers**: Verify mandatory constraints
- **Completeness Scanners**: Check for required fields
- **Consistency Validators**: Ensure logical consistency
- **Relevance Filters**: Remove outdated context

### Manual Validation
- **Peer Review**: Cross-validation by team members
- **Checklist Verification**: Systematic checklist completion
- **Context Audits**: Periodic context quality assessment
- **User Acceptance**: Stakeholder validation

## Error Handling

### Context Loss Detection
- **Missing Constraint Alerts**: Automatic detection of missing constraints
- **Incomplete Context Warnings**: Alerts for incomplete context
- **Stale Context Notifications**: Warnings for outdated information
- **Inconsistency Flags**: Flags for conflicting information

### Recovery Procedures
1. **Immediate Stop**: Pause work when context loss detected
2. **Assessment**: Evaluate extent of context loss
3. **Reconstruction**: Use memory bank to reconstruct
4. **Validation**: Verify reconstructed context
5. **Resume**: Continue with validated context

## Performance Optimization

### Context Compression
- **Summarization**: Extract key points only
- **References**: Use pointers to memory bank
- **Lazy Loading**: Load detailed context on demand
- **Caching**: Cache frequently used context

### Transfer Optimization
- **Incremental Updates**: Send only changes
- **Prioritization**: Transfer critical context first
- **Batch Processing**: Group related context
- **Async Transfer**: Non-blocking context transfer

## Monitoring and Analytics

### Context Health Metrics
- **Completeness Score**: Percentage of required context present
- **Freshness Score**: How current the context is
- **Usage Score**: How frequently context is accessed
- **Loss Rate**: Frequency of context loss incidents

### Performance Metrics
- **Transfer Time**: Time to pass context
- **Validation Time**: Time to validate context
- **Recovery Time**: Time to recover lost context
- **User Satisfaction**: Stakeholder satisfaction scores

## Integration Points

### Tool Integration
- **IDE Plugins**: Context validation in development environment
- **CI/CD Pipelines**: Automated context validation
- **Project Management**: Context tracking in task management
- **Documentation Systems**: Automated context documentation

### System Integration
- **Memory Bank**: Central context storage
- **Task Management**: Context-aware task creation
- **Mode System**: Context-preserving mode transitions
- **Quality System**: Context validation integration

## Maintenance and Evolution

### Regular Updates
- **Template Updates**: Keep templates current
- **Process Refinement**: Improve passing mechanisms
- **Training Updates**: Update team training
- **Tool Updates**: Maintain integration tools

### Version Management
- **Context Versioning**: Track context versions
- **Compatibility Matrix**: Ensure version compatibility
- **Migration Paths**: Support context migration
- **Deprecation Notices**: Handle deprecated context

## Best Practices

### For Context Creators
1. **Be Comprehensive**: Include all relevant context
2. **Use Templates**: Always use standardized templates
3. **Validate Early**: Validate context before transfer
4. **Document Assumptions**: Clearly document assumptions

### For Context Consumers
1. **Verify Completeness**: Check for all required elements
2. **Validate Relevance**: Ensure context is still relevant
3. **Ask Questions**: Seek clarification for unclear context
4. **Provide Feedback**: Report context quality issues

### For System Administrators
1. **Monitor Health**: Track context system health
2. **Update Regularly**: Keep system current
3. **Train Users**: Ensure team understands system
4. **Gather Feedback**: Collect and act on feedback