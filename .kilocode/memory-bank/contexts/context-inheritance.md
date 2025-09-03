# Context Inheritance Rules

## Overview

Context inheritance ensures that critical information, constraints, and requirements are automatically passed from parent tasks to subtasks, preventing loss of important details during task decomposition.

## Inheritance Levels

### Level 1: Mandatory Constraints
**Always inherited - no exceptions**
- Project working directory (`~/zk0/flower/examples/quickstart-smolvla`)
- Environment requirements (conda "zk0")
- Technical focus (SmolVLA + SO-100)
- Scope limitations (no changes to siblings/parents)

### Level 2: Task Context
**Inherited for all subtasks**
- Success criteria from parent task
- Key technical decisions
- Critical assumptions
- Risk assessments

### Level 3: Operational Context
**Inherited based on relevance**
- Current system state
- Recent changes
- Known issues
- Performance baselines

## Inheritance Mechanism

### Automatic Inheritance
When creating subtasks, automatically include:

1. **Project Constraints Block**
   ```
   ## Project Constraints (MANDATORY)
   - [ ] Work ONLY under `~/zk0/flower/examples/quickstart-smolvla`
   - [ ] No changes to sibling or parent directories
   - [ ] Use conda environment "zk0"
   - [ ] Focus on SmolVLA model and SO-100 real-world datasets
   - [ ] Borrow structure from quickstart-lerobot but adapt for new requirements
   ```

2. **Parent Task Reference**
   - Parent task ID
   - Parent success criteria
   - Parent context summary

3. **Technical Context**
   - Current working state
   - Dependencies
   - Known constraints

### Manual Inheritance
Explicitly pass:

1. **Business Logic**
   - Why this subtask exists
   - How it contributes to parent goal
   - Success criteria alignment

2. **Technical Decisions**
   - Architecture choices
   - Design patterns
   - Implementation constraints

3. **Risk Context**
   - Known risks
   - Mitigation strategies
   - Fallback plans

## Context Validation

### Pre-Subtask Validation
Before starting a subtask:
- [ ] All Level 1 constraints included
- [ ] Parent task context summarized
- [ ] Success criteria defined
- [ ] Dependencies identified

### During Execution
Regular validation:
- [ ] Context remains relevant
- [ ] Assumptions still valid
- [ ] Constraints not violated
- [ ] Progress aligns with parent goals

### Post-Subtask Validation
After completion:
- [ ] Context preserved for next subtask
- [ ] Lessons learned captured
- [ ] Memory bank updated
- [ ] Handover documentation complete

## Context Loss Prevention

### Risk Identification
High-risk areas for context loss:
- Mode transitions
- Task decomposition
- External interruptions
- Time gaps between subtasks

### Prevention Strategies
1. **Template Usage**: Always use context templates
2. **Documentation**: Document all assumptions
3. **Validation Points**: Regular context checkpoints
4. **Backup Context**: Maintain context backups

## Recovery Procedures

### If Context is Lost
1. **Stop Work**: Pause current activity
2. **Reference Memory Bank**: Use stored context
3. **Reconstruct**: Rebuild missing context
4. **Document Gap**: Record what was lost
5. **Resume**: Continue with full context

### Context Reconstruction
1. **Parent Task Review**: Review parent task documentation
2. **Memory Bank Check**: Check stored context
3. **Stakeholder Consultation**: Consult with task creator
4. **Validation**: Verify reconstructed context

## Best Practices

### For Task Creators
- Use context templates consistently
- Be explicit about inheritance requirements
- Document all assumptions
- Include validation checkpoints

### For Task Executors
- Verify inherited context completeness
- Ask questions about unclear context
- Document additional context discovered
- Update memory bank with findings

### For Mode Transitions
- Preserve all critical context
- Document transition rationale
- Validate target mode compatibility
- Include rollback procedures

## Quality Metrics

### Context Completeness Score
- Level 1 constraints: 100% (mandatory)
- Level 2 context: >90% (recommended)
- Level 3 context: >70% (situational)

### Context Preservation Rate
- Track context loss incidents
- Measure recovery time
- Monitor prevention effectiveness

## Maintenance

- **Regular Review**: Audit context inheritance effectiveness
- **Template Updates**: Keep templates current
- **Training**: Ensure team understands inheritance rules
- **Feedback Loop**: Incorporate lessons learned