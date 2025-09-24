# Context Management Strategies

**Created**: 2025-09-10
**Last Updated**: 2025-09-10
**Version**: 1.0.0
**Author**: Kilo Code

## Session Transition Template

### Pre-Transition Checklist
- [ ] Document current task completion status
- [ ] Identify next immediate steps
- [ ] Capture critical technical decisions
- [ ] Note any unresolved issues or blockers
- [ ] Update memory bank with recent changes

### Essential Context to Preserve
1. **Environment Setup**: Conda environments, dependencies, configurations
2. **Project State**: Current completion status, recent achievements
3. **Technical Context**: Key technologies, constraints, standards
4. **Immediate Focus**: What was being worked on when transition occurs
5. **Critical Paths**: Important file locations, command patterns

### Fresh Session Starter Format
```
[Memory Bank: Active]
Project: [project name]
Status: [current state summary]
Last Updated: [date]
Key Technical Context:
- Environment: [conda env, key dependencies]
- Architecture: [core components, patterns]
- Constraints: [critical rules, standards]
Current Focus: [immediate next task]
```

## Context Health Monitoring

### Indicators of Approaching Limits
- Increased frequency of "earlier in our conversation" references
- Responses taking longer to formulate
- More frequent memory bank consultations
- Less focused or comprehensive responses
- User having to remind me of recent context

### Optimal Transition Points
- After completing major features or milestones
- Before starting complex debugging sessions
- When conversation exceeds 50 messages
- When working time exceeds 2-3 hours continuously
- When introducing new complex concepts

## Recovery Strategies

### If Context Loss Occurs
1. **Immediate Assessment**: Identify what context was lost
2. **Memory Bank Review**: Consult all memory bank files
3. **Gap Documentation**: Document what needs reconstruction
4. **User Verification**: Confirm understanding with user
5. **Reconstruction**: Build missing context from available sources

### Prevention Measures
- Regular memory bank updates during long sessions
- Clear documentation of complex decisions
- Proactive transition suggestions
- Session summaries at logical breakpoints

## Integration with Subtask Workflow

### Subtask Context Management
When breaking down complex tasks:
1. **Pre-Branch**: Document current context state
2. **Branch Creation**: Create focused subtask with essential context only
3. **Branch Completion**: Update memory bank with results
4. **Main Task Resume**: Load updated context for continuation

### Mode-Specific Optimization
- **Architect Mode**: Use for context planning and memory bank updates
- **Code Mode**: Keep focused on implementation within fresh context windows
- **Debug Mode**: Benefit from isolated context for problem-solving
- **Test Mode**: Clean context for reliable test execution
- **Ask Mode**: Ideal for context management discussions and planning

### Branch Management
- **Clean Handoffs**: Use subtask boundaries as natural transition points
- **Context Chunking**: Break complex projects into documented chunks
- **Dependency Mapping**: Track relationships between context chunks
- **Selective Inheritance**: Load only relevant context for specific branches

## Performance Optimization

### Context Window Efficiency
- **Essential Only**: Include only critical context in new sessions
- **Progressive Loading**: Load additional context as needed
- **Memory Bank Caching**: Use memory bank for persistent knowledge
- **Reference Optimization**: Minimize back-references through better documentation

### Quality Maintenance
- **Fresh Focus**: New sessions maintain high response quality
- **Error Reduction**: Less context confusion leads to fewer mistakes
- **Productivity Boost**: Optimal context windows improve efficiency
- **Scalability**: Handle increasingly complex projects effectively

## Memory Bank Update Rules for Context Transitions

### When to Update Memory Bank
**MANDATORY Updates:**
- **Before Session Transitions**: Update context.md with current task status
- **After Major Milestones**: Document completed features and decisions
- **When Critical Decisions Made**: Capture architectural choices and trade-offs
- **Before Complex Debugging**: Document current state and known issues
- **After Subtask Completion**: Update progress and results

**RECOMMENDED Updates:**
- **Regular Checkpoints**: Every 20-30 messages during long sessions
- **Before Mode Switches**: Document current focus and handoff requirements
- **When New Patterns Discovered**: Add to tasks.md for future reference
- **After Error Resolution**: Document solutions and prevention measures

### What to Update During Transitions

#### context.md Updates
```markdown
## Recent Changes
- [ ] Document current task completion status
- [ ] Note any unresolved issues or blockers
- [ ] Record critical technical decisions made
- [ ] Update project status and next steps
- [ ] Document any new patterns or approaches discovered
```

#### tasks.md Updates (if applicable)
```markdown
## [New Task Pattern]
**Last performed:** [current date]
**Context:** [transition context]
**Files modified:** [list relevant files]
**Key decisions:** [important choices made]
**Lessons learned:** [what worked/didn't work]
```

#### project-constraints.md Updates (rare)
- Only update if new constraints or standards are established
- Document any process improvements discovered

### Update Process Checklist
- [ ] **Assess Changes**: Identify what has changed since last update
- [ ] **Prioritize Information**: Focus on actionable, future-relevant details
- [ ] **Update context.md**: Current status and immediate next steps
- [ ] **Update tasks.md**: If new patterns or reusable workflows discovered
- [ ] **Verify Completeness**: Ensure critical context is preserved
- [ ] **Test Continuity**: Verify new session can effectively continue work

### Quality Standards for Updates
- **Actionable Focus**: Include only information needed for future continuation
- **Concise Documentation**: Avoid verbose descriptions of completed work
- **Decision Rationale**: Document why important choices were made
- **Pattern Recognition**: Note reusable approaches for similar future tasks
- **Version Tracking**: Update version numbers and dates appropriately

## Implementation Guidelines

### Transition Triggers
- **Proactive**: Suggest transitions before performance degradation
- **User-Initiated**: Honor user requests for session transitions
- **Automatic**: Trigger on conversation length/complexity thresholds
- **Milestone-Based**: Transition after completing major deliverables

### Documentation Standards
- **Structured Summaries**: Use consistent format for transition summaries
- **Essential Information**: Focus on actionable context, not exhaustive history
- **Clear Next Steps**: Always identify immediate continuation points
- **Version Control**: Track context evolution across sessions

## Success Metrics

### Effectiveness Indicators
- **Reduced Errors**: Fewer context-related mistakes
- **Improved Quality**: More focused and comprehensive responses
- **Better Continuity**: Smoother transitions between sessions
- **Enhanced Productivity**: Faster completion of complex tasks

### Continuous Improvement
- **Feedback Loop**: Monitor effectiveness and gather user feedback
- **Process Refinement**: Adjust strategies based on usage patterns
- **Tool Enhancement**: Improve automation and integration
- **Best Practice Evolution**: Update guidelines based on experience