# Context Packaging

**Created**: 2025-09-06
**Last Updated**: 2025-09-06
**Version**: 1.0.0
**Author**: Kilo Code

## Core Components - Context Packaging

**How to package:**
- Use standardized templates
- Include validation checklists
- Add metadata (timestamps, versions)
- Compress for efficiency when needed

## Context Formats

### Template-Based Format
```markdown
## Project Constraints (MANDATORY)
See [.kilocode/rules/memory-bank/project-constraints.md](.kilocode/rules/memory-bank/project-constraints.md) for the complete list of project constraints.

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

## Subtask Creation Protocol - Context Packaging

2. **Context Packaging**
   - Use task context template
   - Include all mandatory constraints
   - Add subtask-specific context