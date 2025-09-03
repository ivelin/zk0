# zk0 [zee-ˈkō]


zk0 is an Open Source humanoid built collaboratively by a community of contributors of code, compute and data. 

<img src="https://github.com/user-attachments/assets/9dd876a0-6668-4b9f-ad0d-94a540353418" width=100>


# Why

Why do we need yet another humanoid project? 
As of 2024 AI technology has [advanced enough to speculate](https://x.com/elonmusk/status/1786367513137233933) that within a decade most people will have their own humanoid buddy. By some estimates humanoids will become $100 Trillion market (10B humanoids * $10,000 per unit).

[Today's leading closed source humanoid](https://x.com/Tesla_Optimus/status/1846294753144361371) is trained on [100,000 GPU farm](https://nvidianews.nvidia.com/news/spectrum-x-ethernet-networking-xai-colossus) with real world data collected from millions of cars labeled by able human drivers. 
This is an enormous scale of compute and data that is hard to compete with as a centrazlied entity. 
However it would be interesting to see if a decentralized approach might produce useful results over time.
On the chance that proprietary humanoids ever go rogue, it would be nice to have open source alternatives.

# How

zk0 is composed of several major building blocks:
- Generative AI: 
  * [HuggingFace LeRobot](https://huggingface.co/lerobot) for the Open Source 3D printed robot parts and end-to-end vision language action models.
- Federated Learning: 
  * [Flower](https://flower.ai/) for collaborative training of AI models
- Zero Knowledge Proofs:
  * [EZKL](https://ezkl.xyz/) for verification of contributed model checkpoints trained on local data.


# Memory Bank System

The zk0 Memory Bank System is a critical component that ensures consistency, prevents context loss, and maintains project knowledge across all development activities. It serves as a centralized repository for project context, rules, workflows, and quality standards.

## Purpose and Benefits

The memory bank system addresses the fundamental challenge of maintaining consistency in complex, multi-step development workflows. Its primary purposes include:

- **Context Loss Prevention**: Captures and preserves critical project context that might otherwise be lost between development sessions
- **Consistency Assurance**: Ensures all team members and AI assistants work within the same project constraints and guidelines
- **Knowledge Preservation**: Documents project-specific rules, patterns, and decisions for future reference
- **Quality Control**: Provides checklists and standards to maintain code quality and project integrity

## Structure Overview

The memory bank is organized in the `.kilocode/memory-bank/` directory with the following structure:

```
.kilocode/memory-bank/
├── README.md                 # System overview and key constraints
├── contexts/                 # Context management documentation
│   ├── context-inheritance.md
│   ├── context-passing.md
│   └── task-context-template.md
├── modes/                    # Mode-specific guidelines
│   ├── architect-mode.md
│   └── code-mode.md
├── quality/                  # Quality assurance checklists
│   ├── implementation-checklist.md
│   ├── post-implementation-checklist.md
│   └── pre-implementation-checklist.md
├── rules/                    # Project rules and constraints
│   ├── project-constraints.md
│   └── technical-specifications.md
└── workflows/                # Workflow patterns and procedures
```

## Usage Guidelines

### Referencing the Memory Bank

When working on tasks, always reference the memory bank to ensure consistency:

1. **Check Project Constraints**: Review `.kilocode/memory-bank/rules/project-constraints.md` before starting any work
2. **Follow Mode Guidelines**: Consult the appropriate mode documentation in `.kilocode/memory-bank/modes/`
3. **Use Quality Checklists**: Apply pre and post-implementation checklists from `.kilocode/memory-bank/quality/`
4. **Maintain Context**: Use context templates from `.kilocode/memory-bank/contexts/` for complex tasks

### Key Project Constraints (Always Include)

1. Work ONLY under `~/zk0/flower/examples/quickstart-smolvla`
2. No changes to sibling or parent directories
3. Use conda environment "zk0"
4. Focus on SmolVLA model and SO-100 real-world datasets
5. Borrow structure from quickstart-lerobot but adapt for new requirements

## Maintenance Procedures

To keep the memory bank current as the project evolves:

### Regular Updates
- **Weekly Review**: Review and update project constraints as requirements change
- **After Major Changes**: Update relevant documentation when project structure or goals shift
- **New Patterns**: Document new workflow patterns or coding standards as they emerge

### Adding New Content
1. Identify the appropriate subdirectory (contexts/, modes/, quality/, rules/, workflows/)
2. Create or update markdown files following existing naming conventions
3. Ensure content is clear, actionable, and aligned with project goals
4. Update the main memory bank README.md if new categories are added

### Quality Assurance
- Maintain consistent formatting across all documentation
- Regularly audit for outdated information
- Ensure all team members can easily find and understand the content

## Examples of Context Loss Prevention

### Example 1: Project Scope Consistency
**Without Memory Bank**: A developer might accidentally work on the wrong directory or use incorrect datasets, leading to wasted effort and integration issues.

**With Memory Bank**: By checking `project-constraints.md`, the developer immediately sees they must work only in `~/zk0/flower/examples/quickstart-smolvla` and focus on SmolVLA models, preventing scope drift.

### Example 2: Quality Standards Maintenance
**Without Memory Bank**: Code quality might vary across different development sessions as standards aren't consistently applied.

**With Memory Bank**: Pre and post-implementation checklists ensure consistent quality standards are applied to all work, maintaining project integrity.

### Example 3: Knowledge Transfer
**Without Memory Bank**: New team members or AI assistants might miss critical project context, leading to repeated mistakes.

**With Memory Bank**: All project-specific knowledge, constraints, and patterns are documented and easily accessible, enabling smooth onboarding and consistent execution.

The memory bank system transforms zk0 from a collection of code into a well-documented, maintainable project that can scale effectively with community contributions.

# Directory Structure

```shell
zk0
│
├── lerobot             # clone of lerobot repo: 
│                       #    https://github.com/huggingface/lerobot.git
│
├── flower              # clone of flower repo: 
│                       #     https://github.com/adap/flower/tree/main/examples/quickstart-huggingface
│
├── federate            # This project's core source files: 
│
│
└── README.md           # This README file
```

