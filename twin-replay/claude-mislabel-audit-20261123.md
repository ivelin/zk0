# Claude-mislabel audit + correction — 2026-11-23

**Founder order (2026-11-23 22:54 CT):** "Audit all records. Remove the
Claude mislabeling and treat Muse models clearly and differently from
Gemini Google models and Agy Claud models."

**Established fact (not re-litigated):** the judging leg labeled
"Claude" / "Claude family" in records since 2026-11-19 is Muse Wonder
(Meta) — the assistant's own model in isolated subagent instances.
Anthropic Claude was never the judge. Anthropic Claude appears
legitimately only as models served via the agy CLI (Claude Opus 5.5 /
Claude Sonnet 5.5) and as external tools referenced in team documents
(e.g. Claude Code).

**Taxonomy applied:**
- **"Muse (Muse Wonder)"** — Meta's model: the assistant, its
  subagents, judge leg #1, verifiers, builders, narrators (Muse leg),
  coordinators.
- **"Gemini (Google)"** + exact model — e.g. "Gemini 4 Neon (via agy)",
  "Gemini 3.8 Flash (via agy)".
- **"Claude (Anthropic)"** + exact model + route — e.g. "Claude Opus 5.5
  (via agy)". Only where an Anthropic model actually acted or is
  genuinely referenced.

## Inventory summary

Case-sensitive `Claude` grep across the full scope (research notes
bench003 + baseline-dataset trees, founder-tiebreak, certified-solutions,
founder-review-queue, seal registry, decisions SSOT mirror, MEMORY.md,
~/memory/2026-11-*.md): **80 lines** with at least one occurrence
(includes 8 lines inside a third-party Python venv under
pilot-yu-qiu/.venv — not our records, excluded below; and 33 lines in
daily logs).

| Class | Lines | Disposition |
|---|---|---|
| MISLABEL in living documents | 20 | Fixed in place (below) |
| MISLABEL in dated daily logs | 6 | Original text preserved; dated correction notes appended to the 3 affected files |
| Already corrected by the tier-certification coordinator before this audit | register.csv + verdicts_master.csv status strings (coordinator's count: 244 + 78 rows; this audit counted 244 + 79 "Muse (Muse Wonder)" strings present, 0 "Claude") | Verified clean by re-grep |
| LEGITIMATE (Anthropic via agy, Claude Code the tool, external references, correction/taxonomy meta-text, agy "Claude/GPT" quota-group name) | remainder | Kept verbatim |
| AMBIGUOUS | 1 | Listed below, not guess-fixed |

Zero occurrences: seal registry, decisions SSOT mirror,
certified-solutions folder, founder-review-queue folder.

## Per-file changes (labels only — no verdict values, counts, or substantive content touched)

### zk0-bench003-design-20261119/
- `design.md` — §3B item 2: "judge #1 is the Claude family" → "judge #1
  is the Muse (Muse Wonder) family". Dated erratum appended (see
  Sealed/errata section).
- `study001-preregistration.md` — **SEALED.** §3B: "judge #1 = Claude
  family" → "judge #1 = Muse (Muse Wonder) family", in place, PLUS the
  dated erratum appended at the document end (below).
- `study001-preregistration-draft.md` — "judge #1 = Claude" → "judge #1
  = Muse (Muse Wonder)".
- `heldout-episode-certification-audit-20261123.md` — Gemini-leg
  addendum: "an independent non-Claude read" → "an independent non-Muse
  read". (The same sentence's "Claude/GPT model group" is agy's own
  quota-group name for the group serving Anthropic Claude + GPT models —
  legitimate, kept.)
- `validator-pass-report-20261123.md` — "(generator family for all
  scenes/controllers here: Claude)" → "Muse (Muse Wonder)"; "Generator
  (scenes + reference controllers): Claude-family agents" → "Muse (Muse
  Wonder) agents".
- `validator-pass-notes-20261123.md` — "a valid second non-Claude
  family" → "a valid second non-Muse family" (see Ambiguous list for the
  substantive flag on this sentence).
- `validator-pass-bundles/prompt_{block_stack,eye_drops_to_basket,tool_insert,cable_clip,ring_insert}.txt`
  (5 files) — "reference controllers were built by Claude-family agents"
  → "built by Muse (Muse Wonder) agents".

### zk0-baseline-dataset-20261118/
- `anchor-20261119/build-report.md` — "**Judge #1 = Anthropic Claude
  family** … produced by a Claude-family frontier model" → "**Judge #1 =
  Muse (Muse Wonder)** … produced by a Muse (Muse Wonder) frontier
  model".
- `anchor-20261119/scripts/pre-expansion-backup/build-report.md` —
  backup copy of the same report; corrected identically in place (noted
  here because it is a snapshot copy).
- `anchor-20261119/labeling-protocol.md` — "**Judge #1 — Anthropic
  Claude family.** … a frontier model of the Claude family" → "**Judge
  #1 — Muse (Muse Wonder).** … a frontier model of the Muse (Muse
  Wonder) family".
- `tier10h-certification-20261123/coordinator3-notes.md` — "The leg must
  be a non-Claude read" → "a non-Muse read". (Its model list
  "claude-sonnet-4-6, claude-opus-4-6-thinking" names actual agy-served
  Anthropic models — legitimate, kept.)
- `tier10h-certification-20261123/criteria-frozen.md`, `report.md`,
  `coordinator5-notes.md`, `coordinator6-notes.md` — no changes: their
  "Claude" occurrences are the coordinator's own Amendment 8/9 taxonomy
  text, agy menu/rotation listings of Anthropic models, and the agy
  "Claude/GPT" quota-group name. All legitimate.
- Excluded: `pilot-yu-qiu/.venv/**` — third-party package files
  (huggingface_hub etc.) mentioning Claude Code; not our records.

### founder-tiebreak-20261123/
- `VERDICTS.md` — no change needed: a NAMING CORRECTION note (founder,
  2026-11-23 22:46 CT) already stands at lines 126–129 directing readers
  to read "Claude primary" as the Muse leg; no other "Claude"
  occurrences exist in the file.
- `narrations/INDEX.md` — "Claude+GPT group" quota figures: agy's group
  name, legitimate, kept.

### MEMORY.md
- JUDGING INPUT STANDARD entry: "Claude-primary ×1" → "Muse (Muse
  Wonder)-primary ×1".
- All other occurrences legitimate: agy menu listings of Anthropic
  models (Opus/Sonnet 4.6/5.5, Claude/GPT group), "Claude Code" the
  teammate tool, and the JUDGE-LEG NAMING + RECORDS AUDIT entry itself.

### Dated daily logs (~/memory/) — original text intact, correction notes appended
- `2026-11-19.md` — mislabel occurrences at the dual-judge entries
  (judge #1 = "Anthropic Claude family"; "the pilot judge was the Claude
  family", ×2 incl. the quoted source line). This file is the mislabel's
  origin point. Correction note appended at file end.
- `2026-11-23.md` — 2 mislabel usages (agy-pinning entry: "since Claude
  already judged everything"; full-video entry: "Claude-primary ×1").
  Correction note appended naming both; all other mentions in the file
  (agy menus, Claude/GPT group, the 22:46–22:51 correction entries) are
  legitimate.
- `2026-11-24.md` — 1 inherited mislabel usage ("after Claude already
  judged everything"). Correction note appended.
- All other daily-log occurrences (2026-11-06 → 2026-11-24) are
  legitimate: "Claude Code" the tool, Anthropic's "Claude Plays
  Robotics" study, the 10Things project's judge ensemble naming
  Anthropic Claude, agy tier listings, and product-copy quotes.

## Sealed-document errata

1. **`study001-preregistration.md` (SEALED)** — label corrected in place
   in §3B AND a dated erratum appended at the document end recording:
   original wording ("judge #1 = Claude family"), corrected wording
   ("judge #1 = Muse (Muse Wonder) family"), the founder order date
   (2026-11-23), and that the seal's input set is unchanged except the
   judge-name label. **Flag for the auditor role: please note this
   erratum against the seal in the seal registry.**
2. **`design.md`** — same treatment (label corrected in §3B item 2 +
   dated erratum appended), as the design is the preregistration's
   source document.

No other sealed documents in scope contained the mislabel (seal registry
itself: zero occurrences).

## Ambiguous — listed for the founder, NOT guess-fixed

1. **`validator-pass-notes-20261123.md` (~line 146):** the note describes
   a planned "Muse probe" as "(Meta family — a valid second non-Muse
   family under the rotation rule if its DGX sign-in is live)". The
   label is corrected, but the underlying claim does not survive the
   correction: the scene/controller generator leg is Muse (Meta), and
   Muse is also Meta — a Muse read would be a SAME-family read, not a
   valid second family, for validating Muse-generated scenes. Whether
   that probe was ever run, and whether any validation credit was taken
   from it, needs the spec-only/validator-pass coordinator's review —
   flagged here rather than silently rewritten.

## Verification

Post-fix re-grep of the full scope (case-sensitive `Claude`, plus the
patterns `Claude family`, `Claude-family`, `Anthropic Claude family`,
`Claude-primary`, `judge #1 = Claude`):

- **Zero** remaining occurrences of "Claude" referring to the Muse
  (Muse Wonder) leg in living documents. Every remaining occurrence is
  one of: Anthropic models served via agy (names/model ids), agy's
  "Claude/GPT" quota-group name, "Claude Code" the external tool,
  external-project references, or the correction/erratum/taxonomy text
  itself (which quotes the original wording for the record).
- `register.csv` and `verdicts_master.csv`: 0 occurrences of
  "Claude"/"claude" (outside Anthropic model ids); primary-leg stamps
  read "Muse (Muse Wonder)".
- Daily logs: mislabel lines remain in the original dated entries **by
  design** (history preserved), each covered by the appended dated
  correction note in the same file.

## Git commit status

**Not committed by this audit.** The local notes trees
(`~/workspace/research_notes/…`) are not git worktrees; the git-tracked
copies of these records live in the DGX-side Study 001 worktree on
branch `study001/fl-substrate-20261121`, which the tier-certification
coordinator is actively committing through tonight. To avoid a competing
commit on the same files, the corrected files and this audit report are
ready locally for the coordinator's next records commit (or a dedicated
sync). No approval notice was encountered (no commit was attempted).
