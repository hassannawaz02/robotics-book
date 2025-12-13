# Implementation Plan: Frontend Fixes — Home + Navbar

**Branch**: `1-frontend-home-navbar-fixes` | **Date**: 2025-12-12 | **Spec**: [specs/1-frontend-home-navbar-fixes/spec.md](../specs/1-frontend-home-navbar-fixes/spec.md)
**Input**: Feature specification from `/specs/[###-feature-name]/spec.md`

**Note**: This template is filled in by the `/sp.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

Update the navigation bar by removing the GitHub button and replacing Tutorial with Home. Update the homepage to display 4 module cards with icons/images and descriptions in a clean, professional layout. Ensure all changes are compatible with Docusaurus v3.

## Technical Context

**Language/Version**: JavaScript/TypeScript, Docusaurus v3 compatible
**Primary Dependencies**: Docusaurus framework, React components
**Storage**: N/A (frontend only changes)
**Testing**: Visual verification, browser compatibility testing
**Target Platform**: Web browsers (Chrome, Firefox, Safari, Edge)
**Project Type**: Web frontend (Docusaurus documentation site)
**Performance Goals**: No performance degradation, maintain fast load times
**Constraints**: Must maintain responsive design, Docusaurus v3 compatibility
**Scale/Scope**: Single page updates (navbar and homepage), affects UI/UX

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- **Content-First Approach**: ✅ Changes enhance user navigation and module discovery experience
- **Modular Architecture**: ✅ Changes are isolated to theme components, maintaining modularity
- **Authentication-Driven Access Control**: N/A (no authentication changes)
- **Docusaurus Integration Excellence**: ✅ Changes follow Docusaurus patterns and maintain compatibility
- **Simulation-Ready Design**: N/A (frontend UI changes)
- **AI-Enhanced Learning Experience**: ✅ Improved navigation helps users find content more easily

## Project Structure

### Documentation (this feature)

```text
specs/1-frontend-home-navbar-fixes/
├── plan.md              # This file (/sp.plan command output)
├── research.md          # Phase 0 output (/sp.plan command)
├── data-model.md        # Phase 1 output (/sp.plan command)
├── quickstart.md        # Phase 1 output (/sp.plan command)
├── contracts/           # Phase 1 output (/sp.plan command)
└── tasks.md             # Phase 2 output (/sp.tasks command - NOT created by /sp.plan)
```

### Source Code (repository root)

```text
robotics-book/src/theme/
├── Navbar/
│   └── index.js         # Navigation bar component
├── Homepage/
│   └── index.js         # Homepage component
└── components/
    └── ModuleCard.js    # Module card component
```

**Structure Decision**: Docusaurus theme customization approach - components will be created in the robotics-book/src/theme directory to override default Docusaurus components.

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| [e.g., 4th project] | [current need] | [why 3 projects insufficient] |
| [e.g., Repository pattern] | [specific problem] | [why direct DB access insufficient] |