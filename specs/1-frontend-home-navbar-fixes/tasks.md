---
description: "Task list for frontend fixes - navbar and homepage"
---

# Tasks: Frontend Fixes — Home + Navbar

**Input**: Design documents from `/specs/1-frontend-home-navbar-fixes/`
**Prerequisites**: plan.md (required), spec.md (required for user stories), research.md, data-model.md

**Tests**: The examples below include test tasks. Tests are OPTIONAL - only include them if explicitly requested in the feature specification.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

- **Docusaurus project**: `robotics-book/src/`, `robotics-book/docusaurus.config.ts`

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and basic structure

- [ ] T001 Verify project structure and dependencies in robotics-book/
- [ ] T002 [P] Locate existing navbar configuration in docusaurus.config.ts
- [ ] T003 [P] Locate existing homepage components in robotics-book/src/pages/index.tsx and robotics-book/src/components/HomepageFeatures/

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [ ] T004 Verify current navbar configuration and identify GitHub button and Tutorial link locations
- [ ] T005 [P] Verify current homepage structure and identify module card locations
- [ ] T006 [P] Identify appropriate robotics module icons/images for homepage cards

**Checkpoint**: Foundation ready - user story implementation can now begin in parallel

---

## Phase 3: User Story 1 - Update Navigation Bar (Priority: P1) 🎯 MVP

**Goal**: Remove the "GitHub" button from the navigation bar completely and replace the "Tutorial" link with "Home"

**Independent Test**: The navigation bar should display "Home" instead of "Tutorial" and no "GitHub" button should appear anywhere in the navigation

### Implementation for User Story 1

- [ ] T007 Update navbar configuration in robotics-book/docusaurus.config.ts to replace "Tutorial" with "Home" (depends on T004)
- [ ] T008 Update navbar configuration in robotics-book/docusaurus.config.ts to remove GitHub button completely (depends on T004)
- [ ] T009 Test navbar changes by running local Docusaurus server and verifying navigation

**Checkpoint**: At this point, User Story 1 should be fully functional and testable independently

---

## Phase 4: User Story 2 - Update Homepage Layout (Priority: P1)

**Goal**: Display 4 module cards on the homepage with icons/images and short descriptions in a clean, professional layout

**Independent Test**: The homepage should display 4 distinct module cards with icons/images and short descriptions in a clean, professional layout

### Implementation for User Story 2

- [ ] T010 [P] Create or update module data in robotics-book/src/components/HomepageFeatures/index.tsx to define 4 robotics modules
- [ ] T011 [P] Update module icons/images in robotics-book/src/components/HomepageFeatures/index.tsx for each of the 4 modules
- [ ] T012 [P] Add short descriptions for each module in robotics-book/src/components/HomepageFeatures/index.tsx
- [ ] T013 Update styling if needed in robotics-book/src/components/HomepageFeatures/styles.module.css for clean professional layout
- [ ] T014 Test homepage changes by running local Docusaurus server and verifying module cards display correctly

**Checkpoint**: At this point, User Stories 1 AND 2 should both work independently

---

## Phase 5: User Story 3 - Apply Clean Professional Styling (Priority: P2)

**Goal**: Ensure all styling appears clean and professional and is compatible with Docusaurus v3

**Independent Test**: The styling should look clean and professional across all updated components, with no visual regressions

### Implementation for User Story 3

- [ ] T015 [P] Review and refine styling in robotics-book/src/components/HomepageFeatures/styles.module.css for consistency
- [ ] T016 [P] Verify responsive design works across different screen sizes for updated components
- [ ] T017 Test Docusaurus v3 compatibility by building and running the site to ensure no errors

**Checkpoint**: All user stories should now be independently functional

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories

- [ ] T018 [P] Verify all changes work together harmoniously
- [ ] T019 Documentation updates if needed
- [ ] T020 Code cleanup and refactoring if needed
- [ ] T021 Performance optimization if needed
- [ ] T022 Run final validation by testing site functionality across browsers

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Stories (Phase 3+)**: All depend on Foundational phase completion
  - User stories can then proceed in parallel (if staffed)
  - Or sequentially in priority order (P1 → P2 → P3)
- **Polish (Final Phase)**: Depends on all desired user stories being complete

### User Story Dependencies

- **User Story 1 (P1)**: Can start after Foundational (Phase 2) - No dependencies on other stories
- **User Story 2 (P1)**: Can start after Foundational (Phase 2) - No dependencies on other stories
- **User Story 3 (P2)**: Can start after User Stories 1 and 2 are complete

### Within Each User Story

- Core implementation before integration
- Story complete before moving to next priority

### Parallel Opportunities

- All Setup tasks marked [P] can run in parallel
- All Foundational tasks marked [P] can run in parallel (within Phase 2)
- User Story 1 and User Story 2 can run in parallel after foundational phase
- All implementation tasks within a story marked [P] can run in parallel

---

## Parallel Example: User Stories 1 and 2

```bash
# Launch User Story 1 and User Story 2 in parallel after foundational phase:
Task: "Update navbar configuration in robotics-book/docusaurus.config.ts to replace Tutorial with Home"
Task: "Create or update module data in robotics-book/src/components/HomepageFeatures/index.tsx to define 4 robotics modules"
Task: "Update module icons/images in robotics-book/src/components/HomepageFeatures/index.tsx for each of the 4 modules"
```

---

## Implementation Strategy

### MVP First (User Stories 1 and 2)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (CRITICAL - blocks all stories)
3. Complete Phase 3: User Story 1 - Update Navigation Bar
4. Complete Phase 4: User Story 2 - Update Homepage Layout
5. **STOP and VALIDATE**: Test both user stories independently
6. Deploy/demo if ready

### Incremental Delivery

1. Complete Setup + Foundational → Foundation ready
2. Add User Story 1 → Test independently → Deploy/Demo
3. Add User Story 2 → Test independently → Deploy/Demo
4. Add User Story 3 → Test independently → Deploy/Demo
5. Each story adds value without breaking previous stories

### Parallel Team Strategy

With multiple developers:

1. Team completes Setup + Foundational together
2. Once Foundational is done:
   - Developer A: User Story 1 (Navigation Bar)
   - Developer B: User Story 2 (Homepage Layout)
   - Developer C: User Story 3 (Styling)
3. Stories complete and integrate independently

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story for traceability
- Each user story should be independently completable and testable
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
- Avoid: vague tasks, same file conflicts, cross-story dependencies that break independence