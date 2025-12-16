# Data Model: Module 3 - AI-Robot Brain (NVIDIA Isaac™)

## Entities

### Lesson Content
- **id**: string (unique identifier for the lesson)
- **title**: string (lesson title)
- **content**: string (Markdown content of the lesson)
- **module**: string (module identifier, e.g., "module-3")
- **lessonNumber**: number (position in the module sequence)
- **type**: string (lesson, tutorial, exercise)
- **assets**: array of strings (references to diagrams, images)
- **codeExamples**: array of strings (references to Python code files)
- **createdAt**: timestamp
- **updatedAt**: timestamp

### User Access
- **userId**: string (unique user identifier)
- **lessonId**: string (reference to lesson content)
- **accessLevel**: string (learner, admin)
- **jwtToken**: string (JWT token for authentication)
- **lastAccessed**: timestamp
- **progress**: number (percentage of lesson completed)

### Module Structure
- **moduleId**: string (unique module identifier, e.g., "module-3")
- **title**: string (module title: "Module 3 – AI-Robot Brain")
- **description**: string (module description: "Isaac Sim, Isaac ROS, Nav2")
- **position**: number (module position in the book, e.g., 3)
- **lessons**: array of Lesson Content references
- **assets**: object (path to diagrams, images, code examples)
- **dependencies**: array of strings (required software/environment)

### Code Examples
- **id**: string (unique identifier)
- **moduleId**: string (reference to parent module)
- **lessonId**: string (reference to parent lesson)
- **filename**: string (name of the Python file)
- **content**: string (Python code content)
- **description**: string (what the code demonstrates)
- **language**: string (programming language, e.g., "python")
- **createdAt**: timestamp
- **updatedAt**: timestamp

## Relationships
- Module Structure contains multiple Lesson Content items
- Lesson Content may reference multiple Code Examples
- Lesson Content may reference multiple assets (diagrams, images)
- User Access links users to specific Lesson Content with access permissions

## Validation Rules
- Lesson content must have valid Markdown syntax
- User access requires valid JWT token
- Module structure must include all 5 required lesson files (01-intro.md through 05-exercises.md)
- Code examples must be in Python format for Isaac ROS scripts
- Assets paths must be valid and accessible

## State Transitions
- Lesson Content: draft → reviewed → published
- User Access: unauthenticated → authenticated → authorized
- Module Structure: planned → in-progress → complete