# Data Model for AI-Driven Physical AI & Humanoid Robotics Book

## User Entity
- **id**: UUID (primary key)
- **email**: String (unique, required, validated)
- **password_hash**: String (required, stored securely)
- **role**: Enum ['learner', 'admin'] (default: 'learner')
- **first_name**: String (optional)
- **last_name**: String (optional)
- **created_at**: DateTime (auto-generated)
- **updated_at**: DateTime (auto-generated)
- **is_active**: Boolean (default: true)
- **last_login**: DateTime (nullable)

## Module Entity
- **id**: UUID (primary key)
- **title**: String (required)
- **description**: Text (required)
- **module_number**: Integer (unique, 1-4)
- **learning_objectives**: JSON (array of strings)
- **estimated_duration_hours**: Integer (required)
- **is_published**: Boolean (default: false)
- **created_at**: DateTime (auto-generated)
- **updated_at**: DateTime (auto-generated)

## Lesson Entity
- **id**: UUID (primary key)
- **module_id**: UUID (foreign key to Module)
- **title**: String (required)
- **content**: Text (required, markdown format)
- **lesson_number**: Integer (within module)
- **duration_minutes**: Integer (estimated)
- **is_free**: Boolean (default: false)
- **is_published**: Boolean (default: false)
- **prerequisites**: JSON (array of lesson IDs)
- **assets**: JSON (array of asset file paths)
- **code_examples**: JSON (array of code example file paths)
- **created_at**: DateTime (auto-generated)
- **updated_at**: DateTime (auto-generated)

## UserProgress Entity
- **id**: UUID (primary key)
- **user_id**: UUID (foreign key to User)
- **lesson_id**: UUID (foreign key to Lesson)
- **status**: Enum ['not_started', 'in_progress', 'completed'] (default: 'not_started')
- **progress_percentage**: Integer (0-100, default: 0)
- **last_accessed**: DateTime (nullable)
- **completed_at**: DateTime (nullable)
- **created_at**: DateTime (auto-generated)
- **updated_at**: DateTime (auto-generated)

## Asset Entity
- **id**: UUID (primary key)
- **lesson_id**: UUID (foreign key to Lesson)
- **filename**: String (required)
- **file_path**: String (required)
- **asset_type**: Enum ['image', 'diagram', 'video', 'code', 'pdf'] (required)
- **description**: String (optional)
- **created_at**: DateTime (auto-generated)

## ChatSession Entity
- **id**: UUID (primary key)
- **user_id**: UUID (foreign key to User, nullable for anonymous)
- **title**: String (auto-generated from first query)
- **created_at**: DateTime (auto-generated)
- **updated_at**: DateTime (auto-generated)

## ChatMessage Entity
- **id**: UUID (primary key)
- **session_id**: UUID (foreign key to ChatSession)
- **role**: Enum ['user', 'assistant'] (required)
- **content**: Text (required)
- **timestamp**: DateTime (auto-generated)
- **metadata**: JSON (for future AI model tracking)

## Relationships
- User has many UserProgress records
- Module has many Lessons
- Lesson has many UserProgress records
- Lesson has many Assets
- ChatSession has many ChatMessages
- User has many ChatSessions (optional)

## Validation Rules
- User email must be valid email format
- User password must meet security requirements (min 8 chars, mixed case, numbers, symbols)
- Module number must be between 1 and 4
- Lesson number within module must be unique
- Progress percentage must be between 0 and 100
- Asset file paths must exist in the assets directory

## State Transitions
- User: active <-> inactive (admin action)
- Module: draft -> published (admin action)
- Lesson: draft -> published (admin action)
- UserProgress: not_started -> in_progress -> completed (user actions)